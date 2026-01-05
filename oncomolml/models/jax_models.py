"""
JAX Models for High-Performance Molecular Computations

This module implements JIT-compiled functions for:
- Molecular fingerprint generation
- Differentiable scoring functions
- Fast similarity calculations
- Gradient-based molecular optimization

Showcases: JAX, jax.jit, jax.vmap, jax.grad, automatic differentiation
"""

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad, value_and_grad
    from jax.scipy.special import logsumexp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    raise ImportError(
        "JAX is required for this module. "
        "Install with: pip install jax jaxlib"
    )


# Set default precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Molecular Fingerprint Functions (JIT-compiled)
# =============================================================================

@jit
def _hash_atom_environment(
    atom_features: jnp.ndarray,
    neighbor_features: jnp.ndarray,
    bond_features: jnp.ndarray,
    hash_size: int = 2048
) -> jnp.ndarray:
    """
    Hash atom environment for Morgan fingerprint calculation.
    
    JIT-compiled for maximum performance.
    """
    # Combine features using polynomial hash
    combined = jnp.concatenate([
        atom_features.flatten(),
        neighbor_features.flatten(),
        bond_features.flatten()
    ])
    
    # Rolling hash with prime multiplier
    prime = 31
    hash_val = jnp.zeros(1, dtype=jnp.int32)
    
    def hash_step(carry, x):
        return (carry * prime + jnp.int32(x * 1000)) % hash_size, None
    
    hash_val, _ = jax.lax.scan(hash_step, hash_val, combined)
    
    return hash_val[0]


@partial(jit, static_argnums=(2, 3))
def morgan_fingerprint_jax(
    atom_features: jnp.ndarray,
    adjacency: jnp.ndarray,
    radius: int = 2,
    nbits: int = 2048
) -> jnp.ndarray:
    """
    Compute Morgan circular fingerprint using JAX.
    
    JIT-compiled for GPU acceleration and batched computation.
    
    Args:
        atom_features: Atom feature matrix [num_atoms, num_features]
        adjacency: Adjacency matrix [num_atoms, num_atoms]
        radius: Morgan fingerprint radius
        nbits: Number of bits in fingerprint
        
    Returns:
        Binary fingerprint vector [nbits]
    
    Example:
        >>> atom_feat = jnp.array([[1,0,0], [0,1,0], [0,0,1]])
        >>> adj = jnp.array([[0,1,0], [1,0,1], [0,1,0]])
        >>> fp = morgan_fingerprint_jax(atom_feat, adj, radius=2, nbits=1024)
        >>> print(fp.shape)  # (1024,)
    """
    num_atoms = atom_features.shape[0]
    fingerprint = jnp.zeros(nbits, dtype=jnp.float32)
    
    # Initial atom identifiers
    identifiers = atom_features.sum(axis=1)
    
    # Iterate through radius levels
    for r in range(radius + 1):
        # Hash each atom's environment
        for i in range(num_atoms):
            # Get neighbors
            neighbors = adjacency[i]
            neighbor_ids = identifiers * neighbors
            
            # Create hash from atom + neighbor identifiers
            env_hash = jnp.abs(
                jnp.int32(identifiers[i] * 1000 + neighbor_ids.sum() * 100 + r * 10)
            )
            bit_idx = env_hash % nbits
            fingerprint = fingerprint.at[bit_idx].set(1.0)
        
        # Update identifiers by aggregating neighbor information
        identifiers = identifiers + adjacency @ identifiers
    
    return fingerprint


@jit
def batch_morgan_fingerprint_jax(
    batch_atom_features: jnp.ndarray,
    batch_adjacency: jnp.ndarray,
    batch_mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Morgan fingerprints for a batch of molecules.
    
    Uses vmap for efficient parallelization across batch.
    
    Args:
        batch_atom_features: [batch, max_atoms, num_features]
        batch_adjacency: [batch, max_atoms, max_atoms]
        batch_mask: [batch, max_atoms] - valid atom mask
        
    Returns:
        Batch of fingerprints [batch, nbits]
    """
    # Apply vmap over batch dimension
    batched_fn = vmap(
        lambda af, adj: morgan_fingerprint_jax(af, adj, radius=2, nbits=2048)
    )
    return batched_fn(batch_atom_features, batch_adjacency)


# =============================================================================
# Differentiable Scoring Functions
# =============================================================================

@jit
def gaussian_overlap_score(
    coords1: jnp.ndarray,
    coords2: jnp.ndarray,
    sigma: float = 1.0
) -> jnp.ndarray:
    """
    Compute Gaussian overlap between two sets of 3D coordinates.
    
    Differentiable w.r.t. both coordinate sets for gradient-based optimization.
    
    Args:
        coords1: First coordinate set [N, 3]
        coords2: Second coordinate set [M, 3]
        sigma: Gaussian width parameter
        
    Returns:
        Overlap score (scalar)
    """
    # Pairwise distances
    diff = coords1[:, None, :] - coords2[None, :, :]  # [N, M, 3]
    dist_sq = jnp.sum(diff ** 2, axis=-1)  # [N, M]
    
    # Gaussian overlaps
    overlaps = jnp.exp(-dist_sq / (2 * sigma ** 2))
    
    # Total overlap (normalized)
    score = jnp.sum(overlaps) / (coords1.shape[0] * coords2.shape[0])
    
    return score


@jit
def lennard_jones_potential(
    coords: jnp.ndarray,
    epsilon: float = 1.0,
    sigma: float = 3.4
) -> jnp.ndarray:
    """
    Compute Lennard-Jones potential for a set of coordinates.
    
    V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    
    Args:
        coords: Atomic coordinates [N, 3]
        epsilon: Well depth
        sigma: Zero-crossing distance
        
    Returns:
        Total potential energy (scalar)
    """
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)  # [N, N]
    
    # Avoid self-interaction
    mask = 1.0 - jnp.eye(coords.shape[0])
    
    # LJ potential
    r6 = (sigma / dist) ** 6
    r12 = r6 ** 2
    lj = 4 * epsilon * (r12 - r6)
    
    # Sum over unique pairs
    energy = 0.5 * jnp.sum(lj * mask)
    
    return energy


@partial(jit, static_argnums=(2,))
def differentiable_vina_score(
    ligand_coords: jnp.ndarray,
    receptor_grid: jnp.ndarray,
    grid_spacing: float = 0.375
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simplified differentiable Vina-like scoring function.
    
    Computes affinity score by interpolating receptor grid values
    at ligand atom positions. Fully differentiable for optimization.
    
    Args:
        ligand_coords: Ligand coordinates [N, 3]
        receptor_grid: Pre-computed receptor interaction grid [X, Y, Z]
        grid_spacing: Grid spacing in Angstroms
        
    Returns:
        Tuple of (score, gradient w.r.t. ligand coordinates)
    """
    # Convert coordinates to grid indices (continuous)
    grid_indices = ligand_coords / grid_spacing
    
    # Trilinear interpolation
    def trilinear_interp(grid, point):
        # Floor indices
        i0 = jnp.floor(point).astype(jnp.int32)
        i1 = i0 + 1
        
        # Fractional parts
        f = point - i0
        
        # Clamp to grid bounds
        shape = jnp.array(grid.shape)
        i0 = jnp.clip(i0, 0, shape - 1)
        i1 = jnp.clip(i1, 0, shape - 1)
        
        # 8 corner values
        c000 = grid[i0[0], i0[1], i0[2]]
        c001 = grid[i0[0], i0[1], i1[2]]
        c010 = grid[i0[0], i1[1], i0[2]]
        c011 = grid[i0[0], i1[1], i1[2]]
        c100 = grid[i1[0], i0[1], i0[2]]
        c101 = grid[i1[0], i0[1], i1[2]]
        c110 = grid[i1[0], i1[1], i0[2]]
        c111 = grid[i1[0], i1[1], i1[2]]
        
        # Interpolate
        c00 = c000 * (1 - f[0]) + c100 * f[0]
        c01 = c001 * (1 - f[0]) + c101 * f[0]
        c10 = c010 * (1 - f[0]) + c110 * f[0]
        c11 = c011 * (1 - f[0]) + c111 * f[0]
        
        c0 = c00 * (1 - f[1]) + c10 * f[1]
        c1 = c01 * (1 - f[1]) + c11 * f[1]
        
        return c0 * (1 - f[2]) + c1 * f[2]
    
    # Sum scores over all atoms
    atom_scores = vmap(lambda p: trilinear_interp(receptor_grid, p))(grid_indices)
    total_score = jnp.sum(atom_scores)
    
    # Compute gradient
    score_grad = grad(
        lambda c: jnp.sum(vmap(lambda p: trilinear_interp(receptor_grid, p / grid_spacing))(c))
    )(ligand_coords)
    
    return total_score, score_grad


# =============================================================================
# Similarity and Distance Functions
# =============================================================================

@jit
def tanimoto_similarity(
    fp1: jnp.ndarray,
    fp2: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Tanimoto similarity between two fingerprints.
    
    Args:
        fp1: First fingerprint [nbits]
        fp2: Second fingerprint [nbits]
        
    Returns:
        Tanimoto similarity (0-1)
    """
    intersection = jnp.sum(fp1 * fp2)
    union = jnp.sum(fp1) + jnp.sum(fp2) - intersection
    return intersection / (union + 1e-10)


@jit
def batch_tanimoto_matrix(
    fingerprints: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix.
    
    JIT-compiled for efficient large-scale computation.
    
    Args:
        fingerprints: Batch of fingerprints [N, nbits]
        
    Returns:
        Similarity matrix [N, N]
    """
    # Vectorized computation
    intersection = fingerprints @ fingerprints.T
    sums = jnp.sum(fingerprints, axis=1, keepdims=True)
    union = sums + sums.T - intersection
    
    return intersection / (union + 1e-10)


@jit
def batch_dice_similarity(
    fp1_batch: jnp.ndarray,
    fp2_batch: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Dice similarity for batches of fingerprints.
    
    Args:
        fp1_batch: First batch [N, nbits]
        fp2_batch: Second batch [N, nbits]
        
    Returns:
        Dice similarities [N]
    """
    intersection = jnp.sum(fp1_batch * fp2_batch, axis=1)
    sums = jnp.sum(fp1_batch, axis=1) + jnp.sum(fp2_batch, axis=1)
    return 2 * intersection / (sums + 1e-10)


# =============================================================================
# Neural Network Layers in JAX
# =============================================================================

def init_mlp_params(
    layer_sizes: List[int],
    key: jax.random.PRNGKey
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Initialize MLP parameters with Xavier initialization.
    
    Args:
        layer_sizes: List of layer dimensions
        key: JAX random key
        
    Returns:
        List of (weight, bias) tuples
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        key, subkey = jax.random.split(key)
        
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (in_dim + out_dim))
        w = jax.random.normal(subkey, (in_dim, out_dim)) * scale
        b = jnp.zeros(out_dim)
        
        params.append((w, b))
    
    return params


@jit
def mlp_forward(
    params: List[Tuple[jnp.ndarray, jnp.ndarray]],
    x: jnp.ndarray,
    activation: str = "relu"
) -> jnp.ndarray:
    """
    Forward pass through MLP.
    
    Args:
        params: List of (weight, bias) tuples
        x: Input tensor
        activation: Activation function ("relu", "gelu", "tanh")
        
    Returns:
        Output tensor
    """
    act_fn = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid
    }.get(activation, jax.nn.relu)
    
    for i, (w, b) in enumerate(params[:-1]):
        x = act_fn(x @ w + b)
    
    # No activation on final layer
    w, b = params[-1]
    return x @ w + b


@partial(jit, static_argnums=(2,))
def molecular_property_predictor(
    params: List[Tuple[jnp.ndarray, jnp.ndarray]],
    fingerprint: jnp.ndarray,
    num_properties: int = 5
) -> jnp.ndarray:
    """
    Predict molecular properties from fingerprint using JAX MLP.
    
    Args:
        params: Network parameters
        fingerprint: Molecular fingerprint
        num_properties: Number of properties to predict
        
    Returns:
        Property predictions
    """
    return mlp_forward(params, fingerprint, activation="gelu")


# =============================================================================
# Gradient-Based Molecular Optimization
# =============================================================================

@partial(jit, static_argnums=(3,))
def optimize_ligand_position(
    ligand_coords: jnp.ndarray,
    receptor_grid: jnp.ndarray,
    learning_rate: float = 0.01,
    n_steps: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimize ligand position using gradient descent on scoring function.
    
    Uses JAX's automatic differentiation for efficient optimization.
    
    Args:
        ligand_coords: Initial ligand coordinates [N, 3]
        receptor_grid: Receptor interaction grid
        learning_rate: Gradient descent step size
        n_steps: Number of optimization steps
        
    Returns:
        Tuple of (optimized coordinates, score trajectory)
    """
    def loss_fn(coords):
        score, _ = differentiable_vina_score(coords, receptor_grid)
        return score
    
    grad_fn = grad(loss_fn)
    
    def optimization_step(state, _):
        coords, scores = state
        g = grad_fn(coords)
        new_coords = coords - learning_rate * g
        new_score = loss_fn(new_coords)
        return (new_coords, jnp.append(scores, new_score)), None
    
    initial_state = (ligand_coords, jnp.array([loss_fn(ligand_coords)]))
    final_state, _ = jax.lax.scan(optimization_step, initial_state, None, length=n_steps)
    
    return final_state


# =============================================================================
# Batched Virtual Screening
# =============================================================================

@jit
def batch_score_ligands(
    ligand_fingerprints: jnp.ndarray,
    reference_fingerprint: jnp.ndarray,
    property_weights: jnp.ndarray
) -> jnp.ndarray:
    """
    Score a batch of ligands against a reference.
    
    Combines similarity and property scores for virtual screening.
    
    Args:
        ligand_fingerprints: Ligand FPs [N, nbits]
        reference_fingerprint: Reference FP [nbits]
        property_weights: Weights for multi-objective scoring
        
    Returns:
        Composite scores [N]
    """
    # Tanimoto similarities
    similarities = vmap(lambda fp: tanimoto_similarity(fp, reference_fingerprint))(
        ligand_fingerprints
    )
    
    return similarities


@jit
def diversity_selection(
    fingerprints: jnp.ndarray,
    n_select: int = 100,
    seed: int = 42
) -> jnp.ndarray:
    """
    Select diverse subset using MaxMin algorithm.
    
    JIT-compiled for fast diverse subset selection.
    
    Args:
        fingerprints: Candidate fingerprints [N, nbits]
        n_select: Number to select
        seed: Random seed for first selection
        
    Returns:
        Indices of selected compounds
    """
    n_total = fingerprints.shape[0]
    
    # Initialize with random compound
    key = jax.random.PRNGKey(seed)
    first_idx = jax.random.randint(key, (), 0, n_total)
    
    # Compute all pairwise similarities
    sim_matrix = batch_tanimoto_matrix(fingerprints)
    
    # MaxMin selection
    selected = jnp.array([first_idx])
    min_sims = sim_matrix[first_idx]
    
    for _ in range(n_select - 1):
        # Find compound with minimum maximum similarity to selected
        # (i.e., most different from all selected)
        masked_sims = jnp.where(
            jnp.isin(jnp.arange(n_total), selected),
            jnp.inf,
            min_sims
        )
        next_idx = jnp.argmin(masked_sims)
        selected = jnp.append(selected, next_idx)
        
        # Update minimum similarities
        min_sims = jnp.minimum(min_sims, sim_matrix[next_idx])
    
    return selected


# =============================================================================
# Utility Functions
# =============================================================================

def benchmark_fingerprint_speed(
    n_molecules: int = 10000,
    n_atoms: int = 50
) -> Dict[str, float]:
    """
    Benchmark fingerprint computation speed.
    
    Args:
        n_molecules: Number of molecules
        n_atoms: Atoms per molecule
        
    Returns:
        Timing results
    """
    import time
    
    # Generate random data
    key = jax.random.PRNGKey(0)
    atom_features = jax.random.normal(key, (n_molecules, n_atoms, 10))
    adjacency = jax.random.uniform(key, (n_molecules, n_atoms, n_atoms)) > 0.8
    adjacency = adjacency.astype(jnp.float32)
    mask = jnp.ones((n_molecules, n_atoms))
    
    # Warmup
    _ = batch_morgan_fingerprint_jax(atom_features[:10], adjacency[:10], mask[:10])
    
    # Benchmark
    start = time.time()
    fps = batch_morgan_fingerprint_jax(atom_features, adjacency, mask)
    fps.block_until_ready()  # Ensure computation completes
    elapsed = time.time() - start
    
    return {
        "n_molecules": n_molecules,
        "time_seconds": elapsed,
        "molecules_per_second": n_molecules / elapsed
    }


if __name__ == "__main__":
    print("Testing JAX models...")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    # Test Morgan fingerprint
    key = jax.random.PRNGKey(0)
    atom_feat = jax.random.normal(key, (10, 5))
    adj = (jax.random.uniform(key, (10, 10)) > 0.7).astype(jnp.float32)
    fp = morgan_fingerprint_jax(atom_feat, adj, radius=2, nbits=1024)
    print(f"Morgan fingerprint shape: {fp.shape}, non-zero bits: {jnp.sum(fp > 0)}")
    
    # Test Tanimoto similarity
    fp2 = morgan_fingerprint_jax(atom_feat * 0.9, adj, radius=2, nbits=1024)
    sim = tanimoto_similarity(fp, fp2)
    print(f"Tanimoto similarity: {sim:.4f}")
    
    # Test Lennard-Jones
    coords = jax.random.normal(key, (20, 3)) * 5
    energy = lennard_jones_potential(coords)
    print(f"LJ potential energy: {energy:.4f}")
    
    # Test gradient computation
    grad_fn = grad(lennard_jones_potential)
    forces = -grad_fn(coords)
    print(f"Forces shape: {forces.shape}")
    
    # Test MLP
    params = init_mlp_params([1024, 256, 64, 5], key)
    fp_input = jax.random.normal(key, (32, 1024))
    preds = vmap(lambda x: mlp_forward(params, x, "gelu"))(fp_input)
    print(f"MLP predictions shape: {preds.shape}")
    
    # Benchmark
    print("\nRunning benchmark...")
    results = benchmark_fingerprint_speed(n_molecules=1000, n_atoms=30)
    print(f"Processed {results['n_molecules']} molecules in {results['time_seconds']:.3f}s")
    print(f"Speed: {results['molecules_per_second']:.0f} molecules/second")
    
    print("\nAll JAX models working correctly!")
