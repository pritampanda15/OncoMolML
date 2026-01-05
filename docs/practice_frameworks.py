#!/usr/bin/env python3
"""
OncoMolML - Interactive Practice Script
Run this to understand how each framework works!

Usage:
    python practice_frameworks.py

Each section can be run independently.
"""

import numpy as np
print("=" * 60)
print("OncoMolML Framework Practice")
print("=" * 60)

# ==============================================================================
# SECTION 1: PyTorch Basics
# ==============================================================================
print("\n" + "=" * 60)
print("SECTION 1: PyTorch")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    print(f"✓ PyTorch {torch.__version__} loaded")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # Create a simple model
    class SimpleDrugEncoder(nn.Module):
        """Minimal drug response model to understand PyTorch."""
        def __init__(self, gene_dim=100, mol_dim=256, hidden=64):
            super().__init__()  # ALWAYS call this!
            
            # nn.Sequential chains layers
            self.gene_encoder = nn.Sequential(
                nn.Linear(gene_dim, hidden),
                nn.LayerNorm(hidden),  # Normalizes across features
                nn.GELU(),             # Smooth activation
                nn.Dropout(0.1)        # Regularization
            )
            
            self.mol_encoder = nn.Sequential(
                nn.Linear(mol_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU()
            )
            
            # Final prediction head
            self.predictor = nn.Linear(hidden * 2, 1)
        
        def forward(self, gene_expr, mol_fp):
            # Forward pass - defines computation
            g = self.gene_encoder(gene_expr)
            m = self.mol_encoder(mol_fp)
            combined = torch.cat([g, m], dim=-1)  # Concatenate features
            return self.predictor(combined)
    
    # Test the model
    model = SimpleDrugEncoder()
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create fake data
    batch_size = 8
    gene_expr = torch.randn(batch_size, 100)  # [batch, gene_dim]
    mol_fp = torch.randn(batch_size, 256)     # [batch, mol_dim]
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradients for inference
        predictions = model(gene_expr, mol_fp)
    
    print(f"Input shapes: genes={gene_expr.shape}, molecules={mol_fp.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze().numpy()[:3]}...")
    
    # Training step example
    model.train()  # Set to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    fake_targets = torch.randn(batch_size, 1)
    
    optimizer.zero_grad()                      # Clear old gradients
    pred = model(gene_expr, mol_fp)            # Forward pass
    loss = F.mse_loss(pred, fake_targets)      # Compute loss
    loss.backward()                            # Backpropagation
    optimizer.step()                           # Update weights
    
    print(f"\nTraining step completed, loss: {loss.item():.4f}")
    
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")

# ==============================================================================
# SECTION 2: TensorFlow Basics
# ==============================================================================
print("\n" + "=" * 60)
print("SECTION 2: TensorFlow")
print("=" * 60)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    
    print(f"✓ TensorFlow {tf.__version__} loaded")
    print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    
    class SimpleVariantPredictor(Model):
        """Minimal variant effect predictor to understand TensorFlow."""
        def __init__(self, seq_len=50, hidden=64):
            super().__init__()
            
            # Conv1D for sequence patterns
            self.conv1 = layers.Conv1D(hidden, 7, padding='same', activation='relu')
            self.conv2 = layers.Conv1D(hidden, 5, padding='same', activation='relu')
            
            # Pool and classify
            self.pool = layers.GlobalAveragePooling1D()
            self.dense = layers.Dense(3)  # Benign, VUS, Pathogenic
        
        @tf.function  # Compiles to optimized graph!
        def call(self, x, training=None):
            h = self.conv1(x)
            h = self.conv2(h)
            h = self.pool(h)
            return self.dense(h)
    
    # Test the model
    model = SimpleVariantPredictor()
    
    # One-hot encoded DNA sequence [batch, seq_len, 4]
    batch_size = 8
    seq_len = 50
    fake_seq = tf.random.uniform((batch_size, seq_len, 4))
    
    # First call triggers graph compilation
    predictions = model(fake_seq, training=False)
    
    print(f"\nModel parameters: {model.count_params()}")
    print(f"Input shape: {fake_seq.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions (logits): {predictions[0].numpy()}")
    
    # Custom training step with GradientTape
    @tf.function
    def train_step(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True)
            loss = tf.reduce_mean(loss)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    fake_labels = tf.random.uniform((batch_size,), 0, 3, dtype=tf.int32)
    
    loss = train_step(model, fake_seq, fake_labels, optimizer)
    print(f"\nTraining step completed, loss: {loss.numpy():.4f}")
    
except ImportError as e:
    print(f"✗ TensorFlow not installed: {e}")

# ==============================================================================
# SECTION 3: JAX Basics
# ==============================================================================
print("\n" + "=" * 60)
print("SECTION 3: JAX")
print("=" * 60)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    
    print(f"✓ JAX {jax.__version__} loaded")
    print(f"  Devices: {jax.devices()}")
    
    # JAX is FUNCTIONAL - no classes with state!
    
    # 1. JIT compilation
    @jit
    def tanimoto_similarity(fp1, fp2):
        """Compute Tanimoto coefficient between two fingerprints."""
        intersection = jnp.sum(fp1 * fp2)
        union = jnp.sum(fp1) + jnp.sum(fp2) - intersection
        return intersection / (union + 1e-10)
    
    # Test Tanimoto
    fp1 = jnp.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=jnp.float32)
    fp2 = jnp.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=jnp.float32)
    sim = tanimoto_similarity(fp1, fp2)
    print(f"\nTanimoto similarity: {sim:.4f}")
    
    # 2. Automatic differentiation
    @jit
    def lennard_jones(coords, sigma=3.4, epsilon=1.0):
        """LJ potential - DIFFERENTIABLE for optimization!"""
        diff = coords[:, None, :] - coords[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
        mask = 1 - jnp.eye(coords.shape[0])
        r6 = (sigma / dist) ** 6
        return 0.5 * jnp.sum(4 * epsilon * (r6**2 - r6) * mask)
    
    # Get gradient function automatically!
    grad_lj = grad(lennard_jones)
    
    coords = jax.random.normal(jax.random.PRNGKey(0), (5, 3)) * 5
    energy = lennard_jones(coords)
    forces = -grad_lj(coords)  # Force = -gradient of potential
    
    print(f"\nLJ potential energy: {energy:.4f}")
    print(f"Forces shape: {forces.shape}")
    print(f"Force on atom 0: {forces[0]}")
    
    # 3. vmap for automatic batching
    @jit
    def process_single(x):
        return jnp.sum(x ** 2)
    
    # Automatically batch!
    process_batch = vmap(process_single)
    
    batch = jax.random.normal(jax.random.PRNGKey(1), (100, 10))
    results = process_batch(batch)
    print(f"\nvmap: Processed batch of {batch.shape[0]} items")
    print(f"Results shape: {results.shape}")
    
    # 4. Key JAX gotcha: IMMUTABLE arrays
    arr = jnp.array([1, 2, 3, 4, 5])
    # arr[0] = 10  # THIS WOULD FAIL!
    arr = arr.at[0].set(10)  # This works - returns NEW array
    print(f"\nImmutable update: {arr}")
    
except ImportError as e:
    print(f"✗ JAX not installed: {e}")

# ==============================================================================
# SECTION 4: Numba Basics
# ==============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Numba")
print("=" * 60)

try:
    import numba
    from numba import njit, prange
    
    print(f"✓ Numba {numba.__version__} loaded")
    print(f"  Threads: {numba.get_num_threads()}")
    
    # 1. Basic JIT
    @njit(cache=True)  # cache=True saves compiled code to disk
    def gc_content(sequence):
        """
        Compute GC content.
        sequence: ASCII-encoded numpy array (A=65, C=67, G=71, T=84)
        """
        gc = 0
        total = 0
        for base in sequence:
            if base == 71 or base == 67:  # G or C
                gc += 1
            if base != 78:  # Not N
                total += 1
        return gc / total if total > 0 else 0.0
    
    # Test GC content
    seq = np.array([65, 67, 71, 84, 67, 71, 65, 84], dtype=np.uint8)  # ACGTCGAT
    gc = gc_content(seq)
    print(f"\nGC content of ACGTCGAT: {gc:.2%}")
    
    # 2. Parallel processing
    @njit(parallel=True, cache=True)
    def parallel_gc_batch(sequences):
        """Compute GC for multiple sequences in parallel."""
        n = sequences.shape[0]
        results = np.zeros(n)
        for i in prange(n):  # prange = parallel range
            results[i] = gc_content(sequences[i])
        return results
    
    # Generate batch of sequences
    n_seqs = 10000
    seq_len = 150
    bases = np.array([65, 67, 71, 84], dtype=np.uint8)  # ACGT
    sequences = np.random.choice(bases, (n_seqs, seq_len))
    
    import time
    
    # Warmup (first call compiles)
    _ = parallel_gc_batch(sequences[:10])
    
    # Benchmark
    start = time.time()
    gc_values = parallel_gc_batch(sequences)
    elapsed = time.time() - start
    
    print(f"\nProcessed {n_seqs} sequences in {elapsed*1000:.2f} ms")
    print(f"Speed: {n_seqs/elapsed:.0f} sequences/second")
    print(f"Mean GC: {gc_values.mean():.2%}")
    
    # 3. Smith-Waterman local alignment
    @njit(cache=True)
    def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-2):
        """Local alignment score."""
        m, n = len(seq1), len(seq2)
        H = np.zeros((m + 1, n + 1), dtype=np.int32)
        max_score = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                diag = H[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
                H[i, j] = max(0, diag, H[i-1, j] + gap, H[i, j-1] + gap)
                max_score = max(max_score, H[i, j])
        
        return max_score
    
    s1 = np.array([65, 67, 71, 84, 65], dtype=np.uint8)  # ACGTA
    s2 = np.array([65, 67, 84, 84, 65], dtype=np.uint8)  # ACTTA
    score = smith_waterman(s1, s2)
    print(f"\nSmith-Waterman score (ACGTA vs ACTTA): {score}")
    
except ImportError as e:
    print(f"✗ Numba not installed: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("SUMMARY: When to Use Each Framework")
print("=" * 60)

print("""
┌─────────────┬────────────────────────────────────────┐
│ Framework   │ Best For                               │
├─────────────┼────────────────────────────────────────┤
│ PyTorch     │ Research, complex architectures,       │
│             │ attention mechanisms, multi-task       │
├─────────────┼────────────────────────────────────────┤
│ TensorFlow  │ Production deployment, @tf.function,   │
│             │ mixed precision, serving               │
├─────────────┼────────────────────────────────────────┤
│ JAX         │ Differentiable computing, grad(),      │
│             │ vmap batching, scientific computing    │
├─────────────┼────────────────────────────────────────┤
│ Numba       │ NumPy speedup, parallel loops,         │
│             │ NGS processing, no GPU needed          │
└─────────────┴────────────────────────────────────────┘
""")

print("Practice complete! Now try modifying these examples. 🎯")
