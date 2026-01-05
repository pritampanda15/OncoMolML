# ML Frameworks Interview Prep Guide
## OncoMolML Portfolio Project

---

# 1. PyTorch Deep Dive

## Core Concepts You MUST Know

### What is PyTorch?
**Answer**: PyTorch is a dynamic deep learning framework with eager execution (define-by-run). Unlike TensorFlow 1.x which used static graphs, PyTorch builds the computation graph on-the-fly, making debugging easier.

### Key Components
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

| Component | Purpose |
|-----------|---------|
| `torch.Tensor` | Multi-dimensional array (like NumPy but GPU-enabled) |
| `nn.Module` | Base class for all neural network layers |
| `nn.Sequential` | Container for stacking layers |
| `F` (functional) | Stateless operations (activations, losses) |
| `torch.optim` | Optimizers (Adam, SGD, etc.) |

## How I Used PyTorch in OncoMolML

### 1. DrugResponseNet - Multi-head Attention for Drug Sensitivity
```python
class DrugResponseNet(nn.Module):
    """
    Predicts IC50/drug response from:
    - Gene expression (978 L1000 landmark genes)
    - Molecular fingerprints (2048-bit Morgan)
    """
    def __init__(self, gene_dim=978, mol_dim=2048, hidden_dim=512):
        super().__init__()  # ALWAYS call parent init
        
        # Separate encoders for each modality
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # Better than BatchNorm for variable batches
            nn.GELU(),                   # Smoother than ReLU
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for gene-drug interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True  # Important! [batch, seq, features]
        )
    
    def forward(self, gene_expr, mol_fp):
        gene_encoded = self.gene_encoder(gene_expr)
        mol_encoded = self.mol_encoder(mol_fp)
        
        # Stack as sequence for attention
        combined = torch.stack([gene_encoded, mol_encoded], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        
        return self.prediction_head(attn_out.flatten(1))
```

**Interview Q**: Why use `nn.LayerNorm` instead of `nn.BatchNorm1d`?
**Answer**: LayerNorm normalizes across features for each sample independently, so it works with any batch size (even batch=1). BatchNorm normalizes across the batch dimension, requiring larger batches and behaving differently during training vs inference.

### 2. Multi-task Learning with Uncertainty
```python
class MolPropertyPredictor(nn.Module):
    """Predicts multiple ADMET properties with uncertainty."""
    
    def forward(self, x, return_uncertainty=False):
        features = self.shared_encoder(x)
        
        predictions = torch.cat([
            head(features) for head in self.property_heads
        ], dim=-1)
        
        if return_uncertainty:
            # Heteroscedastic uncertainty - model learns its own confidence
            uncertainties = torch.cat([
                F.softplus(head(features))  # Ensure positive
                for head in self.uncertainty_heads
            ], dim=-1)
            return predictions, uncertainties
        
        return predictions
```

**Interview Q**: What is heteroscedastic uncertainty?
**Answer**: Uncertainty that varies with the input. The model predicts both the value AND its confidence. High uncertainty means "I'm not sure about this prediction." Useful for ADMET where some molecular structures are harder to predict.

### 3. Custom Loss Functions
```python
class DrugResponseLoss(nn.Module):
    """Loss with optional uncertainty weighting."""
    
    def forward(self, pred, target, uncertainty=None):
        if uncertainty is not None:
            # Negative log-likelihood of Gaussian with learned variance
            # -log N(y | pred, σ²) = 0.5 * (log(σ²) + (y-pred)²/σ²)
            loss = 0.5 * (torch.log(uncertainty) + (pred - target)**2 / uncertainty)
            return loss.mean()
        return F.huber_loss(pred, target)  # Robust to outliers
```

## Common Interview Questions

**Q1**: Explain `model.train()` vs `model.eval()`
**A**: `train()` enables dropout and uses batch statistics for BatchNorm. `eval()` disables dropout and uses running statistics. Always switch modes!
```python
model.train()   # For training
model.eval()    # For inference
with torch.no_grad():  # Also disable gradient computation for inference
    output = model(input)
```

**Q2**: What's the difference between `nn.Linear` and `F.linear`?
**A**: `nn.Linear` is a class with learnable parameters (weight, bias). `F.linear` is a stateless function - you must pass weights yourself.

**Q3**: How do you move a model to GPU?
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)  # Don't forget the data!
```

**Q4**: Explain `torch.cat` vs `torch.stack`
**A**: 
- `cat`: Concatenates along existing dimension - shapes must match except concat dim
- `stack`: Creates NEW dimension - all shapes must match exactly
```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
torch.cat([a, b], dim=0)   # Shape: [6, 4]
torch.stack([a, b], dim=0) # Shape: [2, 3, 4] - new dim added
```

---

# 2. TensorFlow Deep Dive

## Core Concepts

### TensorFlow 2.x vs 1.x
- **TF 2.x**: Eager execution by default (like PyTorch)
- **`@tf.function`**: Converts Python to optimized graph for production
- **Keras API**: High-level model building integrated into TF

### Key Components
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
```

## How I Used TensorFlow in OncoMolML

### 1. VariantEffectPredictor - Transformer for Genomics
```python
class VariantEffectPredictor(Model):
    """
    Predicts pathogenicity of genetic variants.
    Input: One-hot encoded DNA (101bp context around variant)
    Output: Benign / VUS / Pathogenic classification
    """
    
    def __init__(self, sequence_length=101, filters=256):
        super().__init__()
        
        # Multi-scale dilated convolutions capture different context sizes
        self.conv_blocks = [
            ConvBlock(filters, kernel_size=7, dilation_rate=1),   # Local
            ConvBlock(filters, kernel_size=5, dilation_rate=2),   # Medium
            ConvBlock(filters, kernel_size=3, dilation_rate=4),   # Wide
            ConvBlock(filters, kernel_size=3, dilation_rate=8),   # Very wide
        ]
        
        # Transformer for long-range dependencies
        self.attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=filters // 8
        )
        
        self.classifier = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(3)  # 3 classes
        ])
    
    @tf.function  # CRITICAL: Compiles to optimized graph
    def call(self, x, training=None):
        # x shape: [batch, 101, 4] (one-hot DNA)
        h = x
        for conv in self.conv_blocks:
            h = conv(h, training=training)
        
        # Self-attention
        h = h + self.attention(h, h, training=training)
        
        return self.classifier(h)
```

**Interview Q**: Why use dilated convolutions?
**Answer**: Dilated convolutions expand receptive field without increasing parameters. A dilation_rate=4 with kernel_size=3 sees 9 positions spread across 11bp instead of consecutive 3bp. Essential for capturing long-range genomic context.

### 2. DNA Sequence One-Hot Encoding
```python
@tf.function
def one_hot_encode_sequence(sequence):
    """
    Convert DNA string to one-hot tensor.
    A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # In practice, use tf.lookup.StaticHashTable
    indices = tf.map_fn(lambda x: mapping.get(x, 0), sequence)
    return tf.one_hot(indices, depth=4)
```

### 3. Positional Encoding for Transformers
```python
def create_positional_encoding(max_len, d_model):
    """
    Sinusoidal positional encoding (from "Attention is All You Need").
    Allows model to learn position-dependent patterns.
    """
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])  # Even indices
    angles[:, 1::2] = np.cos(angles[:, 1::2])  # Odd indices
    
    return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
```

### 4. Mixed Precision Training
```python
# Enable mixed precision for 2-3x speedup on modern GPUs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# IMPORTANT: Output layer must be float32 for numerical stability
layers.Dense(num_classes, dtype='float32')
```

## Common Interview Questions

**Q1**: What does `@tf.function` do?
**A**: Converts Python code to a TensorFlow graph for optimization. Benefits:
- Automatic operation fusion
- Parallel execution
- XLA compilation (optional)
- Portable graphs (can save/load)

**Gotcha**: First call is slow (tracing), subsequent calls are fast.

**Q2**: `model.fit()` vs custom training loop?
```python
# Simple: model.fit()
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)

# Custom: more control
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

**Q3**: Explain `tf.GradientTape`
**A**: Records operations for automatic differentiation. Everything inside the context is "watched" for gradient computation.

---

# 3. JAX Deep Dive

## Core Philosophy
JAX = **NumPy + Autodiff + JIT + vmap**

Unlike PyTorch/TensorFlow, JAX is:
- **Functional**: No hidden state, pure functions
- **Transformable**: Apply `jit`, `grad`, `vmap` to any function
- **NumPy-compatible**: Same API, different backend

## Key Transformations

| Transform | Purpose |
|-----------|---------|
| `jit` | Compile to XLA for speed |
| `grad` | Automatic differentiation |
| `vmap` | Auto-vectorization (batch any function) |
| `pmap` | Parallel execution across devices |

## How I Used JAX in OncoMolML

### 1. JIT-Compiled Molecular Fingerprints
```python
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

@partial(jit, static_argnums=(2, 3))  # Compile; args 2,3 are static (not traced)
def morgan_fingerprint_jax(atom_features, adjacency, radius=2, nbits=2048):
    """
    Compute Morgan circular fingerprint.
    100x+ faster than RDKit for large batches!
    """
    fingerprint = jnp.zeros(nbits, dtype=jnp.float32)
    identifiers = atom_features.sum(axis=1)
    
    for r in range(radius + 1):
        for i in range(atom_features.shape[0]):
            neighbors = adjacency[i]
            neighbor_ids = identifiers * neighbors
            
            # Hash to fingerprint bit
            env_hash = jnp.abs(jnp.int32(
                identifiers[i] * 1000 + neighbor_ids.sum() * 100 + r * 10
            ))
            bit_idx = env_hash % nbits
            fingerprint = fingerprint.at[bit_idx].set(1.0)
        
        # Update: aggregate neighbor info
        identifiers = identifiers + adjacency @ identifiers
    
    return fingerprint
```

**Interview Q**: What is `static_argnums`?
**Answer**: These arguments are treated as compile-time constants. JAX traces different versions for different values. Use for integers that control loop iterations, array shapes, etc.

### 2. vmap for Automatic Batching
```python
# Without vmap: manual batching
def compute_batch_manually(batch_data):
    return jnp.stack([morgan_fingerprint_jax(x, adj) for x, adj in batch_data])

# With vmap: automatic! Way cleaner and faster
batched_fingerprint = vmap(morgan_fingerprint_jax, in_axes=(0, 0))
# Now accepts [batch, atoms, features] instead of [atoms, features]

@jit
def batch_morgan_fingerprint_jax(batch_atoms, batch_adj):
    return vmap(lambda af, adj: morgan_fingerprint_jax(af, adj))(batch_atoms, batch_adj)
```

**Interview Q**: Explain `in_axes`
**Answer**: Specifies which axis to vectorize over for each argument. `(0, 0)` means "batch over axis 0 of both args." `(0, None)` means "batch first arg, broadcast second."

### 3. Differentiable Docking Score
```python
@jit
def lennard_jones_potential(coords, epsilon=1.0, sigma=3.4):
    """
    V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    
    FULLY DIFFERENTIABLE - can optimize molecular positions!
    """
    diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
    
    mask = 1.0 - jnp.eye(coords.shape[0])  # No self-interaction
    
    r6 = (sigma / dist) ** 6
    r12 = r6 ** 2
    energy = 0.5 * jnp.sum(4 * epsilon * (r12 - r6) * mask)
    
    return energy

# Get gradients (forces!) automatically
forces_fn = grad(lennard_jones_potential)
forces = -forces_fn(coords)  # Negative gradient = force
```

### 4. Gradient-Based Ligand Optimization
```python
@partial(jit, static_argnums=(3,))
def optimize_ligand_position(ligand_coords, receptor_grid, lr=0.01, n_steps=100):
    """
    Optimize ligand pose using gradient descent on docking score.
    This is DIFFERENTIABLE DOCKING!
    """
    def loss_fn(coords):
        score, _ = differentiable_vina_score(coords, receptor_grid)
        return score
    
    grad_fn = grad(loss_fn)
    
    def step(state, _):
        coords, scores = state
        g = grad_fn(coords)
        new_coords = coords - lr * g
        return (new_coords, jnp.append(scores, loss_fn(new_coords))), None
    
    # Efficient loop using jax.lax.scan (JIT-compiled)
    final_state, _ = jax.lax.scan(step, (ligand_coords, jnp.array([])), None, length=n_steps)
    return final_state
```

### 5. Tanimoto Similarity
```python
@jit
def tanimoto_similarity(fp1, fp2):
    """Tanimoto coefficient for binary fingerprints."""
    intersection = jnp.sum(fp1 * fp2)
    union = jnp.sum(fp1) + jnp.sum(fp2) - intersection
    return intersection / (union + 1e-10)

# Batch: compute full similarity matrix
@jit
def batch_tanimoto_matrix(fingerprints):
    """All-vs-all Tanimoto. [N, nbits] -> [N, N]"""
    return vmap(lambda fp: vmap(lambda fp2: tanimoto_similarity(fp, fp2))(fingerprints))(fingerprints)
```

## Common Interview Questions

**Q1**: JAX vs NumPy - what's the catch?
**A**: JAX arrays are immutable! Can't do `arr[0] = 5`. Instead:
```python
# NumPy way (doesn't work in JAX)
arr[0] = 5

# JAX way
arr = arr.at[0].set(5)  # Returns new array
```

**Q2**: When does JIT compilation happen?
**A**: First call to a `@jit` function triggers compilation (tracing). Shape changes cause recompilation. Use `static_argnums` for values that change but control program structure.

**Q3**: Why use `jax.lax.scan` instead of Python loop?
**A**: `scan` is JIT-compatible. Python loops unroll and create huge graphs. `scan` compiles to efficient loops.

---

# 4. Numba JIT Deep Dive

## Core Concept
Numba compiles Python/NumPy to machine code at runtime using LLVM. No need to rewrite in C!

## Key Decorators

| Decorator | Purpose |
|-----------|---------|
| `@jit` | General JIT compilation |
| `@njit` | "No Python" - faster, stricter |
| `@njit(parallel=True)` | Enable parallel execution |
| `@vectorize` | Create NumPy ufuncs |
| `@guvectorize` | Generalized ufuncs |

## How I Used Numba in OncoMolML

### 1. Parallel FASTQ Quality Filtering
```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def fast_quality_filter(qualities, min_qual=30, min_length=50):
    """
    Filter sequencing reads by quality.
    
    100,000+ reads/second on a laptop!
    
    Args:
        qualities: 2D array [num_reads, read_length] of Phred scores
        min_qual: Minimum average quality
        min_length: Minimum high-quality bases
    
    Returns:
        Boolean mask of passing reads
    """
    num_reads = qualities.shape[0]
    mask = np.zeros(num_reads, dtype=np.bool_)
    
    # prange: parallel range - each iteration runs on different thread
    for i in prange(num_reads):
        read_qual = qualities[i]
        mean_qual = np.mean(read_qual)
        high_qual_bases = np.sum(read_qual >= min_qual)
        
        if mean_qual >= min_qual and high_qual_bases >= min_length:
            mask[i] = True
    
    return mask
```

**Interview Q**: `prange` vs `range`?
**Answer**: `prange` enables OpenMP-style parallelization. Iterations are distributed across CPU cores. Only use when iterations are independent!

### 2. K-mer Counting with 2-bit Encoding
```python
@njit(cache=True)
def _encode_base(base):
    """Encode nucleotide to 2 bits: A=0, C=1, G=2, T=3"""
    if base == 65:    return 0  # ASCII 'A'
    elif base == 67:  return 1  # ASCII 'C'
    elif base == 71:  return 2  # ASCII 'G'
    elif base == 84:  return 3  # ASCII 'T'
    else:             return -1 # N or other

@njit(cache=True)
def _encode_kmer(sequence, start, k):
    """Encode k-mer to integer. 4^21 < 2^63, so k≤31 fits in int64."""
    kmer_int = 0
    for i in range(k):
        base_code = _encode_base(sequence[start + i])
        if base_code < 0:
            return -1  # Contains N
        kmer_int = (kmer_int << 2) | base_code  # Shift and OR
    return kmer_int

@njit(parallel=True, cache=True)
def parallel_kmer_count(sequences, k=21):
    """Count k-mers across multiple sequences in parallel."""
    num_seqs = sequences.shape[0]
    max_kmers = 4 ** k
    
    # Each thread gets its own count array (no race conditions)
    thread_counts = np.zeros((num_seqs, max_kmers), dtype=np.int32)
    
    for i in prange(num_seqs):
        thread_counts[i] = count_kmers_single(sequences[i], k)
    
    return np.sum(thread_counts, axis=0)
```

**Interview Q**: Why 2-bit encoding for DNA?
**Answer**: DNA has 4 bases, so 2 bits is sufficient. A 21-mer fits in 42 bits (< 64-bit integer). This enables:
- Direct array indexing for counting
- Fast hashing
- SIMD operations

### 3. Smith-Waterman Local Alignment
```python
@njit(cache=True)
def local_alignment_score(seq1, seq2, match=2, mismatch=-1, gap=-2):
    """
    Smith-Waterman local alignment.
    
    Unlike global alignment (Needleman-Wunsch), local alignment:
    - Can start/end anywhere
    - Returns highest-scoring substring alignment
    """
    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=np.int32)
    max_score = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Diagonal: match or mismatch
            if seq1[i-1] == seq2[j-1]:
                diag = H[i-1, j-1] + match
            else:
                diag = H[i-1, j-1] + mismatch
            
            # Gaps
            up = H[i-1, j] + gap
            left = H[i, j-1] + gap
            
            # LOCAL: can reset to 0 (start new alignment)
            H[i, j] = max(0, diag, up, left)
            max_score = max(max_score, H[i, j])
    
    return max_score
```

### 4. Sliding Window Quality Trimming
```python
@njit(cache=True)
def sliding_window_quality_trim(qualities, window_size=4, min_qual=20):
    """
    Like Trimmomatic's SLIDINGWINDOW.
    Finds high-quality region of read.
    
    Returns (start, end) positions.
    """
    read_len = len(qualities)
    
    # Find 5' trim position (scan forward)
    start = 0
    for i in range(read_len - window_size + 1):
        if np.mean(qualities[i:i + window_size]) >= min_qual:
            start = i
            break
    
    # Find 3' trim position (scan backward)
    end = read_len
    for i in range(read_len - window_size, -1, -1):
        if np.mean(qualities[i:i + window_size]) >= min_qual:
            end = i + window_size
            break
    
    return start, end
```

## Common Interview Questions

**Q1**: `@jit` vs `@njit`?
**A**: `@njit` = `@jit(nopython=True)`. Forces compilation to pure machine code with no Python fallback. Faster but stricter - no Python objects allowed.

**Q2**: What can't Numba compile?
**A**: 
- Python objects (lists, dicts) - use `numba.typed.List/Dict`
- String operations (limited support)
- Most library calls (only NumPy supported)
- Classes (use `@jitclass` with type annotations)

**Q3**: What does `cache=True` do?
**A**: Caches compiled functions to disk. First run compiles and saves; subsequent runs load from cache. Eliminates startup compilation time.

**Q4**: How to debug Numba errors?
```python
# Temporarily disable JIT to get Python errors
@jit(nopython=False)  # or remove decorator entirely
def my_function():
    ...
```

---

# 5. Quick Reference - Code to Write from Memory

## PyTorch Model Template
```python
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```

## TensorFlow Model Template
```python
class MyModel(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='gelu')
        self.dense2 = layers.Dense(1)
    
    @tf.function
    def call(self, x, training=None):
        x = self.dense1(x)
        return self.dense2(x)

# Training
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

## JAX Pattern
```python
import jax.numpy as jnp
from jax import jit, grad, vmap

@jit
def loss_fn(params, x, y):
    pred = model_forward(params, x)
    return jnp.mean((pred - y) ** 2)

grad_fn = grad(loss_fn)  # Differentiable!
batched_fn = vmap(single_fn)  # Vectorized!

# Update step
grads = grad_fn(params, x, y)
params = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
```

## Numba Pattern
```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def fast_function(data):
    n = data.shape[0]
    result = np.zeros(n)
    for i in prange(n):  # Parallel loop
        result[i] = compute_single(data[i])
    return result
```

---

# 6. Framework Comparison Table

| Feature | PyTorch | TensorFlow | JAX | Numba |
|---------|---------|------------|-----|-------|
| **Paradigm** | Dynamic graph | Dynamic/Static | Functional | NumPy JIT |
| **Autodiff** | `backward()` | `GradientTape` | `grad()` | ❌ |
| **Batching** | Manual | Manual | `vmap()` | Manual |
| **GPU** | ✅ | ✅ | ✅ | ✅ (CUDA) |
| **Best for** | Research | Production | Scientific | NumPy speedup |
| **My use** | Drug response models | Variant prediction | Differentiable docking | NGS processing |

---

# 7. Behavioral Questions

**Q**: "Why did you choose these specific frameworks?"

**A**: "I chose each framework for its strengths:
- **PyTorch** for drug response models because attention mechanisms and multi-task learning are well-supported, and the dynamic graph makes debugging easier when experimenting with architectures.
- **TensorFlow** for variant effect prediction because the `@tf.function` graph compilation and mixed precision give production-ready performance for serving predictions.
- **JAX** for differentiable scoring functions because `grad` and `jit` let me optimize molecular positions directly through gradient descent - something not easily done in other frameworks.
- **Numba** for NGS preprocessing because I needed to process millions of reads quickly, and `@njit(parallel=True)` gave me near-C performance without leaving Python."

**Q**: "What challenges did you face?"

**A**: "The biggest challenge was JAX's functional paradigm - you can't mutate arrays in-place. I had to rethink algorithms to use `arr.at[idx].set(val)` instead of `arr[idx] = val`. Also, Numba's type inference sometimes fails silently, so I learned to test small cases first and use explicit type hints when needed."

---

Good luck with your interview! 🎯
