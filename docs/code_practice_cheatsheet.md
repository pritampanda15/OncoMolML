# Code Patterns to Practice Writing
## Interview Coding Cheat Sheet

---

## 🔥 PyTorch Patterns

### 1. Basic Neural Network
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

### 2. Attention Layer
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)  # Q, K, V all same
        return self.norm(x + attn_out)    # Residual connection
```

### 3. Training Loop
```python
model = MLP(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()           # Clear gradients
        pred = model(x)                 # Forward
        loss = criterion(pred, y)       # Compute loss
        loss.backward()                 # Backprop
        optimizer.step()                # Update weights
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(val_x), val_y)
```

### 4. GPU Usage
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

---

## 🔥 TensorFlow Patterns

### 1. Keras Model
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class MyModel(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='gelu')
        self.dense2 = layers.Dense(1)
    
    @tf.function  # Compile to graph
    def call(self, x, training=None):
        return self.dense2(self.dense1(x))
```

### 2. Custom Training Step
```python
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

### 3. One-Hot Encoding
```python
# DNA sequence one-hot
def one_hot_dna(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = [mapping.get(b, 0) for b in sequence]
    return tf.one_hot(indices, depth=4)
```

### 4. Conv1D for Sequences
```python
model = tf.keras.Sequential([
    layers.Conv1D(64, 7, activation='relu', padding='same'),
    layers.Conv1D(128, 5, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(3)  # 3 classes
])
```

---

## 🔥 JAX Patterns

### 1. JIT Compilation
```python
import jax.numpy as jnp
from jax import jit

@jit
def fast_function(x, y):
    return jnp.dot(x, y.T)

# Static args (won't be traced)
from functools import partial

@partial(jit, static_argnums=(1,))
def with_static(data, num_iterations):
    for _ in range(num_iterations):
        data = process(data)
    return data
```

### 2. Automatic Differentiation
```python
from jax import grad, value_and_grad

def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# Get gradient function
grad_fn = grad(loss)
grads = grad_fn(params, x, y)

# Get both loss and gradients
loss_and_grad_fn = value_and_grad(loss)
loss_val, grads = loss_and_grad_fn(params, x, y)
```

### 3. vmap for Batching
```python
from jax import vmap

# Single-item function
def process_one(x):
    return jnp.sum(x ** 2)

# Auto-batched version
process_batch = vmap(process_one)

# With axis specification
batched = vmap(fn, in_axes=(0, None))  # Batch arg1, broadcast arg2
```

### 4. Immutable Array Updates
```python
# WRONG - JAX arrays are immutable
# arr[0] = 5  

# CORRECT - returns new array
arr = arr.at[0].set(5)
arr = arr.at[1:3].add(1)
arr = arr.at[mask].set(0)
```

### 5. Pure JAX Neural Network
```python
def init_layer(key, in_dim, out_dim):
    w = jax.random.normal(key, (in_dim, out_dim)) * 0.01
    b = jnp.zeros(out_dim)
    return (w, b)

@jit
def forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b
```

---

## 🔥 Numba Patterns

### 1. Basic JIT
```python
from numba import njit

@njit(cache=True)
def fast_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total
```

### 2. Parallel Processing
```python
from numba import njit, prange

@njit(parallel=True)
def parallel_process(data):
    n = data.shape[0]
    result = np.zeros(n)
    for i in prange(n):  # Parallel loop
        result[i] = expensive_compute(data[i])
    return result
```

### 3. DNA K-mer Encoding
```python
@njit(cache=True)
def encode_base(base):
    # ASCII: A=65, C=67, G=71, T=84
    if base == 65:   return 0
    elif base == 67: return 1
    elif base == 71: return 2
    elif base == 84: return 3
    return -1  # N

@njit(cache=True)
def encode_kmer(seq, start, k):
    kmer = 0
    for i in range(k):
        code = encode_base(seq[start + i])
        if code < 0:
            return -1
        kmer = (kmer << 2) | code  # Shift 2 bits, add base
    return kmer
```

### 4. Smith-Waterman (Local Alignment)
```python
@njit(cache=True)
def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    H = np.zeros((m+1, n+1), dtype=np.int32)
    max_score = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            diag = H[i-1,j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch)
            H[i,j] = max(0, diag, H[i-1,j]+gap, H[i,j-1]+gap)
            max_score = max(max_score, H[i,j])
    
    return max_score
```

---

## 🎯 Common Interview Coding Tasks

### Task 1: Softmax from Scratch
```python
# NumPy/JAX
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# PyTorch
def softmax_torch(x, dim=-1):
    return F.softmax(x, dim=dim)
```

### Task 2: Cross-Entropy Loss
```python
# From scratch
def cross_entropy(pred, target):
    # pred: [batch, classes] logits
    # target: [batch] class indices
    log_softmax = pred - torch.logsumexp(pred, dim=-1, keepdim=True)
    return -log_softmax.gather(1, target.unsqueeze(1)).mean()

# PyTorch built-in
loss = nn.CrossEntropyLoss()(pred, target)
```

### Task 3: Batch Normalization
```python
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### Task 4: Attention Score Computation
```python
def attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, heads, seq, dim]
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

### Task 5: GC Content (Numba)
```python
@njit(cache=True)
def gc_content(sequence):
    """sequence is ASCII-encoded numpy array"""
    gc = 0
    total = 0
    for base in sequence:
        if base == 71 or base == 67:  # G or C
            gc += 1
        if base != 78:  # Not N
            total += 1
    return gc / total if total > 0 else 0.0
```

---

## 📝 Quick Syntax Reference

### PyTorch Tensor Operations
```python
torch.cat([a, b], dim=0)      # Concatenate along existing dim
torch.stack([a, b], dim=0)    # Create new dim
x.view(batch, -1)             # Reshape (-1 = infer)
x.permute(0, 2, 1)            # Transpose dimensions
x.unsqueeze(1)                # Add dimension
x.squeeze(1)                  # Remove dimension
```

### TensorFlow Operations
```python
tf.concat([a, b], axis=0)
tf.stack([a, b], axis=0)
tf.reshape(x, [batch, -1])
tf.transpose(x, perm=[0, 2, 1])
tf.expand_dims(x, axis=1)
tf.squeeze(x, axis=1)
```

### JAX Operations
```python
jnp.concatenate([a, b], axis=0)
jnp.stack([a, b], axis=0)
x.reshape(batch, -1)
jnp.transpose(x, axes=(0, 2, 1))
jnp.expand_dims(x, axis=1)
jnp.squeeze(x, axis=1)

# Immutable updates
x = x.at[0].set(5)
x = x.at[1:3].add(1)
```

---

## 🧪 Practice Problems

1. **Write a 2-layer MLP in PyTorch** with dropout and layer norm
2. **Implement `@tf.function` training step** with gradient tape
3. **Write a JAX loss function** and get its gradient
4. **Parallelize array processing** with Numba `prange`
5. **Implement Tanimoto similarity** in JAX with `vmap` batching
6. **Write Smith-Waterman** in Numba with `@njit`

---

Good luck! Practice these until you can write them without looking. 🚀
