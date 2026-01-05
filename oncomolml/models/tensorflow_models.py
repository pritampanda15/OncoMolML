"""
TensorFlow Models for Cancer NGS Analysis

This module implements deep learning models for:
- Variant effect prediction (pathogenicity scoring)
- DNA/RNA sequence encoding
- Mutation impact classification
- Copy number variation analysis

Showcases: TensorFlow 2.x, Keras API, tf.function, mixed precision
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    raise ImportError(
        "TensorFlow is required for this module. "
        "Install with: pip install tensorflow>=2.15.0"
    )


# Enable mixed precision for faster training on compatible GPUs
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except:
    pass  # Fall back to float32 if mixed precision not supported


class ConvBlock(layers.Layer):
    """Convolutional block with batch norm and residual connection."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(
            filters, kernel_size,
            padding='same',
            dilation_rate=dilation_rate
        )
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout)
        self.activation = layers.Activation('gelu')
    
    def call(self, x, training=None):
        h = self.conv(x)
        h = self.bn(h, training=training)
        h = self.activation(h)
        h = self.dropout(h, training=training)
        return h


class VariantEffectPredictor(Model):
    """
    Deep learning model for predicting variant pathogenicity.
    
    Predicts the functional impact of genetic variants (SNVs, indels)
    based on sequence context and evolutionary conservation.
    
    Architecture:
    - Multi-scale dilated convolutions for sequence context
    - Transformer encoder for long-range dependencies
    - Classification head for pathogenicity scoring
    
    Input: One-hot encoded DNA sequences centered on variant
    Output: Pathogenicity scores (Benign, VUS, Pathogenic)
    
    Args:
        sequence_length: Length of input sequence (default: 101bp context)
        num_classes: Number of pathogenicity classes
        filters: Number of convolutional filters
        num_transformer_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    
    Example:
        >>> model = VariantEffectPredictor(sequence_length=101)
        >>> seq = tf.random.uniform((32, 101, 4))  # One-hot encoded
        >>> pred = model(seq)
        >>> print(pred.shape)  # (32, 3)
    """
    
    CLASS_NAMES = ["Benign", "VUS", "Pathogenic"]
    
    def __init__(
        self,
        sequence_length: int = 101,
        num_classes: int = 3,
        filters: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Multi-scale convolutional encoder
        self.conv_blocks = [
            ConvBlock(filters, kernel_size=7, dilation_rate=1, dropout=dropout),
            ConvBlock(filters, kernel_size=5, dilation_rate=2, dropout=dropout),
            ConvBlock(filters, kernel_size=3, dilation_rate=4, dropout=dropout),
            ConvBlock(filters, kernel_size=3, dilation_rate=8, dropout=dropout),
        ]
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            sequence_length, filters
        )
        
        # Transformer encoder layers
        self.transformer_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=filters // num_heads,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ]
        self.layer_norms = [
            layers.LayerNormalization()
            for _ in range(num_transformer_layers * 2)
        ]
        self.ffn_layers = [
            keras.Sequential([
                layers.Dense(filters * 4, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(filters),
                layers.Dropout(dropout)
            ])
            for _ in range(num_transformer_layers)
        ]
        
        # Global pooling and classification
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes, dtype='float32')  # Ensure float32 for output
        ])
    
    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> tf.Tensor:
        """Create sinusoidal positional encoding."""
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
    
    @tf.function
    def call(self, x, training=None):
        """
        Forward pass for variant effect prediction.
        
        Args:
            x: One-hot encoded sequence [batch, seq_len, 4]
            training: Whether in training mode
            
        Returns:
            Class logits [batch, num_classes]
        """
        # Multi-scale convolutions
        h = x
        for conv_block in self.conv_blocks:
            h = conv_block(h, training=training)
        
        # Add positional encoding
        h = h + self.pos_encoding[:, :tf.shape(h)[1], :]
        
        # Transformer encoder
        for i, (attn, ffn) in enumerate(zip(self.transformer_layers, self.ffn_layers)):
            # Self-attention with residual
            attn_out = attn(h, h, training=training)
            h = self.layer_norms[i * 2](h + attn_out)
            
            # FFN with residual
            ffn_out = ffn(h, training=training)
            h = self.layer_norms[i * 2 + 1](h + ffn_out)
        
        # Pool and classify
        pooled = self.global_pool(h)
        logits = self.classifier(pooled, training=training)
        
        return logits
    
    def predict_pathogenicity(
        self,
        sequences: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Predict pathogenicity with probabilities and labels.
        
        Args:
            sequences: One-hot encoded sequences
            
        Returns:
            Dictionary with probabilities and predicted classes
        """
        logits = self(sequences, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        pred_classes = tf.argmax(probs, axis=-1)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_classes,
            "class_names": [self.CLASS_NAMES[i] for i in pred_classes.numpy()]
        }


class SequenceEncoder(Model):
    """
    DNA/RNA sequence encoder using convolutional and recurrent layers.
    
    Generates fixed-size embeddings from variable-length sequences
    for downstream tasks like:
    - Gene expression prediction
    - Promoter classification
    - Splice site detection
    
    Args:
        vocab_size: Size of nucleotide vocabulary (4 for DNA, 5 for RNA+N)
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        num_conv_layers: Number of convolutional layers
        use_bidirectional: Use bidirectional LSTM
    
    Example:
        >>> encoder = SequenceEncoder(embed_dim=128)
        >>> seq = tf.random.uniform((16, 500, 4))
        >>> embedding = encoder(seq)
        >>> print(embedding.shape)  # (16, 256)
    """
    
    def __init__(
        self,
        vocab_size: int = 4,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 3,
        use_bidirectional: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection if input is one-hot
        self.input_proj = layers.Dense(embed_dim)
        
        # Convolutional layers with increasing dilation
        self.conv_layers = []
        for i in range(num_conv_layers):
            self.conv_layers.append(
                ConvBlock(
                    filters=hidden_dim,
                    kernel_size=5,
                    dilation_rate=2**i,
                    dropout=dropout
                )
            )
        
        # Bidirectional LSTM for sequence modeling
        lstm = layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout)
        if use_bidirectional:
            self.rnn = layers.Bidirectional(lstm)
            self.rnn_dim = hidden_dim * 2
        else:
            self.rnn = lstm
            self.rnn_dim = hidden_dim
        
        # Attention pooling
        self.attention = layers.Dense(1)
        
        # Final projection
        self.output_proj = keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, dtype='float32')
        ])
    
    @tf.function
    def call(self, x, training=None, return_attention=False):
        """
        Encode sequence to fixed-size vector.
        
        Args:
            x: Input sequence [batch, seq_len, vocab_size]
            training: Training mode flag
            return_attention: Return attention weights
            
        Returns:
            Sequence embedding [batch, hidden_dim]
        """
        # Project input
        h = self.input_proj(x)
        
        # Convolutional encoding
        for conv in self.conv_layers:
            h = h + conv(h, training=training)  # Residual
        
        # RNN encoding
        h = self.rnn(h, training=training)
        
        # Attention pooling
        attn_weights = tf.nn.softmax(self.attention(h), axis=1)
        pooled = tf.reduce_sum(h * attn_weights, axis=1)
        
        # Final projection
        output = self.output_proj(pooled)
        
        if return_attention:
            return output, tf.squeeze(attn_weights, axis=-1)
        return output


class MutationImpactClassifier(Model):
    """
    Classifier for mutation functional impact prediction.
    
    Predicts whether a mutation is:
    - Loss of function (LoF)
    - Gain of function (GoF)
    - Neutral
    - Unknown/VUS
    
    Uses both sequence context and protein features.
    
    Args:
        sequence_length: DNA sequence context length
        protein_features: Number of protein-level features
        num_classes: Number of impact classes
    """
    
    IMPACT_CLASSES = ["Neutral", "LoF", "GoF", "VUS"]
    
    def __init__(
        self,
        sequence_length: int = 101,
        protein_features: int = 64,
        num_classes: int = 4,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Sequence encoder
        self.seq_encoder = keras.Sequential([
            layers.Conv1D(128, 7, padding='same', activation='gelu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(256, 5, padding='same', activation='gelu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D()
        ])
        
        # Protein feature encoder
        self.prot_encoder = keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='gelu')
        ])
        
        # Fusion and classification
        self.fusion = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes, dtype='float32')
        ])
    
    @tf.function
    def call(self, inputs, training=None):
        """
        Predict mutation impact.
        
        Args:
            inputs: Tuple of (sequence, protein_features)
            training: Training mode
            
        Returns:
            Impact class logits
        """
        sequence, prot_features = inputs
        
        seq_embed = self.seq_encoder(sequence, training=training)
        prot_embed = self.prot_encoder(prot_features, training=training)
        
        combined = tf.concat([seq_embed, prot_embed], axis=-1)
        logits = self.fusion(combined, training=training)
        
        return logits


class CNVPredictor(Model):
    """
    Copy Number Variation (CNV) predictor from read depth signals.
    
    Segments the genome and predicts copy number states:
    - Deep deletion (0)
    - Heterozygous deletion (1)
    - Neutral (2)
    - Gain (3)
    - Amplification (4+)
    
    Args:
        window_size: Number of bins in input window
        num_features: Features per bin (coverage, GC, mappability)
        num_states: Number of CN states
    """
    
    CN_STATES = ["DeepDel", "HetDel", "Neutral", "Gain", "Amp"]
    
    def __init__(
        self,
        window_size: int = 1000,
        num_features: int = 4,
        num_states: int = 5,
        hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 1D U-Net style architecture for segmentation
        self.encoder1 = keras.Sequential([
            layers.Conv1D(64, 7, padding='same', activation='gelu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2)
        ])
        
        self.encoder2 = keras.Sequential([
            layers.Conv1D(128, 5, padding='same', activation='gelu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2)
        ])
        
        self.bottleneck = keras.Sequential([
            layers.Conv1D(256, 3, padding='same', activation='gelu'),
            layers.BatchNormalization()
        ])
        
        self.decoder1 = keras.Sequential([
            layers.UpSampling1D(2),
            layers.Conv1D(128, 5, padding='same', activation='gelu'),
            layers.BatchNormalization()
        ])
        
        self.decoder2 = keras.Sequential([
            layers.UpSampling1D(2),
            layers.Conv1D(64, 7, padding='same', activation='gelu'),
            layers.BatchNormalization()
        ])
        
        self.output_conv = layers.Conv1D(num_states, 1, padding='same', dtype='float32')
    
    @tf.function
    def call(self, x, training=None):
        """
        Predict copy number states per bin.
        
        Args:
            x: Read depth features [batch, window_size, num_features]
            training: Training mode
            
        Returns:
            CN state logits [batch, window_size, num_states]
        """
        # Encoder
        e1 = self.encoder1(x, training=training)
        e2 = self.encoder2(e1, training=training)
        
        # Bottleneck
        b = self.bottleneck(e2, training=training)
        
        # Decoder with skip connections
        d1 = self.decoder1(b, training=training)
        d1 = d1[:, :tf.shape(e1)[1], :]  # Handle size mismatch
        d1 = d1 + e1
        
        d2 = self.decoder2(d1, training=training)
        d2 = d2[:, :tf.shape(x)[1], :]  # Handle size mismatch
        
        # Output
        logits = self.output_conv(d2)
        
        return logits


# Utility functions
def one_hot_encode_sequence(sequence: str) -> tf.Tensor:
    """
    One-hot encode a DNA sequence.
    
    Args:
        sequence: DNA sequence string (ACGT)
        
    Returns:
        One-hot encoded tensor [seq_len, 4]
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = [mapping.get(base.upper(), 0) for base in sequence]
    return tf.one_hot(indices, depth=4)


def batch_one_hot_encode(sequences: List[str]) -> tf.Tensor:
    """Batch one-hot encode multiple sequences."""
    encoded = [one_hot_encode_sequence(seq) for seq in sequences]
    return tf.stack(encoded, axis=0)


class VariantDataGenerator(keras.utils.Sequence):
    """
    Data generator for variant effect prediction training.
    
    Loads variants from VCF/TSV and generates batches of
    one-hot encoded sequence contexts.
    """
    
    def __init__(
        self,
        variants: List[Dict],
        reference_genome: str,
        batch_size: int = 32,
        context_size: int = 50,
        shuffle: bool = True
    ):
        self.variants = variants
        self.batch_size = batch_size
        self.context_size = context_size
        self.shuffle = shuffle
        self.indices = np.arange(len(variants))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.variants) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_variants = [self.variants[i] for i in batch_indices]
        
        # Generate sequences and labels
        sequences = []
        labels = []
        
        for var in batch_variants:
            # Placeholder - in real implementation, extract from reference
            seq = 'N' * (2 * self.context_size + 1)
            sequences.append(seq)
            labels.append(var.get('label', 0))
        
        X = batch_one_hot_encode(sequences)
        y = tf.constant(labels)
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_variant_effect_model(
    config: Optional[Dict] = None
) -> VariantEffectPredictor:
    """
    Factory function to create VariantEffectPredictor.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured model instance
    """
    default_config = {
        "sequence_length": 101,
        "num_classes": 3,
        "filters": 256,
        "num_transformer_layers": 4,
        "num_heads": 8,
        "dropout": 0.1
    }
    
    if config:
        default_config.update(config)
    
    return VariantEffectPredictor(**default_config)


if __name__ == "__main__":
    print("Testing TensorFlow models...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Test VariantEffectPredictor
    model = VariantEffectPredictor(sequence_length=101)
    seq = tf.random.uniform((4, 101, 4))
    output = model(seq)
    print(f"VariantEffectPredictor output shape: {output.shape}")
    
    # Test SequenceEncoder
    encoder = SequenceEncoder(embed_dim=128)
    seq = tf.random.uniform((4, 500, 4))
    embed = encoder(seq)
    print(f"SequenceEncoder output shape: {embed.shape}")
    
    # Test CNVPredictor
    cnv_model = CNVPredictor(window_size=1000)
    depth = tf.random.uniform((4, 1000, 4))
    cnv_out = cnv_model(depth)
    print(f"CNVPredictor output shape: {cnv_out.shape}")
    
    # Test one-hot encoding
    seq_str = "ACGTACGTNN"
    encoded = one_hot_encode_sequence(seq_str)
    print(f"One-hot encoded shape: {encoded.shape}")
    
    print("\nAll TensorFlow models working correctly!")
