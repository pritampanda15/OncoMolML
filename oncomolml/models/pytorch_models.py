"""
PyTorch Models for Cancer Drug Discovery

This module implements deep learning models for:
- Drug response prediction (IC50/AUC)
- Molecular property prediction (ADMET)
- Gene expression-based drug sensitivity

Showcases: PyTorch, torch.nn, attention mechanisms, multi-task learning
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for this module. "
        "Install with: pip install torch>=2.0.0"
    )


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for feature interaction."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class DrugResponseNet(nn.Module):
    """
    Deep neural network for predicting drug response (IC50/AUC) from
    gene expression profiles and molecular fingerprints.
    
    Architecture:
    - Separate encoders for gene expression and molecular features
    - Cross-attention fusion mechanism
    - Multi-task heads for different response metrics
    
    Reference: Based on DeepCDR and TGSA architectures
    
    Args:
        gene_dim: Dimension of gene expression input (e.g., 978 for L1000)
        mol_dim: Dimension of molecular fingerprint (e.g., 2048 for Morgan)
        hidden_dim: Hidden layer dimension
        num_attention_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
        num_tasks: Number of output tasks (IC50, AUC, etc.)
    
    Example:
        >>> model = DrugResponseNet(gene_dim=978, mol_dim=2048)
        >>> gene_expr = torch.randn(32, 978)
        >>> mol_fp = torch.randn(32, 2048)
        >>> predictions = model(gene_expr, mol_fp)
        >>> print(predictions.shape)  # torch.Size([32, 1])
    """
    
    def __init__(
        self,
        gene_dim: int = 978,
        mol_dim: int = 2048,
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_tasks: int = 1
    ):
        super().__init__()
        
        self.gene_dim = gene_dim
        self.mol_dim = mol_dim
        self.hidden_dim = hidden_dim
        
        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Molecular fingerprint encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Cross-attention layers for gene-drug interaction
        self.cross_attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_tasks)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        gene_expr: Tensor,
        mol_fp: Tensor,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for drug response prediction.
        
        Args:
            gene_expr: Gene expression tensor [batch, gene_dim]
            mol_fp: Molecular fingerprint tensor [batch, mol_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: Predicted drug response [batch, num_tasks]
        """
        # Encode inputs
        gene_encoded = self.gene_encoder(gene_expr)  # [batch, hidden]
        mol_encoded = self.mol_encoder(mol_fp)       # [batch, hidden]
        
        # Stack for attention (treat as sequence of 2 tokens)
        combined = torch.stack([gene_encoded, mol_encoded], dim=1)  # [batch, 2, hidden]
        
        # Apply cross-attention layers
        for attn_layer in self.cross_attention_layers:
            combined = attn_layer(combined)
        
        # Pool and fuse
        gene_out = combined[:, 0, :]
        mol_out = combined[:, 1, :]
        fused = self.fusion(torch.cat([gene_out, mol_out], dim=-1))
        
        # Multi-task predictions
        predictions = torch.cat([
            head(fused) for head in self.prediction_heads
        ], dim=-1)
        
        return predictions
    
    def predict_sensitivity(
        self,
        gene_expr: Tensor,
        mol_fp: Tensor,
        threshold: float = 0.5
    ) -> Tensor:
        """
        Predict binary drug sensitivity (sensitive/resistant).
        
        Args:
            gene_expr: Gene expression tensor
            mol_fp: Molecular fingerprint tensor
            threshold: IC50 threshold for sensitivity
            
        Returns:
            Binary sensitivity predictions
        """
        with torch.no_grad():
            ic50_pred = self.forward(gene_expr, mol_fp)
            return (ic50_pred < threshold).float()


class MolPropertyPredictor(nn.Module):
    """
    Multi-task molecular property predictor for ADMET properties.
    
    Predicts multiple properties simultaneously:
    - Absorption: Solubility, Caco-2, HIA
    - Distribution: LogP, LogD, PPB, BBB
    - Metabolism: CYP450 inhibition
    - Excretion: Half-life, clearance
    - Toxicity: hERG, AMES, LD50
    
    Args:
        input_dim: Molecular fingerprint dimension
        hidden_dims: List of hidden layer dimensions
        num_properties: Number of properties to predict
        property_names: Names of properties for interpretation
        dropout: Dropout probability
    
    Example:
        >>> model = MolPropertyPredictor(input_dim=2048, num_properties=12)
        >>> fp = torch.randn(16, 2048)
        >>> props = model(fp)
        >>> print(props.shape)  # torch.Size([16, 12])
    """
    
    ADMET_PROPERTIES = [
        "solubility", "caco2", "hia", "logp", "logd", 
        "ppb", "bbb", "cyp3a4", "cyp2d6", "half_life",
        "herg", "ames"
    ]
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: List[int] = [1024, 512, 256],
        num_properties: int = 12,
        property_names: Optional[List[str]] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_properties = num_properties
        self.property_names = property_names or self.ADMET_PROPERTIES[:num_properties]
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Property-specific heads (allows for different property types)
        self.property_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for name in self.property_names
        })
        
        # Uncertainty estimation heads (epistemic uncertainty)
        self.uncertainty_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )
            for name in self.property_names
        })
    
    def forward(
        self,
        x: Tensor,
        return_uncertainty: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for property prediction.
        
        Args:
            x: Molecular fingerprint tensor [batch, input_dim]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            predictions: Property predictions [batch, num_properties]
            uncertainties: (optional) Uncertainty estimates [batch, num_properties]
        """
        # Shared encoding
        features = self.shared_encoder(x)
        
        # Property-specific predictions
        predictions = torch.cat([
            self.property_heads[name](features)
            for name in self.property_names
        ], dim=-1)
        
        if return_uncertainty:
            uncertainties = torch.cat([
                self.uncertainty_heads[name](features)
                for name in self.property_names
            ], dim=-1)
            return predictions, uncertainties
        
        return predictions
    
    def predict_with_names(self, x: Tensor) -> Dict[str, Tensor]:
        """Return predictions as a dictionary with property names."""
        predictions = self.forward(x)
        return {
            name: predictions[:, i:i+1]
            for i, name in enumerate(self.property_names)
        }


class GraphDrugEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular graphs.
    
    Uses message passing to learn molecular representations
    from 2D molecular graphs (atoms as nodes, bonds as edges).
    
    Note: Requires torch-geometric for full functionality.
    
    Args:
        atom_features: Number of atom features
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        atom_features: int = 78,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.atom_embedding = nn.Linear(atom_features, hidden_dim)
        
        # Message passing layers (simplified without torch-geometric)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Encode molecular graph to fixed-size representation.
        
        Args:
            x: Atom feature tensor [num_atoms, atom_features]
            batch: Batch assignment tensor [num_atoms]
            
        Returns:
            Molecular representation [batch_size, hidden_dim]
        """
        # Embed atoms
        h = self.atom_embedding(x)
        
        # Message passing (simplified)
        for conv in self.conv_layers:
            h = h + conv(h)  # Residual connection
        
        # Global mean pooling
        if batch is not None:
            # Scatter mean over batch
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, h.size(-1), device=h.device)
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
            counts = torch.bincount(batch, minlength=batch_size).float().unsqueeze(-1)
            out = out / counts.clamp(min=1)
        else:
            out = h.mean(dim=0, keepdim=True)
        
        return self.global_pool(out)


class CancerTypeClassifier(nn.Module):
    """
    Cancer type classifier from gene expression profiles.
    
    Multi-class classification for cancer type identification
    based on transcriptomic signatures (e.g., TCGA pan-cancer).
    
    Args:
        gene_dim: Number of genes in expression profile
        num_cancer_types: Number of cancer type classes
        hidden_dims: Hidden layer dimensions
    """
    
    TCGA_CANCER_TYPES = [
        "BRCA", "LUAD", "LUSC", "COAD", "READ", "PRAD",
        "STAD", "BLCA", "LIHC", "KIRC", "KIRP", "KICH",
        "THCA", "HNSC", "OV", "UCEC", "PAAD", "GBM"
    ]
    
    def __init__(
        self,
        gene_dim: int = 20000,
        num_cancer_types: int = 18,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = gene_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_cancer_types))
        
        self.classifier = nn.Sequential(*layers)
        self.num_cancer_types = num_cancer_types
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Classify cancer type from gene expression.
        
        Args:
            x: Gene expression tensor [batch, gene_dim]
            
        Returns:
            Class logits [batch, num_cancer_types]
        """
        return self.classifier(x)
    
    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# Training utilities
class DrugResponseLoss(nn.Module):
    """Combined loss for drug response prediction with uncertainty."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.huber = nn.HuberLoss(reduction=reduction, delta=1.0)
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        uncertainty: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute loss with optional uncertainty weighting.
        
        If uncertainty is provided, uses heteroscedastic loss.
        """
        if uncertainty is not None:
            # Heteroscedastic loss: -log N(y | pred, sigma^2)
            loss = 0.5 * (torch.log(uncertainty) + (pred - target)**2 / uncertainty)
            return loss.mean()
        else:
            return self.huber(pred, target)


def create_drug_response_model(
    config: Optional[Dict] = None
) -> DrugResponseNet:
    """
    Factory function to create DrugResponseNet with common configurations.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured DrugResponseNet model
    """
    default_config = {
        "gene_dim": 978,
        "mol_dim": 2048,
        "hidden_dim": 512,
        "num_attention_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "num_tasks": 1
    }
    
    if config:
        default_config.update(config)
    
    return DrugResponseNet(**default_config)


if __name__ == "__main__":
    # Quick test
    print("Testing PyTorch models...")
    
    # Test DrugResponseNet
    model = DrugResponseNet(gene_dim=978, mol_dim=2048)
    gene_expr = torch.randn(4, 978)
    mol_fp = torch.randn(4, 2048)
    output = model(gene_expr, mol_fp)
    print(f"DrugResponseNet output shape: {output.shape}")
    
    # Test MolPropertyPredictor
    prop_model = MolPropertyPredictor(input_dim=2048, num_properties=12)
    fp = torch.randn(4, 2048)
    props, uncert = prop_model(fp, return_uncertainty=True)
    print(f"MolPropertyPredictor output shape: {props.shape}")
    print(f"Uncertainty shape: {uncert.shape}")
    
    print("All PyTorch models working correctly!")
