from oncomolml.models.pytorch_models import DrugResponseNet, MolPropertyPredictor
import torch

# Predict drug response from gene expression + molecular features
model = DrugResponseNet(
    gene_dim=978,      # L1000 landmark genes
    mol_dim=2048,      # Morgan fingerprint
    hidden_dim=512
)

# Example prediction
gene_expr = torch.randn(32, 978)      # Batch of gene expressions
mol_fp = torch.randn(32, 2048)        # Drug fingerprints
ic50_pred = model(gene_expr, mol_fp)  # Predicted IC50 values

print("Predicted IC50 values:", ic50_pred)


