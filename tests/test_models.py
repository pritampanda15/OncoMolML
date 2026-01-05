"""
Test suite for OncoMolML package.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from typing import Optional


# =============================================================================
# PyTorch Model Tests
# =============================================================================

class TestPyTorchModels:
    """Test PyTorch-based models."""
    
    @pytest.fixture
    def torch_available(self):
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_drug_response_net_forward(self, torch_available):
        """Test DrugResponseNet forward pass."""
        import torch
        from oncomolml.models.pytorch_models import DrugResponseNet
        
        model = DrugResponseNet(
            gene_dim=978,
            mol_dim=2048,
            hidden_dim=256,
            num_layers=2
        )
        
        batch_size = 4
        gene_expr = torch.randn(batch_size, 978)
        mol_fp = torch.randn(batch_size, 2048)
        
        output = model(gene_expr, mol_fp)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
    
    def test_drug_response_net_multitask(self, torch_available):
        """Test DrugResponseNet with multiple tasks."""
        import torch
        from oncomolml.models.pytorch_models import DrugResponseNet
        
        model = DrugResponseNet(
            gene_dim=500,
            mol_dim=1024,
            num_tasks=3
        )
        
        gene_expr = torch.randn(8, 500)
        mol_fp = torch.randn(8, 1024)
        
        output = model(gene_expr, mol_fp)
        
        assert output.shape == (8, 3)
    
    def test_mol_property_predictor(self, torch_available):
        """Test MolPropertyPredictor."""
        import torch
        from oncomolml.models.pytorch_models import MolPropertyPredictor
        
        model = MolPropertyPredictor(
            input_dim=2048,
            num_properties=12
        )
        
        fp = torch.randn(16, 2048)
        
        # Test without uncertainty
        output = model(fp)
        assert output.shape == (16, 12)
        
        # Test with uncertainty
        output, uncertainty = model(fp, return_uncertainty=True)
        assert output.shape == (16, 12)
        assert uncertainty.shape == (16, 12)
        assert (uncertainty > 0).all()  # Uncertainty should be positive
    
    def test_cancer_type_classifier(self, torch_available):
        """Test CancerTypeClassifier."""
        import torch
        from oncomolml.models.pytorch_models import CancerTypeClassifier
        
        model = CancerTypeClassifier(
            gene_dim=5000,
            num_cancer_types=18
        )
        
        gene_expr = torch.randn(8, 5000)
        
        logits = model(gene_expr)
        probs = model.predict_proba(gene_expr)
        
        assert logits.shape == (8, 18)
        assert probs.shape == (8, 18)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


# =============================================================================
# TensorFlow Model Tests
# =============================================================================

class TestTensorFlowModels:
    """Test TensorFlow-based models."""
    
    @pytest.fixture
    def tf_available(self):
        try:
            import tensorflow as tf
            return True
        except ImportError:
            pytest.skip("TensorFlow not installed")
    
    def test_variant_effect_predictor(self, tf_available):
        """Test VariantEffectPredictor."""
        import tensorflow as tf
        from oncomolml.models.tensorflow_models import VariantEffectPredictor
        
        model = VariantEffectPredictor(
            sequence_length=101,
            num_classes=3,
            filters=128,
            num_transformer_layers=2
        )
        
        # One-hot encoded sequences
        batch_size = 8
        seq_length = 101
        sequences = tf.random.uniform((batch_size, seq_length, 4))
        
        output = model(sequences)
        
        assert output.shape == (batch_size, 3)
    
    def test_sequence_encoder(self, tf_available):
        """Test SequenceEncoder."""
        import tensorflow as tf
        from oncomolml.models.tensorflow_models import SequenceEncoder
        
        encoder = SequenceEncoder(
            embed_dim=128,
            hidden_dim=256
        )
        
        sequences = tf.random.uniform((16, 500, 4))
        
        # Without attention
        embedding = encoder(sequences)
        assert embedding.shape == (16, 256)
        
        # With attention
        embedding, attention = encoder(sequences, return_attention=True)
        assert embedding.shape == (16, 256)
        assert attention.shape == (16, 500)
    
    def test_one_hot_encoding(self, tf_available):
        """Test one-hot encoding utility."""
        from oncomolml.models.tensorflow_models import (
            one_hot_encode_sequence,
            batch_one_hot_encode
        )
        
        seq = "ACGT"
        encoded = one_hot_encode_sequence(seq)
        
        assert encoded.shape == (4, 4)
        # Check A encoding
        assert encoded[0, 0] == 1
        # Check C encoding
        assert encoded[1, 1] == 1
    
    def test_cnv_predictor(self, tf_available):
        """Test CNVPredictor."""
        import tensorflow as tf
        from oncomolml.models.tensorflow_models import CNVPredictor
        
        model = CNVPredictor(
            window_size=256,
            num_features=4,
            num_states=5
        )
        
        depth_data = tf.random.uniform((8, 256, 4))
        
        output = model(depth_data)
        
        assert output.shape == (8, 256, 5)


# =============================================================================
# JAX Model Tests
# =============================================================================

class TestJAXModels:
    """Test JAX-based models."""
    
    @pytest.fixture
    def jax_available(self):
        try:
            import jax
            return True
        except ImportError:
            pytest.skip("JAX not installed")
    
    def test_morgan_fingerprint(self, jax_available):
        """Test Morgan fingerprint calculation."""
        import jax.numpy as jnp
        from oncomolml.models.jax_models import morgan_fingerprint_jax
        
        # Simple molecule representation
        num_atoms = 10
        atom_features = jnp.ones((num_atoms, 5))
        adjacency = jnp.eye(num_atoms)
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        
        fp = morgan_fingerprint_jax(atom_features, adjacency, radius=2, nbits=1024)
        
        assert fp.shape == (1024,)
        assert jnp.sum(fp) > 0  # Should have some bits set
    
    def test_tanimoto_similarity(self, jax_available):
        """Test Tanimoto similarity calculation."""
        import jax.numpy as jnp
        from oncomolml.models.jax_models import tanimoto_similarity
        
        fp1 = jnp.array([1, 1, 0, 0, 1])
        fp2 = jnp.array([1, 0, 0, 1, 1])
        
        sim = tanimoto_similarity(fp1, fp2)
        
        # intersection = 2, union = 4
        expected = 2 / 4
        assert jnp.isclose(sim, expected, atol=1e-5)
    
    def test_batch_tanimoto_matrix(self, jax_available):
        """Test batch Tanimoto similarity matrix."""
        import jax.numpy as jnp
        from oncomolml.models.jax_models import batch_tanimoto_matrix
        
        fps = jnp.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
        ])
        
        sim_matrix = batch_tanimoto_matrix(fps)
        
        assert sim_matrix.shape == (3, 3)
        # Diagonal should be 1 (self-similarity)
        assert jnp.allclose(jnp.diag(sim_matrix), 1.0, atol=1e-5)
        # Should be symmetric
        assert jnp.allclose(sim_matrix, sim_matrix.T, atol=1e-5)
    
    def test_lennard_jones(self, jax_available):
        """Test Lennard-Jones potential."""
        import jax.numpy as jnp
        from jax import grad
        from oncomolml.models.jax_models import lennard_jones_potential
        
        coords = jnp.array([
            [0.0, 0.0, 0.0],
            [3.4, 0.0, 0.0],  # At sigma distance
        ])
        
        energy = lennard_jones_potential(coords, sigma=3.4)
        
        # Energy should be finite
        assert jnp.isfinite(energy)
        
        # Test gradient computation
        grad_fn = grad(lennard_jones_potential)
        forces = -grad_fn(coords)
        
        assert forces.shape == coords.shape
        assert jnp.all(jnp.isfinite(forces))
    
    def test_mlp_forward(self, jax_available):
        """Test JAX MLP forward pass."""
        import jax
        import jax.numpy as jnp
        from oncomolml.models.jax_models import init_mlp_params, mlp_forward
        
        key = jax.random.PRNGKey(42)
        params = init_mlp_params([100, 50, 10], key)
        
        x = jax.random.normal(key, (32, 100))
        output = mlp_forward(params, x, activation="gelu")
        
        assert output.shape == (32, 10)


# =============================================================================
# Numba JIT Tests
# =============================================================================

class TestNumbaJIT:
    """Test Numba JIT-compiled functions."""
    
    @pytest.fixture
    def numba_available(self):
        try:
            import numba
            return True
        except ImportError:
            pytest.skip("Numba not installed")
    
    def test_fast_quality_filter(self, numba_available):
        """Test fast quality filtering."""
        from oncomolml.utils.jit_utils import fast_quality_filter
        
        # Generate test data
        np.random.seed(42)
        qualities = np.random.randint(15, 42, (1000, 150), dtype=np.int32)
        
        mask = fast_quality_filter(qualities, min_qual=30, min_length=50)
        
        assert mask.dtype == np.bool_
        assert len(mask) == 1000
        assert 0 < mask.sum() < 1000  # Some should pass, some fail
    
    def test_gc_content(self, numba_available):
        """Test GC content calculation."""
        from oncomolml.utils.jit_utils import fast_gc_content, batch_gc_content
        
        # ACGT = A(65), C(67), G(71), T(84)
        seq = np.array([65, 67, 71, 84], dtype=np.uint8)  # 50% GC
        
        gc = fast_gc_content(seq)
        assert np.isclose(gc, 0.5, atol=1e-5)
        
        # Batch test
        seqs = np.array([
            [67, 71, 67, 71],  # All GC
            [65, 84, 65, 84],  # All AT
        ], dtype=np.uint8)
        
        gc_batch = batch_gc_content(seqs)
        assert np.isclose(gc_batch[0], 1.0, atol=1e-5)
        assert np.isclose(gc_batch[1], 0.0, atol=1e-5)
    
    def test_kmer_counting(self, numba_available):
        """Test k-mer counting."""
        from oncomolml.utils.jit_utils import count_kmers_single, parallel_kmer_count
        
        # ACGT repeated
        seq = np.array([65, 67, 71, 84, 65, 67, 71, 84], dtype=np.uint8)
        
        counts = count_kmers_single(seq, k=2)
        
        assert counts.shape == (4**2,)
        assert np.sum(counts) == 7  # 7 2-mers in length-8 sequence
    
    def test_edit_distance(self, numba_available):
        """Test edit distance calculation."""
        from oncomolml.utils.jit_utils import edit_distance
        
        seq1 = np.array([65, 67, 71, 84], dtype=np.uint8)  # ACGT
        seq2 = np.array([65, 67, 84, 84], dtype=np.uint8)  # ACTT
        
        dist = edit_distance(seq1, seq2)
        
        assert dist == 1  # One substitution G->T
    
    def test_local_alignment(self, numba_available):
        """Test local alignment scoring."""
        from oncomolml.utils.jit_utils import local_alignment_score
        
        seq1 = np.array([65, 67, 71, 84], dtype=np.uint8)  # ACGT
        seq2 = np.array([65, 67, 71, 84], dtype=np.uint8)  # ACGT
        
        score = local_alignment_score(seq1, seq2, match=2, mismatch=-1, gap=-2)
        
        assert score == 8  # Perfect match: 4 * 2
    
    def test_sequence_hash(self, numba_available):
        """Test sequence hashing."""
        from oncomolml.utils.jit_utils import sequence_hash, batch_sequence_hash
        
        seq1 = np.array([65, 67, 71, 84], dtype=np.uint8)
        seq2 = np.array([65, 67, 71, 84], dtype=np.uint8)
        seq3 = np.array([84, 67, 71, 65], dtype=np.uint8)
        
        h1 = sequence_hash(seq1)
        h2 = sequence_hash(seq2)
        h3 = sequence_hash(seq3)
        
        assert h1 == h2  # Same sequence, same hash
        assert h1 != h3  # Different sequence, different hash


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelines:
    """Test pipeline functionality."""
    
    def test_pipeline_config(self):
        """Test pipeline configuration."""
        from oncomolml.pipelines import PipelineConfig
        
        config = PipelineConfig(
            min_read_quality=25,
            min_variant_freq=0.1,
            batch_size=64
        )
        
        assert config.min_read_quality == 25
        assert config.min_variant_freq == 0.1
        assert config.batch_size == 64
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from oncomolml.pipelines import CancerDrugPipeline
        
        pipeline = CancerDrugPipeline(
            variant_model="transformer",
            drug_model="attention",
            use_gpu=False
        )
        
        assert pipeline is not None
        assert pipeline.variant_model_type == "transformer"
    
    def test_pipeline_run_synthetic(self):
        """Test pipeline with synthetic data."""
        from oncomolml.pipelines import CancerDrugPipeline
        
        pipeline = CancerDrugPipeline(use_gpu=False)
        
        # Run with synthetic data (no files)
        results = pipeline.run(sample_id="test")
        
        assert results.sample_id == "test"
        assert len(results.variants) > 0
        assert len(results.drug_candidates) > 0
    
    def test_pipeline_results(self):
        """Test PipelineResults functionality."""
        from oncomolml.pipelines import (
            PipelineResults, 
            VariantResult, 
            DrugCandidate
        )
        
        results = PipelineResults(
            sample_id="test",
            qc_metrics={"pass_rate": 0.95},
            variants=[
                VariantResult("chr1", 100, "A", "G", "TP53", 0.9, "Pathogenic", "High")
            ],
            pathogenic_variants=[
                VariantResult("chr1", 100, "A", "G", "TP53", 0.9, "Pathogenic", "High")
            ],
            drug_candidates=[
                DrugCandidate("d1", "Drug1", "CCO", 0.5, 0.8, 0.7, "kinase"),
                DrugCandidate("d2", "Drug2", "CCC", 0.3, 0.9, 0.6, "protease"),
            ],
            summary={}
        )
        
        top = results.top_candidates(1)
        assert len(top) == 1
        assert top[0].drug_id == "d2"  # Lower IC50
        
        df = results.to_dataframe()
        assert len(df) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_framework_versions(self):
        """Test that we can detect available frameworks."""
        from oncomolml import get_available_backends
        
        backends = get_available_backends()
        assert isinstance(backends, list)
    
    def test_package_info(self):
        """Test package metadata."""
        import oncomolml
        
        assert hasattr(oncomolml, "__version__")
        assert oncomolml.__version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
