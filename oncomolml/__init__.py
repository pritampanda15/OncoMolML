"""
OncoMolML: High-Performance ML Toolkit for Cancer Drug Discovery and NGS Analysis

A unified framework demonstrating PyTorch, TensorFlow, JAX, and JIT compilation
for computational oncology applications.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from typing import List

# Lazy imports to avoid loading all frameworks at once
_AVAILABLE_BACKENDS: List[str] = []


def _check_pytorch() -> bool:
    try:
        import torch
        _AVAILABLE_BACKENDS.append("pytorch")
        return True
    except ImportError:
        return False


def _check_tensorflow() -> bool:
    try:
        import tensorflow
        _AVAILABLE_BACKENDS.append("tensorflow")
        return True
    except ImportError:
        return False


def _check_jax() -> bool:
    try:
        import jax
        _AVAILABLE_BACKENDS.append("jax")
        return True
    except ImportError:
        return False


def _check_numba() -> bool:
    try:
        import numba
        _AVAILABLE_BACKENDS.append("numba")
        return True
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """Return list of available ML backends."""
    if not _AVAILABLE_BACKENDS:
        _check_pytorch()
        _check_tensorflow()
        _check_jax()
        _check_numba()
    return _AVAILABLE_BACKENDS.copy()


def print_system_info():
    """Print system and backend information."""
    print(f"OncoMolML v{__version__}")
    print("-" * 40)
    
    backends = get_available_backends()
    
    if "pytorch" in backends:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    
    if "tensorflow" in backends:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    if "jax" in backends:
        import jax
        print(f"JAX: {jax.__version__}")
        print(f"  Devices: {jax.devices()}")
    
    if "numba" in backends:
        import numba
        print(f"Numba: {numba.__version__}")


# Expose main classes at package level for convenience
__all__ = [
    "__version__",
    "get_available_backends",
    "print_system_info",
]
