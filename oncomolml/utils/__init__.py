"""Utilities submodule - JIT-compiled functions and data utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jit_utils import (
        fast_quality_filter,
        parallel_kmer_count,
        batch_gc_content,
        edit_distance,
        local_alignment_score,
    )

__all__ = [
    "fast_quality_filter",
    "parallel_kmer_count",
    "batch_gc_content",
    "edit_distance",
    "local_alignment_score",
]
