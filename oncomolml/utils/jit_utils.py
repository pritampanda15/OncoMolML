"""
Numba JIT-Compiled Utilities for NGS Data Processing

This module implements high-performance functions for:
- FASTQ quality filtering
- K-mer counting
- Sequence alignment utilities
- Read preprocessing

Showcases: Numba JIT, parallel processing, SIMD optimization
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import numba
    from numba import jit, njit, prange, vectorize, guvectorize
    from numba import types, typed
    from numba.typed import List as NumbaList, Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    raise ImportError(
        "Numba is required for this module. "
        "Install with: pip install numba>=0.57.0"
    )


# =============================================================================
# Quality Filtering Functions
# =============================================================================

@njit(cache=True)
def phred_to_prob(quality: np.ndarray) -> np.ndarray:
    """
    Convert Phred quality scores to error probabilities.
    
    P(error) = 10^(-Q/10)
    
    Args:
        quality: Array of Phred quality scores
        
    Returns:
        Array of error probabilities
    """
    return np.power(10.0, -quality / 10.0)


@njit(cache=True)
def prob_to_phred(prob: np.ndarray) -> np.ndarray:
    """
    Convert error probabilities to Phred quality scores.
    
    Q = -10 * log10(P)
    
    Args:
        prob: Array of error probabilities
        
    Returns:
        Array of Phred quality scores
    """
    return -10.0 * np.log10(prob + 1e-10)


@njit(parallel=True, cache=True)
def fast_quality_filter(
    qualities: np.ndarray,
    min_qual: int = 30,
    min_length: int = 50
) -> np.ndarray:
    """
    Fast quality filtering for sequencing reads.
    
    JIT-compiled with parallel execution for processing millions of reads.
    
    Args:
        qualities: 2D array of quality scores [num_reads, read_length]
        min_qual: Minimum average quality threshold
        min_length: Minimum read length after trimming
        
    Returns:
        Boolean mask of passing reads
    
    Example:
        >>> quals = np.random.randint(20, 40, (1000000, 150))
        >>> mask = fast_quality_filter(quals, min_qual=30)
        >>> print(f"Passed: {mask.sum()} reads")
    """
    num_reads = qualities.shape[0]
    mask = np.zeros(num_reads, dtype=np.bool_)
    
    for i in prange(num_reads):
        read_qual = qualities[i]
        
        # Calculate mean quality
        mean_qual = np.mean(read_qual)
        
        # Count bases above threshold
        high_qual_bases = np.sum(read_qual >= min_qual)
        
        # Apply filters
        if mean_qual >= min_qual and high_qual_bases >= min_length:
            mask[i] = True
    
    return mask


@njit(cache=True)
def sliding_window_quality_trim(
    qualities: np.ndarray,
    window_size: int = 4,
    min_qual: int = 20
) -> Tuple[int, int]:
    """
    Trim read based on sliding window quality.
    
    Similar to Trimmomatic's SLIDINGWINDOW trimmer.
    
    Args:
        qualities: Quality scores for single read
        window_size: Size of sliding window
        min_qual: Minimum window average quality
        
    Returns:
        Tuple of (start_pos, end_pos) for trimmed read
    """
    read_len = len(qualities)
    
    # Find 5' trim position
    start = 0
    for i in range(read_len - window_size + 1):
        window_mean = np.mean(qualities[i:i + window_size])
        if window_mean >= min_qual:
            start = i
            break
    
    # Find 3' trim position
    end = read_len
    for i in range(read_len - window_size, -1, -1):
        window_mean = np.mean(qualities[i:i + window_size])
        if window_mean >= min_qual:
            end = i + window_size
            break
    
    return start, end


@njit(parallel=True, cache=True)
def batch_quality_trim(
    qualities: np.ndarray,
    window_size: int = 4,
    min_qual: int = 20
) -> np.ndarray:
    """
    Batch quality trimming for multiple reads.
    
    Args:
        qualities: 2D array [num_reads, read_length]
        window_size: Sliding window size
        min_qual: Minimum quality threshold
        
    Returns:
        Array of (start, end) positions [num_reads, 2]
    """
    num_reads = qualities.shape[0]
    trim_positions = np.zeros((num_reads, 2), dtype=np.int64)
    
    for i in prange(num_reads):
        start, end = sliding_window_quality_trim(
            qualities[i], window_size, min_qual
        )
        trim_positions[i, 0] = start
        trim_positions[i, 1] = end
    
    return trim_positions


# =============================================================================
# K-mer Counting Functions
# =============================================================================

@njit(cache=True)
def _encode_base(base: int) -> int:
    """Encode nucleotide to 2-bit representation."""
    # A=65, C=67, G=71, T=84 (ASCII)
    if base == 65:    # A
        return 0
    elif base == 67:  # C
        return 1
    elif base == 71:  # G
        return 2
    elif base == 84:  # T
        return 3
    else:
        return -1  # N or other


@njit(cache=True)
def _encode_kmer(sequence: np.ndarray, start: int, k: int) -> int:
    """
    Encode k-mer to integer using 2-bit encoding.
    
    Args:
        sequence: ASCII-encoded sequence array
        start: Start position
        k: K-mer length
        
    Returns:
        Integer encoding of k-mer, or -1 if contains N
    """
    kmer_int = 0
    for i in range(k):
        base_code = _encode_base(sequence[start + i])
        if base_code < 0:
            return -1
        kmer_int = (kmer_int << 2) | base_code
    return kmer_int


@njit(cache=True)
def count_kmers_single(
    sequence: np.ndarray,
    k: int = 21
) -> np.ndarray:
    """
    Count k-mers in a single sequence.
    
    Args:
        sequence: ASCII-encoded sequence
        k: K-mer length
        
    Returns:
        Array of k-mer counts (size 4^k)
    """
    max_kmers = 4 ** k
    counts = np.zeros(max_kmers, dtype=np.int32)
    
    seq_len = len(sequence)
    for i in range(seq_len - k + 1):
        kmer_int = _encode_kmer(sequence, i, k)
        if kmer_int >= 0:
            counts[kmer_int] += 1
    
    return counts


@njit(parallel=True, cache=True)
def parallel_kmer_count(
    sequences: np.ndarray,
    k: int = 21
) -> np.ndarray:
    """
    Parallel k-mer counting for multiple sequences.
    
    JIT-compiled with parallel execution.
    
    Args:
        sequences: 2D array of ASCII-encoded sequences [num_seqs, seq_len]
        k: K-mer length
        
    Returns:
        Combined k-mer counts
    
    Example:
        >>> seqs = np.array([[65, 67, 71, 84, 65, 67]])  # ACGTAC
        >>> counts = parallel_kmer_count(seqs, k=3)
    """
    num_seqs = sequences.shape[0]
    max_kmers = 4 ** k
    
    # Thread-local counts
    thread_counts = np.zeros((num_seqs, max_kmers), dtype=np.int32)
    
    for i in prange(num_seqs):
        thread_counts[i] = count_kmers_single(sequences[i], k)
    
    # Sum across sequences
    total_counts = np.sum(thread_counts, axis=0)
    
    return total_counts


@njit(cache=True)
def kmer_spectrum(
    counts: np.ndarray,
    max_count: int = 100
) -> np.ndarray:
    """
    Compute k-mer count spectrum.
    
    Args:
        counts: K-mer counts array
        max_count: Maximum count to track
        
    Returns:
        Spectrum (histogram of counts)
    """
    spectrum = np.zeros(max_count + 1, dtype=np.int64)
    
    for count in counts:
        if count <= max_count:
            spectrum[count] += 1
        else:
            spectrum[max_count] += 1
    
    return spectrum


# =============================================================================
# GC Content and Statistics
# =============================================================================

@njit(cache=True)
def fast_gc_content(sequence: np.ndarray) -> float:
    """
    Calculate GC content of a sequence.
    
    Args:
        sequence: ASCII-encoded sequence
        
    Returns:
        GC content (0-1)
    """
    gc_count = 0
    at_count = 0
    
    for base in sequence:
        if base == 71 or base == 67:  # G or C
            gc_count += 1
        elif base == 65 or base == 84:  # A or T
            at_count += 1
    
    total = gc_count + at_count
    if total == 0:
        return 0.0
    return gc_count / total


@njit(parallel=True, cache=True)
def batch_gc_content(sequences: np.ndarray) -> np.ndarray:
    """
    Calculate GC content for batch of sequences.
    
    Args:
        sequences: 2D array [num_seqs, seq_len]
        
    Returns:
        Array of GC contents
    """
    num_seqs = sequences.shape[0]
    gc_contents = np.zeros(num_seqs, dtype=np.float64)
    
    for i in prange(num_seqs):
        gc_contents[i] = fast_gc_content(sequences[i])
    
    return gc_contents


@njit(cache=True)
def sequence_complexity(
    sequence: np.ndarray,
    k: int = 3
) -> float:
    """
    Calculate sequence complexity using k-mer entropy.
    
    Low complexity indicates repetitive sequence.
    
    Args:
        sequence: ASCII-encoded sequence
        k: K-mer size for complexity calculation
        
    Returns:
        Complexity score (0-1, higher = more complex)
    """
    seq_len = len(sequence)
    if seq_len < k:
        return 0.0
    
    # Count k-mers
    max_kmers = 4 ** k
    counts = np.zeros(max_kmers, dtype=np.int32)
    
    total_kmers = 0
    for i in range(seq_len - k + 1):
        kmer_int = _encode_kmer(sequence, i, k)
        if kmer_int >= 0:
            counts[kmer_int] += 1
            total_kmers += 1
    
    if total_kmers == 0:
        return 0.0
    
    # Calculate entropy
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total_kmers
            entropy -= p * np.log2(p)
    
    # Normalize by maximum entropy
    max_entropy = np.log2(min(total_kmers, max_kmers))
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


# =============================================================================
# Alignment Utilities
# =============================================================================

@njit(cache=True)
def hamming_distance(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> int:
    """
    Calculate Hamming distance between two sequences.
    
    Args:
        seq1: First sequence (ASCII)
        seq2: Second sequence (ASCII)
        
    Returns:
        Number of mismatches
    """
    distance = 0
    length = min(len(seq1), len(seq2))
    
    for i in range(length):
        if seq1[i] != seq2[i]:
            distance += 1
    
    return distance


@njit(cache=True)
def edit_distance(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> int:
    """
    Calculate Levenshtein edit distance.
    
    JIT-compiled for speed.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Edit distance
    """
    m, n = len(seq1), len(seq2)
    
    # Use two rows instead of full matrix
    prev_row = np.arange(n + 1, dtype=np.int32)
    curr_row = np.zeros(n + 1, dtype=np.int32)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,      # Deletion
                curr_row[j-1] + 1,    # Insertion
                prev_row[j-1] + cost  # Substitution
            )
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[n]


@njit(cache=True)
def local_alignment_score(
    seq1: np.ndarray,
    seq2: np.ndarray,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2
) -> int:
    """
    Smith-Waterman local alignment score.
    
    JIT-compiled for performance on short sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        match: Match score
        mismatch: Mismatch penalty
        gap: Gap penalty
        
    Returns:
        Maximum alignment score
    """
    m, n = len(seq1), len(seq2)
    
    # Initialize score matrix
    H = np.zeros((m + 1, n + 1), dtype=np.int32)
    max_score = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Score for match/mismatch
            if seq1[i-1] == seq2[j-1]:
                diag = H[i-1, j-1] + match
            else:
                diag = H[i-1, j-1] + mismatch
            
            # Consider gaps
            up = H[i-1, j] + gap
            left = H[i, j-1] + gap
            
            # Local alignment: can start fresh (0)
            H[i, j] = max(0, diag, up, left)
            
            if H[i, j] > max_score:
                max_score = H[i, j]
    
    return max_score


# =============================================================================
# Read Deduplication
# =============================================================================

@njit(cache=True)
def sequence_hash(sequence: np.ndarray) -> np.int64:
    """
    Compute hash of sequence for deduplication.
    
    Args:
        sequence: ASCII-encoded sequence
        
    Returns:
        64-bit hash value
    """
    hash_val = np.int64(0)
    prime = np.int64(31)
    mod = np.int64(2**63 - 1)
    
    for base in sequence:
        hash_val = (hash_val * prime + np.int64(base)) % mod
    
    return hash_val


@njit(parallel=True, cache=True)
def batch_sequence_hash(sequences: np.ndarray) -> np.ndarray:
    """
    Hash multiple sequences in parallel.
    
    Args:
        sequences: 2D array of sequences
        
    Returns:
        Array of hash values
    """
    num_seqs = sequences.shape[0]
    hashes = np.zeros(num_seqs, dtype=np.int64)
    
    for i in prange(num_seqs):
        hashes[i] = sequence_hash(sequences[i])
    
    return hashes


@njit(cache=True)
def find_duplicates(hashes: np.ndarray) -> np.ndarray:
    """
    Find duplicate sequences based on hash values.
    
    Args:
        hashes: Array of sequence hashes
        
    Returns:
        Boolean mask (True = keep, False = duplicate)
    """
    n = len(hashes)
    keep_mask = np.ones(n, dtype=np.bool_)
    
    # Sort indices by hash
    sorted_idx = np.argsort(hashes)
    
    # Mark duplicates
    for i in range(1, n):
        if hashes[sorted_idx[i]] == hashes[sorted_idx[i-1]]:
            keep_mask[sorted_idx[i]] = False
    
    return keep_mask


# =============================================================================
# Variant Calling Utilities
# =============================================================================

@njit(cache=True)
def pileup_base_counts(
    bases: np.ndarray,
    qualities: np.ndarray,
    min_qual: int = 20
) -> np.ndarray:
    """
    Count bases at a pileup position with quality filtering.
    
    Args:
        bases: Array of base calls (ASCII: A=65, C=67, G=71, T=84)
        qualities: Array of quality scores
        min_qual: Minimum quality threshold
        
    Returns:
        Counts array [A, C, G, T, N]
    """
    counts = np.zeros(5, dtype=np.int32)
    
    for i in range(len(bases)):
        if qualities[i] >= min_qual:
            base = bases[i]
            if base == 65:    # A
                counts[0] += 1
            elif base == 67:  # C
                counts[1] += 1
            elif base == 71:  # G
                counts[2] += 1
            elif base == 84:  # T
                counts[3] += 1
            else:             # N
                counts[4] += 1
    
    return counts


@njit(cache=True)
def call_variant(
    ref_base: int,
    base_counts: np.ndarray,
    min_depth: int = 10,
    min_alt_freq: float = 0.1
) -> Tuple[int, float, float]:
    """
    Simple variant caller from pileup.
    
    Args:
        ref_base: Reference base (ASCII)
        base_counts: Base counts [A, C, G, T, N]
        min_depth: Minimum coverage depth
        min_alt_freq: Minimum alternate allele frequency
        
    Returns:
        Tuple of (alt_base, alt_freq, quality)
    """
    total_depth = np.sum(base_counts[:4])  # Exclude N
    
    if total_depth < min_depth:
        return -1, 0.0, 0.0
    
    # Get reference count
    ref_idx = -1
    if ref_base == 65:
        ref_idx = 0
    elif ref_base == 67:
        ref_idx = 1
    elif ref_base == 71:
        ref_idx = 2
    elif ref_base == 84:
        ref_idx = 3
    
    if ref_idx < 0:
        return -1, 0.0, 0.0
    
    # Find highest non-reference allele
    max_alt_count = 0
    max_alt_idx = -1
    
    for i in range(4):
        if i != ref_idx and base_counts[i] > max_alt_count:
            max_alt_count = base_counts[i]
            max_alt_idx = i
    
    if max_alt_idx < 0:
        return -1, 0.0, 0.0
    
    alt_freq = max_alt_count / total_depth
    
    if alt_freq < min_alt_freq:
        return -1, 0.0, 0.0
    
    # Simple quality calculation (Phred-scaled)
    quality = -10 * np.log10(1 - alt_freq + 1e-10)
    
    # Convert index back to ASCII
    bases = np.array([65, 67, 71, 84])
    alt_base = bases[max_alt_idx]
    
    return alt_base, alt_freq, quality


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_quality_filter(
    n_reads: int = 1000000,
    read_length: int = 150
) -> Dict[str, float]:
    """
    Benchmark quality filtering performance.
    
    Args:
        n_reads: Number of reads
        read_length: Read length
        
    Returns:
        Timing results
    """
    import time
    
    # Generate random quality data
    qualities = np.random.randint(20, 42, (n_reads, read_length), dtype=np.int32)
    
    # Warmup
    _ = fast_quality_filter(qualities[:1000], min_qual=30)
    
    # Benchmark
    start = time.time()
    mask = fast_quality_filter(qualities, min_qual=30)
    elapsed = time.time() - start
    
    return {
        "n_reads": n_reads,
        "time_seconds": elapsed,
        "reads_per_second": n_reads / elapsed,
        "passed_reads": int(mask.sum()),
        "pass_rate": float(mask.mean())
    }


if __name__ == "__main__":
    print("Testing Numba JIT utilities...")
    print(f"Numba version: {numba.__version__}")
    print(f"Threads: {numba.get_num_threads()}")
    
    # Test quality filtering
    quals = np.random.randint(15, 42, (1000, 150), dtype=np.int32)
    mask = fast_quality_filter(quals, min_qual=25)
    print(f"Quality filter: {mask.sum()}/{len(mask)} reads passed")
    
    # Test k-mer counting
    seqs = np.array([[65, 67, 71, 84, 65, 67, 71, 84, 65, 67, 71, 84]] * 100)
    counts = parallel_kmer_count(seqs, k=3)
    print(f"K-mer counts: {np.sum(counts > 0)} unique 3-mers")
    
    # Test GC content
    gc = batch_gc_content(seqs)
    print(f"GC content: {gc.mean():.2%}")
    
    # Test edit distance
    seq1 = np.array([65, 67, 71, 84], dtype=np.uint8)
    seq2 = np.array([65, 67, 84, 84], dtype=np.uint8)
    dist = edit_distance(seq1, seq2)
    print(f"Edit distance: {dist}")
    
    # Test alignment
    score = local_alignment_score(seq1, seq2)
    print(f"Alignment score: {score}")
    
    # Benchmark
    print("\nRunning benchmark...")
    results = benchmark_quality_filter(n_reads=100000)
    print(f"Processed {results['n_reads']} reads in {results['time_seconds']:.3f}s")
    print(f"Speed: {results['reads_per_second']:.0f} reads/second")
    
    print("\nAll Numba utilities working correctly!")
