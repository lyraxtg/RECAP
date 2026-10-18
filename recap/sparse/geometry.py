"""3-D Hilbert order of the video tokens, so that each 128-token block is a compact
spatio-temporal region.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def validate_permutation(permutation: np.ndarray, grid: Sequence[int]) -> np.ndarray:
    """Check a reordered-position -> native-index permutation."""
    grid = tuple(int(x) for x in grid)
    n = int(np.prod(grid))
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (n,):
        raise ValueError(f"permutation shape {permutation.shape} != ({n},)")
    if permutation.min(initial=0) != 0 or permutation.max(initial=-1) != n - 1:
        raise ValueError("permutation is outside [0, prod(grid))")
    if np.unique(permutation).size != n:
        raise ValueError("permutation is not one-to-one")
    return permutation


def hilbert_permutation(grid: Sequence[int]) -> np.ndarray:
    """Video tokens ordered along a 3-D Hilbert curve over ``grid = (T, H, W)``.

    Returns the reordered-position -> native-index permutation (native index
    ``t*H*W + y*W + x``).
    """
    frames, height, width = (int(x) for x in grid)
    bits = max(1, math.ceil(math.log2(max(frames, height, width))))
    t, y, x = np.meshgrid(
        np.arange(frames), np.arange(height), np.arange(width), indexing="ij"
    )
    native = (t * (height * width) + y * width + x).ravel()
    # Axes to transposed Hilbert index, vectorised over tokens, x as the leading axis.
    axes = [x.ravel().astype(np.int64), y.ravel().astype(np.int64),
            t.ravel().astype(np.int64)]
    top = 1 << (bits - 1)
    q = top
    while q > 1:
        low = q - 1
        for i in range(3):
            hit = (axes[i] & q) != 0
            axes[0] = np.where(hit, axes[0] ^ low, axes[0])
            swap = np.where(hit, 0, (axes[0] ^ axes[i]) & low)
            axes[0] ^= swap
            axes[i] ^= swap
        q >>= 1
    for i in range(1, 3):
        axes[i] ^= axes[i - 1]
    flip = np.zeros_like(axes[0])
    q = top
    while q > 1:
        flip = np.where((axes[2] & q) != 0, flip ^ (q - 1), flip)
        q >>= 1
    for i in range(3):
        axes[i] ^= flip
    # Interleave the transposed bits, most significant first, into one index.
    index = np.zeros_like(axes[0])
    for b in range(bits - 1, -1, -1):
        for i in range(3):
            index = (index << 1) | ((axes[i] >> b) & 1)
    return validate_permutation(native[np.argsort(index, kind="stable")], grid)
