"""CUDA and Triton kernels.

``block_sparse``   exact attention over the selected key blocks.
``coarse_tail``    corrected coarse attention over the omitted blocks.
``closure_stats``  per-block key variance and pooled value-key covariance.
``detail_cache``   writing and applying the attention refinement.
"""
