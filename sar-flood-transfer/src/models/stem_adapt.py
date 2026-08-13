"""Adapt a 3-channel (RGB) conv stem to single-channel (grayscale) input.

Per CHANNEL_COMPATIBILITY.md: averaging a real ImageNet-pretrained stem's RGB
channel weights down to 1 channel is a well-established, unremarkable
technique -- the RGB channels being averaged are physically meaningful
photographic channels, so the average preserves the filter's learned
response shape. This is legitimate specifically because the source weights
come from a genuine RGB backbone (e.g. ResNet50V2 ImageNet weights); it would
NOT be legitimate for weights whose channels have no physical meaning (that
was the reasoning for rejecting a similar-looking averaging of SpillNet's
RGB-JPEG-rendered channels in the superseded design).
"""

from __future__ import annotations

import numpy as np


def average_rgb_stem_to_grayscale(rgb_weights: np.ndarray) -> np.ndarray:
    """Average a Conv2D kernel's input-channel axis from 3 down to 1.

    Args:
        rgb_weights: array of shape (kh, kw, 3, out_channels) -- a standard
            Keras Conv2D kernel with 3 input channels.

    Returns:
        array of shape (kh, kw, 1, out_channels), the per-output-filter mean
        across the 3 input channels.
    """
    rgb_weights = np.asarray(rgb_weights)
    if rgb_weights.ndim != 4:
        raise ValueError(f"Expected a 4D Conv2D kernel (kh, kw, in_ch, out_ch), got shape {rgb_weights.shape}")
    in_channels = rgb_weights.shape[2]
    if in_channels != 3:
        raise ValueError(f"Expected 3 input channels (RGB), got {in_channels}")

    grayscale_weights = rgb_weights.mean(axis=2, keepdims=True)
    assert grayscale_weights.shape == (*rgb_weights.shape[:2], 1, rgb_weights.shape[3])
    return grayscale_weights.astype(rgb_weights.dtype)
