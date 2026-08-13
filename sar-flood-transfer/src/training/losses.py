"""Losses that respect Sen1Floods11's -1 (no-data) label, matching metrics.py."""

from __future__ import annotations

import tensorflow as tf


def masked_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Binary crossentropy, excluding pixels where y_true == -1.

    y_true: int/float tensor, any shape, values in {-1, 0, 1}.
    y_pred: float tensor, same shape, sigmoid outputs in [0, 1].
    """
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_clipped = tf.clip_by_value(y_true, 0.0, 1.0)  # avoid -1 reaching the BCE formula

    elementwise = tf.keras.backend.binary_crossentropy(y_true_clipped, y_pred)
    masked = elementwise * mask
    return tf.reduce_sum(masked) / (tf.reduce_sum(mask) + 1e-7)
