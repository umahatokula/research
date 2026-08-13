import numpy as np
import tensorflow as tf

from src.training.losses import masked_binary_crossentropy


def test_ignores_negative_one_pixels():
    # Two identical predictions except the -1 pixel's y_pred is wildly wrong;
    # loss should be identical since that pixel is masked out.
    y_true_a = tf.constant([[1.0, 0.0, -1.0]])
    y_true_b = tf.constant([[1.0, 0.0, -1.0]])
    y_pred_a = tf.constant([[0.9, 0.1, 0.5]])
    y_pred_b = tf.constant([[0.9, 0.1, 0.999]])  # wildly different at the ignored pixel

    loss_a = masked_binary_crossentropy(y_true_a, y_pred_a).numpy()
    loss_b = masked_binary_crossentropy(y_true_b, y_pred_b).numpy()
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5)


def test_matches_plain_bce_when_no_ignored_pixels():
    y_true = tf.constant([[1.0, 0.0, 1.0, 0.0]])
    y_pred = tf.constant([[0.8, 0.2, 0.6, 0.3]])
    ours = masked_binary_crossentropy(y_true, y_pred).numpy()
    plain = tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true, y_pred)).numpy()
    np.testing.assert_allclose(ours, plain, rtol=1e-5)


def test_all_ignored_returns_near_zero_not_nan():
    y_true = tf.constant([[-1.0, -1.0]])
    y_pred = tf.constant([[0.1, 0.9]])
    loss = masked_binary_crossentropy(y_true, y_pred).numpy()
    assert np.isfinite(loss)


def test_loss_is_higher_for_worse_predictions():
    y_true = tf.constant([[1.0, 0.0]])
    good_pred = tf.constant([[0.95, 0.05]])
    bad_pred = tf.constant([[0.05, 0.95]])
    good_loss = masked_binary_crossentropy(y_true, good_pred).numpy()
    bad_loss = masked_binary_crossentropy(y_true, bad_pred).numpy()
    assert bad_loss > good_loss
