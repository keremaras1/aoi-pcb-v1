"""Custom evaluation metric for IC keypoint alignment quality.

Measures prediction quality along three axes:

- **X offset error** — mean horizontal displacement of the predicted IC centre.
- **Y offset error** — mean vertical displacement of the predicted IC centre.
- **Angle error** — mean rotation angle difference between predicted and true
  edge vectors, computed via the dot-product angle formula.

The three errors are combined as a weighted sum, with weights configurable
via ``config.json`` under the ``metrics`` section.
"""

import tensorflow as tf

# Selects the edge vector between corner 0 and corner 2 (the "left" edge),
# used as the reference direction for angle calculation.
# Shape: (4, 1)
_EDGE_SELECTOR = tf.constant(
    [[1.], [0.], [-1.], [0.]],
    dtype=tf.float64,
)


class KeypointAlignmentMetric:
    """Weighted alignment metric comparing predicted and true IC keypoints.

    Computes a scalar metric from the mean centre-position error (x, y) and
    the mean rotation angle error, combined as a weighted dot product.

    Example::

        metric = KeypointAlignmentMetric(ref_center, ref_coords, config)
        score = metric(y_true, y_pred)
    """

    def __init__(
        self,
        ref_center: tf.Tensor,
        ref_coords: tf.Tensor,
        config: object,
    ) -> None:
        """Initialise the metric with reference keypoints and weight config.

        Args:
            ref_center: Reference IC centre position, shape ``(2,)``.
            ref_coords: Reference corner coordinates, shape ``(8,)``.
            config: Config instance; reads ``config.metrics.x_weight``,
                ``config.metrics.y_weight``, and ``config.metrics.angle_weight``.
        """
        self.R_center = tf.cast(tf.reshape(ref_center, [-1, 2]), dtype=tf.float64)
        self.R_c = tf.cast(tf.transpose(tf.reshape(ref_coords, [-1, 2])), dtype=tf.float64)
        self.weights = tf.transpose(
            tf.constant(
                [[config.metrics.x_weight, config.metrics.y_weight, config.metrics.angle_weight]],
                dtype=tf.float64,
            )
        )

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the weighted alignment metric for a batch of predictions.

        Args:
            y_true: Ground-truth keypoint coordinates, shape ``(batch, 8)``.
            y_pred: Predicted keypoint coordinates, shape ``(batch, 8)``.

        Returns:
            Scalar metric tensor (float64). Lower is better.
        """
        # Reshape to (batch, 4, 2) then transpose to (batch, 2, 4)
        C_p = tf.einsum('bij->bji', tf.cast(tf.reshape(y_pred, [-1, 4, 2]), dtype=tf.float64))
        C_t = tf.einsum('bij->bji', tf.cast(tf.reshape(y_true, [-1, 4, 2]), dtype=tf.float64))

        # Compute predicted and true IC centres as the mean of the four corners
        center_t = tf.concat([
            tf.reshape(tf.math.reduce_mean(C_t[:, 0, :], axis=1), [-1, 1]),
            tf.reshape(tf.math.reduce_mean(C_t[:, 1, :], axis=1), [-1, 1]),
        ], axis=1)
        center_p = tf.concat([
            tf.reshape(tf.math.reduce_mean(C_p[:, 0, :], axis=1), [-1, 1]),
            tf.reshape(tf.math.reduce_mean(C_p[:, 1, :], axis=1), [-1, 1]),
        ], axis=1)

        # Centre offsets relative to the reference position
        offset_t = self.R_center - center_t
        offset_p = self.R_center - center_p

        # Mean x/y error across the batch
        delta_xy = tf.reshape(tf.reduce_mean(offset_t - offset_p, axis=0), [1, -1])

        # Extract the reference edge vector and the corresponding predicted/true edges
        e_r = tf.linalg.matmul(self.R_c, _EDGE_SELECTOR)

        et_13 = tf.reshape(tf.linalg.matmul(C_t, _EDGE_SELECTOR), [-1, 1, 2])
        ep_13 = tf.reshape(tf.linalg.matmul(C_p, _EDGE_SELECTOR), [-1, 1, 2])

        # Compute angles via dot-product formula: angle = arccos(a·b / (|a||b|))
        e_r_norm = tf.norm(e_r)
        et_13_norm = tf.reshape(tf.norm(et_13, axis=2), [-1, 1, 1])
        ep_13_norm = tf.reshape(tf.norm(ep_13, axis=2), [-1, 1, 1])

        angle_t = tf.math.acos(
            tf.math.divide_no_nan(
                tf.linalg.matmul(et_13, e_r),
                tf.math.multiply_no_nan(et_13_norm, e_r_norm),
            )
        )
        angle_p = tf.math.acos(
            tf.math.divide_no_nan(
                tf.linalg.matmul(ep_13, e_r),
                tf.math.multiply_no_nan(ep_13_norm, e_r_norm),
            )
        )

        # Mean angle error across the batch
        delta_angle = tf.reshape(tf.reduce_mean(angle_t - angle_p), [1, -1])

        # Weighted sum of |x error|, |y error|, |angle error|
        delta = tf.math.abs(tf.concat([delta_xy, delta_angle], axis=1))
        return tf.linalg.matmul(delta, self.weights)
