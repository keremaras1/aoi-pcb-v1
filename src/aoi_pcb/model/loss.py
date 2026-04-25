"""Custom loss function combining coordinate MSE with a perpendicularity penalty.

The loss has two components:

1. **MSE loss** — standard mean squared error between predicted and true
   keypoint coordinates, penalising positional inaccuracy.

2. **Perpendicularity loss** — enforces that the four predicted edge vectors
   form right angles, encouraging the model to output valid rectangles rather
   than arbitrary quadrilaterals. Computed as the mean squared cosine of each
   interior angle; this term is zero when all edges are perpendicular.

The two terms are summed to form the final scalar loss.
"""

import tensorflow as tf

# Selects and signs the four edge vectors from the corner coordinate matrix.
# Each column encodes one edge as the signed sum of two corner positions.
# Shape: (4, 4) — applied as C @ _EDGE_SELECTOR where C is (batch, 2, 4).
_EDGE_SELECTOR = tf.constant(
    [[-1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, -1.0, -1.0, 0.0]],
    dtype=tf.float64,
)

# Reorders norm entries so that norm[i] aligns with its adjacent norm[i+1]
# for computing pairwise products in the perpendicularity denominator.
# Shape: (4, 4) — cyclic permutation of the identity.
_NORMS_SELECTOR = tf.constant(
    [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=tf.float64,
)


@tf.function
def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # pragma: no cover
    """Compute the combined MSE + perpendicularity loss.

    Args:
        y_true: Ground-truth keypoint coordinates, shape ``(batch, 8)``.
            Flattened as [x0, y0, x1, y1, x2, y2, x3, y3].
        y_pred: Predicted keypoint coordinates, same shape as ``y_true``.

    Returns:
        Scalar loss tensor (float64).
    """
    # Reshape to (batch, 4, 2) then transpose to (batch, 2, 4) so each
    # column holds the x and y coordinates of one corner
    C_p = tf.cast(tf.reshape(y_pred, [-1, 4, 2]), dtype=tf.float64)
    C_p = tf.einsum("bij->bji", C_p)

    # Compute the four edge vectors: E = C @ _EDGE_SELECTOR
    # Each column of E is a 2D vector representing one side of the rectangle
    E_p = tf.linalg.matmul(C_p, _EDGE_SELECTOR)
    E_p_T = tf.einsum("bij->bji", E_p)

    # E^T @ E is symmetric; its diagonal holds squared edge norms and its
    # off-diagonal entries hold pairwise dot products between edges
    E = tf.linalg.matmul(E_p_T, E_p)

    # Extract edge norms from the diagonal, reshape for broadcasting
    norms = tf.math.sqrt(tf.linalg.diag_part(E))
    norms = tf.reshape(norms, [-1, 1, norms.shape[1]])

    # Compute reciprocal of each adjacent norm product (the cos denominator)
    norms_denominator = tf.math.reciprocal(norms * tf.linalg.matmul(norms, _NORMS_SELECTOR))
    norms_denominator_T = tf.einsum("bij->bji", norms_denominator)

    # Collect the four adjacent edge dot products (cos numerators):
    # edges 0-1, 1-2, 2-3 are on the first subdiagonal; edge 3-0 is E[-1, 0]
    scalar_123 = tf.linalg.diag_part(E, k=-1)
    scalar_4 = tf.reshape(E[:, -1, 0], [-1, 1])
    e_scalar = tf.reshape(tf.concat([scalar_123, scalar_4], 1), [-1, 1, 4])

    # cos(angle_i) = dot(e_i, e_{i+1}) / (|e_i| * |e_{i+1}|)
    # perpendicularity_fac → 0 when all angles are 90°
    perpendicularity_fac = tf.linalg.matmul(e_scalar, norms_denominator_T)

    perp_loss = tf.reduce_mean(tf.math.square(perpendicularity_fac))
    mse_loss = tf.reduce_mean(tf.math.square(tf.cast(y_true - y_pred, dtype=tf.float64)))

    return mse_loss + perp_loss
