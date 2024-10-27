import tensorflow as tf


class CustomMetric:
    def __init__(self, ref_center, R_c, config):
        self.ref_center = ref_center
        self.R_c = R_c
        self.weights = tf.transpose(tf.constant([[config.metrics.x_weight, config.metrics.y_weight, config.metrics.angle_weight]], dtype=tf.float64))

    def custom_metric(self, y_true, y_pred):
        # reshape labels of each element of the batch such that the rows are the coord dimensions
        C_p = tf.cast(tf.reshape(y_pred, [-1, 4, 2]), dtype=tf.float64)
        # transpose each entry in the tensor
        C_p = tf.einsum('bij->bji', C_p)

        # previous two operations for the actual labels
        C_t = tf.cast(tf.reshape(y_true, [-1, 4, 2]), dtype=tf.float64)
        C_t = tf.einsum('bij->bji', C_t)

        # helper matrix to calculate the necessary edge vectors
        selector = tf.constant([[1.],
                                [0.],
                                [-1.],
                                [0.]], dtype=tf.float64)

        # calculating the center of all four points for both the actual and predicted labels
        center_xt = tf.reshape(tf.math.reduce_mean(C_t[:, 0, :], axis=1), [-1, 1])
        center_yt = tf.reshape(tf.math.reduce_mean(C_t[:, 1, :], axis=1), [-1, 1])

        center_xp = tf.reshape(tf.math.reduce_mean(C_p[:, 0, :], axis=1), [-1, 1])
        center_yp = tf.reshape(tf.math.reduce_mean(C_p[:, 1, :], axis=1), [-1, 1])

        center_t = tf.concat([center_xt, center_yt], axis=1)
        center_p = tf.concat([center_xp, center_yp], axis=1)

        # difference between the properly placed center and the misplaced center
        offset_t = self.ref_center - center_t
        offset_p = self.ref_center - center_p

        # mean error of the x and y offset predictions
        delta_xy = tf.reshape(tf.reduce_mean(offset_t - offset_p, axis=0), [1, -1])

        # extract the necessary edge vector from the actual and predicted labels
        et_13 = tf.linalg.matmul(C_t, selector)
        et_13 = tf.reshape(et_13, [-1, 1, et_13.shape[1]])

        ep_13 = tf.linalg.matmul(C_p, selector)
        ep_13 = tf.reshape(ep_13, [-1, 1, ep_13.shape[1]])

        # reference edge for angle calculation
        e_r = tf.linalg.matmul(self.R_c, selector)

        # scalar product for the left edge and the upward vector (numerator)
        scalar_t = tf.linalg.matmul(et_13, e_r)

        # scalar product for the left edge and the upward vector (numerator)
        scalar_p = tf.linalg.matmul(ep_13, e_r)

        # norms of the all vectors (denominator)
        e_r_norm = tf.norm(e_r)

        et_13_norm = tf.norm(et_13, axis=2)
        et_13_norm = tf.reshape(et_13_norm, [-1, 1, et_13_norm.shape[1]])

        ep_13_norm = tf.norm(ep_13, axis=2)
        ep_13_norm = tf.reshape(ep_13_norm, [-1, 1, ep_13_norm.shape[1]])

        # angle calculation for the actual and predicted labels (arccos of angle formula for angle between two vectors)
        angle_t = tf.math.acos(tf.math.divide_no_nan(scalar_t, tf.math.multiply_no_nan(et_13_norm, e_r_norm)))
        angle_p = tf.math.acos(tf.math.divide_no_nan(scalar_p, tf.math.multiply_no_nan(ep_13_norm, e_r_norm)))

        # mean error of the angle prediction
        delta_angle = tf.reshape(tf.reduce_mean(angle_t - angle_p), [1, -1])

        # unify the errors of the coordinates and angles in a single vector
        delta = tf.math.abs(tf.concat([delta_xy, delta_angle], axis=1))

        # weighted sum of all mean errors as our metric
        metric = tf.linalg.matmul(delta, self.weights)

        return metric
