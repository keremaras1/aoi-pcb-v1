import tensorflow as tf


@tf.function
def custom_loss(y_true, y_pred):
    # Reshape labels of each element of the batch such that the rows are the coord dimensions
    C_p = tf.cast(tf.reshape(y_pred, [-1, 4, 2]), dtype=tf.float64)
    # Transpose each entry in the tensor
    C_p = tf.einsum('bij->bji', C_p)

    # Create matrix to select the points to be subtracted
    c_selector = tf.constant([[-1.0, 0.0, 0.0, -1.0],
                              [0.0, 0.0, 1.0, 1.0],
                              [1.0, 1.0, 0.0, 0.0],
                              [0.0, -1.0, -1.0, 0.0]], dtype=tf.float64)

    # Dot product of C_p and C_selector for each element of the batch (implicit broadcasting)
    E_p = tf.linalg.matmul(C_p, c_selector)

    # Transpose each entry of the tensor
    E_p_transpose = tf.einsum('bij->bji', E_p)

    # Take the dot product of the matrix with itself to create a new matrix.
    # New matrix is symmetrical and contains the square of the norms of each e_vector on its main diag
    # and the scalar of each combination of e_vectors
    E = tf.linalg.matmul(E_p_transpose, E_p)

    # Select diagonals of each matrix to get the square of the norms for each element in the batch
    norms = tf.math.sqrt(tf.linalg.diag_part(E))
    norms = tf.reshape(norms, [-1, 1, norms.shape[1]])

    # norm_selector helps us in reordering the entries in our norms vector
    # is used to get the product of the desired norms by aligning the entries correctly
    norms_selector = tf.constant([[0., 0., 0., 1.],
                                  [1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.]], dtype=tf.float64)

    # vector containing the reciprocal of the products of the norms
    norms_denominator = tf.math.reciprocal(norms * tf.linalg.matmul(norms, norms_selector))
    norms_denominator_transpose = tf.einsum('bij->bji', norms_denominator)

    # Creating a vector containing each of the desired scalar products of E

    # First three entries are in the first subdiagonal below the main diagonal
    scalar_123 = tf.linalg.diag_part(E, k=-1)

    # Fourth entry is the last one in the first column (reshape, to match ranks for concatonation)
    scalar_4 = tf.reshape(E[:, -1, 0], [-1, 1])

    # Concatenate the fourth entry to the vector with the first three entries to get desired vector
    e_scalar = tf.concat([scalar_123, scalar_4], 1)

    # Reshape to match the dimensions
    e_scalar = tf.reshape(e_scalar, [-1, 1, e_scalar.shape[1]])

    # Scalar product of norms_denominator and e_scalar
    # Returns the sum of all cos values for each angle as the perpendicularity_fac
    perpendicularity_fac = tf.linalg.matmul(e_scalar, norms_denominator_transpose)

    # Calculate loss
    perp_sqr_error = tf.math.square(perpendicularity_fac)
    perp_mean_sqr_error = tf.reduce_mean(perp_sqr_error)

    error = tf.cast(y_true - y_pred, dtype=tf.float64)
    sqr_error = tf.math.square(error)
    mean_sqr_error = tf.reduce_mean(sqr_error)

    loss = mean_sqr_error + perp_mean_sqr_error

    return loss
