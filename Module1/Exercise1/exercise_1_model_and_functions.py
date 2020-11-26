import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from generate_mask import adjacency_mask
import scipy.linalg


class GaussianModel(tf.keras.layers.Layer):
    def __init__(self, shape, mask=None, mask_type="orthogonal", params=None):
        super(GaussianModel, self).__init__()
        if mask is None:
            self.mask = adjacency_mask(shape=(shape[0], shape[1]), mask_type=mask_type)
        else:
            self.mask = mask
        if params is None:
            par = Params(shape, self.mask)
        else:
            par = params

        self.loc = par.mu
        self.precision_matrix = par.precision_matrix

        self.dim = np.prod(shape)

    def __call__(self, data):
        y_halfway = tf.einsum("ij,kj->ki", self.precision_matrix, data)
        y = tf.einsum("ki,ki->k", data, y_halfway)
        return tf.exp(-0.5 * y) / self.z

    def build(self):
        self.built = True

    @property
    def covariance_matrix(self):
        return tf.linalg.inv(self.precision_matrix)

    @property
    def z(self):
        return (2 * np.pi) ** (0.5 * self.dim) * tf.abs(
            tf.linalg.det(self.covariance_matrix)
        ) ** 0.5

    def grad_log_model(self, data):
        grad_x = -tf.einsum("ij,sj->si", self.precision_matrix, (data - self.loc))
        return grad_x

    def laplace_log_model(self, data):
        return -tf.einsum("ii", self.precision_matrix)

    def model_loss(self, data):
        # 1/N sum i from 1 to N (1/2*||grad_x log model(xi)||^2 + laplace log model(xi))
        N = len(data)
        loss = 0
        for sample in range(N):
            loss = (
                loss
                + 1 / 2 * tf.linalg.norm(self.grad_log_model(data[sample])) ** 2
                + self.laplace_log_model(data[sample])
            )

        loss = 1 / N * loss
        return loss

    def data_point_weights(
        self, data, det_noise_precision_matrix, noise_precision_matrix
    ):
        weights = tf.sqrt(
            tf.math.abs(
                det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
            )
        ) * tf.exp(
            tf.einsum(
                "ki,ki->k",
                data,
                tf.einsum(
                    "ij,kj->ki", noise_precision_matrix - self.precision_matrix, data
                ),
            )
        )
        return weights

    def NCE_real_data_terms(
        self, real_data, det_noise_precision_matrix, noise_precision_matrix
    ):
        real_weights = self.data_point_weights(
            real_data, det_noise_precision_matrix, noise_precision_matrix
        )
        N = tf.shape(real_data,out_type=tf.float64)[0]
        term_3 = -1 / N * tf.reduce_sum(tf.math.log(real_weights + 1))
        return term_3

    def NCE_noise_data_terms(
        self, noise_data, nu, det_noise_precision_matrix, noise_precision_matrix
    ):
        M = tf.shape(noise_data,out_type=tf.float64)[0]
        noise_weights = self.data_point_weights(
            noise_data, det_noise_precision_matrix, noise_precision_matrix
        )

        tmp_cnst = nu / M

        term_1 = (
            tmp_cnst
            * (
                tf.einsum(
                    "ki,ki",
                    noise_data,
                    tf.einsum(
                        "ij,kj->ki",
                        noise_precision_matrix - self.precision_matrix,
                        noise_data,
                    ),
                )
            )
        )
        term_4 = -nu / M * tf.reduce_sum(tf.math.log(nu * noise_weights + 1))
        return term_1 + term_4

    def NCE_precision_matrix_term(self, nu, det_noise_precision_matrix):
        term_2 = (
            -nu
            / 2
            * tf.math.log(
                nu ** 2
                * tf.math.abs(
                    det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
                )
                + np.finfo(float).eps
            )
        )
        return term_2

    def NCE_zero(self):
        return tf.constant(0,dtype=tf.float64)

    def NCE(
        self,
        real_data,
        noise_data,
        det_noise_precision_matrix,
        noise_precision_matrix,
        nu,
    ):
        loss = 0
        loss = loss + tf.cond(
            tf.shape(real_data)[0] != 0,
            lambda: self.NCE_real_data_terms(
                real_data, det_noise_precision_matrix, noise_precision_matrix
            ),
            lambda: self.NCE_zero(),
        )
        loss = loss + tf.cond(
            tf.shape(noise_data)[0] != 0,
            lambda: self.NCE_noise_data_terms(
                noise_data, nu, det_noise_precision_matrix, noise_precision_matrix
            ),
            lambda: self.NCE_zero(),
        )
        loss = loss + self.NCE_precision_matrix_term(nu, det_noise_precision_matrix)

        return loss


class Params:
    def __init__(self, shape, mask, loc=None, precision_matrix=None, only_ones=False):
        rows = shape[0]
        cols = shape[1]
        self.dim = rows * cols
        if loc is None:
            self.mu = tf.zeros((1, self.dim), dtype=tf.float64)
        else:
            self.mu = loc

        if precision_matrix is None:
            if only_ones:
                tmp = tf.Variable(
                    np.ones((self.dim, self.dim)) * mask, dtype=tf.float64
                )
            else:
                arr = tf.math.abs(
                    tf.random.uniform((self.dim, self.dim), 0.1, 1, dtype=tf.float64)
                )

                tmp = tf.Variable(make_symmetric(arr, mask=mask))

            self.precision_matrix = tmp
        else:
            self.precision_matrix = precision_matrix

    def get_distribution(self):
        tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.mu,
            covariance_matrix=self.covariance_matrix,
            validate_args=False,
            allow_nan_stats=True,
            name="MultivariateNormalFullCovariance",
        )

    def generate_samples(self, num, slide_samples=False):
        if slide_samples:
            eps = np.random.multivariate_normal(
                np.zeros(self.dim), np.eye(self.dim), size=num
            )
            A = scipy.linalg.sqrtm(self.covariance_matrix)
            prod = np.einsum("ij,kj->ki", A, eps)

            return prod + self.mu
        else:
            return np.random.multivariate_normal(
                mean=np.squeeze(self.mu),
                cov=self.covariance_matrix,
                size=num,
                check_valid="ignore",
            ).astype(np.float32)

    @property
    def covariance_matrix(self):
        return tf.linalg.inv(self.precision_matrix)


def make_symmetric(input, sym_type="upper", mask=None):
    if sym_type == "upper":
        part = tf.linalg.band_part(input, 0, -1)
    elif sym_type == "lower":
        part = tf.linalg.band_part(input, -1, 0)
    else:
        raise ValueError

    sym = part + tf.linalg.matrix_transpose(part) - tf.linalg.band_part(part, 0, 0)

    if mask is not None:
        sym = sym * mask

    return sym
