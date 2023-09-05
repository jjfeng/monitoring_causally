import numpy as np

from common import to_safe_prob


class DataGenerator:
    scale = 2

    def __init__(
        self, source_beta: np.ndarray, target_beta: np.ndarray, intercept, x_mean: np.ndarray, propensity_beta: np.ndarray=None, propensity_intercept=0, beta_shift_time:int=None
    ):
        self.source_beta = source_beta
        self.target_beta = target_beta
        self.beta_shift_time = beta_shift_time
        self.x_mean = x_mean.reshape((1, -1))
        self.intercept = intercept
        self.num_p = (source_beta.size - 1) // 2
        assert x_mean.size == self.num_p
        self.propensity_beta = propensity_beta
        self.propensity_intercept = propensity_intercept
        self.curr_time = 0
        
    def update_time(self, curr_time: int):
        self.curr_time = curr_time
    
    @property
    def is_shifted(self):
        return self.curr_time > self.beta_shift_time if self.beta_shift_time is not None else False

    def _get_prob(self, X, A):
        a_x_xa = np.concatenate([A[:, np.newaxis], X, A[:, np.newaxis] * X], axis=1)
        beta = self.source_beta if not self.is_shifted else self.target_beta
        logit = np.matmul(a_x_xa, beta.reshape((-1, 1))) + self.intercept
        return 1 / (1 + np.exp(-logit))

    def _get_propensity(self, X):
        if self.propensity_beta is None:
            return np.ones(X.shape[0]) * 0.5
        else:
            logit = (
                np.matmul(X, self.propensity_beta.reshape((-1, 1)))
                + self.propensity_intercept
            )
            return 1 / (1 + np.exp(-logit))

    def _generate_X(self, num_obs):
        return (
            np.random.normal(scale=self.scale, size=(num_obs, self.num_p)) + self.x_mean
        )

    def _generate_Y(self, X, A):
        probs = self._get_prob(X, A)
        y = np.random.binomial(1, probs.flatten(), size=probs.size)
        return y

    def _generate_A(self, X):
        treatments = self._get_propensity(X)
        A = np.random.binomial(1, treatments.flatten(), size=treatments.size)
        return A

    def generate(self, num_obs):
        X = self._generate_X(num_obs)
        A = self._generate_A(X)
        y = self._generate_Y(X, A)
        return X, y, A
