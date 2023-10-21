import logging
import numpy as np

from matplotlib import pyplot as plt

from subgroups import SubgroupDetector
from common import to_safe_prob


class DataGenerator:
    scale = 2

    def __init__(
        self,
        source_beta: np.ndarray,
        intercept,
        x_mean: np.ndarray,
        propensity_beta: np.ndarray = None,
        propensity_intercept: float = 0,
        beta_shift_time: int = None,
        iter_seeds: np.ndarray = None,
    ):
        self.source_beta = source_beta
        self.target_beta = source_beta
        self.beta_shift_time = beta_shift_time
        self.x_mean = x_mean.reshape((1, -1))
        self.intercept = intercept
        self.num_p = (source_beta.size - 1) // 2
        assert x_mean.size == self.num_p
        self.propensity_beta = propensity_beta
        self.propensity_intercept = propensity_intercept
        self.curr_time = 0
        self.iter_seeds = iter_seeds

    def update_time(self, curr_time: int, set_seed: bool = False):
        self.curr_time = curr_time
        if set_seed and self.iter_seeds is not None:
            np.random.seed(self.iter_seeds[curr_time])

    @property
    def is_shifted(self):
        return (
            self.curr_time > self.beta_shift_time
            if self.beta_shift_time is not None
            else False
        )

    def _get_prob(self, X, A):
        interaction = (A[:, np.newaxis] - 0.5) * 2 * X
        a_x_xa = np.concatenate([A[:, np.newaxis], X, interaction], axis=1)
        beta = self.source_beta if not self.is_shifted else self.target_beta
        logit = np.matmul(a_x_xa, beta.reshape((-1, 1))) + self.intercept
        return 1 / (1 + np.exp(-logit))

    def _get_propensity_inputs(self, X, mdl=None):
        if mdl is not None:
            mdl_pred_prob_a0 = mdl.predict_proba(
                np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
            )[:, 1:]
            mdl_pred_prob_a1 = mdl.predict_proba(
                np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            )[:, 1:]
            mdl_pred_diff = mdl_pred_prob_a1 - mdl_pred_prob_a0
            # print("mdl_pred_diff", np.quantile(np.abs(mdl_pred_diff), [0.1,0.2,0.3,0.7,0.8,0.9]))
            # plt.clf()
            # plt.hist(mdl_pred_diff)
            # plt.show()
        else:
            mdl_pred_diff = np.zeros((X.shape[0], 1))
        X_aug = np.concatenate([mdl_pred_diff, X], axis=1)
        return X_aug

    def _get_propensity(self, X, mdl=None):
        if self.propensity_beta is None:
            return np.ones(X.shape[0]) * 0.5
        else:
            X_aug = self._get_propensity_inputs(X, mdl)
            logit = (
                np.matmul(X_aug, self.propensity_beta.reshape((-1, 1)))
                + self.propensity_intercept
            )
            propensity = 1 / (1 + np.exp(-logit))
            # if mdl is not None:
            #     mdl_pred = mdl.predict_proba(
            #         np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
            #     )[:, 1]
            #     propensity_pos = propensity[mdl_pred < 0.5]
            #     print(
            #         "propensity",
            #         np.sqrt(np.var(propensity_pos)),
            #         np.min(propensity_pos),
            #         np.max(propensity_pos),
            #     )
            #     plt.clf()
            #     plt.hist(propensity_pos)
            #     plt.show()
            return propensity

    def _generate_X(self, num_obs):
        """

        Args:
            num_obs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            np.random.normal(scale=self.scale, size=(num_obs, self.num_p)) + self.x_mean
        )

    def _generate_Y(self, X, A):
        probs = self._get_prob(X, A)
        y = np.random.binomial(1, probs.flatten(), size=probs.size)
        return y

    def _generate_A(self, X, mdl=None):
        treatment_probs = self._get_propensity(X, mdl).flatten()
        A = np.random.binomial(1, treatment_probs, size=treatment_probs.size)
        return A

    def generate(self, num_obs, mdl=None):
        X = self._generate_X(num_obs)
        A = self._generate_A(X, mdl)
        y = self._generate_Y(X, A)
        return X, y, A


class SmallXShiftDataGenerator(DataGenerator):
    def __init__(
        self,
        source_beta: np.ndarray,
        target_beta: np.ndarray,
        intercept,
        prob_shift: float,
        shift_A: int,
        subG: int,
        x_mean: np.ndarray,
        propensity_beta: np.ndarray = None,
        propensity_intercept: float = 0,
        beta_shift_time: int = None,
        iter_seeds: np.ndarray = None,
    ):
        self.source_beta = source_beta
        self.target_beta = target_beta
        self.prob_shift = prob_shift
        self.shift_A = shift_A
        self.subG = subG
        self.beta_shift_time = beta_shift_time
        self.x_mean = x_mean.reshape((1, -1))
        self.intercept = intercept
        self.num_p = (source_beta.size - 1) // 2
        assert x_mean.size == self.num_p
        self.propensity_beta = propensity_beta
        self.propensity_intercept = propensity_intercept
        self.curr_time = 0
        self.iter_seeds = iter_seeds

    def _get_prob(self, X, A):
        interaction = (A[:, np.newaxis] - 0.5) * 2 * X
        a_x_xa = np.concatenate([A[:, np.newaxis], X, interaction], axis=1)
        beta = self.source_beta if not self.is_shifted else self.target_beta
        logit = np.matmul(a_x_xa, beta.reshape((-1, 1))) + self.intercept
        prob = 1 / (1 + np.exp(-logit))
        # print("IS SHIFT?", self.is_shifted)
        if self.is_shifted:
            subG_mask = SubgroupDetector._get_subgroup(X)
            if self.subG == 0:
                subG_mask = np.logical_not(subG_mask)
            logging.info("PREVALENCE of subgroup %f", subG_mask.mean())
            delta_prob = (
                self.prob_shift * subG_mask * (A[:, np.newaxis] == self.shift_A)
            )
            prob = to_safe_prob(prob + delta_prob, eps=0)
        return prob

class SymSmallXShiftDataGenerator(SmallXShiftDataGenerator):
    def _get_prob(self, X, A):
        interaction = (A[:, np.newaxis] - 0.5) * 2 * X
        a_x_xa = np.concatenate([A[:, np.newaxis], X, interaction], axis=1)
        beta = self.source_beta if not self.is_shifted else self.target_beta
        logit = np.matmul(a_x_xa, beta.reshape((-1, 1))) + self.intercept
        prob = 1 / (1 + np.exp(-logit))
        if self.is_shifted:
            subG_mask = SubgroupDetector._get_subgroup(X)
            if self.subG == 0:
                subG_mask = np.logical_not(subG_mask)
            logging.info("PREVALENCE of subgroup %f", subG_mask.mean())
            shift_sign = (prob > 0.5) * -1 + (prob < 0.5)
            delta_prob = (
                shift_sign * self.prob_shift * subG_mask * (A[:, np.newaxis] == self.shift_A)
            )
            prob = to_safe_prob(prob + delta_prob, eps=0)
        return prob