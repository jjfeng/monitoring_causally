import copy
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from subgroups import *
from data_generator import DataGenerator

CLASS_DICT = {
    'npv': 0,
    'ppv': 1
}


class CUSUM:
    alpha_scale = 1

    def __init__(
        self, mdl, batch_size: int, alpha_spending_func, delta: float, n_bootstrap: int
    ):
        self.mdl = mdl
        self.batch_size = batch_size
        self.alpha_spending_func = alpha_spending_func
        self.delta = delta
        self.n_bootstrap = n_bootstrap

    def do_monitor(self, num_iters: int, data_gen):
        raise NotImplementedError()

    def _get_iter_stat(self, y, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def is_fired_alarm(res_df: pd.DataFrame):
        stat_df = res_df[res_df.variable == "stat"]
        dcl_df = res_df[res_df.variable == "dcl"]
        merged_df = stat_df.merge(
            dcl_df, on=["actual_iter"], suffixes=["_stat", "_dcl"]
        )
        is_fired = np.any(merged_df.value_stat > merged_df.value_dcl)
        fire_time = None
        if is_fired:
            fire_time = np.min(np.where(merged_df.value_stat > merged_df.value_dcl))

        return is_fired, fire_time

    def do_bootstrap_update(
        self,
        pred_y_a01: np.ndarray,
        a: np.ndarray,
        eff_count: int,
        alt_overest: str = "less_extreme",
        **kwargs
    ):
        pred_y_a = pred_y_a01[np.arange(a.size), a.flatten()]
        pred_test_sign = (
            -1 if alt_overest == "overest" else
            (1 if alt_overest == "underest" else -((pred_y_a > THRES) - 0.5) * 2)
        )
        print("pred_y_a - pred_test_sign * self.delta", pred_y_a - pred_test_sign * self.delta)
        boot_ys = np.random.binomial(
            n=1,
            p=pred_y_a - pred_test_sign * self.delta,
            size=(self.boot_cumsums.shape[0], pred_y_a.size)
            if self.boot_cumsums is not None
            else (self.n_bootstrap, pred_y_a.size),
        )
        boot_iter_stat, _ = self._get_iter_stat(boot_ys[:, :, np.newaxis, np.newaxis], a, **kwargs)
        self.boot_cumsums = (
            np.concatenate([self.boot_cumsums + boot_iter_stat, boot_iter_stat], axis=1)
            if self.boot_cumsums is not None
            else boot_iter_stat
        )
        boot_cusums = np.maximum(np.max(np.max(np.max(self.boot_cumsums, axis=1), axis=1), axis=1), 0)
        # plt.hist(boot_cusums)
        # plt.show()

        tot_alpha_spent = self.alpha_spending_func(eff_count) * self.alpha_scale
        quantile = self.n_bootstrap * (1 - tot_alpha_spent) / self.boot_cumsums.shape[0]
        # print("quantile", quantile, eff_count, np.max(boot_cusums), tot_alpha_spent)
        if quantile < 1:
            thres = np.quantile(
                boot_cusums,
                q=quantile,
            )
            boot_keep_mask = boot_cusums <= thres
            # print("boot_keep_mask", boot_keep_mask.sum()/self.n_bootstrap, 1 - self.alpha_spending_func(eff_count) * self.alpha_scale)
            self.boot_cumsums = self.boot_cumsums[boot_keep_mask]
        else:
            logging.info("alert: quantile %f", quantile)
            thres = np.max(boot_cusums)
        assert thres >= 0
        return thres

    def _get_mdl_pred_a01(self, x):
        pred_y_a0 = self.mdl.predict_proba(
            np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
        )[:, 1:]
        pred_y_a1 = self.mdl.predict_proba(
            np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        )[:, 1:]
        return np.concatenate([pred_y_a0, pred_y_a1], axis=1)


class CUSUM_naive(CUSUM):
    label = "naive"

    def __init__(
        self,
        mdl,
        threshold: float,
        perf_targets_df: pd.DataFrame,
        alpha_spending_func,
        batch_size: int = 1,
        n_bootstrap: int = 1000,
        delta: float = 0,
        halt_when_fired: bool = True,
        metrics: list[str] = ["npv"],
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.perf_targets = np.array(
            [perf_targets_df.value[perf_targets_df.metric == metric] for metric in metrics]
            ).reshape((1, 1, -1, 1))
        self.alpha_spending_func = alpha_spending_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta
        self.halt_when_fired = halt_when_fired
        self.metrics = metrics
        self.class_mtrs = np.array([CLASS_DICT[metric] for metric in self.metrics]).reshape((1,1,-1,1))
        print("self.class_mtrs", self.class_mtrs)

    def _get_iter_stat(self, y, a, **kwargs):
        pred_class = kwargs["pred_class"]
        pred_mask = pred_class == self.class_mtrs
        a_mask = np.concatenate([a == 0, a == 1], axis=2)
        iter_stats = (self.perf_targets - (y == pred_class) * a_mask) * pred_mask
        print("iter stats shape", iter_stats.shape)
        return np.sum(iter_stats, axis=1, keepdims=True), pred_mask.sum()

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        print("Do %s monitor" % self.label)
        pv_count = 0
        pv_cumsums = None
        pv_cusums = []
        self.boot_cumsums = None
        dcl = []
        actual_iters = []
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            # print("iter", i, pv_count)
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            pred_y_a01 = self._get_mdl_pred_a01(x)
            pred_class_a01 = (pred_y_a01 > self.threshold).astype(int)
            actual_iters.append(i)

            iter_pv_stat, pv_incr = self._get_iter_stat(
                y[np.newaxis, :, np.newaxis, np.newaxis],
                a[np.newaxis, :, np.newaxis, np.newaxis],
                pred_class=pred_class_a01[np.newaxis, :, :, np.newaxis],
            )
            pv_count += pv_incr
            pv_cumsums = (
                np.concatenate([pv_cumsums + iter_pv_stat, iter_pv_stat])
                if pv_cumsums is not None
                else iter_pv_stat
            )
            pv_cusums.append(max(0, np.max(pv_cumsums)))
            logging.info(
                "PV estimate naive %s", pv_cumsums[0] / (i + 1) / self.batch_size
            )

            thres = self.do_bootstrap_update(
                pred_y_a01,
                a[np.newaxis, :, np.newaxis, np.newaxis],
                pv_count,
                pred_class=pred_class_a01[np.newaxis, :, :, np.newaxis],
            )
            dcl.append(thres)

            logging.info("%s control_stat %f", self.label, pv_cusums[-1])
            logging.info("%s dcl %f", self.label, dcl[-1])
            fired = pv_cusums[-1] > dcl[-1]

            if fired and self.halt_when_fired:
                break

        ppv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(pv_cusums), dcl]),
                "actual_iter": np.concatenate([actual_iters, actual_iters]),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        ppv_cusum_df["label"] = self.label
        return ppv_cusum_df


class wCUSUM(CUSUM):
    """Monitor PPV, weighted to equal the original PPV

    PPV_delta = E[(1{y = 1} - PPV) * p(y|x,a) * p_ind(a,x|p(f(x,a) = 1)]
    weightedPPV_delta = E[(1{y = 1} - PPV) * p_ind(a,x|f(x,a) = 1)/p(a,x|f(x,a) = 1) * p(y|x,a)]
    p_ind(a,x|f(x,a) = 1)/p(a,x|f(x,a) = 1) = p_ind(a,x,f(x,a) = 1)/p(a,x,f(x,a) = 1) * p(f(x,a) = 1)/p_ind(f(x,a) = 1)
    assuming f(x,a) = 1, then p_ind(a,x,f(x,a) = 1)/p(a,x,f(x,a) = 1)
                                    = p_ind(a)/p(a|x) = 0.5/p(a|x) if both values are possible
                                    = 1 if only one value is possible
    """

    def __init__(
        self,
        mdl,
        threshold: float,
        perf_targets_df: pd.DataFrame,
        alpha_spending_func,
        subgroup_detector: SubgroupDetectorBase,
        propensity_beta: np.ndarray = None,
        propensity_intercept: float = 0,
        batch_size: int = 1,
        n_bootstrap: int = 10000,
        delta: float = 0,
        halt_when_fired: bool = True,
        metrics: list[str] = ["npv"],
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.perf_targets = np.array(
            [perf_targets_df.value[perf_targets_df.metric == metric] for metric in metrics]
            )[np.newaxis, np.newaxis, :, :]
        print("self.perf_targets", self.perf_targets.shape)
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta
        self.propensity_intercept = propensity_intercept
        self.subgroup_detector = subgroup_detector
        self.delta = delta
        self.n_bootstrap = n_bootstrap
        self.halt_when_fired = halt_when_fired
        self.subg_weights = np.ones(1)
        self.metrics = metrics
        self.class_mtrs = np.array([CLASS_DICT[metric] for metric in self.metrics]).reshape((1,1,-1,1))

    @property
    def label(self):
        return "wCUSUM%s%s" % (
            "_intervene" if self.propensity_beta is not None else "_obs",
            "_subgroup%d" % self.subg_weights.size,
        )

    def _setup(self, data_gen):
        print("DO SETUP")
        data_gen = copy.deepcopy(data_gen)
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
            data_gen.propensity_intercept = self.propensity_intercept

        # estimate class variance
        x, y, a = data_gen.generate(self.n_bootstrap, self.mdl)
        pred_y_a01 = self._get_mdl_pred_a01(x)
        pred_y_a = pred_y_a01[np.arange(a.size), a.flatten()]
        pred_class = (pred_y_a01 > self.threshold).astype(int)
        h = self.subgroup_detector.detect(x, pred_y_a01)
        ha = self.subgroup_detector.detect_with_a(
            x, a.reshape((-1, 1)), pred_y_a.reshape((-1, 1))
        )
        oracle_propensity_a1 = data_gen._get_propensity(x, mdl=self.mdl).flatten()
        oracle_propensity = (
            oracle_propensity_a1 * a + (1 - oracle_propensity_a1) * (1 - a)
        ).reshape((1, -1, 1, 1))
        print("oracle_propensity", np.quantile(oracle_propensity, [0.001, 0.01, 0.1]))
        assert np.max(oracle_propensity) < 1 and np.min(oracle_propensity) > 0
        oracle_weight = 1 / oracle_propensity

        iter_ppv_stats, eff_obs_mask = self._get_iter_stat(
            y.reshape((1, -1, 1, 1)),
            a[np.newaxis, :, np.newaxis, np.newaxis],
            pred_class=pred_class,
            oracle_weight=oracle_weight,
            h=h[np.newaxis, :, np.newaxis, :],
            ha=ha[np.newaxis, :, np.newaxis, :],
            collate=False,
        )
        self.alpha_scale = self.n_bootstrap / np.sum(eff_obs_mask)
        iter_ppv_stats = iter_ppv_stats[0, eff_obs_mask.flatten()]
        subg_var_ests = np.var(iter_ppv_stats, axis=0)
        print(data_gen.propensity_beta, "subg_var_ests", subg_var_ests)

        # calculate weights
        subg_weights = 1 / np.sqrt(subg_var_ests)
        subg_weights[np.isinf(subg_weights)] = 0
        return data_gen, subg_weights[np.newaxis, np.newaxis, :, :]

    def _get_iter_stat(self, y, a, **kwargs):
        pred_class = kwargs["pred_class"][
            np.newaxis, :, np.newaxis, self.subgroup_detector.subg_treatments
        ]
        pred_mask = pred_class == self.class_mtrs
        iter_stats = (
            (
                self.perf_targets
                - (y == pred_class) * kwargs["oracle_weight"] * kwargs["ha"]
            )
            * pred_mask
            * self.subg_weights
            * kwargs["h"]
        )
        print("iter_stats", iter_stats.shape)
        nonzero_mask = (np.sum(np.sum(kwargs["h"], axis=2), axis=2) > 0).flatten()
        if not kwargs["collate"]:
            return iter_stats, nonzero_mask
        else:
            return np.sum(iter_stats, axis=1, keepdims=True), nonzero_mask.sum()

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        data_gen, self.subg_weights = self._setup(data_gen)
        print("Do %s monitor" % self.label)
        logging.info("self.subg_weights %s", self.subg_weights)

        pv_count = 0
        self.boot_cumsums = None
        pv_cumsums = None
        pv_cusums = []
        dcl = []
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            # print("iter", i, len(pv_cusums))
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            pred_y_a01 = self._get_mdl_pred_a01(x)
            pred_y_a = pred_y_a01[np.arange(a.size), a.flatten()]
            pred_class = (pred_y_a01 > self.threshold).astype(int)
            h = (
                self.subgroup_detector.detect(x, pred_y_a01)
                if self.subgroup_detector is not None
                else np.ones(1)
            )
            ha = (
                self.subgroup_detector.detect_with_a(
                    x, a.reshape((-1, 1)), pred_y_a.reshape((-1, 1))
                )
                if self.subgroup_detector is not None
                else np.ones(1)
            )
            oracle_propensity_a1 = data_gen._get_propensity(x, mdl=self.mdl).flatten()
            oracle_propensity = (
                oracle_propensity_a1 * a + (1 - oracle_propensity_a1) * (1 - a)
            ).reshape((1, -1, 1, 1))
            oracle_weight = 1 / oracle_propensity
            # print("weight", oracle_weight, h.shape, self.subg_weights)

            iter_pv_stat, pv_incr = self._get_iter_stat(
                y.reshape((1, -1, 1, 1)),
                a[np.newaxis, :, np.newaxis, np.newaxis],
                pred_class=pred_class,
                oracle_weight=oracle_weight,
                h=h[np.newaxis, :, np.newaxis, :],
                ha=ha[np.newaxis, :, np.newaxis, :],
                collate=True,
            )
            pv_count += pv_incr
            pv_cumsums = (
                np.concatenate([pv_cumsums + iter_pv_stat, iter_pv_stat])
                if pv_cumsums is not None
                else iter_pv_stat
            )
            pv_cusums.append(max(np.max(pv_cumsums), 0))
            logging.info(
                "pv estimate weighted %s",
                pv_cumsums[0, 0] / self.batch_size / (i + 1),
            )

            thres = self.do_bootstrap_update(
                pred_y_a01,
                a[np.newaxis, :, np.newaxis, np.newaxis],
                eff_count=pv_count,
                pred_class=pred_class,
                oracle_weight=oracle_weight,
                h=h[np.newaxis, :, np.newaxis, :],
                ha=ha[np.newaxis, :, np.newaxis, :],
                collate=True,
            )
            dcl.append(thres)

            logging.info("%s control_stat %f", self.label, pv_cusums[-1])
            logging.info("%s dcl %f", self.label, dcl[-1])
            fired = pv_cusums[-1] > dcl[-1]
            if fired and self.halt_when_fired:
                break

        pv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(pv_cusums), dcl]),
                "actual_iter": np.concatenate(
                    [np.arange(len(dcl)), np.arange(len(dcl))]
                ),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        pv_cusum_df["label"] = self.label
        return pv_cusum_df


class CUSUM_score(CUSUM):
    """Score-base CUSUM

    alt_overest=True: p0(x) < f(x) - delta, (f(x) - delta) - y
    alt_overest=False: p0(x) > f(x) + delta, y - (f(x) + delta)
    """

    def __init__(
        self,
        mdl,
        threshold: float,
        alpha_spending_func,
        subgroup_detector: ScoreSubgroupDetector,
        batch_size: int = 1,
        n_bootstrap: int = 1000,
        propensity_beta: np.ndarray = None,
        propensity_intercept: float = 0,
        delta: float = 0,
        halt_when_fired: bool = True,
        alternative: str = "overest",
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.alpha_spending_func = alpha_spending_func
        self.subgroup_detector = subgroup_detector
        self.n_bootstrap = n_bootstrap
        self.delta = delta
        self.propensity_beta = propensity_beta
        self.propensity_intercept = propensity_intercept
        self.halt_when_fired = halt_when_fired
        self.alternative = alternative
        self.subg_weights = np.ones(1)

    @property
    def label(self):
        return "sCUSUM_%s_%s" % (
            ("greater" if self.alternative == "overest" else ("less" if self.alternative == "underest" else "less_extreme")),
            ("intervene" if self.propensity_beta is not None else "obs"),
        )

    def _get_iter_stat(self, y: np.ndarray, a: np.ndarray = None, **kwargs):
        """_summary_

        Args:
            y (np.ndarray): _description_
            a (_type_, optional): IGNORED. Here to unify calls. Defaults to None.

        Returns:
            _type_: _description_
        """
        pred_class = kwargs["mdl_pred"] > THRES
        test_sign = (
            -1 if self.alternative == "overest" else 
            (1 if self.alternative == "underest" else -(pred_class - 0.5) * 2)
        )
        print(y.shape, kwargs["h"].shape, kwargs["mdl_pred"].shape, self.subg_weights.shape)
        iter_stats = (
            (y - (test_sign * self.delta + kwargs["mdl_pred"]))
            * kwargs["h"]
            * test_sign
            * self.subg_weights
        )
        nonzero_mask = (np.sum(np.sum(kwargs["h"], axis=2), axis=2) > 0).flatten()
        print("num_nonzero", nonzero_mask.sum())
        if kwargs["collate"]:
            return np.sum(iter_stats, axis=1, keepdims=True), nonzero_mask.sum()
        else:
            return iter_stats, nonzero_mask

    def _setup(self, data_gen: DataGenerator):
        print("DO SETUP")
        # intervene on the data generator
        data_gen = copy.deepcopy(data_gen)
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
            data_gen.propensity_intercept = self.propensity_intercept
            logging.info(
                "propensity interve %s %s",
                self.propensity_beta,
                self.propensity_intercept,
            )

        # assume random assignment to treatment (so just accounting for prevalence of each subgroup)
        # TODO: this uses an oracle data generator, but better if we use the mdl itself to generate y
        data_gen.update_time(0, set_seed=True)
        x, y, a = data_gen.generate(self.n_bootstrap, mdl=None)
        pred_y_a01 = self._get_mdl_pred_a01(x)
        pred_y_a = pred_y_a01[np.arange(a.size), a.flatten()]
        h = self.subgroup_detector.detect(
            x, a.reshape((-1, 1)), pred_y_a.reshape((-1, 1))
        )

        iter_score_stats, eff_obs_mask = self._get_iter_stat(
            y.reshape((1, -1, 1, 1)),
            mdl_pred=pred_y_a[np.newaxis, :, np.newaxis, np.newaxis],
            h=h[np.newaxis, :, np.newaxis, :],
            collate=False,
        )
        self.alpha_scale = self.n_bootstrap / np.sum(eff_obs_mask)
        logging.info("ALPHA SCALE %f", self.alpha_scale)
        iter_score_stats = iter_score_stats[0, eff_obs_mask.flatten()]

        subg_var_ests = np.var(iter_score_stats, axis=0)
        subg_weights = 1 / np.sqrt(subg_var_ests)
        # remove subgroups if positivity violations are too severe
        subg_weights[np.isinf(subg_weights)] = 0

        return data_gen, subg_weights[np.newaxis, :] / subg_weights.max()

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        data_gen, self.subg_weights = self._setup(data_gen)
        # data_gen, _ = self._setup(data_gen)
        print("Do %s monitor" % self.label)
        print("self.subg_weights", self.subg_weights)
        logging.info("self.subg_weights %s", self.subg_weights)

        self.boot_cumsums = None
        score_cumsums = None
        score_cusums = []
        dcl = []
        score_count = 0
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            # print("iter", i)
            logging.info("iter %d", i)
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            pred_y_a01 = self._get_mdl_pred_a01(x)
            pred_y_a = pred_y_a01[np.arange(a.size), a.flatten()]
            h = self.subgroup_detector.detect(
                x,
                a.reshape((-1, 1)),
                pred_y_a.reshape((-1, 1)),
            )

            iter_score, score_incr = self._get_iter_stat(
                y.reshape((1, -1, 1, 1)),
                mdl_pred=pred_y_a[np.newaxis, :, np.newaxis, np.newaxis],
                h=h[np.newaxis, :, np.newaxis, :],
                collate=True,
            )
            score_count += score_incr

            score_cumsums = (
                np.concatenate([score_cumsums + iter_score, iter_score])
                if score_cumsums is not None
                else iter_score
            )
            score_cusums.append(max(np.max(score_cumsums), 0))
            logging.info(
                "score estimate %s", score_cumsums[0] / self.batch_size / (i + 1)
            )
            logging.info(
                "score max subg=%d t=%d",
                np.argmax(np.amax(score_cumsums, axis=0)),
                np.argmax(np.amax(score_cumsums, axis=2)),
            )

            thres = self.do_bootstrap_update(
                pred_y_a01,
                a=a[np.newaxis, :, np.newaxis, np.newaxis],
                eff_count=score_count,
                alt_overest=self.alternative,
                h=h[np.newaxis, np.newaxis, :],
                mdl_pred=pred_y_a[np.newaxis, :, np.newaxis, np.newaxis],
                collate=True,
            )
            dcl.append(thres)

            logging.info("%s control_stat %f", self.label, score_cusums[-1])
            logging.info("%s dcl %f", self.label, dcl[-1])
            fired = score_cusums[-1] > dcl[-1]
            if fired and self.halt_when_fired:
                break

        score_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(score_cusums), dcl]),
                "actual_iter": np.concatenate(
                    [np.arange(len(dcl)), np.arange(len(dcl))]
                ),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        score_cusum_df["label"] = self.label
        return score_cusum_df
