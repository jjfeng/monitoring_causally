import copy
import logging

import numpy as np
import pandas as pd

from data_generator import DataGenerator


class CUSUM:
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

    def is_fired_alarm(self, res_df: pd.DataFrame):
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
        self, pred_y_a: np.ndarray, eff_count: int, alt_overest: bool = False, **kwargs
    ):
        sign = -1 if alt_overest else 1
        boot_ys = np.random.binomial(
            n=1,
            p=pred_y_a.reshape((1, -1)) + sign * self.delta,
            size=(self.boot_cumsums.shape[0], pred_y_a.size)
            if self.boot_cumsums is not None
            else (self.n_bootstrap, pred_y_a.size),
        )
        boot_iter_stat, _ = self._get_iter_stat(boot_ys[:, :, np.newaxis], **kwargs)
        self.boot_cumsums = (
            np.concatenate([self.boot_cumsums + boot_iter_stat, boot_iter_stat], axis=1)
            if self.boot_cumsums is not None
            else boot_iter_stat
        )
        boot_cusums = np.maximum(np.max(np.max(self.boot_cumsums, axis=1), axis=1), 0)

        quantile = (
            self.n_bootstrap
            * (1 - self.alpha_spending_func(eff_count))
            / self.boot_cumsums.shape[0]
        )
        print("quantile", quantile, eff_count, np.max(boot_cusums), self.alpha_spending_func(eff_count))
        if quantile < 1:
            thres = np.quantile(
                boot_cusums,
                q=quantile,
            )
            boot_keep_mask = boot_cusums <= thres
            print("boot_keep_mask", boot_keep_mask.sum()/self.n_bootstrap, 1 - self.alpha_spending_func(eff_count))
            self.boot_cumsums = self.boot_cumsums[boot_keep_mask]
        else:
            logging.info("alert: quantile %f", quantile)
            thres = np.max(boot_cusums)
        assert thres >= 0
        return thres


class CUSUM_naive(CUSUM):
    label = "naive"

    def __init__(
        self,
        mdl,
        threshold: float,
        expected_vals: pd.Series,
        alpha_spending_func,
        batch_size: int = 1,
        n_bootstrap: int = 1000,
        delta: float = 0,
        halt_when_fired: bool = True,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta
        self.halt_when_fired = halt_when_fired

    def _get_iter_stat(self, y, **kwargs):
        pred_class = kwargs["pred_class"]
        mask = pred_class == 1
        iter_stats = (self.expected_vals["ppv"] - (y == pred_class)) * mask
        return np.sum(iter_stats, axis=1, keepdims=True), mask.sum()

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        print("Do %s monitor" % self.label)
        ppv_count = 0
        ppv_cumsums = None
        ppv_cusums = []
        self.boot_cumsums = None
        dcl = []
        actual_iters = []
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            print("iter", i, ppv_count)
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1]
            pred_class = (pred_y_a > self.threshold).astype(int).reshape((1, -1))
            actual_iters.append(i)

            iter_ppv_stat, ppv_incr = self._get_iter_stat(
                y[np.newaxis, :], pred_class=pred_class
            )
            ppv_count += ppv_incr
            ppv_cumsums = (
                np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])
                if ppv_cumsums is not None
                else iter_ppv_stat
            )
            ppv_cusums.append(max(0, np.max(ppv_cumsums)))
            logging.info(
                "PPV estimate naive %f", ppv_cumsums[0] / (i + 1) / self.batch_size
            )

            thres = self.do_bootstrap_update(
                pred_y_a,
                ppv_count,
                pred_class=pred_class[:, :, np.newaxis],
                alt_overest=True,
            )
            dcl.append(thres)

            logging.info("%s control_stat %f", self.label, ppv_cusums[-1])
            logging.info("%s dcl %f", self.label, dcl[-1])
            fired = ppv_cusums[-1] > dcl[-1]
            if fired and self.halt_when_fired:
                break

        ppv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(ppv_cusums), dcl]),
                "actual_iter": np.concatenate([actual_iters, actual_iters]),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        ppv_cusum_df["label"] = self.label
        return ppv_cusum_df


class wCUSUM(CUSUM):
    def __init__(
        self,
        mdl,
        threshold: float,
        expected_vals: pd.Series,
        alpha_spending_func,
        propensity_beta: np.ndarray = None,
        batch_size: int = 1,
        subgroup_func=None,
        n_bootstrap: int = 10000,
        delta: float = 0,
        halt_when_fired: bool = True,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta
        self.subgroup_func = subgroup_func
        self.delta = delta
        self.n_bootstrap = n_bootstrap
        self.halt_when_fired = halt_when_fired
        self.subg_weights = np.ones(1)

    @property
    def label(self):
        return "wCUSUM%s%s" % (
            "_intervene" if self.propensity_beta is not None else "_obs",
            "_subgroup" if self.subgroup_func is not None else "",
        )

    def _setup(self, data_gen):
        print("DO SETUP")
        data_gen = copy.deepcopy(data_gen)
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
            data_gen.propensity_intercept = 0

        # estimate class variance
        subg_weights = np.ones(1)
        if self.subgroup_func is not None:
            x, y, a = data_gen.generate(self.n_bootstrap, self.mdl)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1].reshape((1, -1, 1))
            pred_class = (pred_y_a > self.threshold).astype(int)
            h = self.subgroup_func(x, pred_y_a.reshape((-1, 1)))
            oracle_propensity = data_gen._get_propensity(x, mdl=self.mdl).reshape(
                (1, -1, 1)
            )
            oracle_weight = 1 / oracle_propensity

            iter_ppv_stats = self._get_iter_stat(
                y.reshape((1, -1, 1)),
                pred_class=pred_class,
                oracle_weight=oracle_weight,
                h=h[np.newaxis, :, :],
                collate=False,
            )[0]
            iter_ppv_stats = iter_ppv_stats[pred_class.flatten() == 1]
            subg_var_ests = np.var(iter_ppv_stats, axis=0)
            subg_weights = 1 / np.sqrt(subg_var_ests)
            subg_weights[np.isinf(subg_weights)] = 0
        return data_gen, subg_weights.reshape((1, 1, -1))

    def _get_iter_stat(self, y, **kwargs):
        mask = kwargs["pred_class"] == 1
        iter_stats = (
            (self.expected_vals["ppv"] - (y == kwargs["pred_class"]))
            * kwargs["oracle_weight"]
            * kwargs["h"]
            * self.subg_weights
        ) * mask
        if not kwargs["collate"]:
            return iter_stats
        else:
            return np.sum(iter_stats, axis=1, keepdims=True), mask.sum()

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        print("Do %s monitor" % self.label)
        data_gen, self.subg_weights = self._setup(data_gen)

        ppv_count = 0
        self.boot_cumsums = None
        ppv_cumsums = None
        # subg_counts = None
        ppv_cusums = []
        dcl = []
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            print("iter", i, len(ppv_cusums))
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1].reshape((1, -1, 1))
            pred_class = (pred_y_a > self.threshold).astype(int)
            h = (
                self.subgroup_func(x, pred_y_a.reshape((-1, 1)))
                if self.subgroup_func is not None
                else np.ones(1)
            )
            oracle_propensity = data_gen._get_propensity(x, mdl=self.mdl).reshape(
                (1, -1, 1)
            )
            oracle_weight = 1 / oracle_propensity
            # print("weight", oracle_weight, h.shape, self.subg_weights)

            iter_ppv_stat, ppv_incr = self._get_iter_stat(
                y.reshape((1, -1, 1)),
                pred_class=pred_class,
                oracle_weight=oracle_weight,
                h=h[np.newaxis, :],
                collate=True,
            )
            ppv_count += ppv_incr
            ppv_cumsums = (
                np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])
                if ppv_cumsums is not None
                else iter_ppv_stat
            )
            # TODO: do we even need this section below?
            # h_sum = h.sum(axis=0, keepdims=True).reshape((1,1,-1))
            # subg_counts = (
            #     np.concatenate([subg_counts + h_sum, h_sum])
            #     if subg_counts is not None
            #     else h_sum
            # )
            ppv_cusums.append(
                max(np.max(ppv_cumsums), 0)  # [subg_counts > 0]) if subg_counts.sum() else 0
            )
            logging.info(
                "PPV estimate weighted %s",
                ppv_cumsums[0, 0] / self.batch_size / (i + 1),
            )

            thres = self.do_bootstrap_update(
                pred_y_a[0],
                ppv_count,
                pred_class=pred_class,
                oracle_weight=oracle_weight,
                h=h,
                alt_overest=True,
                collate=True,
            )
            dcl.append(thres)

            logging.info("%s control_stat %f", self.label, ppv_cusums[-1])
            logging.info("%s dcl %f", self.label, dcl[-1])
            fired = ppv_cusums[-1] > dcl[-1]
            if fired and self.halt_when_fired:
                break

        ppv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(ppv_cusums), dcl]),
                "actual_iter": np.concatenate(
                    [np.arange(len(dcl)), np.arange(len(dcl))]
                ),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        ppv_cusum_df["label"] = self.label
        return ppv_cusum_df


class CUSUM_score(CUSUM):
    """Score-base CUSUM

    alt_overest=True: p0(x) < f(x) - delta, (f(x) - delta) - y
    alt_overest=False: p0(x) > f(x) + delta, y - (f(x) + delta)
    """
    def __init__(
        self,
        mdl,
        threshold: float,
        expected_vals: pd.Series,
        alpha_spending_func,
        subgroup_func,
        batch_size: int = 1,
        n_bootstrap: int = 1000,
        delta: float = 0,
        halt_when_fired: bool = True,
        alt_overest: bool = True,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = batch_size
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.subgroup_func = subgroup_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta
        self.halt_when_fired = halt_when_fired
        self.alt_overest = alt_overest
    
    @property
    def label(self):
        return "sCUSUM_%s" % ('greater' if self.alt_overest else 'less')

    def _get_iter_stat(self, y, **kwargs):
        test_sign = -1 if self.alt_overest else 1
        iter_stats = (y - (test_sign * self.delta + kwargs["mdl_pred"])) * kwargs["h"] * test_sign
        num_nonzero = (np.sum(kwargs["h"], axis=2) > 0).sum()
        print("num_nonzero", num_nonzero)
        return np.sum(iter_stats, axis=1, keepdims=True), num_nonzero

    def do_monitor(self, num_iters: int, data_gen: DataGenerator):
        print("Do %s monitor" % self.label)
        self.boot_cumsums = None
        score_cumsums = None
        score_cusums = []
        dcl = []
        ppv_count = 0
        for i in range(num_iters):
            data_gen.update_time(i, set_seed=True)
            print("iter", i)
            x, y, a = data_gen.generate(self.batch_size, self.mdl)
            propensity_inputs = data_gen._get_propensity_inputs(x, self.mdl)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1].reshape((1, -1, 1))
            h = self.subgroup_func(
                x, pred_y_a.reshape((-1, 1)), a.reshape((-1, 1)), propensity_inputs
            )
            assert np.all(h >= 0)

            iter_score, ppv_incr = self._get_iter_stat(
                y.reshape((1, -1, 1)), mdl_pred=pred_y_a, h=h[np.newaxis, :, :]
            )
            ppv_count += ppv_incr

            score_cumsums = (
                np.concatenate([score_cumsums + iter_score, iter_score])
                if score_cumsums is not None
                else iter_score
            )
            score_cusums.append(max(np.max(score_cumsums), 0))
            logging.info(
                "score estimate %s", score_cumsums[0] / self.batch_size / (i + 1)
            )

            thres = self.do_bootstrap_update(
                pred_y_a,
                eff_count=ppv_count,
                alt_overest=self.alt_overest,
                h=h[np.newaxis, :, :],
                mdl_pred=pred_y_a,
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
