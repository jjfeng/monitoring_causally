import copy

import numpy as np
import pandas as pd


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
            dcl_df, on=["eff_iter", "actual_iter"], suffixes=["_stat", "_dcl"]
        )
        return np.any(merged_df.value_stat > merged_df.value_dcl)

    def do_bootstrap_update(self, pred_y_a: np.ndarray, eff_count: int, **kwargs):
        print("pred_y_a", pred_y_a.shape)
        boot_ys = np.random.binomial(
            n=1,
            p=pred_y_a - self.delta,
            size=self.boot_cumsums.shape[0]
            if self.boot_cumsums is not None
            else self.n_bootstrap,
        )[:, np.newaxis]
        boot_iter_stat = self._get_iter_stat(boot_ys, **kwargs)
        if self.boot_cumsums is not None:
            print("self.boot_cumsums", self.boot_cumsums.shape)
        print("boot_iter_stat", boot_iter_stat.shape)
        self.boot_cumsums = (
            np.concatenate([self.boot_cumsums + boot_iter_stat, boot_iter_stat], axis=1)
            if self.boot_cumsums is not None
            else boot_iter_stat
        )
        print("POST self.boot_cumsums", self.boot_cumsums.shape)
        boot_cusums = np.max(np.max(self.boot_cumsums, axis=1), axis=1)
        print("boot_cusums", boot_cusums.shape)

        thres = np.quantile(
            boot_cusums,
            q=self.n_bootstrap
            * (1 - self.alpha_spending_func(eff_count))
            / self.boot_cumsums.shape[0],
        )
        boot_keep_mask = boot_cusums <= thres
        print("boot_keep_mask", boot_keep_mask.shape)
        self.boot_cumsums = self.boot_cumsums[boot_keep_mask]
        return thres


class CUSUM_naive(CUSUM):
    label = "naive"

    def __init__(
        self,
        mdl,
        threshold: float,
        expected_vals: pd.Series,
        alpha_spending_func,
        n_bootstrap: int = 1000,
        delta: float = 0,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta

    def _get_iter_stat(self, y, **kwargs):
        return (self.expected_vals["ppv"] - (y == kwargs["pred_class"]))[
            :, :, np.newaxis
        ]

    def do_monitor(self, num_iters: int, data_gen):
        ppv_count = 0
        ppv_cumsums = None
        ppv_cusums = []
        self.boot_cumsums = None
        dcl = []
        actual_iters = []
        for i in range(num_iters):
            print("iter", i, ppv_count)
            x, y, a = data_gen.generate(1)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1:]
            pred_class = (pred_y_a > self.threshold).astype(int)

            if pred_class == 1:
                actual_iters.append(i)
                ppv_count += 1
                iter_ppv_stat = self._get_iter_stat(y, pred_class=pred_class)
                ppv_cumsums = (
                    np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])
                    if ppv_cumsums is not None
                    else iter_ppv_stat
                )
                ppv_cusums.append(np.max(ppv_cumsums))

                thres = self.do_bootstrap_update(
                    pred_y_a[0], ppv_count, pred_class=pred_class
                )
                dcl.append(thres)

        ppv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(ppv_cusums), dcl]),
                "eff_iter": np.concatenate([np.arange(len(dcl)), np.arange(len(dcl))]),
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
        subgroup_func=None,
        n_bootstrap: int = 10000,
        delta: float = 0,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta
        self.subgroup_func = subgroup_func
        self.delta = delta
        self.n_bootstrap = n_bootstrap

    @property
    def label(self):
        return "wCUSUM%s%s" % (
            "_intervene" if self.propensity_beta is not None else "_obs",
            "_subgroup" if self.subgroup_func is not None else "",
        )

    def _setup(self, data_gen):
        data_gen = copy.deepcopy(data_gen)
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta

        # estimate class variance
        subg_weights = np.ones(1)
        if self.subgroup_func is not None:
            x, y, a = data_gen.generate(self.n_bootstrap)
            h = self.subgroup_func(x)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1:]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1 / oracle_propensity

            iter_ppv_stats = (
                (self.expected_vals["ppv"] - (y[:, np.newaxis] == pred_class))
                * oracle_weight
                * h
            )
            iter_ppv_stats = iter_ppv_stats[pred_class.flatten() == 1]
            subg_var_ests = np.var(iter_ppv_stats, axis=0)
            print("subg_var_ests", subg_var_ests)
            subg_weights = 1 / np.sqrt(subg_var_ests)
        return data_gen, subg_weights

    def _get_iter_stat(self, y, **kwargs):
        return (
            (self.expected_vals["ppv"] - (y[:, np.newaxis] == kwargs["pred_class"]))
            * kwargs["oracle_weight"]
            * kwargs["h"]
            * self.subg_weights[np.newaxis, :]
        )

    def do_monitor(self, num_iters: int, data_gen):
        data_gen, self.subg_weights = self._setup(data_gen)
        print("self.subg_weights", self.subg_weights)

        ppv_count = 0
        self.boot_cumsums = None
        ppv_cumsums = None
        subg_counts = None
        ppv_cusums = []
        dcl = []
        actual_iters = []
        for i in range(num_iters):
            print("iter", i, len(ppv_cusums))
            x, y, a = data_gen.generate(1)
            h = self.subgroup_func(x) if self.subgroup_func is not None else np.ones(1)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1:]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1 / oracle_propensity
            print("weight", oracle_weight, oracle_weight * h * self.subg_weights)

            if pred_class == 1:
                actual_iters.append(i)
                ppv_count += 1
                iter_ppv_stat = self._get_iter_stat(
                    y, pred_class=pred_class, oracle_weight=oracle_weight, h=h
                )
                ppv_cumsums = (
                    np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])
                    if ppv_cumsums is not None
                    else iter_ppv_stat
                )
                subg_counts = (
                    np.concatenate([subg_counts + h, h])
                    if subg_counts is not None
                    else h
                )
                ppv_cusums.append(np.max(ppv_cumsums[subg_counts > 0]))

                thres = self.do_bootstrap_update(
                    pred_y_a[0],
                    ppv_count,
                    pred_class=pred_class,
                    oracle_weight=oracle_weight,
                    h=h,
                )
                dcl.append(thres)

        ppv_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(ppv_cusums), dcl]),
                "eff_iter": np.concatenate([np.arange(len(dcl)), np.arange(len(dcl))]),
                "actual_iter": np.concatenate([actual_iters, actual_iters]),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        ppv_cusum_df["label"] = self.label
        return ppv_cusum_df


class CUSUM_score(CUSUM):
    label = "sCUSUM"

    def __init__(
        self,
        mdl,
        threshold: float,
        expected_vals: pd.Series,
        alpha_spending_func,
        subgroup_func,
        n_bootstrap: int = 1000,
        delta: float = 0,
    ):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.subgroup_func = subgroup_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta

    def _get_iter_stat(self, y, **kwargs):
        return ((y - kwargs["mdl_pred"]) * kwargs["h"])[:, np.newaxis, :]

    def do_monitor(self, num_iters: int, data_gen):
        self.boot_cumsums = None
        score_cumsums = None
        subg_counts = None
        score_cusums = []
        dcl = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            h = self.subgroup_func(x)
            pred_y_a = self.mdl.predict_proba(
                np.concatenate([x, a[:, np.newaxis]], axis=1)
            )[:, 1]

            iter_score = self._get_iter_stat(y, mdl_pred=pred_y_a, h=h)
            score_cumsums = (
                np.concatenate([score_cumsums + iter_score, iter_score])
                if score_cumsums is not None
                else iter_score
            )
            subg_counts = (
                np.concatenate([subg_counts + h, h]) if subg_counts is not None else h
            )
            score_cusums.append(np.max(score_cumsums))

            thres = self.do_bootstrap_update(
                pred_y_a[0], eff_count=i + 1, h=h, mdl_pred=pred_y_a
            )
            dcl.append(thres)

        score_cusum_df = pd.DataFrame(
            {
                "value": np.concatenate([np.array(score_cusums), dcl]),
                "eff_iter": np.concatenate([np.arange(len(dcl)), np.arange(len(dcl))]),
                "actual_iter": np.concatenate(
                    [np.arange(len(dcl)), np.arange(len(dcl))]
                ),
                "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
            }
        )
        score_cusum_df["label"] = self.label
        return score_cusum_df
