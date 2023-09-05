import copy

import numpy as np
import pandas as pd


class CUSUM:
    def __init__(self, mdl, batch_size: int, alpha_spending_func):
        self.mdl = mdl
        self.batch_size = batch_size
        self.alpha_spending_func = alpha_spending_func
    
    def do_monitor(self, num_iters: int, data_gen):
        raise NotImplementedError()


class CUSUM_naive(CUSUM):
    label = 'naive'
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func, n_bootstrap: int = 1000, delta: float = 0):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.n_bootstrap = n_bootstrap
        self.delta = delta

    def do_monitor(self, num_iters: int, data_gen):
        ppv_cumsums = None
        ppv_cusums = []
        boot_ppv_cumsums = None
        dcl = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1]
            pred_class = (pred_y_a > self.threshold).astype(int)
            
            if pred_class == 1:
                iter_ppv_stat = self.expected_vals["ppv"] - (y == pred_class)
                ppv_cumsums = np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])if ppv_cumsums is not None else iter_ppv_stat
                ppv_cusums.append(np.max(ppv_cumsums))

                boot_ppv_ys = np.random.binomial(n=1, p=pred_y_a - self.delta, size=boot_ppv_cumsums.shape[0] if boot_ppv_cumsums is not None else self.n_bootstrap)[:,np.newaxis]
                boot_iter_ppv_stat = self.expected_vals["ppv"] - (boot_ppv_ys == pred_class)
                boot_ppv_cumsums = np.concatenate([boot_ppv_cumsums + boot_iter_ppv_stat, boot_iter_ppv_stat], axis=1) if boot_ppv_cumsums is not None else boot_iter_ppv_stat
                boot_ppv_cusums = np.max(boot_ppv_cumsums, axis=1)
                
                print("BOOT VS COUNT", boot_ppv_cumsums.shape, len(ppv_cusums))
                thres = np.quantile(
                    boot_ppv_cusums,
                    q=self.n_bootstrap * (1 - self.alpha_spending_func(len(ppv_cusums)))/boot_ppv_cumsums.shape[0])
                dcl.append(thres)
                boot_keep_mask = boot_ppv_cusums <= thres
                print("boot_out_mask", boot_keep_mask.shape, boot_keep_mask.sum())
                print("booting", boot_ppv_cumsums.shape)
                boot_ppv_cumsums = boot_ppv_cumsums[boot_keep_mask]
        
        ppv_cusum_df = pd.DataFrame({
            "value": np.concatenate([np.array(ppv_cusums), dcl]),
            "iter": np.concatenate([np.arange(len(dcl)), np.arange(len(dcl))]),
            "variable": ["stat"] * len(dcl) + ["dcl"] * len(dcl),
        })
        ppv_cusum_df['label'] = self.label
        return ppv_cusum_df

class wCUSUM(CUSUM):
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func, propensity_beta: np.ndarray = None, subgroup_func = None):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta
        self.subgroup_func = subgroup_func
        
    @property
    def label(self):
        return 'wCUSUM%s%s' % (
            '_intervene' if self.propensity_beta is not None else '_obs',
            '_subgroup' if self.subgroup_func is not None else '')
    
    def _setup(self, data_gen):
        data_gen = copy.deepcopy(data_gen)
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
        
        # estimate class variance
        subg_weights = np.ones(1)
        if self.subgroup_func is not None:
            x, y, a = data_gen.generate(10000)
            h = self.subgroup_func(x)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1:]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1/oracle_propensity
            
            iter_ppv_stats = (self.expected_vals["ppv"] - (y[:,np.newaxis] == pred_class)) * oracle_weight * h
            iter_ppv_stats = iter_ppv_stats[pred_class.flatten() == 1]
            subg_var_ests = np.var(iter_ppv_stats, axis=0)
            print("subg_var_ests", subg_var_ests)
            subg_weights = 1/np.sqrt(subg_var_ests)
        return data_gen, subg_weights
    
    def do_monitor(self, num_iters: int, data_gen):
        data_gen, self.subg_weights = self._setup(data_gen)
        print("self.subg_weights", self.subg_weights)
        
        ppv_cumsums = None
        subg_counts = None
        ppv_cusums = []
        for i in range(num_iters):
            print("iter", i, len(ppv_cusums))
            x, y, a = data_gen.generate(1)
            h = self.subgroup_func(x) if self.subgroup_func is not None else np.ones(1)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1:]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1/oracle_propensity
            print("weight", oracle_weight, oracle_weight * h * self.subg_weights)
            
            if pred_class == 1:
                iter_ppv_stat = (self.expected_vals["ppv"] - (y[:, np.newaxis] == pred_class)) * oracle_weight * h * self.subg_weights[np.newaxis, :]
                ppv_cumsums = np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat]) if ppv_cumsums is not None else iter_ppv_stat
                subg_counts = np.concatenate([subg_counts + h, h]) if subg_counts is not None else h
                ppv_cusums.append(np.max(ppv_cumsums[subg_counts > 0]))
                
        ppv_cusum_df = pd.DataFrame({
            "value": np.array(ppv_cusums),
            "iter": np.arange(len(ppv_cusums))
        })
        ppv_cusum_df['label'] = self.label
        return ppv_cusum_df

class CUSUM_score(CUSUM):
    label = 'sCUSUM'
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func, subgroup_func):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.subgroup_func = subgroup_func

    def do_monitor(self, num_iters: int, data_gen):
        score_cumsums = None
        subg_counts = None
        score_cusums = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            h = self.subgroup_func(x)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1]
            
            iter_score = (y - pred_y_a) * h
            score_cumsums = np.concatenate([score_cumsums + iter_score, iter_score]) if score_cumsums is not None else iter_score
            subg_counts = np.concatenate([subg_counts + h, h]) if subg_counts is not None else h
            score_cusums.append(np.max(score_cumsums))
        
        score_cusum_df = pd.DataFrame({
            "value": np.array(score_cusums),
            "iter": np.arange(len(score_cusums))
        })
        score_cusum_df['label'] = self.label
        return score_cusum_df