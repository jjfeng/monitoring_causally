import numpy as np
import pandas as pd

class PerfAccumulator:
    def __init__(self):
        self.ppv_cum_list = []

    def update(self, pred_y_a, y, x, a):
        self.ppv_cum_list.append(pred_y_a - y)


class CUSUM:
    def __init__(self, mdl, batch_size: int, alpha_spending_func):
        self.mdl = mdl
        self.batch_size = batch_size
        self.alpha_spending_func = alpha_spending_func
    
    def do_monitor(self, num_iters: int, data_gen):
        raise NotImplementedError()


class CUSUM_naive(CUSUM):
    label = 'naive'
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func

    def do_monitor(self, num_iters: int, data_gen):
        ppv_cumsums = None
        ppv_cusums = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1]
            pred_class = (pred_y_a > self.threshold).astype(int)
            
            if pred_class == 1:
                iter_ppv_stat = self.expected_vals["ppv"] - (y == pred_class)
                ppv_cumsums = np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat])if ppv_cumsums is not None else iter_ppv_stat
                ppv_cusums.append(np.max(ppv_cumsums))
        
        ppv_cusum_df = pd.DataFrame({
            "value": np.array(ppv_cusums),
            "iter": np.arange(ppv_cumsums.size)
        })
        ppv_cusum_df['label'] = self.label
        return ppv_cusum_df

class wCUSUM(CUSUM):
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func, propensity_beta: np.ndarray = None):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta

    @property
    def label(self):
        return 'wCUSUM%s' % ('_intervene' if self.propensity_beta is not None else '_obs')

    def do_monitor(self, num_iters: int, data_gen):
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
        ppv_cumsums = None
        ppv_cusums = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1/oracle_propensity
            
            if pred_class == 1:
                iter_ppv_stat = (self.expected_vals["ppv"] - (y == pred_class)) * oracle_weight
                ppv_cumsums = np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat]) if ppv_cumsums is not None else iter_ppv_stat
                ppv_cusums.append(np.max(ppv_cumsums))
        
        ppv_cusum_df = pd.DataFrame({
            "value": np.array(ppv_cusums),
            "iter": np.arange(ppv_cumsums.size)
        })
        ppv_cusum_df['label'] = self.label
        return ppv_cusum_df

class wCUSUM_intervention(CUSUM):
    label = 'wCUSUM_intervention'
    def __init__(self, mdl, threshold: float, expected_vals: pd.Series, alpha_spending_func, propensity_beta: np.ndarray = None):
        self.mdl = mdl
        self.threshold = threshold
        self.batch_size = 1
        self.expected_vals = expected_vals
        self.alpha_spending_func = alpha_spending_func
        self.propensity_beta = propensity_beta

    def do_monitor(self, num_iters: int, data_gen):
        if self.propensity_beta is not None:
            data_gen.propensity_beta = self.propensity_beta
        ppv_cumsums = None
        ppv_cusums = []
        for i in range(num_iters):
            print("iter", i)
            x, y, a = data_gen.generate(1)
            pred_y_a = self.mdl.predict_proba(np.concatenate([x, a[:,np.newaxis]], axis=1))[:,1]
            pred_class = (pred_y_a > self.threshold).astype(int)
            oracle_propensity = data_gen._get_propensity(x)
            oracle_weight = 1/oracle_propensity
            
            if pred_class == 1:
                iter_ppv_stat = (self.expected_vals["ppv"] - (y == pred_class)) * oracle_weight
                ppv_cumsums = np.concatenate([ppv_cumsums + iter_ppv_stat, iter_ppv_stat]) if ppv_cumsums is not None else iter_ppv_stat
                ppv_cusums.append(np.max(ppv_cumsums))
        
        ppv_cusum_df = pd.DataFrame({
            "value": np.array(ppv_cusums),
            "iter": np.arange(ppv_cumsums.size)
        })
        ppv_cusum_df['label'] = self.label
        return ppv_cusum_df
