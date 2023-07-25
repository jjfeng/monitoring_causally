import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

def make_loss_func(loss_name: str, ml_mdl: BaseEstimator):
    if loss_name == "brier":
        return lambda x,y: np.power(ml_mdl.predict_proba(x)[:,1] - y.flatten(), 2)
    else:
        raise NotImplementedError()

def to_safe_prob(prob, eps=1e-10):
    return np.maximum(eps, np.minimum(1 - eps, prob))

def convert_logit_to_prob(logit):
    return 1/(1 + np.exp(-logit))

def convert_prob_to_logit(prob, eps=1e-10):
    return np.log(prob/(1 - prob))

def get_complementary_logit(logit):
    return convert_prob_to_logit(1 - convert_logit_to_prob(logit))

def get_sigmoid_deriv(logit):
    p = convert_logit_to_prob(logit)
    return p * (1 - p)

def get_inv_sigmoid_deriv(prob):
    return 1/prob + 1/(1 - prob)

def read_csv(csv_file: str, read_A: bool=False):
    df = pd.read_csv(csv_file)
    Y = df.iloc[:,-1]
    if read_A:
        X = df.iloc[:,:-2]
        A = df.iloc[:,-2]
        return X, A, Y
    else:
        X = df.iloc[:,:-1]
        return X, Y

def get_n_jobs():
    n_cpu = int(os.getenv('OMP_NUM_THREADS')) if os.getenv('OMP_NUM_THREADS') is not None else 0
    n_jobs = max(n_cpu - 1, 1) if n_cpu > 0 else -1
    return n_jobs