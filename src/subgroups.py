import numpy as np

THRES = 0.5
GROUP_THRES = 0

def avg_npv_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
        ],
        axis=1,
    )

def subgroup_npv_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
            # (pred_y_a < 0.25) * pred_class * (a == 0),
            # (pred_y_a < 0.25) * pred_class * (a == 1),
            # (pred_y_a > 0.25) * pred_class * (a == 0),
            # (pred_y_a > 0.25) * pred_class * (a == 1),
            (np.abs(x[:,:1]) < 1) * pred_class * (a == 0),
            (np.abs(x[:,1:2]) < 1) * pred_class * (a == 0),
            # (np.abs(x[:,:1]) < 2) * pred_class * (a == 1),
            # (np.abs(x[:,1:2]) < 2) * pred_class * (a == 1),
        ],
        axis=1,
    )
