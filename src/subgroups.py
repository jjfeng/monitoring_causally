import numpy as np

THRES = 0.5
GROUP_THRES = 0

AVG_TREATMENTS = np.array([0,1])
def avg_npv_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
        ],
        axis=1,
    )

SUBGROUP_TREATMENTS = np.array([0,1,0,0,1,1])
def subgroup_npv_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    subg_mask = (np.abs(x[:,:1]) < 1) | (np.abs(x[:,1:2]) < 1)
    not_subg_mask = np.logical_not(subg_mask)
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
            subg_mask * pred_class * (a == 0),
            not_subg_mask * pred_class * (a == 0),
            subg_mask * pred_class * (a == 1),
            not_subg_mask * pred_class * (a == 1),
        ],
        axis=1,
    )

def score_subgroup_npv_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    subg_mask = (np.abs(x[:,:1]) < 1) | (np.abs(x[:,1:2]) < 1)
    not_subg_mask = np.logical_not(subg_mask)
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
            subg_mask * pred_class * (a == 0),
            not_subg_mask * pred_class * (a == 0),
            subg_mask * pred_class * (a == 1),
            not_subg_mask * pred_class * (a == 1),
        ],
        axis=1,
    )