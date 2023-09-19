import numpy as np

THRES = 0.5
GROUP_THRES = 0

AVG_TREATMENTS = np.array([0,1])
def avg_npv_func(x, pred_y_a01):
    return pred_y_a01 < THRES

def avg_npv_a_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    return np.concatenate(
        [
            pred_class * (a == 0),
            pred_class * (a == 1),
        ],
        axis=1,
    )

SUBGROUP_TREATMENTS = np.array([0,1,0,0,1,1])
def _get_subgroup(X):
    return ((X[:,:1] > -1) & (X[:,:1] < 2)) & (np.abs(X[:,1:2]) < 2.5)

def subgroup_npv_func(x, pred_y_a01):
    pred_class_a01 = pred_y_a01 < THRES
    subg_mask = _get_subgroup(x) 
    not_subg_mask = np.logical_not(subg_mask)
    return np.concatenate(
        [
            pred_class_a01[:,:1],
            pred_class_a01[:,1:],
            subg_mask * pred_class_a01[:,:1],
            not_subg_mask * pred_class_a01[:,:1],
            subg_mask * pred_class_a01[:,1:],
            not_subg_mask * pred_class_a01[:,1:],
        ],
        axis=1,
    )
def subgroup_npv_a_func(x, a, pred_y_a):
    pred_class = pred_y_a < THRES
    subg_mask = _get_subgroup(x)
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
    # pred_class = pred_y_a < THRES
    subg_mask = _get_subgroup(x)
    not_subg_mask = np.logical_not(subg_mask)
    return np.concatenate(
        [
            (a == 0),
            (a == 1),
            subg_mask * (a == 0),
            not_subg_mask * (a == 0),
            subg_mask * (a == 1),
            not_subg_mask * (a == 1),
        ],
        axis=1,
    )