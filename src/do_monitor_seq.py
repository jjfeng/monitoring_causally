import os
import copy
import argparse
import pickle
import logging
import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from common import get_n_jobs, read_csv, to_safe_prob

from cusums import CUSUM, CUSUM_naive, wCUSUM, CUSUM_score

THRES = 0.5

# TODO: unify data generation so they do not differ


def parse_args():
    parser = argparse.ArgumentParser(description="monittor a ML algorithm")
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size for monitoring",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="iters for monitoring",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Type I error",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0,
        help="tolerated calibration error",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="num bootstrap seqs",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--data-gen-template",
        type=str,
        default="_output/dataJOB.csv",
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlJOB.pkl",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="_output/out.csv",
    )
    parser.add_argument(
        "--plot-file-template",
        type=str,
        default="_output/plotJOB.png",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
    )
    args = parser.parse_args()
    args.data_gen_file = args.data_gen_template.replace("JOB", str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.mdl_file = args.mdl_file_template.replace("JOB", str(args.job_idx))
    args.plot_file = args.plot_file_template.replace("JOB", str(args.job_idx))
    return args

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
            (x[:, :1] > 2) * pred_class * (a == 0),
            (x[:, 1:2] < 2) * pred_class * (a == 0),
            pred_class * (a == 0),
            (x[:, :1] > 2) * pred_class * (a == 1),
            (x[:, 1:2] < 2) * pred_class * (a == 1),
            pred_class * (a == 1),
        ],
        axis=1,
    )


def score_under_subgroup_func(x, pred_y_a, a, propensity_inputs):
    pred_class = pred_y_a < THRES
    return np.concatenate(
        [
            (x[:, :1] > 2) * pred_class,
            (x[:, 1:2] < 2) * pred_class,
            pred_class,
            # (x[:, :1] > 2) * pred_class * (a == 0),
            # (x[:, 1:2] < 2) * pred_class * (a == 0),
            # pred_class * (a == 0),
            # (x[:, :1] > 2) * pred_class * (a == 1),
            # (x[:, 1:2] < 2) * pred_class * (a == 1),
            # pred_class * (a == 1),
        ],
        axis=1,
    )


def main():
    args = parse_args()
    seed = args.seed_offset + args.job_idx
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.seed_offset = seed
    with open(args.mdl_file, "rb") as f:
        mdl = pickle.load(f)

    expected_vals = pd.Series({"ppv": 0.66, "npv": 0.8})
    alpha_spending_func = lambda eff_count: min(1, args.alpha / args.num_iters / args.batch_size * eff_count)

        # # WCUSUM avg, no intervention, oracle propensity model
    np.random.seed(seed)
    wcusum = wCUSUM(
        mdl,
        threshold=THRES,
        batch_size=args.batch_size,
        expected_vals=expected_vals,
        subgroup_func=avg_npv_func,
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_res_df = wcusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("wcusum fired? %s", CUSUM.is_fired_alarm(wcusum_res_df))

    # Naive CUSUM
    np.random.seed(seed)
    cusum = CUSUM_naive(
        mdl,
        threshold=THRES,
        batch_size=args.batch_size,
        expected_vals=expected_vals,
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
        metric="npv",
    )
    cusum_res_df = cusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("cusum fired? %s", CUSUM.is_fired_alarm(cusum_res_df))
    
    # SCORE
    np.random.seed(seed)
    score_cusum = CUSUM_score(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        batch_size=args.batch_size,
        alpha_spending_func=alpha_spending_func,
        subgroup_func=score_under_subgroup_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
        alt_overest=False, # check if we underestimated
    )
    score_cusum_res_df_under = score_cusum.do_monitor(
        num_iters=args.num_iters, data_gen=copy.deepcopy(data_gen)
    )
    logging.info(
        "%s fired? %s", score_cusum.label, CUSUM.is_fired_alarm(score_cusum_res_df_under)
    )

    # WCUSUM with Intervention
    np.random.seed(seed)
    propensity_beta_intervene = np.zeros(data_gen.propensity_beta.shape)
    propensity_beta_intervene[0] = 0
    wcusum_int = wCUSUM(
        mdl,
        threshold=THRES,
        batch_size=args.batch_size,
        expected_vals=expected_vals,
        propensity_beta=propensity_beta_intervene,
        subgroup_func=avg_npv_func,
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_int_res_df = wcusum_int.do_monitor(
        num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info("wcusum_int fired? %s", CUSUM.is_fired_alarm(wcusum_int_res_df))

    # WCUSUM with subgroups, no intervention, oracle propensity model
    np.random.seed(seed)
    wcusum_subg = wCUSUM(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        batch_size=args.batch_size,
        alpha_spending_func=alpha_spending_func,
        subgroup_func=subgroup_npv_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_subg_res_df = wcusum_subg.do_monitor(
        num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info(
        "wcusum_subg fired? %s", CUSUM.is_fired_alarm(wcusum_subg_res_df)
    )

    res_df = pd.concat(
        [
            cusum_res_df,
            wcusum_res_df,
            wcusum_int_res_df,
            wcusum_subg_res_df,
            score_cusum_res_df_under,
        ]
    )

    res_df.to_csv(args.out_file, index=False)

    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x="actual_iter", y="value", hue="label", style="variable")
    plt.legend()
    plt.savefig(args.plot_file)
    print(args.plot_file)


if __name__ == "__main__":
    main()
