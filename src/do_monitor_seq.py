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

from cusums import CUSUM_naive, wCUSUM, CUSUM_score


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
        default=10,
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


def subgroup_func(x):
    return np.concatenate(
        [
            x[:, :1] < 0,
            x[:, :1] > 0,
            x[:, 1:2] < 0,
            x[:, 1:2] > 0,
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
    with open(args.mdl_file, "rb") as f:
        mdl = pickle.load(f)

    expected_vals = pd.Series({"ppv": 0.9})
    alpha_spending_func = lambda x: min(1, args.alpha/args.num_iters * x)
    THRES = 0.5

    # Naive CUSUM
    np.random.seed(seed)
    cusum = CUSUM_naive(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    cusum_res_df = cusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("cusum fired? %s", cusum.is_fired_alarm(cusum_res_df))

    # Score monitoring
    np.random.seed(seed)
    score_cusum = CUSUM_score(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        alpha_spending_func=alpha_spending_func,
        subgroup_func=subgroup_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    score_cusum_res_df = score_cusum.do_monitor(
        num_iters=args.num_iters, data_gen=copy.deepcopy(data_gen)
    )
    logging.info(
        "score_cusum fired? %s", score_cusum.is_fired_alarm(score_cusum_res_df)
    )

    # # WCUSUM avg, no intervention, oracle propensity model
    np.random.seed(seed)
    wcusum = wCUSUM(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        propensity_beta=None,
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_res_df = wcusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("wcusum fired? %s", wcusum.is_fired_alarm(wcusum_res_df))

    # # WCUSUM with subgroups, no intervention, oracle propensity model
    np.random.seed(seed)
    wcusum_subg = wCUSUM(
        mdl,
        threshold=THRES,
        expected_vals=expected_vals,
        propensity_beta=None,
        alpha_spending_func=alpha_spending_func,
        subgroup_func=subgroup_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_subg_res_df = wcusum_subg.do_monitor(
        num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info(
        "wcusum_subg fired? %s", wcusum_subg.is_fired_alarm(wcusum_subg_res_df)
    )

    # WCUSUM with Intervention
    np.random.seed(seed)
    wcusum_int = wCUSUM(
        mdl,
        threshold=0.5,
        expected_vals=expected_vals,
        propensity_beta=np.zeros(data_gen.propensity_beta.size),
        alpha_spending_func=alpha_spending_func,
        delta=args.delta,
        n_bootstrap=args.n_boot,
    )
    wcusum_int_res_df = wcusum_int.do_monitor(
        num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info("wcusum_int fired? %s", wcusum_int.is_fired_alarm(wcusum_int_res_df))

    res_df = pd.concat(
        [
            cusum_res_df,
            wcusum_res_df,
            wcusum_int_res_df,
            wcusum_subg_res_df,
            score_cusum_res_df,
        ]
    )

    res_df.to_csv(args.out_file, index=False)

    plt.clf()
    sns.lineplot(data=res_df, x="actual_iter", y="value", hue="label", style="variable")
    plt.legend()
    plt.savefig(args.plot_file)
    print(args.plot_file)


if __name__ == "__main__":
    main()
