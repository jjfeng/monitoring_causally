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

from cusums import CUSUM_naive


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
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
    )
    args = parser.parse_args()
    args.data_gen_file = args.data_gen_template.replace("JOB", str(args.job_idx))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.mdl_file = args.mdl_file_template.replace("JOB", str(args.job_idx))
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
    with open(args.mdl_file, "rb") as f:
        mdl = pickle.load(f)

    expected_vals = pd.Series({
        'ppv': 0.8
    })
    alpha_spending_func = lambda x: 0.001
    cusum = CUSUM_naive(mdl, threshold=0.5, expected_vals=expected_vals, alpha_spending_func=alpha_spending_func)
    res_df = cusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)

    res_df.to_csv(args.out_file, index=False)

    plt.clf()
    sns.lineplot(
        data=res_df,
        x="iter",
        y="value",
        hue="label"
    )
    plt.legend()
    plt.savefig("_output/test.png")


if __name__ == "__main__":
    main()
