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

THRES = 0.5

# TODO: unify data generation so they do not differ


def parse_args():
    parser = argparse.ArgumentParser(description="monittor a ML algorithm")
    parser.add_argument(
        "--res-csv",
        type=str,
    )
    parser.add_argument(
        "--plot-file",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    res_df = pd.read_csv(args.res_csv)
    print("res_df", res_df)

    for proc_label in  res_df.label.unique():
        max_val = res_df.value[res_df.label == proc_label].max()
        print("mas", max_val)
        res_df.value[res_df.label == proc_label] /= max_val

    plt.clf()
    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x="actual_iter", y="value", hue="label", style="variable")
    sns.despine()
    plt.legend()
    plt.savefig(args.plot_file)
    print(args.plot_file)


if __name__ == "__main__":
    main()
