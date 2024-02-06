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
from plot_simulation_estimands import PROC_DICT
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
        "--batch-size",
        type=int,
        default=1,
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

    for proc_label in res_df.label.unique():
        max_val = res_df.value[res_df.label == proc_label].max()
        print("mas", max_val)
        res_df.value[res_df.label == proc_label] /= max_val

    res_df["actual_iter"] = res_df.actual_iter * args.batch_size
    res_df["Procedure"] = res_df.label.replace(PROC_DICT)
    res_df[" "] = res_df.variable.replace({
        'dcl': 'Control limit',
        'stat': 'Chart statistic',
    })

    # res_df = res_df[res_df.Procedure.isin(["Naive", "3I"])]

    plt.clf()
    sns.set_context("paper", font_scale=2.5)
    plt.figure(figsize=(20, 5))
    deep_colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)]
    pastel_colors = [(0.6313725490196078, 0.788235294117647, 0.9568627450980393), (1.0, 0.7058823529411765, 0.5098039215686274), (0.5529411764705883, 0.8980392156862745, 0.6313725490196078)]
    sns.set_palette(
        [
            'black',
            deep_colors[0],
            pastel_colors[0],
            deep_colors[1],
            pastel_colors[1],
            deep_colors[2],
            pastel_colors[2],
        ]
    )
    ax = sns.lineplot(
        data=res_df,
        x="actual_iter",
        y="value",
        hue="Procedure",
        style=" ",
        legend=True,
        linewidth=3,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.3))
    plt.xlabel("Time")
    plt.ylabel("Value")
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.plot_file)
    print(args.plot_file)


if __name__ == "__main__":
    main()
