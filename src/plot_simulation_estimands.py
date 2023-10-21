import os
import argparse
import logging


import scipy.stats as stats
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from cusums import CUSUM

PROC_DICT = {
    "naive": "Naive",
    "wCUSUM_intervene_subgroup2": "1I",
    "wCUSUM_obs_subgroup2": "1O",
    "wCUSUM_obs_subgroup6": "2O",
    "wCUSUM_intervene_subgroup6": "2I",
    "sCUSUM_less_obs": "3O",
    "sCUSUM_less_intervene": "3I",
    "wCUSUM_intervene_subgroup4": "1I",
    "wCUSUM_obs_subgroup4": "1O",
    "wCUSUM_obs_subgroup12": "2O",
    "wCUSUM_intervene_subgroup12": "2I",
    "sCUSUM_less_extreme_obs": "3O",
    "sCUSUM_less_extreme_intervene": "3I",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files, get power of methods"
    )
    parser.add_argument("--shift-time", type=int, default=0)
    parser.add_argument(
        "--result-files",
        type=str,
    )
    parser.add_argument(
        "--do-agg",
        action="store_true",
    )
    parser.add_argument(
        "--omit-naive",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--csv-file",
        type=str,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="_output/log.txt",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    args.result_files = args.result_files.split(",")
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    max_time = 0
    all_res = []
    for idx, f in enumerate(args.result_files):
        if not os.path.exists(f):
            continue
        res = pd.read_csv(f)
        max_time = max(max_time, res.actual_iter.max() + 1)
        fire_dict = {
            "procedure": [],
            "alert_time": [],
            "is_fired": [],
        }
        for procedure_name in res.label.unique():
            is_fired, fire_time = CUSUM.is_fired_alarm(res[res.label == procedure_name])
            fire_dict["procedure"].append(procedure_name)
            fire_dict["alert_time"].append(fire_time if is_fired else max_time * 2)
            fire_dict["is_fired"].append(is_fired)
        fire_df = pd.DataFrame(fire_dict)
        fire_df["seed"] = idx
        all_res.append(fire_df)
    all_res = pd.concat(all_res).reset_index(drop=True)
    all_res["procedure"] = all_res.procedure.replace(PROC_DICT)
    all_res["alert_time"] = all_res.alert_time * args.batch_size
    all_res.to_csv(args.csv_file, index=False)

    print(all_res.groupby('procedure').mean())

    uniq_procedures = all_res.procedure.drop_duplicates()
    print("uniq_procedures", uniq_procedures)

    if args.omit_naive:
        all_res = all_res[all_res.procedure != 'Naive']
    print(all_res)
    sns.set_context("paper", font_scale=2)
    plt.figure(figsize=(8, 5))
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
    ax = sns.ecdfplot(
        data=all_res,
        x="alert_time",
        hue="procedure",
        linewidth=3,
        legend=True
    )

    if args.shift_time > 0:
        plt.axvline(
            x=args.batch_size * (args.shift_time + 1), color="black", linestyle="--"
        )
    plt.xlim(0, max_time * args.batch_size + 1)
    
    ax.set_xlabel("Alarm time")
    plt.tight_layout()
    sns.despine()
    plt.savefig(args.plot)


if __name__ == "__main__":
    main()
