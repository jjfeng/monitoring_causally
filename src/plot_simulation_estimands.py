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
    'naive': '1A: Naive',
    'wCUSUM_intervene_subgroup2': '1int: NPV int',
    'wCUSUM_obs_subgroup2': '1obs: NPV obs',
    'wCUSUM_obs_subgroup6': '2obs: NPV subgroup',
    'wCUSUM_intervene_subgroup6': '2int: NPV subgroup',
    'sCUSUM_less_obs': '3obs: Residuals',
    'sCUSUM_less_intervene': '3int: Residuals',
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files, get power of methods"
    )
    parser.add_argument(
        "--shift-time",
        type=int,
        default=0
    )
    parser.add_argument(
        "--result-files",
        type=str,
    )
    parser.add_argument(
        "--do-agg",
        action="store_true",
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
        res = pd.read_csv(f)
        max_time = max(max_time, res.actual_iter.max() + 1)
        fire_dict = {
            'procedure': [],
            'alert_time': [],
        }
        for procedure_name in res.label.unique():
            is_fired, fire_time = CUSUM.is_fired_alarm(res[res.label == procedure_name])
            fire_dict['procedure'].append(procedure_name)
            fire_dict['alert_time'].append(fire_time if is_fired else max_time * 2)
        fire_df = pd.DataFrame(fire_dict)
        fire_df['seed'] = idx
        all_res.append(fire_df)
    all_res = pd.concat(all_res).reset_index(drop=True)
    all_res['procedure'] = all_res.procedure.replace(PROC_DICT)
    all_res['alert_time'] = all_res.alert_time * args.batch_size
    all_res.to_csv(args.csv_file, index=False)

    print(all_res)
    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(8, 5))
    ax = sns.ecdfplot(
        data=all_res,
        x="alert_time",
        hue='procedure',
        legend=True,
        linewidth=3,
    )
    plt.axvline(x=args.batch_size * (args.shift_time + 1), color="black", linestyle="--")
    plt.xlim(0, max_time * args.batch_size + 1)
    ax.set_xlabel("Alarm time")
    plt.tight_layout()
    sns.despine()
    plt.savefig(args.plot)


if __name__ == "__main__":
    main()
