import os
import argparse
import logging

import scipy.stats as stats
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="aggregate result files, get power of methods"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
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
    
    all_res = []
    for idx, f in enumerate(args.result_files):
        res = pd.read_csv(f)
        res['seed'] = idx
        all_res.append(res)
    all_res = pd.concat(all_res).reset_index()

    bins = np.linspace(-2, 2, 20)
    print(all_res)
    cumsums = all_res[(all_res.idx == 0) & (all_res.label == "cumsum")].value
    wcumsums = all_res[(all_res.idx == 0) & (all_res.label == "wcumsum")].value
    print(cumsums)
    plt.hist(cumsums, label="cusum", alpha=0.5)
    plt.hist(wcumsums, label="wcusum", alpha=0.5)
    plt.legend()
    plt.savefig(args.plot)

if __name__ == "__main__":
    main()
