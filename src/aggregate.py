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


def plot_normal(ax, plot_stats):
    xmin = plot_stats.min()
    xmax = plot_stats.max()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(plot_stats), np.sqrt(np.var(plot_stats)))

    ax.plot(x, p, "k", linewidth=2)


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    all_res = []
    for idx, f in enumerate(args.result_files):
        res = pd.read_csv(f)
        res["seed"] = idx
        all_res.append(res)
    all_res = pd.concat(all_res).reset_index()

    # bins = np.linspace(-20, 20, 40)
    print(all_res)
    fig, axs = plt.subplots(
        all_res.idx.max() + 1, 2, figsize=(10, all_res.idx.max() * 2)
    )
    for idx in all_res.idx.unique():
        print("idx", idx)
        cumsums = all_res[(all_res.idx == idx) & (all_res.label == "cumsum")].value
        wcumsums = all_res[(all_res.idx == idx) & (all_res.label == "wcumsum")].value
        print("CUSUM", cumsums.mean())
        # print("CUSUM", cumsums.mean(), all_res[(all_res.idx == idx) & (all_res.label == "cumsum")].sort_values('value'))
        print("WCUSUM", wcumsums.mean())
        axs[idx, 0].hist(cumsums, bins=20, label="cusum", alpha=0.5, density=True)
        plot_normal(axs[idx, 0], cumsums)
        axs[idx, 1].hist(wcumsums, bins=20, label="wcusum", alpha=0.5, density=True)
        plot_normal(axs[idx, 1], wcumsums)
        axs[idx, 0].legend()
        axs[idx, 1].legend()
    plt.savefig(args.plot)


if __name__ == "__main__":
    main()
