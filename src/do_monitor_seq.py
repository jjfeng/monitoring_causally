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

from cusums import CUSUM, CUSUM_naive, wCUSUM, CUSUM_score
from common import get_n_jobs, read_csv, to_safe_prob
from subgroups import *

THRES = 0.5


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
        "--mean-intervene-beta",
        type=float,
        default=-2,
        help="slope for randomization model for intervention",
    )
    parser.add_argument(
        "--score-intervene-beta",
        type=float,
        default=-2,
        help="slope for randomization model for intervention",
    )
    parser.add_argument(
        "--intervene-intercept",
        type=float,
        default=0,
        help="intercept for randomization model for intervention",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="iters for monitoring",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="overest",
        help="options: overest, underest, less_extreme")
    parser.add_argument(
        "--metrics",
        type=str,
        default="npv",
        help="which metrics to monitor, comma separated")
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
        "--perf-targets-csv",
        type=str,
        default="_output/mdl_perf.csv",
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
    args.metrics = args.metrics.split(",")
    return args


def main():
    args = parse_args()
    seed = args.seed_offset + args.job_idx
    np.random.seed(seed)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    logging.info("SEED %d", seed)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.iter_seeds = np.random.randint(0, high=100000, size=args.num_iters * 2)
    with open(args.mdl_file, "rb") as f:
        mdl = pickle.load(f)
    perf_targets_df = pd.read_csv(args.perf_targets_csv)
    perf_targets_df["value"] = perf_targets_df.value - args.delta
    print("perf_targets", perf_targets_df)

    # uniform alpha spending function
    alpha_spending_func = lambda eff_count: min(
        1, args.alpha / args.num_iters / args.batch_size * eff_count
    )

    score_intervene_beta = np.zeros(data_gen.propensity_beta.shape)
    score_intervene_beta[0] = args.score_intervene_beta
    cusum_intervene_beta = np.zeros(data_gen.propensity_beta.shape)
    cusum_intervene_beta[0] = args.mean_intervene_beta
    intervene_intercept = args.intervene_intercept

    # SCORE
    score_cusum = CUSUM_score(
        mdl,
        batch_size=args.batch_size,
        alpha_spending_func=alpha_spending_func,
        subgroup_detector=ScoreSubgroupDetector(),
        delta=args.delta,
        n_bootstrap=args.n_boot,
        alternative=args.alternative,  # check if we underestimated
    )
    score_cusum_res_df_under = score_cusum.do_monitor(
        num_iters=args.num_iters, data_gen=copy.deepcopy(data_gen)
    )
    logging.info(
        "%s fired? %s",
        score_cusum.label,
        CUSUM.is_fired_alarm(score_cusum_res_df_under),
    )
    
    # WCUSUM with subgroups, intervention, oracle propensity model
    wcusum_subg = wCUSUM(
       mdl,
       perf_targets_df=perf_targets_df,
       batch_size=args.batch_size,
       propensity_beta=cusum_intervene_beta,
       propensity_intercept=intervene_intercept,
       alpha_spending_func=alpha_spending_func,
       subgroup_detector=SubgroupDetector(),
       delta=args.delta,
       n_bootstrap=args.n_boot,
       metrics=args.metrics,
    )
    wcusum_subg_int_res_df = wcusum_subg.do_monitor(
       num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info(
       "wcusum_subg_int fired? %s", CUSUM.is_fired_alarm(wcusum_subg_int_res_df)
    )

    # Naive CUSUM
    cusum = CUSUM_naive(
       mdl,
       batch_size=args.batch_size,
       perf_targets_df=perf_targets_df[perf_targets_df.h_idx <= 1],
       alpha_spending_func=alpha_spending_func,
       delta=args.delta,
       n_bootstrap=args.n_boot,
       metrics=args.metrics,
    )
    cusum_res_df = cusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("cusum fired? %s", CUSUM.is_fired_alarm(cusum_res_df))

    # SCORE -- intervention
    score_cusum = CUSUM_score(
       mdl,
       batch_size=args.batch_size,
       alpha_spending_func=alpha_spending_func,
       subgroup_detector=ScoreSubgroupDetector(),
       propensity_beta=score_intervene_beta,
       propensity_intercept=intervene_intercept,
       delta=args.delta,
       n_bootstrap=args.n_boot,
       alternative=args.alternative,  # check if we underestimated
    )
    score_cusum_int_res_df_under = score_cusum.do_monitor(
       num_iters=args.num_iters, data_gen=copy.deepcopy(data_gen)
    )
    logging.info(
       "%s fired? %s",
       score_cusum.label,
       CUSUM.is_fired_alarm(score_cusum_int_res_df_under),
    )

    
    # WCUSUM with subgroups, no intervention, oracle propensity model
    wcusum_subg = wCUSUM(
       mdl,
       perf_targets_df=perf_targets_df,
       subgroup_detector=SubgroupDetector(),
       batch_size=args.batch_size,
       alpha_spending_func=alpha_spending_func,
       delta=args.delta,
       n_bootstrap=args.n_boot,
       metrics=args.metrics,
    )
    wcusum_subg_res_df = wcusum_subg.do_monitor(
       num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info("wcusum_subg fired? %s", CUSUM.is_fired_alarm(wcusum_subg_res_df))

    # WCUSUM avg, with Intervention
    wcusum_int = wCUSUM(
       mdl,
       batch_size=args.batch_size,
       perf_targets_df=perf_targets_df[perf_targets_df.h_idx < 2],
       propensity_beta=cusum_intervene_beta,
       propensity_intercept=intervene_intercept,
       subgroup_detector=SubgroupDetectorSimple(),
       alpha_spending_func=alpha_spending_func,
       delta=args.delta,
       n_bootstrap=args.n_boot,
       metrics=args.metrics,
    )
    wcusum_int_res_df = wcusum_int.do_monitor(
       num_iters=args.num_iters, data_gen=data_gen
    )
    logging.info("wcusum_int fired? %s", CUSUM.is_fired_alarm(wcusum_int_res_df))

    # WCUSUM avg, no intervention, oracle propensity model
    wcusum = wCUSUM(
       mdl,
       batch_size=args.batch_size,
       perf_targets_df=perf_targets_df[perf_targets_df.h_idx < 2],
       subgroup_detector=SubgroupDetectorSimple(),
       alpha_spending_func=alpha_spending_func,
       delta=args.delta,
       n_bootstrap=args.n_boot,
       metrics=args.metrics,
    )
    wcusum_res_df = wcusum.do_monitor(num_iters=args.num_iters, data_gen=data_gen)
    logging.info("wcusum fired? %s", CUSUM.is_fired_alarm(wcusum_res_df))

    res_df = pd.concat(
        [
            cusum_res_df,
            wcusum_res_df,
            wcusum_int_res_df,
            wcusum_subg_res_df,
            wcusum_subg_int_res_df,
            score_cusum_res_df_under,
            score_cusum_int_res_df_under,
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
