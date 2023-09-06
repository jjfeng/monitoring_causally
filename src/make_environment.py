import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from data_generator import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="make data generator")
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="simple",
    )
    parser.add_argument("--x-mean", type=str, help="x mean")
    parser.add_argument(
        "--propensity-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument(
        "--propensity-intercept", type=float, help="propensity intercept"
    )
    parser.add_argument("--intercept", type=float, help="intercept")
    parser.add_argument(
        "--source-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument(
        "--target-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument("--beta-shift-time", type=int, default=None)
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/data_logJOB.txt",
    )
    parser.add_argument(
        "--out-data-gen-file",
        type=str,
        default="_output/datagen.pkl",
    )
    args = parser.parse_args()
    args.source_beta = np.array(list(map(float, args.source_beta.split(","))))
    args.target_beta = np.array(list(map(float, args.target_beta.split(","))))
    args.propensity_beta = np.array(list(map(float, args.propensity_beta.split(","))))
    args.x_mean = np.array(list(map(float, args.x_mean.split(","))))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.out_data_gen_file = args.out_data_gen_file.replace("JOB", str(args.job_idx))
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # TODO: vary the type of data being returned based on data type string
    dg = DataGenerator(
        source_beta=args.source_beta,
        target_beta=args.target_beta,
        beta_shift_time=args.beta_shift_time,
        intercept=args.intercept,
        x_mean=args.x_mean,
        propensity_beta=args.propensity_beta,
        propensity_intercept=args.propensity_intercept,
    )
    # This is just a sanity check
    X, y, A = dg.generate(10)

    with open(args.out_data_gen_file, "wb") as f:
        pickle.dump(dg, f)


if __name__ == "__main__":
    main()
