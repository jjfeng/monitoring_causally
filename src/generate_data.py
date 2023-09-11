import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from data_generator import DataGenerator, SmallXShiftDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for testing")
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed offset",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="simple",
    )
    parser.add_argument("--x-mean", type=str, help="x mean")
    parser.add_argument("--intercept", type=float, help="intercept")
    parser.add_argument("--shift-type", type=str, choices=["none", "small_x_shift"])
    parser.add_argument(
        "--source-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument(
        "--target-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument(
        "--propensity-beta", type=str, help="comma separated list of coefficients"
    )
    parser.add_argument(
        "--propensity-intercept", type=float, help="propensity intercept"
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        default=100,
    )
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
    parser.add_argument(
        "--out-source-file-template",
        type=str,
        default="_output/dataJOB.csv",
    )
    parser.add_argument(
        "--out-target-file-template",
        type=str,
        default="_output/dataJOB.csv",
    )
    args = parser.parse_args()
    args.source_beta = np.array(list(map(float, args.source_beta.split(","))))
    args.target_beta = np.array(list(map(float, args.target_beta.split(","))))
    args.propensity_beta = np.array(list(map(float, args.propensity_beta.split(","))))
    args.x_mean = np.array(list(map(float, args.x_mean.split(","))))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.out_source_file = args.out_source_file_template.replace(
        "JOB", str(args.job_idx)
    )
    args.out_target_file = args.out_target_file_template.replace(
        "JOB", str(args.job_idx)
    )
    return args


def output_data(dg, args, out_file):
    X, y, A = dg.generate(args.num_obs)
    df = pd.DataFrame(X)
    df["A"] = A
    df["y"] = y
    print("MEAN OUTCOME rate", df.y.mean())
    df.to_csv(out_file, index=False)


def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # TODO: vary the type of data being returned based on data type string
    if args.shift_type == "small_x_shift":
        dg = SmallXShiftDataGenerator(
            source_beta=args.source_beta,
            target_beta=args.target_beta,
            intercept=args.intercept,
            x_mean=args.x_mean,
            beta_shift_time=1)
    else:
        dg = DataGenerator(
            source_beta=args.source_beta,
            target_beta=args.target_beta,
            intercept=args.intercept,
            x_mean=args.x_mean,
            beta_shift_time=1)
    output_data(dg, args, args.out_source_file)

    if args.out_data_gen_file:
        dg.propensity_beta = args.propensity_beta
        dg.propensity_intercept = args.propensity_intercept
        with open(args.out_data_gen_file, "wb") as f:
            pickle.dump(dg, f)

    dg.update_time(2)
    output_data(dg, args, args.out_target_file)


if __name__ == "__main__":
    main()
