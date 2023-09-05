import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from data_generator import DataGenerator


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
    parser.add_argument("--beta", type=str, help="comma separated list of coefficients")
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
        "--out-file-template",
        type=str,
        default="_output/dataJOB.csv",
    )
    args = parser.parse_args()
    args.beta = np.array(list(map(float, args.beta.split(","))))
    args.x_mean = np.array(list(map(float, args.x_mean.split(","))))
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.out_file = args.out_file_template.replace("JOB", str(args.job_idx))
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # TODO: vary the type of data being returned based on data type string
    dg = DataGenerator(source_beta=args.beta, intercept=0, x_mean=args.x_mean)

    X, y, A = dg.generate(args.num_obs)
    df = pd.DataFrame(X)
    df["A"] = A
    df["y"] = y
    print(df)
    df.to_csv(args.out_file, index=False)

    if args.out_data_gen_file:
        with open(args.out_data_gen_file, "wb") as f:
            pickle.dump(dg, f)


if __name__ == "__main__":
    main()
