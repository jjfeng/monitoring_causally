import os
import copy
import argparse
import pickle
import logging
import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

from common import get_n_jobs, read_csv, to_safe_prob


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
        default=10,
        help="batch size for monitoring",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="iters for monitoring",
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
        "--out-file",
        type=str,
        default="_output/out.csv",
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
    return args

def make_propensity_features(x, window_size = 10, backwards=True):
    if backwards:
        x_window = x[-window_size:]
        t = np.arange(-x_window.shape[0],0)[:,np.newaxis]/window_size
    else:
        x_window = x
        t = np.arange(x.shape[0])[:,np.newaxis]/window_size

    xt = np.concatenate([
        x_window, 
        t,
        # x_window * t,
        ], axis=1)
    return xt

def do_monitor(data_gen, mdl, prev_x,prev_a,prev_y, batch_size, num_iters, window_size, null_val):
    propensity_mdl = LogisticRegression(penalty="l1", solver="saga", max_iter=10000)
    wcumsums = np.zeros(num_iters)
    wcusum_stats = np.zeros(num_iters)
    cumsums = np.zeros(num_iters)
    cusum_stats = np.zeros(num_iters)
    for i in range(num_iters):
        x, y, a = data_gen.generate(batch_size)
        pred_y = mdl.predict_proba(np.concatenate([x,a[:, np.newaxis]], axis=1))[:,1]
        loss = np.power(pred_y - y, 2)

        # train propensity model
        prev_xt = make_propensity_features(prev_x, window_size, backwards=True)
        propensity_mdl.fit(prev_xt, prev_a[-window_size:])
        print(propensity_mdl.coef_)

        xt = make_propensity_features(x, window_size, backwards=False)
        propensity_a1 = to_safe_prob(propensity_mdl.predict_proba(xt)[:,1])
        propensity_a0 = 1 - propensity_a1
        # print(propensity_mdl.coef_, propensity_mdl.intercept_)
        
        ipw_loss = loss/propensity_a1 * (a == 1) # + loss/propensity_a0 * (a == 0)
        weights = np.sqrt(propensity_a1) #+ np.sqrt(propensity_a0)
        print("step", np.mean(weights * (ipw_loss - null_val)))
        wcumsums[:i+1] += np.mean(weights * (ipw_loss - null_val))
        cumsums[:i+1] += np.mean((ipw_loss - null_val))
        print("CUM MEAN", wcumsums[0]/(i + 1))
        wcusum_stats[i] = wcumsums[:i + 1].max()
        cusum_stats[i] = cumsums[:i + 1].max()

        # update data
        prev_x = np.concatenate([prev_x, x])
        prev_a = np.concatenate([prev_a, a])
        prev_y = np.concatenate([prev_y, y])
    
    cusum_df = pd.DataFrame({
        "value": cusum_stats,
        "idx": np.arange(num_iters),
        "label": ["cusum"] * num_iters
    })
    wcusum_df = pd.DataFrame({
        "value": wcusum_stats,
        "idx": np.arange(num_iters),
        "label": ["wcusum"] * num_iters
    })
    cumsum_df = pd.DataFrame({
        "value": cumsums,
        "idx": np.arange(num_iters),
        "label": ["cumsum"] * num_iters
    })
    wcumsum_df = pd.DataFrame({
        "value": wcumsums,
        "idx": np.arange(num_iters),
        "label": ["wcumsum"] * num_iters
    })
    return pd.concat([cusum_df, wcusum_df, cumsum_df, wcumsum_df])

def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
    with open(args.mdl_file, "rb") as f:
        mdl = pickle.load(f)

    MANY_OBS_NUM = 100000

    # biased batch monitoring
    x, y, a = data_gen.generate(MANY_OBS_NUM)
    pred_y_a0 = mdl.predict_proba(np.concatenate([x, np.zeros((MANY_OBS_NUM,1))], axis=1))[:,1]
    pred_y_a1 = mdl.predict_proba(np.concatenate([x, np.ones((MANY_OBS_NUM,1))], axis=1))[:,1]
    biased_loss_a0 = 0 #np.power(pred_y_a0 - y, 2)[a == 0].mean()
    biased_loss_a1 = np.power(pred_y_a1 - y, 2)[a == 1].mean()
    # biased_loss = (pred_y - y).mean()
    logging.info("biased_loss %f", biased_loss_a0 + biased_loss_a1)

    # get oracle performance
    oracle_data_gen = copy.deepcopy(data_gen)
    oracle_data_gen.propensity_beta = None
    oracle_x, oracle_y, oracle_a = oracle_data_gen.generate(MANY_OBS_NUM)
    pred_y_a0 = mdl.predict_proba(np.concatenate([oracle_x, np.zeros((MANY_OBS_NUM, 1))], axis=1))[:,1]
    pred_y_a1 = mdl.predict_proba(np.concatenate([oracle_x, np.ones((MANY_OBS_NUM, 1))], axis=1))[:,1]
    oracle_brier_a0 = 0 # np.power(pred_y_a0 - y, 2)[a == 0].mean()
    oracle_brier_a1 = np.power(pred_y_a1 - y, 2)[oracle_a == 1].mean()
    oracle_loss = oracle_brier_a0 + oracle_brier_a1
    # oracle_brier = (pred_y - y).mean()
    logging.info("BRIER %f", oracle_loss)

    # run monitoring
    WINDOW_SIZE = 100000
    res_df = do_monitor(data_gen, mdl, x,a,y, args.batch_size, args.num_iters, WINDOW_SIZE, null_val=oracle_loss)

    res_df.to_csv(args.out_file, index=False)

    # plt.clf()
    # plt.plot(res_dicts[0]["cusum"], label="cusum")
    # plt.plot(res_dicts[0]['wcusum'], label="wcusum")
    # plt.legend()
    # plt.savefig("_output/test.png")


if __name__ == "__main__":
    main()
