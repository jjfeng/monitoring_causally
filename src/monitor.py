import os
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
        wcumsums[:i] += np.mean(weights * (ipw_loss - null_val))
        cumsums[:i] += np.mean((ipw_loss - null_val))
        print("CUM MEAN", wcumsums[0]/(i + 1))
        wcusum_stats[i] = wcumsums[:i + 1].max()
        cusum_stats[i] = cumsums[:i + 1].max()

        # update data
        prev_x = np.concatenate([prev_x, x])
        prev_a = np.concatenate([prev_a, a])
        prev_y = np.concatenate([prev_y, y])
    return {
        "wcusum": wcusum_stats,
        "cusum": cusum_stats,
        "wcumsum": wcumsums,
        "cumsum": cumsums,
    }

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
    print("biased_loss", biased_loss_a0 + biased_loss_a1)

    # run monitoring
    BATCH_SIZE = 10
    NUM_ITERS = 10
    WINDOW_SIZE = 100000
    NULL_VAL = 0.14

    NUM_REPS = 100
    res_dicts = []
    for i in range(NUM_REPS):
        print("rep", i)
        res_dict = do_monitor(data_gen, mdl, x,a,y, BATCH_SIZE, NUM_ITERS, WINDOW_SIZE, NULL_VAL)
        res_dicts.append(res_dict)

    # get oracle performance
    data_gen.propensity_beta = None
    x, y, a = data_gen.generate(MANY_OBS_NUM)
    pred_y_a0 = mdl.predict_proba(np.concatenate([x, np.zeros((MANY_OBS_NUM, 1))], axis=1))[:,1]
    pred_y_a1 = mdl.predict_proba(np.concatenate([x, np.ones((MANY_OBS_NUM, 1))], axis=1))[:,1]
    oracle_brier_a0 = 0 # np.power(pred_y_a0 - y, 2)[a == 0].mean()
    oracle_brier_a1 = np.power(pred_y_a1 - y, 2)[a == 1].mean()
    # oracle_brier = (pred_y - y).mean()
    print("BRIER", oracle_brier_a0 + oracle_brier_a1)

    plt.hist([res_dicts[i]["cumsum"][0] for i in range(NUM_REPS)], label="cusum", alpha=0.5)
    plt.hist([res_dicts[i]["wcumsum"][0] for i in range(NUM_REPS)], label="wcusum", alpha=0.5)
    plt.legend()
    plt.savefig("_output/hist.png")

    plt.clf()
    plt.plot(res_dicts[0]["cusum"], label="cusum")
    plt.plot(res_dicts[0]['wcusum'], label="wcusum")
    plt.legend()
    plt.savefig("_output/test.png")


if __name__ == "__main__":
    main()
