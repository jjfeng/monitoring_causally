import os
import argparse
import pickle
import logging
import json

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

from subgroups import *
from cusums import *
from common import get_n_jobs, read_csv


def parse_args():
    parser = argparse.ArgumentParser(description="train a ML algorithm")
    parser.add_argument(
        "--job-idx",
        type=int,
        default=1,
        help="job idx",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="train frac",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LogisticRegression",
        choices=[
            "RandomForestClassifier",
            "LogisticRegression",
            "GradientBoostingClassifier",
        ],
    )
    parser.add_argument("--param-dict-file", type=str, default="model_dict.json")
    parser.add_argument(
        "--calib-method", type=str, choices=["sigmoid", "isotonic"], default="sigmoid"
    )
    parser.add_argument(
        "--train-dataset-template",
        type=str,
        default="_output/train_dataJOB.csv",
    )
    parser.add_argument(
        "--data-gen-file",
        type=str,
    )
    parser.add_argument(
        "--mdl-file-template",
        type=str,
        default="_output/mdlJOB.pkl",
    )
    parser.add_argument(
        "--perf-csv",
        type=str,
        default="_output/mdl_perf.csv",
    )
    parser.add_argument(
        "--log-file-template",
        type=str,
        default="_output/logJOB.txt",
    )
    parser.add_argument(
        "--plot-source-file-template",
        type=str,
        default="_output/plot_ml_JOB.png",
    )
    parser.add_argument(
        "--plot-target-file-template",
        type=str,
        default="_output/plot_ml_JOB.png",
    )
    args = parser.parse_args()
    args.train_dataset_file = args.train_dataset_template.replace(
        "JOB", str(args.job_idx)
    )
    args.log_file = args.log_file_template.replace("JOB", str(args.job_idx))
    args.mdl_file = args.mdl_file_template.replace("JOB", str(args.job_idx))
    args.plot_source_file = args.plot_source_file_template.replace(
        "JOB", str(args.job_idx)
    )
    args.plot_target_file = args.plot_target_file_template.replace(
        "JOB", str(args.job_idx)
    )
    return args


def do_evaluate_model(mdl, testX, testY, plot_file: str = None, prefix: str = None):
    pred_prob = mdl.predict_proba(testX)[:, 1]
    conf_matrix = confusion_matrix(testY.astype(int), (pred_prob > 0.5).astype(int))
    print("conf_matrix", conf_matrix)
    logging.info("denom %d", (conf_matrix[1, 1] + conf_matrix[0, 1]))
    logging.info(
        "ppv %s %f", prefix, conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    )
    logging.info("denom %d", conf_matrix[0, 0] + conf_matrix[1, 0])
    logging.info(
        "npv %s %f", prefix, conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    )
    print(
        "ppv %s %f", prefix, conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    )
    print(
        "npv %s %f", prefix, conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
    )

    if plot_file is not None:
        RocCurveDisplay.from_estimator(mdl, testX, testY)
        plt.savefig(plot_file)
        print(plot_file)

def do_evaluate_subgroup_all(mdl, testX, testY):
    res_npv = do_evaluate_subgroup(mdl, 0, testX, testY)
    res_ppv = do_evaluate_subgroup(mdl, 1, testX, testY)
    res = pd.concat([res_npv, res_ppv])
    return res

def do_evaluate_subgroup(mdl, pred_label_match: int, testX, testY):
    """Evaluate performance in subgroups

    Args:
        mdl (_type_): _description_
        pred_label_match (int): checking for predictions = 0 or 1
        testX (_type_): covariates
        testY (_type_): outcomes
        out_file (str, optional): file to output results in
    """
    pred_prob = mdl.predict_proba(testX)[:, 1]
    h = SubgroupDetector().detect_with_a(
        testX[:, :-1], testX[:, -1:], pred_prob.reshape((-1, 1)), pred_label_match=pred_label_match,
    )
    npvs = np.sum((testY[:, np.newaxis] == pred_label_match) * h, axis=0) / np.sum(h, axis=0)
    print("subgroup size", np.sum(h, axis=0))
    res = pd.DataFrame(
        {
            "h_idx": np.arange(h.shape[1]),
            "value": npvs,
        }
    )
    res["metric"] = "npv" if pred_label_match == 0 else "ppv"
    logging.info(res)
    return res


def main():
    args = parse_args()
    np.random.seed(args.seed_offset + args.job_idx)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    # Generate training data
    X, y = read_csv(args.train_dataset_file, read_A=False)
    trainX, testX, trainY, testY = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=args.train_frac
    )

    with open(args.param_dict_file, "r") as f:
        full_param_dict = json.load(f)
        param_dict = full_param_dict[args.model_type]

    # Train the original ML model
    n_jobs = get_n_jobs()
    print("n_jobs", n_jobs)
    if args.model_type == "GradientBoostingClassifier":
        base_mdl = GradientBoostingClassifier()
    elif args.model_type == "RandomForestClassifier":
        base_mdl = RandomForestClassifier(n_estimators=30, n_jobs=1)
    elif args.model_type == "LogisticRegression":
        base_mdl = LogisticRegression(penalty="l1", solver="saga")
    else:
        raise NotImplementedError("model type not implemented")
    if max([len(a) for a in param_dict.values()]) > 1:
        # If there is tuning to do
        grid_cv = GridSearchCV(
            estimator=base_mdl, param_grid=param_dict, cv=3, n_jobs=1, verbose=4
        )
        grid_cv.fit(
            trainX,
            trainY,
        )
        logging.info("CV BEST SCORE %f", grid_cv.best_score_)
        logging.info("CV BEST PARAMS %s", grid_cv.best_params_)
        print(grid_cv.best_params_)
        base_mdl.set_params(**grid_cv.best_params_)
    else:
        param_dict0 = {k: v[0] for k, v in param_dict.items()}
        base_mdl.set_params(**param_dict0)
        print(base_mdl)
        logging.info(base_mdl)
    # print("MODEL OOB", mdl.oob_score_)

    if args.model_type != "LogisticRegression":
        mdl = CalibratedClassifierCV(base_mdl, cv=5, method=args.calib_method)
    else:
        mdl = base_mdl

    logging.info("training data %s", X.shape)
    mdl.fit(
        trainX,
        trainY,
    )
    # logging.info(mdl.coef_)
    # logging.info(mdl.intercept_)

    with open(args.mdl_file, "wb") as f:
        pickle.dump(mdl, f)

    # Evaluate the model on source data
    do_evaluate_model(
        mdl,
        testX[testX[:, -1] == 0],
        testY[testX[:, -1] == 0],
        plot_file=args.plot_source_file,
        prefix="source",
    )
    do_evaluate_model(
        mdl,
        testX[testX[:, -1] == 1],
        testY[testX[:, -1] == 1],
        plot_file=args.plot_source_file,
        prefix="source",
    )
    do_evaluate_subgroup_all(mdl, testX, testY)

    # Evaluate the model on biased target data
    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
    NOBS = 10000
    data_gen.seed_offset = 123412

    data_gen.update_time(1000)
    target_x, target_y, target_a = data_gen.generate(NOBS, mdl)
    target_testX = np.concatenate([target_x, target_a.reshape((-1, 1))], axis=1)
    logging.info("biased post")
    do_evaluate_subgroup_all(mdl, target_testX, target_y)

    # Evaluate the model on unbiased target data
    logging.info("oracle pre")
    data_gen.update_time(0, set_seed=True)
    data_gen.propensity_beta = None
    data_gen.propensity_intercept = None
    target_x, target_y, target_a = data_gen.generate(NOBS)
    target_testX = np.concatenate([target_x, target_a.reshape((-1, 1))], axis=1)
    res = do_evaluate_subgroup_all(mdl, target_testX, target_y)
    res.to_csv(args.perf_csv, index=False)

    logging.info("oracle post")
    data_gen.update_time(10000, set_seed=True)
    target_x, target_y, target_a = data_gen.generate(NOBS)
    target_testX = np.concatenate([target_x, target_a.reshape((-1, 1))], axis=1)
    do_evaluate_subgroup_all(mdl, target_testX, target_y)


if __name__ == "__main__":
    main()
