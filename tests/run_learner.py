#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn
from time import time

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline_space.automl_pipeline.learner import Learner
from pipeline_space.hdl import hdl2cs, layering_config
from pipeline_space.utils import generate_grid


def run_linear_model():
    X, y, cat = load("/data/Project/AutoML/ML-Pipeline-Experiment/189864.bz2")
    y = y.astype(int)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    HDL = {
        "learner(choice)": {
            "LinearSVC": {
                "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1]},
                "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
            },
            "LogisticRegression": {
                "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1]},
                "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
            },
        }
    }
    CS = hdl2cs(HDL)
    for config in generate_grid(CS):
        config = config.get_dictionary()
        layered_config = layering_config(config)
        print(layered_config)
        AS, HP = layered_config['learner'].popitem()
        learner = Learner(AS, HP)
        start_time = time()
        learner.fit(X_train, y_train)
        score = learner.score(X_test, y_test)
        cost_time = time() - start_time
        print(f"score {score:.3f}s")
        print(f"cost  {cost_time:.3f}s")
        print()


def run_boosting_model():
    X, y, cat = load("/data/Project/AutoML/ML-Pipeline-Experiment/189864.bz2")
    y = y.astype(int)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    HDL = {
        "learner(choice)": {
            "LGBMClassifier": {
                # "n_estimators": {"_type": "ordinal", "_value": [50, 100, 200]},
                "num_leaves": {"_type": "int_quniform", "_value": [30, 150, 30]},
                # "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                # "subsample": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                # "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1e-2, 1e-1, 1]}
            },
            "XGBClassifier": {
                # "n_estimators": {"_type": "ordinal", "_value": [50, 100, 200]},
                "max_depth": {"_type": "int_quniform", "_value": [10, 90, 20]},
                # "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                # "subsample": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                # "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1e-2, 1e-1, 1]}
            },
        }
    }
    CS = hdl2cs(HDL)
    for config in generate_grid(CS):
        config = config.get_dictionary()
        layered_config = layering_config(config)
        print(layered_config)
        AS, HP = layered_config['learner'].popitem()
        learner = Learner(AS, HP)
        start_time = time()
        learner.fit(X_train, y_train)
        score = learner.score(X_test, y_test)
        cost_time = time() - start_time
        print(f"score {score:.3f}s")
        print(f"cost  {cost_time:.3f}s")
        print()


def run_bagging_model():
    X, y, cat = load("/data/Project/AutoML/ML-Pipeline-Experiment/189864.bz2")
    y = y.astype(int)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    HDL = {
        "learner(choice)": {
            "RandomForestClassifier": {
                "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},
                "bootstrap": {"_type": "choice", "_value": [True, False]},
            },
            "ExtraTreesClassifier": {
                "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},
                "bootstrap": {"_type": "choice", "_value": [True, False]},
            },
        }
    }
    CS = hdl2cs(HDL)
    for config in generate_grid(CS):
        config = config.get_dictionary()
        layered_config = layering_config(config)
        print(layered_config)
        AS, HP = layered_config['learner'].popitem()
        learner = Learner(AS, HP)
        start_time = time()
        learner.fit(X_train, y_train)
        score = learner.score(X_test, y_test)
        cost_time = time() - start_time
        print(f"score {score:.3f}s")
        print(f"cost  {cost_time:.3f}s")
        print()


def run_knn_model():
    X, y, cat = load("/data/Project/AutoML/ML-Pipeline-Experiment/189864.bz2")
    y = y.astype(int)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    HDL = {
        "learner(choice)": {
            "KNeighborsClassifier": {
                "n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 2]},
            }
        }
    }
    CS = hdl2cs(HDL)
    for config in generate_grid(CS):
        config = config.get_dictionary()
        layered_config = layering_config(config)
        print(layered_config)
        AS, HP = layered_config['learner'].popitem()
        learner = Learner(AS, HP)
        start_time = time()
        learner.fit(X_train, y_train)
        score = learner.score(X_test, y_test)
        cost_time = time() - start_time
        print(f"score {score:.3f}s")
        print(f"cost  {cost_time:.3f}s")
        print()


if __name__ == '__main__':
    # run_linear_model()
    # run_boosting_model()
    # run_bagging_model()
    run_knn_model()
