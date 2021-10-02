#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn

from time import time

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pipeline_space.hdl import hdl2cs
from pipeline_space.hdl import layering_config

from pipeline_space.automl_pipeline.construct_pipeline import construct_pipeline

X, y, cat = load("/media/tqc/doc/Project/metalearn_experiment/data/189864.bz2")
y = y.astype(int)
X = MinMaxScaler().fit_transform(X)
HDL = {
    "scaler(choice)": {
        "MinMaxScaler": {
        },
        "StandardScaler": {
        },
        "RobustScaler": {
        },
    },
    "selector(choice)": {
        "LogisticRegression": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5]}
        },
        "LinearSVC": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5]}
        },
        "XGBClassifier": {
            "max_depth": {"_type": "int_quniform", "_value": [10, 70, 20]},
        },
        "LGBMClassifier": {
            "num_leaves": {"_type": "int_quniform", "_value": [30, 90, 30]},
        },
        "RandomForestClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 5]},
        },
        "ExtraTreesClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 5]},
        },
        "None": {}
    },
    "learner(choice)": {
        "LGBMClassifier": {
            # "n_estimators": {"_type": "ordinal", "_value": [50, 100, 200]},
            "num_leaves": {"_type": "int_quniform", "_value": [30, 150, 30]},
            "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
            "subsample": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
            "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1e-2, 1e-1, 1]}
        },
        "XGBClassifier": {
            # "n_estimators": {"_type": "ordinal", "_value": [50, 100, 200]},
            "max_depth": {"_type": "int_quniform", "_value": [10, 90, 20]},
            "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
            "subsample": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
            "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1e-2, 1e-1, 1]}
        },
        "LinearSVC": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1]},
            "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
        },
        "LogisticRegression": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1]},
            "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
        },
        "RandomForestClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},
            "bootstrap": {"_type": "choice", "_value": [True, False]},
        },
        "ExtraTreesClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},
            "bootstrap": {"_type": "choice", "_value": [True, False]},
        },
        "KNeighborsClassifier": {
            "n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 2]},
        }
    }
}
CS = hdl2cs(HDL)
CS.seed(50)
layered_dict = layering_config(CS.sample_configuration().get_dictionary())
print(layered_dict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

start_time = time()
pipeline = construct_pipeline(layered_dict)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
print(time() - start_time)

start_time = time()
pipeline = construct_pipeline(layered_dict)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
print(time() - start_time)
