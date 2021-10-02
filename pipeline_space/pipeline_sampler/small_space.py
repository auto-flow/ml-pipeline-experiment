#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-02
# @Contact    : qichun.tang@bupt.edu.cn
from .base import BasePipelineSampler


class SmallPipelineSampler(BasePipelineSampler):

    def get_HDL(self):
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
                    "max_depth": {"_type": "int_quniform", "_value": [10, 50, 20]},
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
                    "num_leaves": {"_type": "int_quniform", "_value": [30, 150, 30]},  # 30 60 90 120 150
                    "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                    "reg_lambda": {"_type": "choice", "_value": [1e-3, 1]}
                },
                "XGBClassifier": {
                    "max_depth": {"_type": "int_quniform", "_value": [10, 90, 20]},  # 10 30 50 70 90
                    "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                    "reg_lambda": {"_type": "choice", "_value": [1e-3, 1]}
                },
                "LinearSVC": {
                    "C": {"_type": "quniform", "_value": [0.01, 1, 0.066]},
                    "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
                },
                "LogisticRegression": {
                    "C": {"_type": "quniform", "_value": [0.01, 1, 0.066]},
                    "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
                },
                "RandomForestClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},  # 2 7 12 15 22
                    "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 5]},  # 1 6 11
                    "bootstrap": {"_type": "choice", "_value": [True, False]},
                },
                "ExtraTreesClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 5]},  # 2 7 12 15 22
                    "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 5]},  # 1 6 11
                    "bootstrap": {"_type": "choice", "_value": [True, False]},
                },
                "KNeighborsClassifier": {
                    "n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 2]},  # 3 5 7
                    "p": {"_type": "int_quniform", "_value": [1, 2, 1]},  # 1 2
                }
            }
        }
        return HDL
