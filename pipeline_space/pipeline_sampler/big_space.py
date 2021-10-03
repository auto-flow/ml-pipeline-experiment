#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-02
# @Contact    : qichun.tang@bupt.edu.cn
from pipeline_space.pipeline_sampler.base import BasePipelineSampler
from pipeline_space.utils import generate_grid
from pipeline_space.hdl import hdl2cs


class BigPipelineSampler(BasePipelineSampler):

    def get_HDL(self):
        HDL = {
            "scaler(choice)": {
                "MinMaxScaler": {},
                "StandardScaler": {},
                "RobustScaler": {}
            },
            "selector(choice)": {
                # "LogisticRegression": {
                #     "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1, 5]}
                # },
                "LinearSVC": {
                    "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5, 1, 5]}
                },
                "XGBClassifier": {
                    "max_depth": {"_type": "int_quniform", "_value": [10, 50, 10]},
                },
                "LGBMClassifier": {
                    "num_leaves": {"_type": "int_quniform", "_value": [30, 90, 15]},
                },
                "RandomForestClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 2]},
                },
                "ExtraTreesClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 2]},
                },
                "None": {}
            },
            "learner(choice)": {
                "LGBMClassifier": {
                    "num_leaves": {"_type": "int_quniform", "_value": [30, 150, 15]},
                    "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                    "lambda_l1": {"_type": "ordinal", "_value": [1e-5, 0.1, 0.5]},
                    "lambda_l2": {"_type": "ordinal", "_value": [1e-5, 0.1, 0.5]},
                },
                "XGBClassifier": {
                    "max_depth": {"_type": "int_quniform", "_value": [10, 100, 10]},
                    "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                    "reg_lambda": {"_type": "ordinal", "_value": [1e-5, 0.1, 0.5]},
                    "reg_alpha": {"_type": "ordinal", "_value": [1e-5, 0.1, 0.5]},
                },
                "LinearSVC": {
                    "C": {"_type": "quniform", "_value": [0.01, 1, 0.033]},
                    "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
                },
                # "LogisticRegression": {
                #     "C": {"_type": "quniform", "_value": [0.01, 1, 0.033]},
                #     "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
                # },
                "RandomForestClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 2]},  # 2 7 12 15 22
                    "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 2]},  # 1 6 11
                    "bootstrap": {"_type": "choice", "_value": [True, False]},
                    # "max_features": {"_type": "choice", "_value": ["log2", "sqrt"]},
                },
                "ExtraTreesClassifier": {
                    "min_samples_split": {"_type": "int_quniform", "_value": [2, 22, 2]},  # 2 7 12 15 22
                    "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 2]},  # 1 6 11
                    "bootstrap": {"_type": "choice", "_value": [True, False]},
                    # "max_features": {"_type": "choice", "_value": ["log2", "sqrt"]},
                },
                "KNeighborsClassifier": {
                    "n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 2]},  # 3 5 7
                    "p": {"_type": "int_quniform", "_value": [1, 2, 1]},  # 1 2
                }
            }
        }
        return HDL

if __name__ == '__main__':
    print(len(generate_grid(hdl2cs(BigPipelineSampler().get_HDL()))))