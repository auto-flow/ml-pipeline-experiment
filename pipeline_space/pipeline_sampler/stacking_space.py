#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-02
# @Contact    : qichun.tang@bupt.edu.cn
from joblib import dump

from pipeline_space.hdl import hdl2cs
from pipeline_space.pipeline_sampler.base import BasePipelineSampler
from pipeline_space.utils import generate_grid


class BaggingPipelineSampler(BasePipelineSampler):

    def all_learner(self):
        return {
            "LGBMClassifier": {
                # "num_leaves": {"_type": "int_quniform", "_value": [30, 150, 30]},  # 30 60 90 120 150
                "colsample_bytree": {"_type": "quniform", "_value": [0.6, 1, 0.2]},
                "reg_lambda": {"_type": "choice", "_value": [1e-3, 1]}
            },
            "XGBClassifier": {
                # "max_depth": {"_type": "int_quniform", "_value": [10, 90, 20]},  # 10 30 50 70 90
                "colsample_bytree": {"_type": "quniform", "_value": [0.6, 1, 0.2]},
                "reg_lambda": {"_type": "choice", "_value": [1e-3, 1]}
            },
            "LinearSVC": {
                "C": {"_type": "quniform", "_value": [0.01, 1.01, 0.5]},
                "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
            },
            "LogisticRegression": {
                "C": {"_type": "quniform", "_value": [0.01, 1.01, 0.5]},
                "penalty": {"_type": "choice", "_value": ["l2", "l1"]},
            },
            "RandomForestClassifier": {
                "min_samples_split": {"_type": "int_quniform", "_value": [2, 12, 5]},  # 2 7 12 15 22
                # "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 5]},  # 1 6 11
                "bootstrap": {"_type": "choice", "_value": [True, False]},
            },
            "ExtraTreesClassifier": {
                "min_samples_split": {"_type": "int_quniform", "_value": [2, 12, 5]},  # 2 7 12 15 22
                # "min_samples_leaf": {"_type": "int_quniform", "_value": [1, 11, 5]},  # 1 6 11
                "bootstrap": {"_type": "choice", "_value": [True, False]},
            },
            "KNeighborsClassifier": {
                "n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 2]},  # 3 5 7
                "p": {"_type": "int_quniform", "_value": [1, 2, 1]},  # 1 2
            }
        }

    def calc_every_learner_counts(self):
        res = {}
        for learner, HP in self.all_learner().items():
            CS = hdl2cs({learner: HP})
            grids = generate_grid(CS)
            print(f"{learner}:", len(grids))
            res[learner] = grids
        return res

    def calc_bagging_learner_counts(self):
        grids = generate_grid(hdl2cs(self.get_HDL()))
        print(f"stacked:", len(grids))
        return grids

    def get_HDL(self):
        HDL = {}
        for i, (learner, HP) in enumerate(self.all_learner().items()):
            HDL[f"learner{i}(choice)"] = {learner: HP, "None": {}}
        return HDL


if __name__ == '__main__':
    sampler = BaggingPipelineSampler()
    learner_grids = sampler.calc_every_learner_counts()
    bagging_grids = sampler.calc_bagging_learner_counts()
    grids = {
        'learner_grids': learner_grids,
        'bagging_grids': bagging_grids,
    }
    dump(grids, "bagging_grids.pkl")
