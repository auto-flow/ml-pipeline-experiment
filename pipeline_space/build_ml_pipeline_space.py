#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-21
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter, defaultdict

from pipeline_space.hdl import hdl2cs
from pipeline_space.hdl import layering_config
from pipeline_space.utils import generate_grid
from pipeline_space.utils import get_hash_of_str


def get_HDL():
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
                "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1]}
            },
            "XGBClassifier": {
                "max_depth": {"_type": "int_quniform", "_value": [10, 90, 20]},  # 10 30 50 70 90
                "colsample_bytree": {"_type": "quniform", "_value": [0.8, 1, 0.1]},
                "reg_lambda": {"_type": "ordinal", "_value": [1e-3, 1]}
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


def get_hyperopt_space():
    from hyperopt import hp
    space = {
        "scaler": hp.choice("scaler", [
            "MinMaxScaler", "StandardScaler", "RobustScaler"
        ]),
        "selector": hp.choice("selector", [
            {"LogisticRegression": {"C": hp.choice("selector.LogisticRegression.C", [0.1, 0.25, 0.5])}},
            {"LinearSVC": {"C": hp.choice("selector.LinearSVC.C", [0.1, 0.25, 0.5])}},
            {"XGBClassifier": {"max_depth": 10 + hp.quniform("selector.XGBClassifier.max_depth", 0, 40, 20)}},
            {"LGBMClassifier": {"num_leaves": 30 + hp.quniform("selector.LGBMClassifier.num_leaves", 0, 60, 30)}},
            {"RandomForestClassifier": {
                "min_samples_split": 12 + hp.quniform("selector.RandomForestClassifier.min_samples_split", 0, 10, 5)}},
            {"ExtraTreesClassifier": {
                "min_samples_split": 12 + hp.quniform("selector.ExtraTreesClassifier.min_samples_split", 0, 10, 5)}},
            "None"
        ]),
        "learner": hp.choice("learner", [
            {"LGBMClassifier": {
                "num_leaves": hp.quniform("learner.LGBMClassifier.num_leaves", 30, 150, 30),
                "colsample_bytree": hp.quniform("learner.LGBMClassifier.colsample_bytree", 0.8, 1, 0.1),
                "reg_lambda": hp.choice("learner.LGBMClassifier.reg_lambda", [1e-3, 1])
            }},
            {"XGBClassifier": {
                "max_depth": 10 + hp.quniform("learner.XGBClassifier.max_depth", 0, 80, 20),
                "colsample_bytree": hp.quniform("learner.XGBClassifier.colsample_bytree", 0.8, 1, 0.1),
                "reg_lambda": hp.choice("learner.XGBClassifier.reg_lambda", [1e-3, 1])
            }},
            {"LinearSVC": {
                "C": 0.01 + hp.quniform("learner.LinearSVC.C", 0, 1 - 0.01, 0.066),
                "penalty": hp.choice("learner.LinearSVC.penalty", ["l2", "l1"]),
            }},
            {"LogisticRegression": {
                "C": 0.01 + hp.quniform("learner.LogisticRegression.C", 0, 1 - 0.01, 0.066),
                "penalty": hp.choice("learner.LogisticRegression.penalty", ["l2", "l1"]),
            }},
            {"RandomForestClassifier": {
                "min_samples_split":
                    2 + hp.quniform("learner.RandomForestClassifier.min_samples_split", 0, 20, 5),
                "min_samples_leaf":
                    1 + hp.quniform("learner.RandomForestClassifier.min_samples_leaf", 0, 10, 5),
                "bootstrap": hp.choice("learner.RandomForestClassifier.bootstrap", [True, False])
            }},
            {"ExtraTreesClassifier": {
                "min_samples_split":
                    2 + hp.quniform("learner.ExtraTreesClassifier.min_samples_split", 0, 20, 5),
                "min_samples_leaf":
                    1 + hp.quniform("learner.ExtraTreesClassifier.min_samples_leaf", 0, 10, 5),
                "bootstrap": hp.choice("learner.ExtraTreesClassifier.bootstrap", [True, False])
            }},
            {"KNeighborsClassifier": {
                "n_neighbors":
                    1 + hp.quniform("learner.KNeighborsClassifier.n_neighbors", 2, 6, 2),
                "p": hp.quniform("learner.KNeighborsClassifier.p", 1, 2, 1),
            }},
        ]),
    }
    return space


def get_all_configs():
    HDL = get_HDL()
    CS = hdl2cs(HDL)
    # print(CS)
    grids = generate_grid(CS)
    # print(len(grids))
    configs = []
    index = 0
    for grid in grids:
        config = layering_config(grid.get_dictionary())
        configs.append(config)
        # index += 1
        # if index % 5000 == 0:
        #     print(index)
        #     print(config)
    print('len(configs):', len(configs))
    learners = [list(config['learner'].keys())[0] for config in configs]
    selector = [list(config['selector'].keys())[0] if config['selector'] is not None else "None"
                for config in configs]
    print("Counter(learners):", Counter(learners))
    print("Counter(selector):", Counter(selector))
    config_ids = [get_hash_of_str(str(config)) for config in configs]
    config_id_to_config = defaultdict(list)
    for config, config_id in zip(configs, config_ids):
        config_id_to_config[config_id].append(config)
    config_ids_cnt = Counter(config_ids)
    abnormal_config_ids = [config_id for config_id, cnt in config_ids_cnt.items() if cnt > 1]
    return configs, config_id_to_config
