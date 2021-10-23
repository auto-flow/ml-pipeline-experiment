#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-02
# @Contact    : qichun.tang@bupt.edu.cn
from pipeline_space.pipeline_sampler.base import BasePipelineSampler
from pipeline_space.utils import generate_grid, get_hash_of_dict
from pipeline_space.hdl import hdl2cs


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
                "C": {"_type": "ordinal", "_value": [0.1, 1, 5]},
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

    def calc_every_learner_grids(self):
        learner2grids = {}
        for learner, HP in self.all_learner().items():
            CS = hdl2cs({learner: HP})
            grids = generate_grid(CS)
            print(f"{learner}:", len(grids))
            learner2grids[learner] = grids
        return learner2grids

    def calc_bagging_learner_counts(self):
        grids = generate_grid(hdl2cs(self.get_HDL()))
        print(f"stacked:", len(grids))
        return grids

    def get_HDL(self):
        HDL = {}
        for i, (learner, HP) in enumerate(self.all_learner().items()):
            HDL[f"learner{i}(choice)"] = {learner: HP, "None": {}}
        return HDL

    def get_config_id(self, config: dict):
        for module, AS_HP in config.items():
            if AS_HP == {'None': {}}:
                # 和预处理程序对齐
                config[module] = None
        return get_hash_of_dict(config)


def test1():
    import hyperopt.pyll.stochastic
    from pipeline_space.hdl import hdl2cs, layering_config

    sampler = BaggingPipelineSampler()
    print(hyperopt.pyll.stochastic.sample(
        sampler.get_hyperopt_space()))
    HDL = sampler.get_HDL()
    CS = hdl2cs(HDL)
    print(layering_config(CS.sample_configuration()))


if __name__ == '__main__':
    import pandas as pd
    import hyperopt.pyll.stochastic

    path='processed_data/bagging_d146594_processed.csv'
    df=pd.read_csv(path)
    df.set_index("config_id",inplace=True)
    sampler = BaggingPipelineSampler()
    space=sampler.get_hyperopt_space()
    for _ in range(10000):
        config=hyperopt.pyll.stochastic.sample(space)
        config_id=sampler.get_config_id(config)
        row=df.loc[config_id]
        print(row)




