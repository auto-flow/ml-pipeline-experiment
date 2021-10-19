#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-04
# @Contact    : qichun.tang@bupt.edu.cn

import json
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from pipeline_space.hdl import layering_config
from pipeline_space.pipeline_sampler import BaggingPipelineSampler
from pipeline_space.utils import get_hash_of_dict

root = '/data/Project/AutoML/ML-Pipeline-Experiment'


class HyperoptEvaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        df.set_index("config_id", inplace=True)
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        df[metric] = df["metrics"].apply(lambda x: json.loads(x)["f1"])
        self.global_min = 1 - df[metric].max()
        print(self.global_min)
        self.sampler = BaggingPipelineSampler()
        self.space = self.sampler.get_hyperopt_space()

    def __call__(self, config):
        config_id = self.sampler.get_config_id(config)
        return 1 - float(self.df.loc[config_id, self.metric])


class UltraoptEvaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        df.set_index("config_id", inplace=True)
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        df[metric] = df["metrics"].apply(lambda x: json.loads(x)["f1"])
        self.global_min = 1 - df[metric].max()
        print(self.global_min)
        self.sampler = BaggingPipelineSampler()

        self.losses = []

    def __call__(self, config):
        layered_config = layering_config(config)
        layered_config_ = deepcopy(layered_config)
        # 和预处理程序对齐
        for module, AS_HP in layered_config_.items():
            if AS_HP is None:
                layered_config_[module] = {}
        config_id = get_hash_of_dict(layered_config_)
        loss = 1 - float(self.df.loc[config_id, self.metric])
        self.losses.append(loss)
        return loss


def test_hyperopt():
    from hyperopt import tpe, fmin, Trials
    print('test_hyperopt')
    path = root + '/processed_data/bagging_d146594_processed.csv'
    df = pd.read_csv(path)
    evaluator = HyperoptEvaluator(df, "f1")
    losses = []
    for i in range(20):
        trials = Trials()
        best = fmin(
            evaluator, evaluator.space, algo=partial(tpe.suggest, n_startup_jobs=50),
            max_evals=200,
            rstate=np.random.RandomState(i * 10), trials=trials,

        )
        losses.append(np.min(trials.losses()))
    print(np.mean(losses))


if __name__ == '__main__':
    # test_hyperopt()
    # exit(0)
    from ultraopt import fmin
    from ultraopt.optimizer import ETPEOptimizer

    path = root + '/processed_data/bagging_d146594_processed.csv'
    df = pd.read_csv(path)
    losses = []
    evaluator = UltraoptEvaluator(df, "f1")
    for i in range(5):
        evaluator.losses = []
        HDL = evaluator.sampler.get_HDL()
        optimizer = ETPEOptimizer(min_points_in_model=50)
        ret = fmin(
            evaluator, HDL, optimizer, random_state=i * 10,
            n_iterations=200,
        )
        # losses.append(evaluator.losses[-1])
        losses.append(np.min(evaluator.losses))
    print(np.mean(losses))
