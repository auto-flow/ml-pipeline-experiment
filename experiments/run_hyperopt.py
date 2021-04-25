#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import tpe, fmin, Trials
import sys
from pipeline_space.build_ml_pipeline_space import get_hyperopt_space

# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv('processed_data/d146594_processed.csv')
# data = pd.read_csv('processed_data/d189863_processed.csv')
# data = pd.read_csv('processed_data/d189864_processed.csv')
space = get_hyperopt_space()

def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m

class Evaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        self.global_min = 1 - df[metric].max()
        print(self.global_min)

    def __call__(self, config):
        layered_config_ = deepcopy(config)
        df = self.df
        for component in ['scaler', 'selector', 'learner']:
            sub_config = layered_config_[component]
            if isinstance(sub_config, str):
                df_ = df.loc[df[component] == sub_config]
                df = df_
                continue
            AS, HP = sub_config.popitem()
            df_ = df.loc[df[component] == AS, :]
            df = df_
            for k, v in HP.items():
                name = f"{component}.{AS}.{k}"
                # 对于浮点数考虑精度误差
                if isinstance(v, (float)):
                    df_ = df.loc[np.abs(df[name] - v) < 1e-8, :]
                else:
                    df_ = df.loc[df[name] == v, :]
                df = df_
        assert df.shape[0] == 1
        return 1 - float(df[self.metric].values[0])


evaluator = Evaluator(data, 'balanced_accuracy')

trials = Trials()
best = fmin(
    evaluator, space, algo=partial(tpe.suggest, n_startup_jobs=40), max_evals=200,
    rstate=np.random.RandomState(50), trials=trials,
)
