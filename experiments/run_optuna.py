#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import optuna
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from ultraopt import fmin
from ultraopt.hdl import hdl2cs
from ultraopt.hdl import layering_config
from ultraopt.optimizer import ETPEOptimizer

from pipeline_space.build_ml_pipeline_space import get_HDL

class Evaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        self.global_min = 1 - df[metric].max()
        print(self.global_min)

    def __call__(self, trial):
        df = self.df
        for component in ['scaler', 'selector', 'learner']:
            sub_config = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
            if sub_config is None:
                df_ = df.loc[df[component] == "None"]
                df = df_
                continue
            AS, HP = sub_config.popitem()
            df_ = df.loc[df[component] == AS, :]
            df = df_
            for k, v in HP.items():
                name = f"{component}.{AS}.{k}"
                # 对于浮点数考虑精度误差
                if isinstance(v, float):
                    df_ = df.loc[np.abs(df[name] - v) < 1e-8, :]
                else:
                    df_ = df.loc[df[name] == v, :]
                df = df_
        assert df.shape[0] == 1
        return 1 - float(df[self.metric].values[0])
