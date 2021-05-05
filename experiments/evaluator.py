#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-05-05
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy

import numpy as np
import pandas as pd
from optuna import Trial
from ultraopt.hdl import layering_config

from pipeline_space.build_ml_pipeline_space import get_HDL


def HDL_define_by_run(trial: Trial, df: pd.DataFrame, sub_HDL: dict, name):
    choices = list(sub_HDL.keys())
    choice = trial.suggest_categorical(name, choices)
    df_ = df.loc[df[name] == choice, :]
    df = df_
    if choice == "None":
        return df
    HP = sub_HDL[choice]
    for hp_name, hp_define in HP.items():
        _type = hp_define["_type"]
        _value = hp_define["_value"]
        com_hp_name = f"{name}.{choice}.{hp_name}"
        if _type in ("ordinal", "choice"):
            v = trial.suggest_categorical(com_hp_name, _value)
        elif _type in ("int_quniform", "quniform"):
            v = trial.suggest_discrete_uniform(com_hp_name, *_value)
        else:
            raise NotImplementedError
        if isinstance(v, float):
            df_ = df.loc[np.abs(df[com_hp_name] - v) < 1e-8, :]
        else:
            df_ = df.loc[df[com_hp_name] == v, :]
        df = df_
    return df


class OptunaEvaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        self.global_min = 1 - df[metric].max()
        print(self.global_min)
        self.losses = []
        self.HDL = get_HDL()

    def __call__(self, trial: Trial):
        df = self.df
        scaler = trial.suggest_categorical("scaler", ["MinMaxScaler", "StandardScaler", "RobustScaler"])
        df = df.loc[df["scaler"] == scaler]
        for component in ['selector', 'learner']:
            sub_HDL = self.HDL[f"{component}(choice)"]
            df = HDL_define_by_run(trial, df, sub_HDL, component)
        assert df.shape[0] == 1
        loss = 1 - float(df[self.metric].values[0])
        self.losses.append(loss)
        return loss


class HyperoptEvaluator():
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


class UltraoptEvaluator():
    def __init__(self, df: pd.DataFrame, metric):
        self.metric = metric
        self.df = df
        # 打印全局最优解数值
        print('Global minimum: ', end="")
        self.global_min = 1 - df[metric].max()
        print(self.global_min)
        self.losses = []

    def __call__(self, config):
        layered_config = layering_config(config)
        layered_config_ = deepcopy(layered_config)
        df = self.df
        for component in ['scaler', 'selector', 'learner']:
            sub_config = layered_config_[component]
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
        loss = 1 - float(df[self.metric].values[0])
        self.losses.append(loss)
        return loss