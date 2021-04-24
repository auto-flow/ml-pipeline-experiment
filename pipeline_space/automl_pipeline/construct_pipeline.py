#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-21
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from pipeline_space.automl_pipeline.learner import Learner
from pipeline_space.automl_pipeline.selector import Selector

scalers = [RobustScaler, MinMaxScaler, StandardScaler]


def construct_scaler(config):
    AS, HP = config.popitem()
    return ('scaler', eval(AS)())


def construct_selector(config, n_jobs=None):
    if config is None:
        return ('selector', 'passthrough')
    AS, HP = config.popitem()
    selector = Selector(AS, **HP, n_jobs=n_jobs)
    return ('selector', selector)


def construct_learner(config, n_jobs=None):
    AS, HP = config.popitem()
    learner = Learner(AS, HP, n_jobs=n_jobs)
    return ('learner', learner)


def construct_pipeline(config, memory="/tmp", verbose=True, n_jobs=None):
    config = deepcopy(config)
    steps = [
        construct_scaler(config['scaler']),
        construct_selector(config['selector'], n_jobs=n_jobs),
        construct_learner(config['learner'], n_jobs=n_jobs),
    ]
    return Pipeline(steps, memory=memory, verbose=verbose)
