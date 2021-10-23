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
