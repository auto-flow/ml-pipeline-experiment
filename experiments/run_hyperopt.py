#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import tpe, fmin, Trials

from pipeline_space.build_ml_pipeline_space import get_hyperopt_space

# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
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
repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
setup_runs = int(sys.argv[4])
print(f"repetitions={repetitions}, max_iter={max_iter}, setup_runs={setup_runs}")
res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))
for trial in range(repetitions):
    trials = Trials()
    best = fmin(
        evaluator, space, algo=partial(tpe.suggest, n_startup_jobs=setup_runs),
        max_evals=max_iter,
        rstate=np.random.RandomState(trial * 10), trials=trials,

    )
    losses = trials.losses()
    res[f"trial-{trial}"] = losses
res = raw2min(res)
m = res.mean(1)
s = res.std(1)
final_result = {
    "global_min": evaluator.global_min,
    "mean": m.tolist(),
    "std": s.tolist(),
    "q10": res.quantile(0.10, 1).tolist(),
    "q25": res.quantile(0.25, 1).tolist(),
    "q75": res.quantile(0.75, 1).tolist(),
    "q90": res.quantile(0.90, 1).tolist()
}
Path(f'experiments/results/hyperopt-TPE-{dataset_id}.json').write_text(
    json.dumps(final_result)
)
