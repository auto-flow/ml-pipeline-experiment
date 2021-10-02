#!/usr/bin/env python20
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn

import json
import sys
from pathlib import Path

import optuna
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from optuna.samplers import TPESampler

from experiments.evaluator import OptunaEvaluator
from experiments.utils import raw2min

# 146594, 189863, 189864

dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')

repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
n_startup_trials = int(sys.argv[4])
mode = (sys.argv[5])
assert mode in ['multi', 'uni']
print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))


def evaluate(trial):
    gamma_ = lambda x: min(int(np.ceil(0.25 * np.sqrt(x))), 25)
    evaluator = OptunaEvaluator(data, 'balanced_accuracy')
    global_min = evaluator.global_min
    tpe = TPESampler(
        n_startup_trials=n_startup_trials, multivariate=True if mode == 'multi' else False,
        gamma=gamma_,
        seed=trial * 10)
    study = optuna.create_study(sampler=tpe, )
    study.optimize(evaluator, n_trials=max_iter)
    # res[f"trial-{trial}"] =
    return trial, evaluator.losses, global_min


for trial, losses, global_min in Parallel(
        backend="multiprocessing",n_jobs=-1)(
        delayed(evaluate)(trial) for trial in range(repetitions)
):
    res[f"trial-{trial}"] = losses
res = raw2min(res)
m = res.mean(1)
s = res.std(1)
final_result = {
    "global_min": global_min,
    "mean": m.tolist(),
    "std": s.tolist(),
    "q10": res.quantile(0.10, 1).tolist(),
    "q25": res.quantile(0.25, 1).tolist(),
    "q75": res.quantile(0.75, 1).tolist(),
    "q90": res.quantile(0.90, 1).tolist()
}
Path(f'experiments/results/optuna-TPE-{mode}-{dataset_id}.json').write_text(
    json.dumps(final_result)
)
