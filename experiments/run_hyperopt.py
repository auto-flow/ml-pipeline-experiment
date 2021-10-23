#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.evaluator import PipelineHyperoptEvaluator
from experiments.evaluator import BaggingHyperoptEvaluator
from experiments.utils import raw2min
from hyperopt import tpe, fmin, Trials
from joblib import Parallel, delayed
from pipeline_space.pipeline_sampler import BaggingPipelineSampler
from pipeline_space.pipeline_sampler import SmallPipelineSampler

# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
benchmark_type = sys.argv[2]
print(benchmark_type)
repetitions = int(sys.argv[3])
max_iter = int(sys.argv[4])
n_startup_trials = int(sys.argv[5])

if benchmark_type == "pipeline":
    space=SmallPipelineSampler().get_hyperopt_space()
    data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
    evaluator = PipelineHyperoptEvaluator(data, 'balanced_accuracy')
elif benchmark_type == "bagging":
    space=BaggingPipelineSampler().get_hyperopt_space()
    data = pd.read_csv(f'processed_data/bagging_d{dataset_id}_processed.csv')
    evaluator = BaggingHyperoptEvaluator(data, 'f1')
else:
    raise NotImplementedError



print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))


def evaluate(trial):
    trials = Trials()
    best = fmin(
        evaluator, space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_trials),
        max_evals=max_iter,
        rstate=np.random.RandomState(trial * 10), trials=trials,
    )
    losses = trials.losses()
    df[f"trial-{trial}"] = losses
    return trial, losses


for trial, losses in Parallel(
        backend="multiprocessing", n_jobs=-1)(
    delayed(evaluate)(trial) for trial in range(repetitions)
):
    df[f"trial-{trial}"] = losses


res = raw2min(df)
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
fname=f'experiments/results/hyperopt-TPE-{dataset_id}-{benchmark_type}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
df.to_csv(f'{fname}.csv', index=False)
