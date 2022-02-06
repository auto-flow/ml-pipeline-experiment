#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys
from pathlib import Path

import pandas as pd
from experiments.evaluator import PipelineUltraoptEvaluator, BaggingUltraoptEvaluator
from experiments.utils import raw2min
from joblib import Parallel, delayed
from ultraopt import fmin

dataset_id = sys.argv[1]
benchmark_type = sys.argv[2]
print(dataset_id)
print(benchmark_type)

repetitions = int(sys.argv[3])
max_iter = int(sys.argv[4])
n_startup_trials = int(sys.argv[5])
print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                  index=range(max_iter))

if benchmark_type == "pipeline":
    data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
    evaluator = PipelineUltraoptEvaluator(data, 'balanced_accuracy')
elif benchmark_type == "bagging":
    data = pd.read_csv(f'processed_data/bagging_d{dataset_id}_processed.csv')
    evaluator = BaggingUltraoptEvaluator(data, 'f1')
else:
    raise NotImplementedError
HDL = evaluator.sampler.get_HDL()
global_min = evaluator.global_min


def evaluate(trial):
    ret = fmin(
        evaluator, HDL, "Random", random_state=trial * 10,
        n_iterations=max_iter,
    )
    losses = ret["budget2obvs"][1]["losses"]
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
    "global_min": global_min,
    "mean": m.tolist(),
    "std": s.tolist(),
    "q10": res.quantile(0.10, 1).tolist(),
    "q25": res.quantile(0.25, 1).tolist(),
    "q75": res.quantile(0.75, 1).tolist(),
    "q90": res.quantile(0.90, 1).tolist()
}
fname = f'experiments/results/Random-{dataset_id}-{benchmark_type}'

Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
df.to_csv(f'{fname}.csv', index=False)
