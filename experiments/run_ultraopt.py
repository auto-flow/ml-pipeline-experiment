#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys
from pathlib import Path

import pandas as pd
from ultraopt import fmin
from ultraopt.hdl import hdl2cs
from ultraopt.optimizer import ETPEOptimizer

from experiments.evaluator import UltraoptEvaluator
from experiments.utils import raw2min
from pipeline_space.build_ml_pipeline_space import get_HDL

# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
HDL = get_HDL()

evaluator = UltraoptEvaluator(data, 'balanced_accuracy')
CS = hdl2cs(HDL)
repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
n_startup_trials = int(sys.argv[4])
print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))
for trial in range(repetitions):
    optimizer = ETPEOptimizer(
        min_points_in_model=n_startup_trials,
    )
    ret = fmin(
        evaluator, HDL, optimizer, random_state=trial * 10,
        n_iterations=max_iter,
    )
    losses = ret["budget2obvs"][1]["losses"]
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
Path(f'experiments/results/ultraopt-ETPE-{dataset_id}.json').write_text(
    json.dumps(final_result)
)
