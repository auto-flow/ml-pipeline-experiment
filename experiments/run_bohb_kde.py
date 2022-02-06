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
from experiments.utils import replace_ordinal_to_cat
from hpbandster.core.worker import Worker
from hpbandster.optimizers.config_generators.bohb import BOHB
from joblib import Parallel
from sklearn.utils import delayed
from tqdm import tqdm
from ultraopt.hdl import hdl2cs
# 146594, 189863, 189864
from ultraopt.structure import Job

dataset_id = sys.argv[1]
benchmark_type = sys.argv[2]
print(dataset_id)
print(benchmark_type)

repetitions = int(sys.argv[3])
max_iter = int(sys.argv[4])
n_startup_trials = int(sys.argv[5])
print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
print(dataset_id)

if benchmark_type == "pipeline":
    data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
    evaluator = PipelineUltraoptEvaluator(data, 'balanced_accuracy')
elif benchmark_type == "bagging":
    data = pd.read_csv(f'processed_data/bagging_d{dataset_id}_processed.csv')
    evaluator = BaggingUltraoptEvaluator(data, 'f1')
else:
    raise NotImplementedError


class MyWorker(Worker):
    evaluator = None

    def compute(self, config, budget, **kwargs):
        return {"loss": self.evaluator(config)}


df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))

HDL = evaluator.sampler.get_HDL()

HDL = replace_ordinal_to_cat(HDL)
CS = hdl2cs(HDL)
bohb = BOHB(configspace=CS, min_points_in_model=n_startup_trials,
            random_fraction=0, bandwidth_factor=1, num_samples=24)


def evaluate(trial):
    CS.seed(trial)
    losses = []
    for i in tqdm(range(max_iter)):
        config = bohb.get_config(1)[0]
        loss = evaluator(config)
        job = Job(id=i, budget=1, config=config)
        job.result = {'loss': loss}
        bohb.new_result(job)
        losses.append(loss)
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
fname=f'experiments/results/BOHB-KDE-{dataset_id}-{benchmark_type}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
df.to_csv(f'{fname}.csv', index=False)
