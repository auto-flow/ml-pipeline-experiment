#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
'''
smac==0.10.0
'''
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.evaluator import BaggingUltraoptEvaluator
from experiments.evaluator import PipelineUltraoptEvaluator
from experiments.utils import raw2min
from joblib import Parallel, delayed
from pipeline_space.pipeline_sampler import SmallPipelineSampler
from pipeline_space.pipeline_sampler import BaggingPipelineSampler
from pipeline_space.build_ml_pipeline_space import get_HDL
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from ultraopt.hdl import hdl2cs

# 146594 10 100 30
# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
benchmark_type = sys.argv[2]
print(benchmark_type)
repetitions = int(sys.argv[3])
max_iter = int(sys.argv[4])
n_startup_trials = int(sys.argv[5])
dataset_id = sys.argv[1]

data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
HDL = get_HDL()

if benchmark_type == "pipeline":
    HDL = SmallPipelineSampler().get_HDL()
    data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
    evaluator = PipelineUltraoptEvaluator(data, 'balanced_accuracy')
elif benchmark_type == "bagging":
    HDL = BaggingPipelineSampler().get_HDL()
    data = pd.read_csv(f'processed_data/bagging_d{dataset_id}_processed.csv')
    evaluator = BaggingUltraoptEvaluator(data, 'f1')
else:
    raise NotImplementedError


print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                  index=range(max_iter))


random_fraction = 0.33
n_trees = 10
max_feval = 4
global_min = evaluator.global_min


def evaluate(trial):
    cs = hdl2cs(HDL)
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": max_iter,
        "cs": cs,
        "deterministic": "false",
        "initial_incumbent": "RANDOM",
        "output_dir": "",
    })
    tae = ExecuteTAFuncDict(evaluator, use_pynisher=False)
    initial_configurations = cs.sample_configuration(n_startup_trials)
    # cs.seed(trial)
    smac = SMAC(
        scenario=scenario, tae_runner=tae,
        rng=np.random.RandomState(trial),
        initial_configurations=initial_configurations
    )
    # probability for random configurations

    smac.solver.random_configuration_chooser.prob = random_fraction
    smac.solver.model.rf_opts.num_trees = n_trees
    # only 1 configuration per SMBO iteration
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    # maximum number of function evaluations per configuration
    smac.solver.intensifier.maxR = max_feval

    smac.optimize()

    # runhistory = smac.runhistory
    # configs = runhistory.get_all_configs()
    # losses = [runhistory.get_cost(config) for config in configs]

    print('finish', trial)
    return trial, evaluator.losses[:max_iter]



for trial, losses in Parallel(
        backend="multiprocessing", n_jobs=10)(
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
fname = f'experiments/results/SMAC-{dataset_id}-{benchmark_type}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
df.to_csv(f'{fname}.csv', index=False)
