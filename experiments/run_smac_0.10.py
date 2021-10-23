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
from experiments.evaluator import UltraoptEvaluator
from experiments.utils import raw2min
from joblib import Parallel, delayed
from pipeline_space.build_ml_pipeline_space import get_HDL
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from ultraopt.hdl import hdl2cs

# 146594, 189863, 189864

dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
HDL = get_HDL()

evaluator = UltraoptEvaluator(data, 'balanced_accuracy')
cs = hdl2cs(HDL)
repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
n_startup_trials = int(sys.argv[4])
print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")
res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))

scenario = Scenario({
    "run_obj": "quality",
    "runcount-limit": max_iter,
    "cs": cs,
    "deterministic": "false",
    "initial_incumbent": "RANDOM",
    "output_dir": ""
})

tae = ExecuteTAFuncDict(evaluator, use_pynisher=False)

random_fraction = 0.33
n_trees = 10
max_feval = 4


def evaluate(trial):
    smac = SMAC(scenario=scenario, tae_runner=tae,
                rng=np.random.RandomState(trial))
    # probability for random configurations

    smac.solver.random_configuration_chooser.prob = random_fraction
    smac.solver.model.rf_opts.num_trees = n_trees
    # only 1 configuration per SMBO iteration
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    # maximum number of function evaluations per configuration
    smac.solver.intensifier.maxR = max_feval

    smac.optimize()

    runhistory = smac.runhistory
    configs = runhistory.get_all_configs()
    losses = [runhistory.get_cost(config) for config in configs]

    print('finish', trial)
    return trial, losses


global_min = evaluator.global_min

for trial, losses in Parallel(
        backend="multiprocessing", n_jobs=5)(
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
Path(f'experiments/results/SMAC-{dataset_id}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
