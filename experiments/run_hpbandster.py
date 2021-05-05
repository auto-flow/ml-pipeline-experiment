#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys
import time
from pathlib import Path

import hpbandster.core.nameserver as hpns
import numpy as np
import pandas as pd
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB
from ultraopt.hdl import hdl2cs

from experiments.evaluator import UltraoptEvaluator
from experiments.utils import raw2min
from pipeline_space.build_ml_pipeline_space import get_no_ordinal_HDL

# 146594, 189863, 189864
dataset_id = sys.argv[1]
print(dataset_id)
data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')


class MyWorker(Worker):
    evaluator = None

    def compute(self, config, budget, **kwargs):
        return {"loss": self.evaluator(config)}





hb_run_id = f'{dataset_id}'

NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

num_workers = 1

repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
n_startup_trials = int(sys.argv[4])

res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                   index=range(max_iter))
evaluator = UltraoptEvaluator(data, 'balanced_accuracy')
for trial in range(repetitions):
    worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                      run_id=hb_run_id,
                      id=0)
    evaluator = UltraoptEvaluator(data, 'balanced_accuracy')
    worker.evaluator = evaluator
    worker.run(background=True)
    HDL = get_no_ordinal_HDL()
    CS = hdl2cs(HDL)
    CS.seed(trial * 10 + 5)
    bohb = BOHB(configspace=CS,
                run_id=hb_run_id,
                # just test KDE
                eta=2, min_budget=1, max_budget=1,
                nameserver=ns_host,
                nameserver_port=ns_port,
                num_samples=64,
                random_fraction=33,
                bandwidth_factor=3,
                ping_interval=10, min_bandwidth=.3)

    results = bohb.run(max_iter, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    res[f"trial-{trial}"] = evaluator.losses
NS.shutdown()
time.sleep(1)
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
Path(f'experiments/results/hpbandster-KDE-{dataset_id}.json').write_text(
    json.dumps(final_result)
)
