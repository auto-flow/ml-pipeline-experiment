#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
from pathlib import Path

import numpy as np
import pylab as plt

info = {
    "hyperopt-TPE": ("hyperopt-TPE", "purple",),
    "ultraopt-ETPE": ("ultraopt-ETPE", "r",),
    # "ultraopt-ETPE_g3": ("ultraopt-ETPE_g3", "r",),
    # "ultraopt-ETPE_g4": ("ultraopt-ETPE_g4", "g",),
    # "ultraopt-ETPE_g5": ("ultraopt-ETPE_g5", "brown",),
    # "ultraopt-ETPE_g6": ("ultraopt-ETPE_g6", "b",),
    # "ultraopt-ETPE_g8": ("ultraopt-ETPE_g8", "k",),
}
# 146594, 189863, 189864

dataset_ids = [
    146594,
    189863,
    189864,
]
benchmark_types = [
    'pipeline',
    'bagging',
]
benchmarks = [
    [146594, 'pipeline', (0.145, 0.154)],
    [146594, 'bagging', (0.1623, 0.18)],
    [189863, 'pipeline', (1.26e-1, 1.35e-1)],
    [189863, 'bagging', (0.137, 0.15)],
    [189864, 'pipeline', (0.2075, 0.22)],
    [189864, 'bagging', (0.227, 0.24)],
]
# 设置字体样式
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (12, 15)

plt.close()
index = 1
iteration_truncate = 500

for dataset_id, benchmark_type, ylim in benchmarks:
    plt.subplot(3, 2, index)
    for fname, (name, color) in info.items():
        if "_g" in fname:
            fname, suffix = fname.split("_")
            path = f"experiments/results/{fname}-{dataset_id}-{benchmark_type}_{suffix}.json"
        else:
            path = f"experiments/results/{fname}-{dataset_id}-{benchmark_type}.json"
        mean_std = json.loads(Path(path).read_text())
        mean = np.array(mean_std["mean"])[:iteration_truncate]
        q1 = np.array(mean_std["q25"])[:iteration_truncate]
        q2 = np.array(mean_std["q75"])[:iteration_truncate]
        iters = range(len(mean))
        plt.ylim(*ylim)
        plt.title(f"{dataset_id}-{benchmark_type}")
        # if not log_scale:
        plt.fill_between(
            iters, q1, q2, alpha=0.1,
            color=color
        )
        plt.plot(
            iters, mean, color=color, label=name, alpha=0.9
        )
    index += 1
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid(alpha=0.4)
    plt.legend(loc="best")
# plt.yscale("log")
# plt.xscale("symlog")
task = "pipeline_space"
plt.suptitle(f"{task}")
plt.tight_layout()
for suffix in ["pdf", "png"]:
    plt.savefig(f"{task}.{suffix}")
plt.show()
