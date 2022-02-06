#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from pathlib import Path

import numpy as np
import pylab as plt

plt.rcParams['figure.figsize'] = (10, 8)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "font.serif": ["Palatino"],
})
plt.rc('legend', fontsize=16)

info = {
    "Random": ("Random", ["r", '.', 'solid']),
    "hyperopt-TPE": ("HyperOpt-TPE", ["purple", 'v', 'solid']),
    "optuna-TPE": ("Optuna-TPE", ["g", '^', 'solid']),
    "BOHB-KDE": ("BOHB-KDE", ["olive", 's', 'solid']),
    "ultraopt-ETPE": ("ETPE", ["b", 's', 'dashed']),
    "ultraopt-ETPE-univar": ("ETPE(univar)", ["brown", "x", 'dashed']),
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
    [146594, 'pipeline'],
    [146594, 'bagging'],
    [189863, 'pipeline'],
    [189863, 'bagging'],
    [189864, 'pipeline'],
    [189864, 'bagging'],
]
# 设置字体样式

ylims = [
    [np.exp(-6.3), 0.009384615384615408],
    [np.exp(-5.5), 0.019285714285714295],
    [np.exp(-7), 0.00861215647026714],
    [np.exp(-6.3), 0.01],
    [np.exp(-6.8), 0.01],
    [np.exp(-6.6), 0.01],
]

plt.close()
dir_name = 'pipeline_benchmark_figures'
os.system(f'mkdir -p {dir_name}')
index = 1
iteration_truncate = 300

metainfo = [
    # NIPS 2003 feature selection challenge
    [146594, 'madelon', 'https://www.openml.org/t/146594', 'https://www.openml.org/d/1485'],
    [189863, 'madeline', 'https://www.openml.org/t/189863', 'https://www.openml.org/d/41144'],
    [189864, 'philippine', 'https://www.openml.org/t/189864', 'https://www.openml.org/d/41145'],
]
openml_taskid_to_datasetName = {
    l[0]: l[1]
    for l in metainfo
}

for i, (dataset_id, benchmark_type) in enumerate(benchmarks):
    for fname, (name, color) in info.items():
        if "_g" in fname:
            fname, suffix = fname.split("_")
            path = f"experiments/results/{fname}-{dataset_id}-{benchmark_type}_{suffix}.json"
        else:
            path = f"experiments/results/{fname}-{dataset_id}-{benchmark_type}.json"
        mean_std = json.loads(Path(path).read_text())
        global_min = mean_std['global_min']
        mean = np.array(mean_std["mean"])[:iteration_truncate]
        q1 = np.array(mean_std["q25"])[:iteration_truncate]
        q2 = np.array(mean_std["q75"])[:iteration_truncate]
        mean -= global_min
        q1 -= global_min
        q2 -= global_min
        y1, y2 = ylims[i]
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        iters = range(len(mean))
        plt.ylim(y1, y2)
        plt.yscale("log")

        # if not log_scale:
        plt.fill_between(
            iters, q1, q2, alpha=0.1,
            color=color[0]
        )
        plt.plot(
            iters, mean, color=color[0], label=name, alpha=0.9,
            marker=color[1], linestyle=color[2], markevery=30
        )
    index += 1
    plt.xlabel("iterations")
    plt.ylabel("immediate regret")
    # plt.grid(alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    openml_dataset_name = openml_taskid_to_datasetName[dataset_id]
    plt.savefig(f"{dir_name}/{benchmark_type}_{dataset_id}_{openml_dataset_name}.png")
    plt.savefig(f"{dir_name}/{benchmark_type}_{dataset_id}_{openml_dataset_name}.pdf")
    plt.show()

target_dir = f'/data/Project/AutoML/ultraopt/paper_figures/{dir_name}'

os.system(f'rm -rf {target_dir}')
os.system(f'cp -r {dir_name} {target_dir}')
