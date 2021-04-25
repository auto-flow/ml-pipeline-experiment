#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
from pathlib import Path

import numpy as np
import pylab as plt

info = {
    "hyperopt-TPE": ("hyperopt-TPE", "r",),
    "ultraopt-ETPE": ("ultraopt-ETPE", "g",),
    "hpbandster-KDE": ("hpbandster-KDE", "k",),
    "Random": ("Random", "b",),
}
# 146594, 189863, 189864

dataset_id = 189864
# 设置字体样式
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (10, 8)

plt.close()
index = 1
truncate = 100
for fname, (name, color,) in info.items():
    mean_std = json.loads(Path(f"experiments/results/{fname}-{dataset_id}.json").read_text())
    mean = np.array(mean_std["mean"])[:truncate]
    q1 = np.array(mean_std["q25"])[:truncate]
    q2 = np.array(mean_std["q75"])[:truncate]
    iters = range(len(mean))
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
plt.yscale("log")
# plt.xscale("symlog")
title = "Comparison between Optimizers"
plt.title(f"{dataset_id}")
plt.tight_layout()
for suffix in ["pdf", "png"]:
    plt.savefig(f"{dataset_id}.{suffix}")
plt.show()
