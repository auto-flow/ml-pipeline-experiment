#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn
from time import time

import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler

from pipeline_space.automl_pipeline.selector import Selector
from pipeline_space.hdl import hdl2cs, layering_config
from pipeline_space.utils import generate_grid

X, y, cat = load("/data/Project/AutoML/ML-Pipeline-Experiment/189864.bz2")
y = y.astype(int)
X = MinMaxScaler().fit_transform(X)
HDL = {
    "selector(choice)": {
        "RandomForestClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 5]},
        },
        "ExtraTreesClassifier": {
            "min_samples_split": {"_type": "int_quniform", "_value": [12, 22, 5]},
        },
        "XGBClassifier": {
            "max_depth": {"_type": "int_quniform", "_value": [10, 70, 20]},
        },
        "LGBMClassifier": {
            "num_leaves": {"_type": "int_quniform", "_value": [30, 90, 30]},
        },
        "LogisticRegression": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5]}
        },
        "LinearSVC": {
            "C": {"_type": "ordinal", "_value": [0.1, 0.25, 0.5]}
        },
        # "None": {}
    },
}
CS = hdl2cs(HDL)
for config in generate_grid(CS):
    config = config.get_dictionary()
    layered_config = layering_config(config)
    print(layered_config)
    AS, HP = layered_config['selector'].popitem()
    selector = Selector(AS, HP)
    start_time = time()
    selector.fit(X, y)
    cost_time = time() - start_time
    print(f"select {(np.count_nonzero(selector.select_) / X.shape[1]) * 100:.2f}% features")
    print(f"cost {cost_time:.3f}s")
    print()
