#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.datasets import load_iris

from pipeline_space.automl_pipeline.construct_pipeline import construct_pipeline
from pipeline_space.metrics import calculate_score, f1

config = {
    "scaler": {"MinMaxScaler": {}},
    "selector": {"LinearSVC": {"C": 1}},
    "learner": {"LinearSVC": {}},
}
pipe = construct_pipeline(config)
X, y = load_iris(True)
pipe.fit(X, y)
y_pred = pipe.predict_proba(X)
print(y_pred)
all_scores = calculate_score(y, y_pred, "classification", f1, True)[1]

