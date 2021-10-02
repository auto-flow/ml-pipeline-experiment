#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-03
# @Contact    : qichun.tang@bupt.edu.cn
import os
import socket
from collections import defaultdict
from pprint import pprint
from random import shuffle, seed
from time import time

import numpy as np
import peewee as pw
from joblib import load, delayed, Parallel
from playhouse.postgres_ext import JSONField
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pipeline_space.automl_pipeline.construct_pipeline import construct_pipeline
# from pipeline_space.build_ml_pipeline_space import get_all_configs
from pipeline_space.metrics import calculate_score, f1
from pipeline_space.pipeline_sampler import SmallPipelineSampler, BigPipelineSampler
from pipeline_space.utils import get_chunks, get_hash_of_str

config = {
    'learner': {
        'LGBMClassifier':
            {'colsample_bytree': 0.8, 'lambda_l1': 1e-05, 'lambda_l2': 0.1, 'num_leaves': 30}},
    'scaler': {'MinMaxScaler': {}}, 'selector': {'LogisticRegression': {'C': 0.01}}}

X, y, cat = load("/media/tqc/doc/Project/metalearn_experiment/data/146594.bz2")
X = X.values
X = X[:, ~np.array(cat)]
y = LabelEncoder().fit_transform(y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(next(cv.split(X, y))[0])


pipeline = construct_pipeline(config, verbose=True, n_jobs=None)
pipeline.fit(X, y)
