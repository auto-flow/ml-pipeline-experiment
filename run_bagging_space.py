#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-03
# @Contact    : qichun.tang@bupt.edu.cn
import os
import socket
import warnings
from time import time

import numpy as np
import pandas as pd
from joblib import dump, load
from lightgbm import LGBMClassifier
from pipeline_space.automl_pipeline.learner import Learner
from pipeline_space.hdl import hdl2cs
from pipeline_space.hdl import layering_config
from pipeline_space.pipeline_sampler.bagging_space import BaggingPipelineSampler
from pipeline_space.utils import get_hash_of_dict, generate_grid_yield, dict_to_csv_str
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

hostname = socket.gethostname()

warnings.filterwarnings('ignore')
savedpath = os.getenv('SAVEDPATH', '.')
import_models = [
    XGBClassifier, LGBMClassifier, RandomForestClassifier, ExtraTreesClassifier,
    LogisticRegression, LinearSVC, KNeighborsClassifier
]
datapath = os.getenv(
    'DATAPATH',
    '/media/tqc/doc/Project/metalearn_experiment/data/146594.bz2'
)
SPLIT_NUM = float(os.getenv('SPLIT_NUM', '0.33'))
if SPLIT_NUM > 1:
    splitter = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
else:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_NUM, random_state=0)
X, y, cat = load(datapath)
X = X.values
X = X[:, ~np.array(cat)]
y = LabelEncoder().fit_transform(y)
n_tests = None
y_true = None

learner2grids = BaggingPipelineSampler().calc_every_learner_grids()
fname = f"{savedpath}/learner2data.pkl"

if os.path.exists(fname):
    learner2data = load(fname)
else:
    learner2data = {}
    for learner, grids in learner2grids.items():
        if "tqc" in hostname and learner not in ['LGBMClassifier', 'XGBClassifier']:
            continue
        for cs in grids:
            config = layering_config(cs)
            config_id = get_hash_of_dict(config, sort=False)
            AS, HP = config.popitem()
            learner_obj = Learner(AS, HP)
            y_pred_list = []
            y_true_list = []
            for train_ix, test_ix in splitter.split(X, y):
                y_test = y[test_ix]
                learner_obj.fit(X[train_ix, :], y[train_ix])
                test_pred = learner_obj.predict_proba(X[test_ix])
                y_pred_list.append(test_pred)
                y_true_list.append(y_test[:, np.newaxis])
            y_pred = np.vstack(y_pred_list)
            if y_true is None:
                y_true = np.vstack(y_true_list).flatten()
                n_tests = y_true.shape[0]
            learner2data[config_id] = y_pred
        print(learner, ', done.')
    dump(learner2data, fname)
n_trials = 10000
bagging_space_sampler = BaggingPipelineSampler()
CS = hdl2cs(bagging_space_sampler.get_HDL())
index = 0
columns = ['config_id', 'config', 'metrics']
data = []
total = 7 ** 7
start_time = time()
out_fname = f'{savedpath}/bagging_space.csv'
exist_file = os.path.exists(out_fname)
solved_config_id_set = set()
if exist_file:
    # 从文件中把已经解决的config_id读出来
    solved_config_id_set = set(pd.read_csv(out_fname)['config_id'].to_list())
if not exist_file:
    with open(out_fname, 'a') as f:
        f.write(",".join(columns) + "\n")

is_test = False
# todo: 可以改成多进程，但是写的时候要加锁
for config in (generate_grid_yield(CS)):  # , total=total
    index += 1
    if is_test and index > n_trials:
        break
    if index % 1000 == 0:
        cost_time = time() - start_time
        p = index / total
        rest_time = cost_time * ((1 - p) / p)
        print(f"index = {index}, rest_time = {rest_time:.2f}, cost_time = {cost_time:.2f}")
    # main logic
    config = layering_config(config)
    line_id = get_hash_of_dict(config)
    if line_id in solved_config_id_set:
        total -= 1
        continue
    y_pred = np.zeros([n_tests, 2])
    n_learners = 0
    for i in range(7):
        key = f"learner{i}"
        sub_dict = config[key]
        if sub_dict == {} or sub_dict is None:
            continue
        n_learners += 1
        config_id = get_hash_of_dict(sub_dict)
        y_pred += learner2data[config_id]
    y_pred /= max(n_learners, 1)
    y_prob = y_pred[:, 1]
    y_pred = np.argmax(y_pred, axis=1)
    metric = {
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        # "precision": precision_score(y_test, y_pred),
        # "recall": recall_score(y_test, y_pred),
    }
    if index % 1000 == 0:
        print(metric)
    with open(out_fname, 'a') as f:
        f.write(",".join([
            line_id,
            dict_to_csv_str(config),
            dict_to_csv_str(metric)
        ]) + "\n")
if is_test:
    cost_time = time() - start_time
    mean_time = cost_time / n_trials
    print(mean_time * total / 3600)
