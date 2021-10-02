#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-21
# @Contact    : qichun.tang@bupt.edu.cn
import multiprocessing as mp
import warnings

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from pipeline_space.utils import softmax

warnings.filterwarnings('ignore')

import_models = [
    XGBClassifier, LGBMClassifier, RandomForestClassifier, ExtraTreesClassifier,
    LogisticRegression, LinearSVC, KNeighborsClassifier
]
linear_models = [
    "LogisticRegression", "LinearSVC"
]
bagging_models = [
    "RandomForestClassifier", "ExtraTreesClassifier"
]
boosting_models = [
    "XGBClassifier", "LGBMClassifier"
]
knn_models = [
    "KNeighborsClassifier"
]


class Learner(BaseEstimator, ClassifierMixin):
    def __init__(self, AS, HP, n_jobs=1):
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs
        self.HP = HP
        self.AS = AS
        # 给除了KNN之外的模型设置随机种子
        if AS not in knn_models:
            self.HP.update(random_state=0)
        # 设置一些默认参数
        if AS == "LogisticRegression":
            self.HP.update(dict(
                solver='liblinear',
            ))
        elif AS == "LinearSVC":
            self.HP.update(dict(
                dual=False
            ))
        elif AS in bagging_models:
            self.HP.update(dict(
                n_jobs=n_jobs,
                n_estimators=100,
            ))
        elif AS in boosting_models:
            self.HP.update(dict(
                n_jobs=n_jobs,
                nthread=n_jobs,
                n_estimators=100,
                learning_rate=0.1,
            ))
        elif AS in knn_models:
            pass
        else:
            raise NotImplementedError(f"Invalid AS {self.AS}")
        self.model = eval(AS)(**self.HP)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except:
            y_pred = softmax(self.model.decision_function(X))
            return y_pred