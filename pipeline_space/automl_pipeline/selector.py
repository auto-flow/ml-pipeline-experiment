#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-21
# @Contact    : qichun.tang@bupt.edu.cn
import multiprocessing as mp
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

import_models = [
    XGBClassifier, LGBMClassifier, RandomForestClassifier, ExtraTreesClassifier,
    LogisticRegression, LinearSVC
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


class Selector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            AS,
            C=None,
            max_depth=None,
            num_leaves=None,
            min_samples_split=None,
            n_jobs=1):
        self.min_samples_split = min_samples_split
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.C = C
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs
        self.AS = AS
        self.HP = dict(random_state=0)
        if AS == "LogisticRegression":
            self.HP.update(
                penalty='l1',
                solver='liblinear',
                C=C
            )
        elif AS == "LinearSVC":
            self.HP.update(
                penalty='l1',
                dual=False,
                C=C
            )
        elif AS in bagging_models:
            self.HP.update(
                n_jobs=n_jobs,
                n_estimators=100,
            )

        elif AS in boosting_models:
            self.HP.update(
                # n_jobs=n_jobs,
                # nthread=n_jobs,
                n_estimators=50,
                learning_rate=0.2,
                min_samples_split=min_samples_split
            )
            if AS == "XGBClassifier":
                self.HP.update(
                    max_depth=max_depth
                )
            else:
                self.HP.update(
                    num_leaves=num_leaves
                )
        else:
            raise NotImplementedError(f"Invalid AS {self.AS}")
        self.model = eval(AS)(**self.HP)
        self.keys = tuple(self.HP.keys())
        self.values = tuple(self.HP.values())

    def fit(self, X, y):
        if self.AS in linear_models:
            self.model.fit(X, y)
            weights = np.abs(self.model.coef_.sum(axis=0))
            self.select_ = weights != 0
        elif self.AS in boosting_models or self.AS in bagging_models:
            rng = np.random.RandomState(0)
            n, m = X.shape
            X_sh = X.copy()
            for i in range(m):
                X_sh[:, i] = X_sh[:, i][rng.permutation(n)]
            X_cat = np.hstack([X, X_sh])
            self.model.fit(X_cat, y)
            feature_importances_ = self.model.feature_importances_
            left_imp = feature_importances_[:m]
            right_imp = feature_importances_[m:]
            self.select_ = left_imp > right_imp
        else:
            raise NotImplementedError
        return self

    def transform(self, X):
        return X[:, self.select_]
