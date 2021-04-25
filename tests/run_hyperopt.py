#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn
import hyperopt.pyll.stochastic
from hyperopt import hp

# space = 0.01 + hp.quniform("learner.LinearSVC.C", 0, 1 - 0.01, 0.066)
space = hp.quniform("learner.LinearSVC.C", 1, 2, 1)
for i in range(100):
    print(hyperopt.pyll.stochastic.sample(space))
