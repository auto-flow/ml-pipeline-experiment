#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn
'''
实验脚本，实验步骤如下：
0. 从DATAPATH加载测试数据（bz2 压缩的 pickle）
1. 根据HDL构造一个空间，然后用网格搜索的方法遍历整个空间，得到4万个配置
2. 根据参数，对这组配置进行切分
3. 对切分后的子配置进行遍历
    3.1 计算配置的hash值，作为主键
        （如果主键存在，跳过）
    3.2 5折交叉，并记录每折的各种metrics
    3.3 对metrics求平均
    3.4 整理好数据，上传数据库
'''
import os
import socket
from collections import defaultdict
from pprint import pprint
from random import seed
from time import time

import numpy as np
from joblib import load, delayed, Parallel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pipeline_space.automl_pipeline.construct_pipeline import construct_pipeline
# from pipeline_space.build_ml_pipeline_space import get_all_configs
from pipeline_space.metrics import calculate_score, f1
from pipeline_space.pipeline_sampler import SmallPipelineSampler, BigPipelineSampler
from pipeline_space.utils import get_chunks, get_hash_of_str, dict_to_csv_str

hostname = socket.gethostname()

# 单机环境变量填写：
# SPLITS=30;INDEX=10;KFOLD=3;SPACE_TYPE=BIG;TABLE_NAME=big_d146594;DATAPATH=/media/tqc/doc/Project/metalearn_experiment/data/146594.bz2
SPLITS = int(os.environ['SPLITS'])
INDEX = int(os.environ['INDEX'])
KFOLD = int(os.environ['KFOLD'])
DATAPATH = os.environ['DATAPATH']
SPACE_TYPE: str = os.environ['SPACE_TYPE']
CONFIG_ID = os.environ.get('CONFIG_ID')  # c478a1b5bde6f36883bc429f39a66b41

if SPACE_TYPE.upper() == "SMALL":
    space_sampler = SmallPipelineSampler()
elif SPACE_TYPE.upper() == "BIG":
    space_sampler = BigPipelineSampler()
else:
    raise NotImplementedError

all_configs, config_id_to_config = space_sampler.get_all_configs()
if CONFIG_ID is None:
    np.random.seed(0)
    np.random.shuffle(all_configs)
    print("all_configs[0]")
    print(all_configs[0])
    print("all_configs[-1]")
    print(all_configs[-1])
    N = len(all_configs)
    split_configs = get_chunks(all_configs, SPLITS)
    sub_configs = split_configs[INDEX]
    pprint(sub_configs[-1])
else:
    sub_configs = [config_id_to_config[CONFIG_ID][0]]
X, y, cat = load(DATAPATH)
X = X.values
X = X[:, ~np.array(cat)]
y = LabelEncoder().fit_transform(y)
cv = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=0)
print(next(cv.split(X, y))[0])

# 单机测试
is_test = False
if "tqc" in hostname:
    n_jobs = 1
    is_test = True
    os.environ['SAVEDPATH'] = 'savedpath'
    os.system(f"rm -rf $SAVEDPATH")
    os.system(f"mkdir -p $SAVEDPATH")
else:
    n_jobs = 1

seed(0)
# shuffle(sub_configs)
config_chunks = get_chunks(sub_configs, 1)

fname = os.environ['SAVEDPATH'] + "/data.csv"
log_fname = os.environ['SAVEDPATH'] + "/info.log"
f = open(fname, 'w+')
log_f = open(log_fname, 'w+')
columns = ['config_id', 'cost_time', 'failed_info', 'all_score', 'config']
f.write(",".join(columns) + "\n")
all_start_time = time()


# for config in tqdm(sub_configs):
def process(configs):
    # 会深拷贝X,y

    total = len(configs)
    for idx, config in enumerate(configs):
        config_id = get_hash_of_str(str(config))
        print(config_id)
        print(config)
        all_scores_list = defaultdict(list)
        start_time = time()

        try:
            for train_ix, test_ix in cv.split(X, y):
                X_train = X[train_ix, :]
                y_train = y[train_ix]
                X_test = X[test_ix, :]
                y_test = y[test_ix]
                pipeline = construct_pipeline(config, verbose=True, n_jobs=None)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict_proba(X_test)
                # 算 metrics
                all_scores = calculate_score(y_test, y_pred, "classification", f1, True)[1]
                for metric_name, score in all_scores.items():
                    all_scores_list[metric_name].append(score)
            all_scores_mean = {}
            for metric_name, scores in all_scores_list.items():
                all_scores_mean[metric_name] = float(np.mean(scores))
            failed_info = ""
        except Exception as e:
            failed_info = str(e)
            all_scores_mean = {}
        cost_time = time() - start_time  # 因为缓存的存在，所以可能不准
        print('accuracy', all_scores_mean.get('accuracy'))
        f.write(",".join([
            config_id,
            str(cost_time),
            failed_info,
            dict_to_csv_str(all_scores_mean),
            dict_to_csv_str(config),
        ]) + "\n")
        all_cost_time = time() - all_start_time
        p = (idx + 1) / total
        rest_time = all_cost_time * ((1 - p) / p)
        info = (f"index = {idx}, rest_time = {rest_time:.2f}, cost_time = {all_cost_time:.2f}")
        print(info)
        log_f.write(info + "\n")
        if is_test and idx > 10:
            print('finish test')
            break


# 开多个进程，对切片的config进行处理
Parallel(backend="multiprocessing", n_jobs=n_jobs)(
    delayed(process)(configs)
    for configs in config_chunks
)

os.system("rm -rf $SAVEDPATH/tmp")
f.close()
log_f.close()
