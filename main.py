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

hostname = socket.gethostname()
# psql -U postgres
db = pw.PostgresqlDatabase(
    database="ml_pipeline_experiment",
    host="123.56.90.56",
    user="postgres",
    password="xenon"
)

# os.environ['OMP_NUM_THREADS'] = "1"


def get_conn(create_table=False):
    class Trial(pw.Model):
        config_id = pw.CharField(primary_key=True)
        cost_time = pw.FloatField(null=True)
        failed_info = pw.TextField(null=True)
        all_score = JSONField(null=True)
        config = JSONField(null=True)

        class Meta:
            database = db
            table_name = os.environ['TABLE_NAME']
    if create_table:
        Trial.create_table(safe=True)
    return Trial


get_conn(create_table=True)
# 单机环境变量填写：
# SPLITS=10;INDEX=0;KFOLD=5;SPACE_TYPE=BIG;TABLE_NAME=small_d146594;DATAPATH=/media/tqc/doc/Project/metalearn_experiment/data/146594.bz2
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
if "tqc" in hostname:
    n_jobs = 5
else:
    n_jobs = 30

seed(0)
shuffle(sub_configs)
config_chunks = get_chunks(sub_configs, n_jobs)


# for config in tqdm(sub_configs):
def process(configs):
    # 会深拷贝X,y
    Trial = get_conn()
    for config in configs:
        config_id = get_hash_of_str(str(config))
        print(config)
        all_scores_list = defaultdict(list)
        start_time = time()
        if len(list(Trial.select(Trial.config_id).where(Trial.config_id == config_id).dicts())) == 0:
            Trial.create(config_id=config_id)
        else:
            print(f'{config_id} exists, continue')
            continue
        try:
            for train_ix, test_ix in cv.split(X, y):
                X_train = X[train_ix, :]
                y_train = y[train_ix]
                X_test = X[test_ix, :]
                y_test = y[test_ix]
                pipeline = construct_pipeline(config, verbose=True, n_jobs=8)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict_proba(X_test)
                # 算 metrics
                all_scores = calculate_score(y_test, y_pred, "classification", f1, True)[1]
                for metric_name, score in all_scores.items():
                    all_scores_list[metric_name].append(score)
            all_scores_mean = {}
            for metric_name, scores in all_scores_list.items():
                all_scores_mean[metric_name] = float(np.mean(scores))
            failed_info = None
        except Exception as e:
            failed_info = str(e)
            all_scores_mean = None
        cost_time = time() - start_time  # 因为缓存的存在，所以可能不准
        print('accuracy', all_scores_mean['accuracy'])
        Trial.update(
            cost_time=cost_time,
            failed_info=failed_info,
            all_score=all_scores_mean,
            config=config
        ).where(Trial.config_id == config_id).execute()
        # 整理数据，上传数据库
        print()


# 开多个进程，对切片的config进行处理
Parallel(backend="multiprocessing", n_jobs=n_jobs)(
    delayed(process)(configs)
    for configs in config_chunks
)

os.system("mkdir rm -rf $SAVEDPATH/tmp")
