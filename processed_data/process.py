#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-24
# @Contact    : qichun.tang@bupt.edu.cn
import json

import pandas as pd
from tqdm import tqdm

datasets = [146594, 189863, 189864]

for dataset in datasets:
    print(dataset)
    df = pd.read_csv(f'd{dataset}.csv', header=None)
    columns = ['config_id', 'cost_time', 'failed_info', 'all_score', 'config']
    df.columns = columns
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i, :]
        # all_score
        all_score = json.loads(row['all_score'])
        for k, v in all_score.items():
            df.loc[i, k] = v
        # config
        config = json.loads(row['config'])
        for component in ['scaler', 'selector', 'learner']:
            sub_config = config[component]
            if sub_config is None:
                df.loc[i, component] = "None"
                continue
            AS, HP = sub_config.popitem()
            df.loc[i, component] = AS
            for k, v in HP.items():
                name = f"{component}.{AS}.{k}"
                df.loc[i, name] = v
    for col in columns:
        if col == 'config_id':
            continue
        df.pop(col)
    df.to_csv(f"d{dataset}_processed.csv", index=False)
