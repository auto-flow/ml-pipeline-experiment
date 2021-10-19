#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-24
# @Contact    : qichun.tang@bupt.edu.cn
import json

import pandas as pd
from tqdm import tqdm

from pipeline_space.utils import get_hash_of_dict

# datasets = [146594, 189863, 189864]
datasets = ["bagging_d146594"]

for dataset in datasets:
    print(dataset)
    df = pd.read_csv(f'{dataset}.csv')
    columns = ['config_id', 'config', 'metrics']
    df.columns = columns
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i, :]
        # config
        config = json.loads(row['config'])
        for component in [f"learner{i}" for i in range(7)]:
            sub_config = config[component]
            if sub_config is None:
                config[component] = {}
        config_id = get_hash_of_dict(config)
        df.loc[i, 'config_id'] = config_id
    df.to_csv(f"{dataset}_processed.csv", index=False)
