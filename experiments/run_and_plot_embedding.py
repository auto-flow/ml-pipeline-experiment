#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-25
# @Contact    : qichun.tang@bupt.edu.cn

import pandas as pd
import pylab as plt
from experiments.evaluator import BaggingUltraoptEvaluator
from experiments.evaluator import PipelineUltraoptEvaluator
from pipeline_space.pipeline_sampler import SmallPipelineSampler, BaggingPipelineSampler
from ultraopt.hdl import hdl2cs

# 146594, 189863, 189864
dataset_id = "189863"
print(dataset_id)
benchmark_type = "pipeline"
print(benchmark_type)
ax = plt.gca()  # gca:get current axis得到当前轴
# 设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.tight_layout()
plt.rcParams['figure.figsize'] = (7.5, 6)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 10,
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

if benchmark_type == "pipeline":
    HDL = SmallPipelineSampler().get_HDL()
    data = pd.read_csv(f'processed_data/d{dataset_id}_processed.csv')
    evaluator = PipelineUltraoptEvaluator(data, 'balanced_accuracy')
elif benchmark_type == "bagging":
    HDL = BaggingPipelineSampler().get_HDL()
    data = pd.read_csv(f'processed_data/bagging_d{dataset_id}_processed.csv')
    evaluator = BaggingUltraoptEvaluator(data, 'f1')
else:
    raise NotImplementedError

CS = hdl2cs(HDL)


def evaluate(trial):
    # gamma_=lambda x:min(int(np.ceil(0.20 * x)), 15)
    print()
    '''
    T=100
    optimizer = ETPEOptimizer(
        limit_max_groups='auto',
        min_points_in_model=T,
    )
    ret = fmin(
        evaluator, HDL, optimizer, random_state=trial,
        n_iterations=T+20,
    )
    losses = ret["budget2obvs"][1]["losses"]
    df_pair= ret.optimizer.config_transformer.embedding_encoder_history[-1][1]
    # learner , scaler , selector
    learner, scaler, selector=df_pair

    learner:pd.DataFrame
    learner.sort_index(inplace=True)
    selector.sort_index(inplace=True)
    # learner.index=[f"{x}-learner" for x in learner.index]
    # selector.index = [f"{x}-selector" for x in selector.index]

    '''
    learner = pd.DataFrame.from_dict({
        'ExtraTreesClassifier': {0: -0.4424110323190689, 1: 0.74439381882548332},
        'KNeighborsClassifier': {0: 1.20873498916625977, 1: -0.712388277053833},
        'LGBMClassifier': {0: 1.9274120330810547, 1: 0.711807250976562},
        'LinearSVC': {0: -0.75643324971199036, 1: -1.2001015663146973},
        'LogisticRegression': {0: -1.513469934463501, 1: -0.9649285674095154},
        'RandomForestClassifier': {0: -1.0295536518096924, 1: 0.43393656611442566},
        'XGBClassifier': {0: 1.40766894817352295, 1: 1.4978851079940796}},
        'index'
    )
    for i, row in learner.iterrows():
        name = (i)
        loc = (row.tolist())
        x, y = loc
        plt.scatter(x, y)
        if name=="LGBMClassifier":
            x-=0.5
            y+=0.1
        elif name=="XGBClassifier":
            y-=0.2
            x-=0.1
        else:
            y+=0.1
        plt.annotate(name, [x, y])
    # plt.title(str(trial))
    # for i,row in selector.iterrows():
    #     name=(i)
    #     loc=(row.tolist())
    #     plt.scatter(loc[0],loc[1],label=name)
    # plt.legend()

    # plt.subplot(1,2,1)
    # for i,row in learner.iterrows():
    #     name=(i)
    #     loc=(row.tolist())
    #     plt.scatter(loc[0],loc[1],label=name)
    # plt.legend()
    # plt.title('learner')
    # plt.subplot(1, 2, 2)
    # for i,row in selector.iterrows():
    #     name=(i)
    #     loc=(row.tolist())
    #     plt.scatter(loc[0],loc[1],label=name)
    # plt.legend()
    # plt.title('selector')
    # plt.suptitle(dataset_id+f" {trial}")
    plt.savefig('embedding_dist.png')
    plt.savefig('embedding_dist.pdf')
    plt.show()



global_min = evaluator.global_min

# for i in range(1,100):
evaluate(5)
#     evaluate(i)
