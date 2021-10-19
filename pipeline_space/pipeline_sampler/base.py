#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-02
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter, defaultdict

from pipeline_space.hdl import hdl2cs
from pipeline_space.hdl import layering_config
from pipeline_space.utils import generate_grid
from pipeline_space.utils import get_hash_of_str


class BasePipelineSampler():
    def get_HDL(self):
        raise NotImplementedError

    def get_no_ordinal_HDL(self):
        HDL = self.get_HDL()
        HDL['selector(choice)']['LogisticRegression']['C']['_type'] = 'choice'
        HDL['selector(choice)']['LinearSVC']['C']['_type'] = 'choice'
        return HDL

    def get_all_configs(self):
        HDL = self.get_HDL()
        CS = hdl2cs(HDL)
        # print(CS)
        grids = generate_grid(CS)
        # print(len(grids))
        configs = []
        index = 0
        for grid in grids:
            config = layering_config(grid.get_dictionary())
            configs.append(config)
            # index += 1
            # if index % 5000 == 0:
            #     print(index)
            #     print(config)
        print('len(configs):', len(configs))
        learners = [list(config['learner'].keys())[0] for config in configs]
        selector = [list(config['selector'].keys())[0] if config['selector'] is not None else "None"
                    for config in configs]
        print("Counter(learners):", Counter(learners))
        print("Counter(selector):", Counter(selector))
        config_ids = [get_hash_of_str(str(config)) for config in configs]
        config_id_to_config = defaultdict(list)
        for config, config_id in zip(configs, config_ids):
            config_id_to_config[config_id].append(config)
        config_ids_cnt = Counter(config_ids)
        abnormal_config_ids = [config_id for config_id, cnt in config_ids_cnt.items() if cnt > 1]
        return configs, config_id_to_config

    def process_hyperopt_param(self, _type: str, _value: list, name):
        from hyperopt import hp
        from hyperopt.pyll import scope

        param_mapper = {
            "ordinal": hp.choice,
            "choice": hp.choice,
            "quniform": hp.quniform,
            "int_quniform": hp.quniform,
        }
        func = param_mapper[_type]
        if 'uniform' in _type:
            low, high, q = _value
            obj = func(name, 0, high - low, q)
            if _type.startswith('int_'):
                obj = scope.int(obj)
            return low + obj
        else:
            return func(name, _value)

    def get_hyperopt_space(self):
        from hyperopt import hp

        space = {}
        HDL = self.get_HDL()

        for module, AS_HP in HDL.items():
            module = module.replace("(choice)", "")
            choices = []
            for AS, HP in AS_HP.items():
                param_dict = {}
                for param, param_define in HP.items():
                    _type = param_define["_type"]
                    _value = param_define["_value"]
                    param_name = f"{module}.{AS}.{param}"
                    param_dict[param] = self.process_hyperopt_param(_type, _value, param_name)
                choices.append({AS: param_dict})
            space[module] = hp.choice(module, choices)
        return space

