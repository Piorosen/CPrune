#%%
import numpy as np
import torch

import torch.fx
from tvm_test import *

from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency, ReshapeDependency, InputChannelDependency, AttentionWeightDependency
import copy

input_shape = [1,3,32,32]
dummy_input = torch.rand(input_shape)

model = load_mode()
layer_dependency = ChannelDependency(model, dummy_input=torch.randn(input_shape)).dependency_sets
layer_dependency
#%%
config_list = [
    {'sparsity': 0.10, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[0])},
    {'sparsity': 0.20, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[1])},
    {'sparsity': 0.30, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[2])},
    {'sparsity': 0.40, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[3])},
    {'sparsity': 0.50, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[4])},
    {'sparsity': 0.60, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[5])},
    {'sparsity': 0.70, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[6])},
    {'sparsity': 0.80, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[7])},
    {'sparsity': 0.90, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[8])},
    {'sparsity': 0.10, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[9])},
    {'sparsity': 0.20, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[10])},
    {'sparsity': 0.30, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[11])},
    {'sparsity': 0.40, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[12])},
    ]
#%%
config_list = [
    {'sparsity': 0.10, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.conv2']},
    {'sparsity': 0.20, 'op_types': ['Conv2d'], 'op_names': ['layer4.1.conv2']},
    # {'sparsity': 0.30, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.shortcut.0']},
]
#%%
model = load_mode()
bf, bp, bs = count_flops_params(model, tuple(input_shape), verbose=False)
mod, params = torch_to_tvm(model)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
ddags = [str(task.compute_dag) for task in tasks]


prune = prune_model(copy.deepcopy(model), config_list,dependency_aware=False)
export_onnx(prune, 'conv1.onnx')
pf, pp, ps = count_flops_params(prune, tuple(input_shape), verbose=False)
mod, params = torch_to_tvm(prune)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
pdags = [str(task.compute_dag) for task in tasks]
print(len(tasks))
print(bf == pf, pf/bf, bf, pf)
print(bp == pp, pp/bp, bp, pp)
print(bs == ps)
print(list(map(lambda x: x[0] == x[1], zip(ddags, pdags))))

