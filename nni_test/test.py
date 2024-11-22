#%%
import numpy as np
import nni
import os
import torch
import json
import math

import torch.fx
from tvm_test import *
print(nni.__version__)
# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_mode()

# %%

mod, params = torch_to_tvm(model)

#%%
# build_mod = relay.build_module.BuildModule()
# with auto_scheduler.ApplyHistoryBest('./tvm_logs/000500.log'):
#     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
#         lib = build_mod.build(mod, params=params, target='llvm -mtriple=aarch64-linux-none')

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')

log_file = './tvm_logs/000500.log'
next_log_file = './tvm_logs/010000.log'
task_id = 2
a = lower_tensor_ir(log_file, tasks[task_id])
b = export_compute_dag_from_tasks(log_file, tasks[task_id])

#%%
print(tasks[task_id].print_best(log_file))
print(print_task_info(tasks[task_id]))
# print(diff(a, b))
# same_as, tune, print_best, apply_best
sch, args = tasks[task_id].apply_best(log_file)

func = tvm.build(sch, args, tasks[task_id].target)

# print(diff(tasks[task_id].print_best(log_file), tasks[task_id].print_best(next_log_file)))
#%%
#%%
func()
# %%
# print(build_mod.get_graph_json())
# print(build_mod.get_params())
print(build_mod.get_function_metadata()['fused_nn_conv2d_add_nn_relu_1'])

#build_mod.get_module()
#%%

tvm.lower(mod['main'], simple_mode=True)
# %%
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency, ReshapeDependency, InputChannelDependency, AttentionWeightDependency
import copy

input_shape = [1,3,32,32]
dummy_input = torch.rand(input_shape)

model = load_mode()
layer_dependency = ChannelDependency(model, dummy_input=torch.randn(input_shape)).dependency_sets
# layer_dependency
sorted(list(layer_dependency[10]))
#%%
config_list = [
    {'sparsity': 0.0625, 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[10])},
    ]
# config_list = [{'sparsity':0.0703125,'op_types':['Conv2d'],'op_names':['layer4.1.conv2']},{'sparsity':0.0703125,'op_types':['Conv2d'],'op_names':['layer4.0.conv2']},{'sparsity':0.15625,'op_types':['Conv2d'],'op_names':['layer1.0.conv2']},{'sparsity':0.15625,'op_types':['Conv2d'],'op_names':['layer1.1.conv2']}]

#%% 기본 모델 튜닝 DAG 

bf, bp, bs = count_flops_params(model, tuple(input_shape), verbose=False)
mod, params = torch_to_tvm(model)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
ddags = [str(task.compute_dag) for task in tasks]
#%%0.0625
layer = sorted(list(layer_dependency[10]))
layer
#%%
# config_list = [
#     {'sparsity': (1 / 64 * 5), 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[1])},
#     ]
config_list = [
    {'sparsity': 0.99, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.shortcut.0']},
    {'sparsity': 0.99, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.conv2']},
    {'sparsity': 0.99, 'op_types': ['Conv2d'], 'op_names': ['layer4.1.conv2']},
    ]

model = load_mode()
prune = prune_model(copy.deepcopy(model), config_list,dependency_aware=True)
export_onnx(prune, 'conv1.onnx')
pf1, pp1, ps1 = count_flops_params(prune, tuple(input_shape), verbose=True)
mod, params = torch_to_tvm(prune)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
adags1 = [str(task.compute_dag) for task in tasks]

print(len(tasks))
print(pf1 == bf, pf1/bf, pf1, bf)
print(pp1 == bp, pp1/bp, pp1, bp)
print(ps1 == bs)
print(list(map(lambda x: x[0] == x[1], zip(adags1, ddags))))

#%%
config_list = [
    {'sparsity': (1 / 64 * 4), 'op_types': ['Conv2d'], 'op_names': list(layer_dependency[1])},
    ]
model = load_mode()
prune = prune_model(copy.deepcopy(model), config_list,dependency_aware=True)
export_onnx(prune, 'conv1.onnx')
pf2, pp2, ps2 = count_flops_params(prune, tuple(input_shape), verbose=True)
mod, params = torch_to_tvm(prune)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
adags2 = [str(task.compute_dag) for task in tasks]
print(len(tasks))
print(pf1 == pf2, pf1/pf2, pf1, pf2)
print(pp1 == pp2, pp1/pp2, pp1, pp2)
print(ps1 == ps2)
print(list(map(lambda x: x[0] == x[1], zip(adags1, adags2))))

#%%
print(adags1[0])
print(adags2[0])
# %%
import logging

logging.disable(logging.CRITICAL)

f = list(filter(lambda x: x['module_type'] == 'Conv2d', bs))
f = list(map(lambda x: x['name'], f))
config_list = list(map(lambda x:  {'sparsity': 0.25, 'op_types': ['Conv2d'], 'op_names': [x]}, f))
for i in range(len(config_list)):
    print(f'############### {i:04} ###############')
    print(config_list[i])
    prune = prune_model(copy.deepcopy(model), [config_list[i]])
    pf, pp, ps = count_flops_params(prune, tuple(input_shape), verbose=False)
    mod, params = torch_to_tvm(prune)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
    pdags = [str(task.compute_dag) for task in tasks]
    print(bf == pf, pf/bf, bf, pf)
    print(bp == pp, pp/bp, bp, pp)
    print(bs == ps)
    print(list(map(lambda x: x[0] == x[1], zip(ddags, pdags))))
logging.disable(logging.NOTSET)

# %%
# %%
pos, subgraph_tasks = extract_task_layer(model, tasks, task_weights)
# %%

def task_to_layer(model, task_id, pos, subgraph_tasks):
    input_shape = [1,3,32,32]
    _,_,bs = count_flops_params(model, tuple(input_shape), verbose=True)
    f = list(filter(lambda x: x['module_type'] == 'Conv2d', bs))
    f = list(map(lambda x: x['name'], f))

    dummy_input = torch.randn(input_shape)
    depen = ChannelDependency(model, dummy_input=dummy_input).dependency
    task_to_nni_index = np.where(np.array(subgraph_tasks) == task_id)[0]
    nni_index_to_layer = np.array(pos)[task_to_nni_index]

    list_depen = []
    list_index = []
    for i in np.array(f)[nni_index_to_layer]:
        for j in list(depen[i]):
            list_depen.append(j)
            list_index.append(f.index(j))

    return np.unique(list_depen), np.unique(list_index)

task_id = 13
layer, index = task_to_layer(model, task_id, pos, subgraph_tasks)
print(layer, index)
#%%

# %%

print(max(subgraph_tasks), subgraph_tasks)
print(len(subgraph_tasks))
# %%
print(max(pos), pos)
print(len(pos))
# %%
# 해야하는것 Task -> Layer
# Task 13 -> Layer Group
task_id = 0
pos[task_id]
np.where(subgraph_tasks == task_id)[0]

# %%
# %%
f = list(filter(lambda x: x['module_type'] == 'Conv2d', bs))
f = list(map(lambda x: x['name'], f))
# len(bs)
f
# %%
# %%
subgraph_tasks = np.array(subgraph_tasks)
# %%
subgraph_tasks[subgraph_tasks == np.array(2)]
# %%
# %%

# %%
