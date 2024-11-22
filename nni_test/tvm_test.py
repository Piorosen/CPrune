import torch.onnx
from tvm import relay, auto_scheduler
import tvm
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
import difflib
import math
import torch
import numpy as np
import os
from resnet import ResNet18
import copy
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency, ReshapeDependency, InputChannelDependency, AttentionWeightDependency

def diff(a:str, b:str):
    # 두 텍스트를 줄 단위로 나눕니다.
    text1_lines = a.splitlines()
    text2_lines = b.splitlines()

    # Differ 객체를 생성하여 차이점을 계산합니다.
    d = difflib.Differ()
    diff = d.compare(text1_lines, text2_lines)

    # 결과를 출력합니다.
    return '\n'.join(diff)

def lower_tensor_ir(log_file, task):
    sch, args = task.apply_best(log_file)
    return tvm.lower(sch, args, simple_mode=True)

def lower_relay_ir(mod):
    return mod['main'].astext(show_meta_data=False)

def export_compute_dag_from_tasks(log_file, task):
    return task.print_best(log_file, print_mode='schedule')


def torch_to_tvm(model):
    dummy_input = torch.rand(1, 3, 32, 32)
    input_shape = [1,3,32,32]
    scripted_model = torch.jit.trace(model, dummy_input).eval()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts),
                                    relay.transform.InferType(),
                                    relay.transform.FoldConstant(),
                                    relay.transform.DeadCodeElimination()])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod, params
# print(tasks[0].print_best(log_file))
# print(tasks[0].print_best(next_log_file))
# tasks[task_id].same_as(tasks[task_id + 1])

def log_split(file = './long_cifar_resnet_rockpi.log', dir = './tvm_logs'):
    lines = []
    with open(file, 'rt') as f:
        lines = f.readlines()

    lines_num = len(lines)
    for i in range(math.ceil(lines_num / 500)):
        max_num = min(lines_num, (i + 1) * 500)
        with open(os.path.join(dir, f'/{max_num:06}.log'), 'wt') as f:
            f.writelines(lines[:max_num])
            
def print_task_info(task):
    print(task.compute_dag)
    print(task.desc)
    print(task.hardware_params)
    print(task.target)
    print(task.workload_key)
    print(task.target_host)
    print(task.target_host)            

def load_mode(pth: str = './cifar10_model_300.pth'):
    device = torch.device("cpu")
    model = ResNet18()
    model.load_state_dict(torch.load(pth))
    model.to(device)
    model.eval()
    return model

def prune_model(model, config_list, dependency_aware:bool = True):
    device = torch.device("cpu")
    input_shape = [1,3,32,32]
    dummy_input = torch.randn(input_shape)
    pruner = PRUNER_DICT['l1'](copy.deepcopy(model), config_list=config_list, dependency_aware=dependency_aware, dummy_input=dummy_input)
    prunemodel = pruner.compress()
    pruner.export_model('model.pth', 'mask.pth')
    model.load_state_dict(torch.load('model.pth'))
    m_speedup = ModelSpeedup(model, dummy_input, 'mask.pth', device)
    m_speedup.speedup_model()
    model.eval()
    os.remove('model.pth')
    os.remove('mask.pth')
    return model

def export_onnx(model, file_name = 'resnet18_prune.onnx'):
    dummy_input = torch.randn((1,3,32,32))
    torch.onnx.export(model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            file_name,       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=12,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['input0'],   # the model's input names 
            output_names = ['output0'], # the model's output names 
            ) 

def extract_task_layer(model, tasks, task_weights):
    dummy_input = torch.randn((1,3,32,32))
    _, _, temp_results = count_flops_params(model, dummy_input)
    conv2d_num = 0
    others_num = 0
    downsample_subgraphs = []
    temp_results_len = len(temp_results)
    for idx in range(temp_results_len):
        if 'downsample' in temp_results[idx].get('name'):
            downsample_subgraphs.append(idx)
        elif 'shortcut' in temp_results[idx].get('name'):
            downsample_subgraphs.append(idx)
        if temp_results[idx].get('module_type') == 'Conv2d':
            conv2d_num+=1
        else:
            others_num+=1
    conv2d_subgraph_chs = [-1 for i in range(conv2d_num)]
    temp_idx = 0
    for idx in range(temp_results_len):
        if temp_results[idx].get('module_type') == 'Conv2d':
            conv2d_subgraph_chs[temp_idx] = temp_results[idx].get('weight_shape')[0]
            temp_idx += 1
    ##################### subgraph_task connection #######################
    pos = []
    last_idx = conv2d_num - 1
    list_filled = [0 for i in range(conv2d_num)]
    for idx in range(conv2d_num):
        n = conv2d_num - 1 - idx
        if list_filled[n] == 1:
            continue
        elif 'downsample' in temp_results[n].get('name'):
            continue
        elif 'shortcut' in temp_results[n].get('name'):
            continue
        else:
            pos.append(n)
            list_filled[n] = 1
        split_name = temp_results[n].get('name').split('.')
        for i in range(conv2d_num):
            if i == n: break
            temp_split = temp_results[i].get('name').split('.')
            if split_name[0] == temp_split[0] and \
            split_name[len(split_name)-1] == temp_split[len(temp_split)-1] and \
            temp_results[n].get('weight_shape') == temp_results[i].get('weight_shape') and \
            temp_results[n].get('flops') == temp_results[i].get('flops') and \
            temp_results[n].get('params') == temp_results[i].get('params'):
                pos.append(i)
                list_filled[i] = 1

    pos = pos + downsample_subgraphs

    subgraph_tasks = [-1 for i in range(conv2d_num)]
    pos_idx = 0
    downsample_idx = 0
    for idx, task in enumerate(tasks):
        if idx < others_num:
            continue
        if len(task.workload_key) < 80:
            continue
        for i in range(task_weights[idx]):
            subgraph_tasks[pos[pos_idx]] = idx
            pos_idx += 1

    return pos, subgraph_tasks, 

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
    for i in np.array(f)[nni_index_to_layer]:
        for j in list(depen[i]):
            list_depen.append(j)

    return list_depen
