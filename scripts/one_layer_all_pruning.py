# 1. 모두다 프루닝
# 2. 정확도 다시 합치고 통합.

#%%
import sys
import json
# sys.stderr = open(os.devnull, 'w')
import logging
logging.getLogger().setLevel(logging.FATAL)

import tvm_test as tt
import torch.onnx
from tvm import relay, auto_scheduler
import tvm
from tvm.contrib import graph_executor, utils, ndk, graph_runtime as runtime
import torch
import os
import numpy as np
from resnet import ResNet18
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency, ReshapeDependency, InputChannelDependency, AttentionWeightDependency
import torch.nn as nn
import torch.nn.functional as F

import copy
input_shape = [1,3,32,32]
dummy_input = torch.randn(input_shape)
class TestBlock(nn.Module):
    def __init__(self, observable = []):
        super(TestBlock, self).__init__()
        self.observable = nn.ModuleList(observable)
        
    def forward(self, x):
        result = x
        for o in self.observable:
            result = o(result)
        
        return F.relu(result)
def safe_int(value):
    try:
        return int(value)
    except ValueError:
        return None  # 혹은 원하는 값을 반환
    



def _get_latest_iter(dir):
        dirs = os.path.join(dir, 'tvm')
        dirs = os.listdir(dirs)
        pk = list(filter(lambda x: x[-3:] == 'pkl', dirs))
        pk = list(filter(lambda x: safe_int(x.split('_')[0]), pk))
        pk = list(filter(lambda x: x.split('_')[-1] == 'op.pkl', pk))
        pk = list(map(lambda x: x.split("_")[0], pk))

        if len(pk) == 0:
            return 0
        else:
            pk_max = max(list(map(lambda x: int(x), pk)))
            return pk_max

def _get_last_epoch(dir, cnt=-1):
    if cnt == -1:
        pk_max = _get_latest_iter(dir)
    else:
        pk_max = cnt
    if pk_max == 0:
        return None
    
    iter = str(pk_max).zfill(3)
    dirs = os.path.join(dir, 'tvm')
    dirs = os.listdir(dirs)
    dirs.sort()
    dd = list(filter(lambda x: x[:3] == iter, dirs))
    epoch = dd[-1].split('.')[0].split('_')[:2]
    return '_'.join(epoch)

#%%
if True:
    device_key = 'sd865-1'
    use_android = True
    tracker_host = '0.0.0.0'
    tracker_port = 9190
    timeout = 200
    runner_number = 10
    runner_repeat = 2
    early_stop = 200
    at_least_trials = 740
    num_per_round = 60

    # layer2.0.shorcut.0 cld[4]  // (1, 64, 32, 32) 
    # layer3.0.shorcut.0 cld[7]     (1, 128, 16, 16)
    # layer4.0.shorcut.0 cld[10]  // (1, 256, 8, 8)
    
    # layer1.1.conv1 print(cld[2]) 64   // (1, 64, 32, 32)
    # layer2.1.conv1 print(cld[5]) 128  // (1, 128, 16, 16)
    # layer3.1.conv1 print(cld[8]) 256  //  (1, 256, 8, 8) 
    # layer4.1.conv1 print(cld[11]) 512 // (1, 512, 4, 4)
    model = tt.load_mode('./cifar10_resnet18_94.7.pth')
    cld = ChannelDependency(model, dummy_input=dummy_input).dependency_sets
    for s in range(1, 512):
        import gc
        gc.collect()
        log_file = f'{device_key}.layer3.1.conv1.alltune.{s:04}.log'
        # sparsity가 소수점이거나 나눠지지 않는다면, 버림함.
        config_list = [{'sparsity': s/256.0,  # 바꿔야함.
                        'op_types':['Conv2d'], 
                        'op_names':list(cld[8])}] # 바꿔야함.
        pmodel = tt.prune_model(model, config_list)
        _ = tt.count_flops_params(pmodel, dummy_input)
        
        # conv1 in Layer 2 만 검증 테스트를 수행함.
        block = TestBlock([pmodel.layer3[1].conv1, pmodel.layer3[1].bn1]) # 바꿔야함.
        
        # mod, params = tt.torch_to_tvm(block, [1, 128, 16, 16])
        mod, params = tt.torch_to_tvm(block, [1, 256, 8, 8]) # 바꿔야함
        if use_android:
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-android')
        else:
            tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
        
        # tune_trials = (at_least_trials + num_per_round) * len(tasks)
        tune_trials = (at_least_trials + num_per_round) * len(tasks) # (740 + 60) * 1
        assert(tune_trials == 800)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
                            num_measure_trials=tune_trials,
                            builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
                            runner=auto_scheduler.RPCRunner(device_key, 
                                                            host=tracker_host, 
                                                            port=tracker_port, 
                                                            timeout=timeout, 
                                                            number=runner_number, 
                                                            repeat=runner_repeat,),
                            
                            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                            verbose=1,
                            early_stopping=int(early_stop),
                            num_measures_per_round = num_per_round,
                        ) 
        tuner.tune(tune_option)
#%%
if False:
    model = tt.load_mode('./cifar10_resnet18_94.7.pth')
    _, _, a = tt.count_flops_params(model, dummy_input)
    cld = ChannelDependency(model, dummy_input=dummy_input).dependency_sets
    input_shape = [1, 64, 32, 32]
    block = model.layer2[0].shortcut
    mod, params = tt.torch_to_tvm(block, input_shape)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
    print([t.workload_key for t in tasks])

# %% 프루닝 가능한 모든 채널 정보 수집
if False:
    result = []
    model = tt.load_mode('./cifar10_resnet18_94.7.pth')
    cld = ChannelDependency(model, dummy_input=dummy_input).dependency_sets
    
    # layer2.0.shorcut.0 cld[4]  // (1, 64, 32, 32) 
    # layer3.0.shorcut.0 cld[7]     (1, 128, 16, 16)
    # layer4.0.shorcut.0 cld[10]  // (1, 256, 8, 8)
    
    # layer1.1.conv1 print(cld[2]) 64   // (1, 64, 32, 32)
    # layer2.1.conv1 print(cld[5]) 128  // (1, 128, 16, 16)
    # layer3.1.conv1 print(cld[8]) 256  //  (1, 256, 8, 8) 
    # layer4.1.conv1 print(cld[11]) 512 // (1, 512, 4, 4)
    s = 0
    # e = 128
    # e = 256
    e = 512
    for s in range(s, e):
        import gc
        gc.collect()
        # file_name = 'resnet18.layer2[0].shortcut.pruning_with_fp.txt'
        # file_name = 'resnet18.layer3[0].shortcut.pruning_with_fp.txt'
        file_name = 'resnet18.layer4[0].shortcut.pruning_with_fp.txt'
        if os.path.exists(file_name):
            with open(file_name, 'rt') as f:
                result = json.load(f)
        # sparsity가 소수점이거나 나눠지지 않는다면, 버림함.
        if s == 0:
            pmodel = copy.deepcopy(model)
        else:
            config_list = [{'sparsity': s/float(e), 
                            'op_types':['Conv2d'], 
                            # 'op_names':list(cld[5])}]
                            'op_names':list(cld[10])}]
            
            pmodel = tt.prune_model(copy.deepcopy(model), config_list)
        
        # block = pmodel.layer2[0].shortcut
        # block = pmodel.layer3[0].shortcut
        block = pmodel.layer4[0].shortcut
            
        # input_shape = [1, 64, 32, 32]
        # input_shape = [1, 128, 16, 16]
        input_shape = [1, 256, 8, 8]
        input = torch.randn(input_shape)
        # block = TestBlock([pmodel.layer4[1].conv1, pmodel.layer4[1].bn1])
        flops, flop_params, a = tt.count_flops_params(block, input)

        # conv1 in Layer 2 만 검증 테스트를 수행함.
        # conv1 in Layer 4 만 검증 테스트를 수행함.
        
        # mod, params = tt.torch_to_tvm(block, [1,3,32,32])
        mod, params = tt.torch_to_tvm(block, input_shape)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
        # print(len(tasks), [t.workload_key for t in tasks])
        result.append([len(tasks), task_weights, tasks[0].workload_key, flops, flop_params])
        with open(file_name, 'wt+') as f:
            json.dump(result, f)

#%% 데이터 합치기
if False:
    result = [[] for i in range(len(data))]
    # 로그를 모두 다 합치고, 하나로 만들자.
    dir = ['/work/experiments/manytime_sd865_resnet18_all_earlystop_50',
        '/work/experiments/manytime_sd865_resnet18_all_none',
        '/work/experiments/manytime_sd865-1_resnet18_all_early10000000.0_min0.1',
        '/work/experiments/manytime_sd865-1_resnet18_error_early300',
        '/work/experiments/manytime_sd865-3_resnet18_all_early10000000.0_min0.2',
        '/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'
        ]
    save_file = './sd865_all_prune.layer2.conv1.log'

    with open(save_file, 'wt+') as f:
        for d in dir:
            log_file = os.path.join(d, 'tvm', _get_last_epoch(d) + '.log')
            file_data = None
            with open(log_file, 'rt') as file:
                file_data = file.readlines()
            for f in file_data:
                hash_id = json.loads(f)['i'][0][0]
                try:
                    index = data.index(hash_id)
                    result[index].append(f.strip())
                except:
                    pass
    for i in range(len(result)):
        print(i, len(result[i]))        
    
# ["a543b40af283ee8e8ccfabc62b2e44e4", 1, 4, 4, 512, 6, 6, 508, 512, 1, 1, 1, 508, 1, 4, 4, 508]