#%%
from dotenv import load_dotenv
import tvm_test as tt
from tvm import relay, auto_scheduler
from tvm.auto_scheduler import dispatcher
import tvm
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
import copy

import os
import torch.utils.data
import torchvision.models as models
import pickle
import random
import string
from parsing_experiments import get_data
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
from nni.compression.pytorch import ModelSpeedup
from utils import *
from c_pruner import CPruner
from resnet import ResNet18

torch.cuda.set_device(1)
def generate_random_string(length=16):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

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
        pk_max = _get_latest_iter()
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
def convert(d, f):
    if f == None:
        return []
    with open(os.path.join(d, 'tvm', f + '_config.pkl'), 'rb') as ff:
        return pickle.load(ff)
#%%
def convert_data(dirs):
# dir = f'/work/experiments/manytime_sd865_resnet18_all_none'    
    dir = dirs
    max_it = _get_latest_iter(dir)
    epoch = [_get_last_epoch(dir, value) for value in range(max_it)]
    epoch = [convert(dir, value) for value in epoch]

    hash_map = []
    result = []
    dummy_input = torch.randn([1,3,32,32])
    for idx in range(len(epoch)):
        result_item = {}
        model = tt.load_mode('./cifar10_resnet18_94.7.pth')
        tt.prune_model(model, epoch[idx])
        mod, params = tt.torch_to_tvm(model)
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
        result_item['tasks'] = len(tasks)
        result_item['task_weights'] = task_weights
        need_retune = 0
        total_cnt = len(tasks)
        for t in tasks:
            if not t.workload_key in hash_map:
                need_retune += 1
                hash_map.append(t.workload_key)
        result_item['need_retune'] = need_retune
        result_item['idx'] = idx
        result.append(result_item)
    return result

# %%
file_list = ['manytime_rockpi_resnet18_all_none',
             'manytime_rockpi_resnet18_error_early300', 
             'manytime_rockpi_resnet18_error_inf',
             'manytime_sd865_resnet18_all_earlystop_50',
             'manytime_sd865_resnet18_all_none',
             'manytime_sd865-1_resnet18_error_early300',
             'manytime_sd865-3_resnet18_error_early10000000.0']

convert_data(f'/work/experiments/{file_list[0]}')
