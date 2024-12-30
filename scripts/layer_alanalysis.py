#%%
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
# sys.stderr = open(os.devnull, 'w')
import logging
logging.getLogger().setLevel(logging.FATAL)
import matplotlib.pyplot as plt
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
import pandas as pd


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

def get_speed_minima(draw_data2, b=False):
    local_minima_indices = argrelextrema(draw_data2, np.less)[0]
    local_minima_values = draw_data2[local_minima_indices]
    filtered_local_minima_indices_refined = []
    filtered_local_minima_values_refined = []
    if b:
        for idx, current_idx in enumerate(local_minima_indices):
            if idx == 0:
                filtered_local_minima_indices_refined.append(current_idx)
                filtered_local_minima_values_refined.append(draw_data2[current_idx])
            else:
                # Compare with the previous local minima
                prev_idx = filtered_local_minima_indices_refined[-1]
                if draw_data2[current_idx] < draw_data2[prev_idx]:
                    filtered_local_minima_indices_refined.append(current_idx)
                    filtered_local_minima_values_refined.append(draw_data2[current_idx])
        return filtered_local_minima_indices_refined, filtered_local_minima_values_refined
    else:
        return local_minima_indices, local_minima_values

def get_speed_maxima(draw_data2, b=False):
    # Find local maxima indices
    local_maxima_indices = argrelextrema(draw_data2, np.greater)[0]
    local_maxima_values = draw_data2[local_maxima_indices]
    if b:
        filtered_local_maxima_indices_refined = []
        filtered_local_maxima_values_refined = []
        for idx, current_idx in enumerate(local_maxima_indices):
            # If it's the first local maxima, include it
            if idx == 0:
                filtered_local_maxima_indices_refined.append(current_idx)
                filtered_local_maxima_values_refined.append(draw_data2[current_idx])
            else:
                # Compare with the previous local maxima
                prev_idx = filtered_local_maxima_indices_refined[-1]
                if draw_data2[current_idx] > draw_data2[prev_idx] * 0.9:
                    filtered_local_maxima_indices_refined.append(current_idx)
                    filtered_local_maxima_values_refined.append(draw_data2[current_idx])

        return filtered_local_maxima_indices_refined, filtered_local_maxima_values_refined
    else:
        return local_maxima_indices, local_maxima_values

def get_data(hint_files = './sd865.layer3.1.conv1/resnet18.layer3.1.conv1.pruning_with_fp.txt',
             dir=['/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'],
             all_prune_per_layer_dir = './sd865.layer3.1.conv1'):
    dirs = all_prune_per_layer_dir
    
    result = []
    data_flops = []
    data_params = []
    with open(hint_files, 'rt') as f:
        data_file = json.load(f)
        data = [d[2] for d in data_file]
        data_flops = [d[3] for d in data_file]
        data_params = [d[4] for d in data_file]
    result = [[] for i in range(len(data))]

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

    items = [os.path.join(dirs, i) for i in sorted(os.listdir(dirs))[2:]]

    for d in items:
        with open(d, 'rt') as file:
            file_data = file.readlines()
        for f in file_data:
            hash_id = json.loads(f)['i'][0][0]
            try:
                index = data.index(hash_id)
                result[index].append(f.strip())
            except:
                pass
    result_data = []
    draw_data = []

    for res in result:
        inner = []
        for r in res:
            parsed_data = json.loads(r)
            r_values = parsed_data["r"][0]
            avg_r = sum(r_values) / len(r_values)
            if avg_r < 100:
                inner.append(avg_r)
        # result_data.append(inner)
        draw_data.append(min(inner))
    
    return np.array(draw_data), \
            np.array(data_flops), \
            np.array(data_params)
#%%
def draw(hint_file, zero_history, prune_dir, title):
    draw_data, data_flops, data_params = get_data(hint_file, zero_history, prune_dir)
    x_values = list(range(len(draw_data)))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title(title)
    ax1.set_xlabel("Pruning Ratio")
    ax1.set_ylabel("Infrence Time(ms)")
    y = data_flops / draw_data / 1000 / 1000 / 1000
    ax2 = ax1.twinx()
    mx, my = get_speed_maxima(y, True)
    ax2.scatter(mx, my)
    ax1.bar(x_values, draw_data * 1000, color='lightcoral')
    ax2.plot(x_values, y)
    ax2.set_ylabel('G Flops (Flops / Time)')
    
    data = {'prune_ratio': x_values,
            'flops': data_flops,
            'params': data_params,
            'infer_time': draw_data * 1000,
            'gflops': data_flops / draw_data / 1000 / 1000 / 1000,
            }
    df = pd.DataFrame(data)
    df['max_070'] = df.index.isin(mx)

    df.to_csv('./csv_result/' + title + '.csv')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    fig.show()
#%%
draw_data = [['./sd865.layer1.1.conv1/resnet18.layer1.1.conv1.pruning_with_fp.txt',
              ['/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'],
              './sd865.layer1.1.conv1',
              'layer1.1.conv1 on SD865'],
             ['./sd865.layer2.1.conv1/resnet18.layer2.1.conv1.pruning_with_fp.txt',
              ['/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'],
              './sd865.layer2.1.conv1',
              'layer2.1.conv1 on SD865'],
             ['./sd865.layer3.1.conv1/resnet18.layer3.1.conv1.pruning_with_fp.txt',
              ['/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'],
              './sd865.layer3.1.conv1',
              'layer3.1.conv1 on SD865'],
             ['./rockpi.layer1.1.conv1/resnet18.layer1.1.conv1.pruning_with_fp.txt',
              ['/work/experiments/manytime_sd865-3_resnet18_error_early10000000.0'],
              './rockpi.layer1.1.conv1',
              'layer1.1.conv1 on RockPI'],
             ['./rockpi.layer2.1.conv1/resnet18.layer2.1.conv1.pruning_with_fp.txt',
              ['/work/experiments/manytime_rockpi_resnet18_all_none'],
              './rockpi.layer2.1.conv1',
              'layer2.1.conv1 on RockPI'],
             ]

for render in draw_data[:]:
    draw(render[0], render[1], render[2], render[3])

#%%
exit(0)
#%%
import os

def rename_files_in_directory(directory_path):
    try:
        if not os.path.isdir(directory_path):
            raise ValueError(f"The path '{directory_path}' is not a valid directory.")
        for filename in os.listdir(directory_path):
            if not os.path.isfile(os.path.join(directory_path, filename)):
                continue
            # Apply renaming logic
            if '-' in filename and filename.startswith("sd865"):
                new_name = filename.replace('-1', '')
                new_name = new_name.replace('-3', '')
                # Rename the file
                old_file_path = os.path.join(directory_path, filename)
                new_file_path = os.path.join(directory_path, new_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_name}")
    except:
        pass
rename_files_in_directory('./sd865.layer3.1.conv1')
# %%
