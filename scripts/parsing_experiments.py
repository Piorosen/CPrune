#%%
import os

import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
import cpruner
import pickle
import numpy as np
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from resnet import ResNet18
import copy

def get_time_from_log(file_name = '/work/experiments/manytime_sd865_resnet18_all_none/time_log.txt'):
    stack_data = pd.read_csv(file_name, header=None)
    stack_data.columns = ['timestamp', 'iter', 'cnt', 'event', 'description']
    stack_data['description'] = stack_data['description'].str.strip()
    stack_data['description'] = stack_data['description'].replace('optimizer_tvm (tune_mode : 0)', 'optimizer_tvm')
    stack_data['description'] = stack_data['description'].replace('optimizer_tvm (tune_mode : 1)', 'optimizer_tvm')
    stack_data['description'] = stack_data['description'].replace('optimizer_tvm (tune_mode : 2)', 'optimizer_tvm')
    stack_data['description'] = stack_data['description'].replace('optimizer_tvm (tune_mode : 3)', 'optimizer_tvm')
    stack_data
    # Displaying the first few rows to understand its structure for the stack-based processing
    stack_data.head()
    # Using separate stacks for each description to handle the calculations
    from collections import defaultdict

    # Initializing a dictionary of stacks for each description
    description_stacks = defaultdict(list)
    durations = []

    # Iterating over rows in the dataset
    for _, row in stack_data.iterrows():
        if row['event'] == 'start':
            if len(description_stacks[row['description']]) > 0:
                pass
            else:
                description_stacks[row['description']].append((row['timestamp'], row['iter'], row['cnt']))    
            
        elif row['event'] == 'end' and row['description'] in description_stacks:
            # Pop the last start event for this description and calculate duration
            if description_stacks[row['description']]:
                start_time, iter_, cnt_ = description_stacks[row['description']].pop()
                if iter_ == row['iter']:
                    duration = row['timestamp'] - start_time
                    durations.append({
                        'description': row['description'],
                        'iter': iter_,
                        'cnt': cnt_,
                        'duration': duration
                    })
    # Converting durations to a DataFrame for visualization
    durations_df = pd.DataFrame(durations)
    return durations_df

def safe_int(value):
    try:
        return int(value)
    except ValueError:
        return None  # 혹은 원하는 값을 반환
    
def _get_latest_iter(dirs):
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

def _get_last_epoch(cnt, dirs):
    _experiment_data_dir = dirs
    pk_max = _get_latest_iter(dirs)

    iter = str(cnt).zfill(3)
    dirs = os.listdir(dirs)
    dirs.sort()
    dd = list(filter(lambda x: x[:3] == iter, dirs))
    epoch = dd[-1].split('.')[0].split('_')[:2]
    return '_'.join(epoch)

def load_model(pth: str = './cifar10_model_300.pth'):
    device = torch.device("cpu")
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(pth, map_location=torch.device('cpu') ))
    model.eval()
    return model
def count_zero_weights(model):
    zero_count = 0
    total_count = 0
    # 모델의 모든 파라미터를 반복
    for param in model.parameters():
        # 0인 값의 개수
        zero_count += (param == 0).sum().item()
        # 전체 값의 개수
        total_count += param.numel()
    
    sparsity = zero_count / total_count  # sparsity 계산 (전체에서 0인 값의 비율)
    return sparsity

def get_data(dir):
    d = dir
    dirs = os.path.join(d, 'tvm')
    pk_max = _get_latest_iter(dirs)
    base_txt = [_get_last_epoch(i, dirs) for i in range(1, pk_max + 1)]
    items = [_get_last_epoch(i, dirs) + '_best_op.pkl' for i in range(1, pk_max + 1)]
    logs = [_get_last_epoch(i, dirs) + '.log' for i in range(1, pk_max + 1)]
    perf = [_get_last_epoch(i, dirs) + '.pkl' for i in range(1, pk_max + 1)]
    times = get_time_from_log(os.path.join(d, 'time_log.txt'))

    baseline_time = times[times['iter'] == 0]['duration'].sum()
    tmp = times[times['description'] == 'sequence_pruning']['duration'].to_list()
    all_time = [baseline_time, *tmp]
    tmp = times[times['description'] == 'fine_tune_train']\
        .groupby(by='iter')['duration']\
        .sum()\
        .to_list()
    train_time = [0, *tmp]

    tmp = times[times['description'] == 'optimizer_tvm']\
        .groupby(by='iter')['duration']\
        .sum()\
        .to_list()
    opt_time = tmp
    tmp = times[times['description'] == 'optimizer_tvm']\
        .groupby(by='iter')['description']\
        .count()
    trial_times = tmp

    perf_item = [os.path.join(dirs, x) for x in perf]
    perf_item = [os.path.join(dirs, 'baseline.pkl'), *perf_item]
    it = [os.path.join(dirs, x) for x in items]
    accuracy = [0.8291]
    op_names = ['']
    op_sparsity = [0]
    op_channel = [0]

    for i in it:
        with open(i, 'rb') as f:
            d = pickle.load(f)
            accuracy.append(d['performance'])
            op_names.append(d['op_name'])
            op_sparsity.append(d['sparsity'])
            op_channel.append(d['ch_num'])
    base = [os.path.join(dirs, x) for x in base_txt]


    total_sparsity = [0]
    for i in base:
        c = i + '_config.pkl'
        m = i +'_model.pth'
        model = load_model(m)
        
        config_list = None
        with open(c, 'rb') as f:
            config_list = pickle.load(f)
        print(config_list)
        input_shape = [1,3,32,32]
        dummy_input = torch.randn(input_shape)
        pruner = PRUNER_DICT['l1'](model, config_list=config_list, dependency_aware=True, dummy_input=dummy_input)
        pruner.compress()
        total_sparsity.append(count_zero_weights(model))
    perf_list = []
    for item in perf_item:
        with open(item, 'rb') as f:
            d = pickle.load(f)
            perf_list.append(d.CurrentLatency.mean())
    len(perf_list)
    print(perf_list)
    print(d)

    index = list(range(1, len(train_time) + 1))
    train_time  = np.array(train_time  )
    trial_times = np.array(trial_times )
    all_time    = np.array(all_time    )
    opt_time    = np.array(opt_time    )
    perf_list   = np.array(perf_list   )
    accuracy    = np.array(accuracy    )
    op_names    = np.array(op_names    )
    op_sparsity = np.array(op_sparsity )
    op_channel  = np.array(op_channel  )
    total_sparsity = np.array(total_sparsity)

    data = {
        "index": index,
        "train_time": train_time,
        "trial_times": trial_times,
        "all_time": all_time - train_time,
        "not_opt": all_time,
        "opt_time": opt_time,
        "perf_list": perf_list,
        'accuracy': accuracy,
        'op_names': op_names,
        'op_sparsity': op_sparsity,
        'op_channel': op_channel,
        'total_sparsity': total_sparsity,
        'all_time_hours': (all_time - train_time) / 3600
    }
    return data
#%%
def draw(data, title, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    plt.rc('font', size=16)        # 기본 폰트 크기

    for i in range(1, len(data['all_time_hours']) - 1):  # 첫 번째와 마지막 인덱스 제외
        if data['all_time_hours'][i] > 10:
            # 앞뒤 평균값으로 대체
            data['all_time_hours'][i] = (data['all_time_hours'][i-1] + data['all_time_hours'][i+1]) / 2
    df = pd.DataFrame(data)
    all_time_avg = df['all_time_hours'][df['all_time_hours'] > 0.1].mean()
    df['all_time_hours'] = df['all_time_hours'].apply(lambda x: all_time_avg if x < 1 else x)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 좌측 y축에 Performance (성능) 플로팅
    ax1.set_xlabel('Trials')  # x축 라벨
    ax1.set_ylabel('Performance', color='tab:blue')  # 좌측 y축 라벨
    l1 = ax1.plot(df['index'], df['perf_list'], color='tab:blue', marker='o', label='Performance')  # 성능
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 우측 y축에 Accuracy (정확도) 플로팅
    ax2 = ax1.twinx()  # 우측 y축 생성
    ax2.set_ylabel('Accuracy', color='tab:red')  # 우측 y축 라벨
    l2 = ax2.plot(df['index'], df['accuracy'], color='tab:red', marker='x', label='Accuracy')  # 정확도
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 그래프 제목
    plt.title(f'Performance and Accuracy (Trials, {title})')

    # 그래프 표시
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(os.path.join(save_dir, 'pa_trials.png'))
    plt.show()
    
    # 그래프 초기화
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 좌측 y축에 Performance (성능) 플로팅
    ax1.set_xlabel('Tuning Time (hours)')  # x축 라벨
    ax1.set_ylabel('Performance', color='tab:blue')  # 좌측 y축 라벨
    ax1.plot(df['all_time_hours'].cumsum(), df['perf_list'], color='tab:blue', marker='o', label='Performance')  # 성능
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 우측 y축에 Accuracy (정확도) 플로팅
    ax2 = ax1.twinx()  # 우측 y축 생성
    ax2.set_ylabel('Accuracy', color='tab:red')  # 우측 y축 라벨
    ax2.plot(df['all_time_hours'].cumsum(), df['accuracy'], color='tab:red', marker='x', label='Accuracy')  # 정확도
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 그래프 제목
    plt.title(f'Performance and Accuracy (Hours, {title})')

    # 그래프 표시
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(os.path.join(save_dir, 'pa_hour.png'))
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 좌측 y축에 Performance (성능) 플로팅
    ax1.set_xlabel('Trials')  # x축 라벨
    ax1.set_ylabel('Performance', color='tab:blue')  # 좌측 y축 라벨
    ax1.plot(df['index'], df['perf_list'], color='tab:blue', marker='o', label='Performance')  # 성능
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 우측 y축에 total_sparsity (전체 희소성) 플로팅
    ax2 = ax1.twinx()  # 우측 y축 생성
    ax2.set_ylabel('Total Sparsity', color='tab:red')  # 우측 y축 라벨
    ax2.plot(df['index'], df['total_sparsity'], color='tab:red', marker='x', label='Total Sparsity')  # 전체 희소성
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 그래프 제목
    plt.title(f'Performance and Total Sparsity (Trials, {title})')

    # 그래프 표시
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig(os.path.join(save_dir, 'pt_trials.png'))
    plt.show()
def to_csv(data, file):
    for i in range(1, len(data['all_time_hours']) - 1):  # 첫 번째와 마지막 인덱스 제외
        if data['all_time_hours'][i] > 10:
            # 앞뒤 평균값으로 대체
            data['all_time_hours'][i] = (data['all_time_hours'][i-1] + data['all_time_hours'][i+1]) / 2
    df = pd.DataFrame(data)
    all_time_avg = df['all_time_hours'][df['all_time_hours'] > 0.1].mean()
    df['all_time_hours'] = df['all_time_hours'].apply(lambda x: all_time_avg if x < 1 else x)
    df.to_csv(file)

#%%
data = get_data('/work/experiments/manytime_rockpi_resnet18_all_none')
to_csv(data, 'rockpi_all_tune.csv')
#%%
draw(data, 'Rock PI, Changed', 'rockpi_changed')
#%%

#%%












#%%
#%%
#%%
#%%
#%%
df.to_csv('./result.csv')

# Plotting each column
plt.figure(figsize=(10, 6))

# Adding labels, legend, and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Visualization of Different Metrics")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %%
df = pd.DataFrame(pick)
df = df.drop(columns=['masks'])
time_conv = np.vectorize(lambda x: f'{int(x // 60)}m {int(x) % 60}s')
# time_train_time = np.where(1000 > time_train_time, time_train_time, 110)
tvm_tune = t- time_train_time
# tvm_tune = np.where(10000 > tvm_tune, tvm_tune, 1000)

max_time = np.where(time_train_time > tvm_tune, time_train_time, tvm_tune) + 6


# tvm_tune = np.where(10000 > tvm_tune, tvm_tune, 1000)
# max_time = np.where(4000 > max_time, max_time + (40*60), max_time)
# max_time = np.where(10000 > max_time, max_time, min(max_time) + 50*60)

df['time'] = time_conv(max_time)
df['train'] = time_conv(time_train_time)
df['tvm_tune'] = time_conv(tvm_tune)

# weight_mask = df.iloc[16]['masks']['weight_mask'] #
# num_ones = torch.sum(weight_mask == 1).item()

# total_elements = weight_mask.numel()
print(df.to_csv())
#%%
#%%
import matplotlib.pyplot as plt
time_index = np.array(t[:99]).cumsum() / 3600
plt.style.use('default')
# Figure와 첫 번째 y축 생성
fig, ax1 = plt.subplots(figsize=(10, 5))

# 첫 번째 y축 - Latency
ax1.plot(df.index + 1, perf_list, label='Latency', color='blue')
ax1.set_xlabel('Time (Hour)')
ax1.set_ylabel('Latency', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 두 번째 y축 - Accuracy
ax2 = ax1.twinx()  # 두 번째 y축 생성
ax2.plot(df.index + 1, df['performance'], label='Accuracy', color='orange')
ax2.set_ylabel('Accuracy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 그래프 제목 설정
plt.title('Figure of Latency and Accuracy to Time')
plt.grid()
plt.legend()
# 레이아웃 및 저장
plt.tight_layout()
plt.savefig('a.png')
plt.show()

plt.figure(figsize=(10, 5))
# plt.plot(df.index + 1, np.array(tvm_tune) / 60, label='TVM Tune')
plt.plot(df.index + 1, np.array(time_train_time) / 60, label='Training')
plt.plot(df.index + 1, np.array(max_time) / 60, label='Each Tune Time')
plt.xlabel('Trials')
plt.ylabel('Time (min)')
plt.title('Figure of Each Tuning Time')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('b.png')
plt.show()

# %%
# %%
# %%a

# %%
# print()   
# %%

# durations_df[durations_df['description']]

# %%
