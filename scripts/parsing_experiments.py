#%%
import pickle
with open('/work/experiments/manytime_rockpi_resnet18_error/tvm/baseline.pkl', 'rb') as f:
    print(pickle.load(f))

#%%
import os

import pandas as pd
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
import cpruner
import pickle

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

dirs = os.path.join('/work/experiments/fast_resnet18_error', 'tvm')
pk_max = _get_latest_iter(dirs)
items = [_get_last_epoch(i, dirs) + '_best_op.pkl' for i in range(1, pk_max + 1)]
logs = [_get_last_epoch(i, dirs) + '.log' for i in range(1, pk_max + 1)]
perf = [_get_last_epoch(i, dirs) + '.pkl' for i in range(1, pk_max + 1)]
#%%
import numpy as np

it = [os.path.join(dirs, d) for d in items]

start_time = os.path.getctime(os.path.join(dirs, 'baseline.log'))
iter_start_time = np.array([os.path.getctime(i) for i in it])
time_data = [start_time, *iter_start_time]
t = []
for i in range(len(time_data) -1):
    t.append(time_data[i + 1] - time_data[i])
t = np.array(t)
len(t)

#%%

time_g = np.array([os.path.getctime(os.path.join(dirs, i)) for i in items])
time_e = np.array([os.path.getctime(os.path.join(dirs, i)) for i in logs])
time_train_time = time_g - time_e
#%%
# %%
itema = [os.path.join(dirs, x) for x in items]
perf_item = [os.path.join(dirs, x) for x in perf]

pick = []
for item in itema:
    with open(item, 'rb') as f:
        d = pickle.load(f)
        pick.append(d)
len(pick)

perf_list = []
for item in perf_item:
    with open(item, 'rb') as f:
        d = pickle.load(f)
        perf_list.append(d.CurrentLatency.mean())
len(perf_list)
print(perf_list)
#%%

# %%
df = pd.DataFrame(pick)
df = df.drop(columns=['masks'])
time_conv = np.vectorize(lambda x: f'{int(x // 60)}m {int(x) % 60}s')
tvm_tune = t- time_train_time
max_time = np.where(time_train_time > tvm_tune, time_train_time, tvm_tune)  + 6
df['time'] = time_conv(max_time)
df['train'] = time_conv(time_train_time)
df['tvm_tune'] = time_conv(tvm_tune)
# weight_mask = df.iloc[16]['masks']['weight_mask'] #
# num_ones = torch.sum(weight_mask == 1).item()

# total_elements = weight_mask.numel()
print(df.to_csv())
#%%

# np.max(time_train_time, t)
# %%
#%%
import matplotlib.pyplot as plt
time_index = np.array(t[:99]).cumsum() / 3600

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
# 레이아웃 및 저장
fig.tight_layout()
plt.savefig('a.png')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(df.index + 1, np.array(tvm_tune) / 60, label='TVM Tune')
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
# %%


# %%a

# %%
# print()
# %%
with open(os.path.join(dirs, 'baseline.pkl'), 'rb') as f:
    print(pickle.load(f).PruneNum)
# %%
