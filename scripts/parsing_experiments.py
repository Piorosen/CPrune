#%%
import os
import pickle
import pandas as pd
import torch

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
    return '_'.join(epoch) + '_best_op.pkl'

dirs = os.path.join('/work/experiments/imagenet_resnet18', 'tvm')
pk_max = _get_latest_iter(dirs)
items = [_get_last_epoch(i, dirs) for i in range(1, pk_max + 1)]

# %%
itema = [os.path.join(dirs, x) for x in items]

pick = []
for item in itema:
    with open(item, 'rb') as f:
        d = pickle.load(f)
        pick.append(d)
# %%
df = pd.DataFrame(pick)
df = df.drop(columns=['masks'])
# weight_mask = df.iloc[16]['masks']['weight_mask'] #
# num_ones = torch.sum(weight_mask == 1).item()

# # Total number of elements
# total_elements = weight_mask.numel()

# # Count the number of 0s
# num_zeros = total_elements - num_ones
 
# %%
print(df.to_markdown())
# %%
