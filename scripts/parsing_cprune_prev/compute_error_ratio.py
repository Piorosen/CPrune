#%%
import pickle as pkl
import numpy as np
import os
import pandas as pd
# %%
l =  sorted(list(filter(lambda x: x[-3:] == 'pkl', os.listdir())))
l
#%%
fourth = 0
total = 0
results = []
output_df = {'model': [], 'name':[], 'size': [], 'fourth': [], 'ratio': [], 'total': []}

for i in l:
    print(i)
    with open(i, 'rb+') as f:
        data = pkl.load(f)
        keys = list(data)
        keys =  sorted(list(filter(lambda x: not 'gpu' in x.lower(), keys)))
    
    result = [] 
    for k in keys:
        print(k)
        config_list = data[k]['config_list']
        size = len(list(data[k]['config_list'].keys()))
        base = np.array(config_list[0]['base'])
        tmp_fourth = 0
        tmp_total = 0
        prev = base
        for conf_index in range(1, size):
            channel = np.array(config_list[conf_index]['fully'])
            diff_index = np.arange(len(channel))[channel != prev]
            p = base[diff_index] - channel[diff_index]
            tmp_fourth += len(p[p % 4 == 0])
            tmp_total += len(p)
            prev = np.array(config_list[conf_index]['fully'])
            # if len(p[p % 4 == 0]) != len(p):
                # print('\t\t', fourth, total, p, diff_index, base[diff_index], channel[diff_index])
                # print(fourth, total)
        fourth += tmp_fourth
        total += tmp_total
        n = i.split('.')[0]
        if tmp_total != 0:
            output_df['model'].append(n)
            output_df['name'].append(f'{k}')
            output_df['size'].append(size)
            output_df['fourth'].append(tmp_fourth)
            output_df['total'].append(tmp_total)
            output_df['ratio'].append(tmp_fourth / tmp_total)
            print(tmp_fourth, tmp_total, '(', tmp_fourth / tmp_total * 100 ,')')
    # break
print('Total', fourth, total, '(', fourth / total * 100 ,')')
df = pd.DataFrame(output_df)
print(df.to_markdown())
# %%
