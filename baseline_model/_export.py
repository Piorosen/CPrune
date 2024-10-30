#%%
import pickle
import os 
import re
import matplotlib as plt
import numpy as np
import pandas as pd

lists = os.listdir('./')
os.getcwd()
lists = list(map(lambda x: os.path.join(os.getcwd(), x), lists))
lists = list(filter(lambda x: x[-3:] == 'pkl', lists))
lists.sort()
# with os.

# %%
totals = []
for file in lists[:]:
    parsed_data = []
    with open(file, 'rb') as f:
        data, time, model, dataset, device  = pickle.load(f)
        # print(data[0])
    # Extracting relevant data (loss, acc1, and acc5)
    for log in data[-2:-1]:
        patterns = {
            "iteration": r"\[\s*(\d+)/391\]",
            "eta": r"eta:\s([\d:]+)",
            "loss_current": r"loss:\s([\d\.]+)\s\(",
            "loss_avg": r"loss:\s[\d\.]+\s\(([\d\.]+)\)",
            "acc1_current": r"acc1:\s([\d\.]+)\s\(",
            "acc1_avg": r"acc1:\s[\d\.]+\s\(([\d\.]+)\)",
            "acc5_current": r"acc5:\s([\d\.]+)\s\(",
            "acc5_avg": r"acc5:\s[\d\.]+\s\(([\d\.]+)\)",
            "time": r"time:\s([\d\.]+)",
            "data": r"data:\s([\d\.]+)",
            "max_mem": r"max mem:\s(\d+)"
        }
        entry = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, log)
            if match:
                entry[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
        entry['model'] = model
        entry['dataset'] = dataset
        entry['device'] = 'A100'
        entry['time'] = time
        del entry['iteration']
        del entry['eta']
        del entry['loss_current']
        del entry['acc1_current']
        del entry['acc5_current']
        del entry['data']
        
        totals.append(entry)
        # Display the parsed data
        # print(entry)

df = pd.DataFrame(totals)
df = df.reindex(['model','acc1_avg','acc5_avg','loss_avg','time','max_mem','dataset','device'],axis=1)
df = df.sort_values('model')
print(df.to_markdown(index=False))
df.to_csv('./convert.csv')
df.to_dict()
# %%
