#%%
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
from c_pruner import CPruner
sys.path.pop(0)
from types import SimpleNamespace
from dotenv import load_dotenv
import copy
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from tvm.auto_scheduler import dispatcher
import os
import torch.utils.data
import torchvision.models as models
import pickle
import random
import string
from parsing_experiments import get_data
from utils import train, test_top1, get_data_dataset
from nni.compression.pytorch import ModelSpeedup
from resnet import ResNet18



def generate_random_string(length=16):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

train_loader, val_loader, criterion = get_data_dataset('cifar10', '/work/dataset', 512, 512)

def short_term_trainer(model, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    args = SimpleNamespace(log_interval=1000)
    for e in range(epochs):
        train(args, model, device, train_loader, criterion, optimizer, e)
    model = model.to(torch.device('cpu'))
    
def evaluator_top1(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result, _ = test_top1(model, device, criterion, val_loader)
    model = model.to(torch.device('cpu'))
    return result, result

dispatcher.DATA_FILE_NAME = f'/work/tmp_get_error_from_tvm_{generate_random_string()}.txt'
# data = get_data('/work/experiments/manytime_rockpi_resnet18_error_inf')

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
def accuracy_convert(dir, export, base_pth = 'cifar10_resnet18_94.7.pth'):
    project = export
    if not os.path.exists(project):
        os.mkdir(project)
    # project = 'manytime_rockpi_resnet18_error_inf'
    # dir = '/work/experiments/manytime_rockpi_resnet18_error_inf'
    max_it = _get_latest_iter(dir)
    epoch = [_get_last_epoch(dir, value) for value in range(max_it)]
    epoch = [convert(dir, f) for f in epoch]

    predict_acc = [(0.9475, 0.9475)]
    model = ResNet18().to('cpu')
    model.load_state_dict(torch.load(base_pth, map_location='cpu')['net'])

    with open(os.path.join(project, 'result.txt'), 'wt+') as f:
        f.write(f'{0:04},{predict_acc[0]}\n')
    

    origin_model = copy.deepcopy(model)
    for index in range(1, len(epoch)):
        e = epoch[index]
        dummy_input = torch.randn([1,3,32,32])
        pruner = PRUNER_DICT['l1'](copy.deepcopy(origin_model), e, dependency_aware=True, dummy_input=dummy_input)
        model_masked = pruner.compress()
        pruner.export_model(os.path.join(project, f'tmp.pth'), 
                            os.path.join(project, f'{index:04}_mask.pth'))
        
        optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        short_term_trainer(model_masked, optimizer, epochs=5)
        acc = evaluator_top1(model_masked)
        predict_acc.append(acc)
        # model.load_state_dict(model_masked.state_dict())
        pruner.export_model(os.path.join(project, f'{index:04}_model.pth'), 
                            os.path.join(project, f'tmp.pth'))
        # m_speedup = ModelSpeedup(origin_model, dummy_input, os.path.join(project, f'{index:04}_mask.pth'), torch.device('cpu'))
        
        origin_model.load_state_dict(torch.load(os.path.join(project, f'{index:04}_model.pth')))
        with open(os.path.join(project, 'result.txt'), 'at+') as f:
            f.write(f'{index:04},{acc}\n')
#%%
file_list = [#'manytime_rockpi_resnet18_all_none',
             #'manytime_rockpi_resnet18_error_early300', 
             #'manytime_rockpi_resnet18_error_inf',
             #'manytime_sd865_resnet18_all_earlystop_50',
             #'manytime_sd865_resnet18_all_none',
             'manytime_sd865-1_resnet18_error_early300',
             'manytime_sd865-3_resnet18_error_early10000000.0']

for item in file_list:
    accuracy_convert(f'/work/experiments/{item}',
                     f'./{item}')
# %%
