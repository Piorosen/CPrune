#%%
import tvm
import os
import torch, torchvision as tv
import pandas as pd
from nni.compression.pytorch import ModelSpeedup
import time
import argparse
import json
from utils import test, get_data_dataset
import pickle

# 1. Load Model
# 2. History Apply
# 3. Run!
# with Parameters.

def get_model_zoo(model):
    checkpoint_dir = os.path.join(os.getenv("TORCH_HOME"), 'hub', 'checkpoints')
    model_dict = {
        "alexnet": [tv.models.alexnet, os.path.join(checkpoint_dir, 'alexnet-owt-4df8aa71.pth')],
        "densenet121": [tv.models.densenet121, os.path.join(checkpoint_dir, 'densenet121-a639ec97.pth')],
        "densenet161": [tv.models.densenet161, os.path.join(checkpoint_dir, 'densenet161-8d451a50.pth')],
        "densenet201": [tv.models.densenet201, os.path.join(checkpoint_dir, 'densenet201-c1103571.pth')],
        "googlenet": [tv.models.googlenet, os.path.join(checkpoint_dir, 'googlenet-1378be20.pth')],
        "inception_v3": [tv.models.inception_v3, os.path.join(checkpoint_dir, 'inception_v3_google-1a9a5a14.pth')],
        "mobilenet_v2": [tv.models.mobilenet_v2, os.path.join(checkpoint_dir, 'mobilenet_v2-b0353104.pth')],
        "resnet18": [tv.models.resnet18, os.path.join(checkpoint_dir, 'resnet18-5c106cde.pth')],
        "resnet34": [tv.models.resnet34, os.path.join(checkpoint_dir, 'resnet34-333f7ec4.pth')],
        "resnet50": [tv.models.resnet50, os.path.join(checkpoint_dir, 'resnet50-19c8e357.pth')],
        "resnet101": [tv.models.resnet101, os.path.join(checkpoint_dir, 'resnet101-5d3b4d8f.pth')],
        "resnet152": [tv.models.resnet152, os.path.join(checkpoint_dir, 'resnet152-b121ed2d.pth')],
        "resnext50_32x4d": [tv.models.resnext50_32x4d, os.path.join(checkpoint_dir, 'resnext50_32x4d-7cdf4587.pth')],
        "resnext101_32x8d": [tv.models.resnext101_32x8d, os.path.join(checkpoint_dir, 'resnext101_32x8d-8ba56ff5.pth')],
        "shufflenet_v2_x0_5": [tv.models.shufflenet_v2_x0_5, os.path.join(checkpoint_dir, 'shufflenetv2_x0.5-F707e7162e.pth')],
        "shufflenet_v2_x1_0": [tv.models.shufflenet_v2_x1_0, os.path.join(checkpoint_dir, 'shufflenetv2_x1-5666bf0f80.pth')],
        "squeezenet1_0": [tv.models.squeezenet1_0, os.path.join(checkpoint_dir, 'squeezenet1_0-a815701f.pth')],
        "squeezenet1_1": [tv.models.squeezenet1_1, os.path.join(checkpoint_dir, 'squeezenet1_1-f364aa15.pth')],
        "vgg11": [tv.models.vgg11, os.path.join(checkpoint_dir, 'vgg11-bbd30ac9.pth')],
        "vgg11_bn": [tv.models.vgg11_bn, os.path.join(checkpoint_dir, 'vgg11_bn-6002323d.pth')],
        "vgg13": [tv.models.vgg13, os.path.join(checkpoint_dir, 'vgg13-c768596a.pth')],
        "vgg13_bn": [tv.models.vgg13_bn, os.path.join(checkpoint_dir, 'vgg13_bn-abd245e5.pth')],
        "vgg16": [tv.models.vgg16, os.path.join(checkpoint_dir, 'vgg16-397923af.pth')],
        "vgg16_bn": [tv.models.vgg16_bn, os.path.join(checkpoint_dir, 'vgg16_bn-6c64b313.pth')],
        "vgg19": [tv.models.vgg19, os.path.join(checkpoint_dir, 'vgg19-bcbb9e9d.pth')],
        "vgg19_bn": [tv.models.vgg19_bn, os.path.join(checkpoint_dir, 'vgg19_bn-c79401a0.pth')],
    }
    m, pth = model_dict[model]
    return m(True, True), pth 

def get_files(model):
    df = pd.read_csv('baseline.csv')
    data = df[df['model'] == model].to_dict()
    cleaned_data = {key: list(value.values())[0] for key, value in data.items()}
    cleaned_data.pop("Unnamed: 0", None)
    return cleaned_data

def execute(model, name, device, args):
    _, val_loader, criterion = get_data_dataset('imagenet', args.dataset_directory, args.batch_size, args.test_batch_size)
    
    start = time.time()
    metric_logger = test(model, device, criterion, val_loader)
    end = time.time() - start
    with open(os.path.join(args.output, f'{name}_{end}.pkl'), 'wb') as f:
        pickle.dump([metric_logger.output, end, name, 'imagenet', device], f)
    return True


#%%

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="TVM model deployment script")
    
    # Add arguments for each of the provided variables
    parser.add_argument("--name", type=str, default='resnet18', help="Name of the model")
    parser.add_argument("--model_pth", type=str, default='/work/experiments/imagenet_resnet18/tvm/003_000000_model.pth', help="Path to the model .pth file")
    parser.add_argument("--mask_pth", type=str, default='/work/experiments/imagenet_resnet18/tvm/003_000000_mask.pth', help="Path to the mask .pth file")
    parser.add_argument('--dataset_directory', type=str, help="Path to set as dataset_directory", default='/work/dataset')
    parser.add_argument('--batch_size', help="Path to set as dataset_directory", default=128, type=int)
    parser.add_argument('--test_batch_size', help="Path to set as dataset_directory", default=128, type=int)
    parser.add_argument('--output', type=str, help="Path to set as Exporting Path", default='/work/scripts/experiments')

    args, _ = parser.parse_known_args()
    
    model, _ = get_model_zoo(args.name)
    info = get_files(args.name)
    dummy_input = torch.randn([4, 3, 224, 224])

    if not (args.model_pth == '' or args.mask_pth == ''):
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(args.model_pth))
        m_speedup = ModelSpeedup(model, dummy_input, args.mask_pth, device)
        m_speedup.speedup_model()

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    input_shape = [1,3,224,224]
    execute(model, args.name, device, args)
    
if __name__ == '__main__':
    main()

# %%
