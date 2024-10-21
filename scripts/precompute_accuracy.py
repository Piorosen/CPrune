#%%
import os
import torch
import torchvision as tv
import os
import argparse
from utils import test, get_data_dataset
import time
import pickle

model_dict = {
    "alexnet": tv.models.alexnet(True, True),
    "densenet121": tv.models.densenet121(True, True),
    "densenet161": tv.models.densenet161(True, True),
    "densenet201": tv.models.densenet201(True, True),
    "googlenet": tv.models.googlenet(True, True),
    "resnet18": tv.models.resnet18(True, True),
    "resnet34": tv.models.resnet34(True, True),
    "resnet50": tv.models.resnet50(True, True),
    "resnet101": tv.models.resnet101(True, True),
    "resnet152": tv.models.resnet152(True, True),
    "inception_v3": tv.models.inception_v3(True, True),
    "vgg11": tv.models.vgg11(True, True),
    "vgg11_bn": tv.models.vgg11_bn(True, True),
    "vgg13": tv.models.vgg13(True, True),
    "vgg13_bn": tv.models.vgg13_bn(True, True),
    "vgg16": tv.models.vgg16(True, True),
    "vgg16_bn": tv.models.vgg16_bn(True, True),
    "vgg19": tv.models.vgg19(True, True),
    "vgg19_bn": tv.models.vgg19_bn(True, True),
    "mobilenet_v2": tv.models.mobilenet_v2(True, True),
    "resnext50_32x4d": tv.models.resnext50_32x4d(True, True),
    "resnext101_32x8d": tv.models.resnext101_32x8d(True, True),
    "shufflenet_v2_x0_5": tv.models.shufflenet_v2_x0_5(True, True),
    "shufflenet_v2_x1_0": tv.models.shufflenet_v2_x1_0(True, True),
    "squeezenet1_0": tv.models.squeezenet1_0(True, True),
    "squeezenet1_1": tv.models.squeezenet1_1(True, True)
}

def execute(name, device, args):
    if name in list(model_dict.keys()):
        return False
    
    model = model_dict[name].to(device)
    _, val_loader, criterion = get_data_dataset('imagenet', args.dataset_directory, args.batch_size, args.test_batch_size)
    
    start = time.time()
    metric_logger = test(model, device, criterion, val_loader)
    end = time.time() - start
    with open(f'{name}_{end}', 'wb') as f:
        pickle.dump([metric_logger, end, name, 'imagenet', device])
    return True

def main():
    parser = argparse.ArgumentParser(description="Read and print TORCH_HOME environment variable.")
    parser.add_argument('--torch_home', type=str, help="Path to set as TORCH_HOME", default='/work/experiments')
    parser.add_argument('--output', type=str, help="Path to set as Exporting Path", default='/work/scripts/experiments')
    parser.add_argument('--dataset_directory', type=str, help="Path to set as dataset_directory", default='/work/dataset')
    parser.add_argument('--batch_size', help="Path to set as dataset_directory", default=512, type=int)
    parser.add_argument('--test_batch_size', help="Path to set as dataset_directory", default=512, type=int)
    parser.add_argument('--model', help="Path to set as dataset_directory", default='all', type=str)
    
    # Parse the arguments
    args, unknown = parser.parse_known_args()

    os.makedirs(args.output, exist_ok=True)
    # If --torch_home is provided, set it as the TORCH_HOME environment variable
    os.environ['TORCH_HOME'] = args.torch_home

    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == 'all':
        for name in list(model_dict.keys()):
            execute(name, device, args)
    else:
        execute(args.model, device, args)
    
if __name__ == "__main__":
    main()

# %%

# %%

# %%

# %%
