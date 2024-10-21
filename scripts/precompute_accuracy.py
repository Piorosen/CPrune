#%%
import os
import torch
import torchvision as tv
import os
import argparse
from utils import test, get_data_dataset
import time
import pickle
import gc

torch.manual_seed(42)

def lazy_load():
    return  {
        "alexnet": tv.models.alexnet,
        "densenet121": tv.models.densenet121,
        "densenet161": tv.models.densenet161,
        "densenet201": tv.models.densenet201,
        "googlenet": tv.models.googlenet,
        "resnet18": tv.models.resnet18,
        "resnet34": tv.models.resnet34,
        "resnet50": tv.models.resnet50,
        "resnet101": tv.models.resnet101,
        "resnet152": tv.models.resnet152,
        "inception_v3": tv.models.inception_v3,
        "vgg11": tv.models.vgg11,
        "vgg11_bn": tv.models.vgg11_bn,
        "vgg13": tv.models.vgg13,
        "vgg13_bn": tv.models.vgg13_bn,
        "vgg16": tv.models.vgg16,
        "vgg16_bn": tv.models.vgg16_bn,
        "vgg19": tv.models.vgg19,
        "vgg19_bn": tv.models.vgg19_bn,
        "mobilenet_v2": tv.models.mobilenet_v2,
        "resnext50_32x4d": tv.models.resnext50_32x4d,
        "resnext101_32x8d": tv.models.resnext101_32x8d,
        "shufflenet_v2_x0_5": tv.models.shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0": tv.models.shufflenet_v2_x1_0,
        "squeezenet1_0": tv.models.squeezenet1_0,
        "squeezenet1_1": tv.models.squeezenet1_1
    }

def execute(model, name, device, args):
    _, val_loader, criterion = get_data_dataset('imagenet', args.dataset_directory, args.batch_size, args.test_batch_size)
    
    start = time.time()
    metric_logger = test(model, device, criterion, val_loader)
    end = time.time() - start
    with open(os.path.join(args.output, f'{name}_{end}.pkl'), 'wb') as f:
        pickle.dump([metric_logger.output, end, name, 'imagenet', device], f)
    return True

def main():
    parser = argparse.ArgumentParser(description="Read and print TORCH_HOME environment variable.")
    parser.add_argument('--torch_home', type=str, help="Path to set as TORCH_HOME", default='/work/experiments')
    parser.add_argument('--output', type=str, help="Path to set as Exporting Path", default='/work/scripts/experiments')
    parser.add_argument('--dataset_directory', type=str, help="Path to set as dataset_directory", default='/work/dataset')
    parser.add_argument('--batch_size', help="Path to set as dataset_directory", default=128, type=int)
    parser.add_argument('--test_batch_size', help="Path to set as dataset_directory", default=128, type=int)
    parser.add_argument('--model', help="Path to set as dataset_directory", default='all', type=str)
    
    # Parse the arguments
    args, unknown = parser.parse_known_args()
    print(args)
    
    os.makedirs(args.output, exist_ok=True)
    # If --torch_home is provided, set it as the TORCH_HOME environment variable
    os.environ['TORCH_HOME'] = args.torch_home
    print(os.getenv("TORCH_HOME"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = lazy_load()
    
    if args.model == 'all':
        for name in list(model_dict.keys()):
            model = model_dict[name](True,True).to(device)
            execute(model, name, device, args)
            del model
        gc.collect()
    else:
        if name in list(model_dict.keys()):
            return False
        model = model_dict[name](True,True).to(device)
        execute(model, name, device, args)
        del model
        
if __name__ == "__main__":
    main()

# %%

# %%

# %%

# %%

# %%
