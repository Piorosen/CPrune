#%%
from tvm import relay, auto_scheduler
import tvm
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
import numpy as np
import os
import torch, torchvision as tv
import pandas as pd
from nni.compression.pytorch import ModelSpeedup
import time
import argparse
import json
import copy

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

def evaluate_tvm(mod, params, input_size, input_name, device_type, tvm_target, tvm_device_key, tvm_host, tvm_port, log_file):
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            if device_type == 'cpu':
                lib = relay.build_module.build(mod, params=params, target=tvm_target)
            else:
                lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=tvm_target)
    
    tmp = utils.tempdir()
    if False:
        lib_fname = tmp.relpath("net.so")
        lib.export_library(lib_fname, ndk.create_shared)
        remote = auto_scheduler.utils.request_remote(tvm_device_key, tvm_host, tvm_port, timeout=200)
        remote.upload(lib_fname)
        rlib = remote.load_module("net.so")
    else:
        lib_fname = tmp.relpath("net.tar")
        lib.export_library(lib_fname)
        remote = auto_scheduler.utils.request_remote(tvm_device_key, tvm_host, tvm_port, timeout=200)
        remote.upload(lib_fname)
        rlib = remote.load_module("net.tar")

    # Create graph executor
    if device_type == 'cpu':
        ctx = remote.cpu()
    else:
        ctx = remote.cl(0)
    module = graph_executor.GraphModule(rlib["default"](ctx))

    data_tvm = tvm.nd.array((np.random.uniform(size=input_size)).astype('float32'))
    module.set_input(input_name, data_tvm)
    ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=10)
    prof_res = np.array(ftimer().results) * 1e3
    current_latency = np.mean(prof_res)
    
    return prof_res
#%%
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
#%%

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="TVM model deployment script")
    
    # Add arguments for each of the provided variables
    parser.add_argument("--name", type=str, default='resnet18', help="Name of the model")
    parser.add_argument("--model_pth", type=str, default='/work/experiments/imagenet_resnet18/tvm/001_000002_model.pth', help="Path to the model .pth file")
    parser.add_argument("--mask_pth", type=str, default='/work/experiments/imagenet_resnet18/tvm/001_000002_mask.pth', help="Path to the mask .pth file")
    parser.add_argument("--tvm_log", type=str, default='/work/experiments/imagenet_resnet18/tvm/001_000002.log', help="Path to the TVM log file")
    parser.add_argument("--tvm_target", type=str, default="llvm -mtriple=aarch64-linux-none", help="TVM target configuration")
    parser.add_argument("--tvm_devicekey", type=str, default='rockpi', help="TVM device key")
    parser.add_argument("--tvm_host", type=str, default='127.0.0.1', help="TVM tracker host address")
    parser.add_argument("--tvm_port", type=int, default=9190, help="TVM tracker port")

    args, _ = parser.parse_known_args()
    
    device = torch.device('cpu')
    
    
    
    dirs = os.path.join('/work/experiments/imagenet_resnet18', 'tvm')
    pk_max = _get_latest_iter(dirs)
    items = [_get_last_epoch(i, dirs) for i in range(1, pk_max + 1)]
    rrr = []
    for item in items:
        o_itme = item
        item = os.path.join(dirs, item)
        args.model_pth = item + '_model_train.pth'
        args.mask_pth = item + '_mask.pth'
        args.tvm_log = item + '.log'
        
        model, _ =  get_model_zoo(args.name)
        model = copy.deepcopy(model)
        info = get_files(args.name)
        dummy_input = torch.randn([4, 3, 224, 224])
        model.to(device)
        model.eval()

        model.load_state_dict(torch.load(args.model_pth))
        m_speedup = ModelSpeedup(model, dummy_input, args.mask_pth, device)
        m_speedup.speedup_model()

        # dummy_input = torch.randn((1,3,224,224))
        # torch.onnx.export(model,         # model being run 
        #     dummy_input,       # model input (or a tuple for multiple inputs) 
        #     "resnet18.onnx",       # where to save the model  
        #     export_params=True,  # store the trained parameter weights inside the model file 
        #     opset_version=12,    # the ONNX version to export the model to 
        #     do_constant_folding=True,  # whether to execute constant folding for optimization 
        #     input_names = ['input0'],   # the model's input names 
        #     output_names = ['output0'], # the model's output names 
        #     ) 
        input_shape = [1,3,224,224]
        dummy_input = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, dummy_input).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        relay.transform.InferType(),
                                        relay.transform.FoldConstant(),
                                        relay.transform.DeadCodeElimination()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
            
        result = evaluate_tvm(mod, params, input_shape, input_name, 'cpu', args.tvm_target, args.tvm_devicekey, args.tvm_host, args.tvm_port, args.tvm_log)
        with open(f'{o_itme}.json', 'w') as f:
            json.dump({'inference': result.tolist(), 'info': info, 'param': vars(args)}, f)
    print(rrr)
#%%
if __name__ == '__main__':
    main()

# %%

#%%
