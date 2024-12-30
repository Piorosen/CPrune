#%%
from nni.compression.pytorch import ModelSpeedup
import os
import torchvision as tv
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
import copy
import torch
from nni.compression.pytorch.utils.counter import count_flops_params

#%%
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
from models.implements.cnn.cifar10.resnet import ResNet18

checkpoint_dir = os.path.join(os.getenv("TORCH_HOME"), 'hub', 'checkpoints')

config_list = [
    {"op_types": [
            "Conv2d"
        ], "op_names": [
            "layer4.1.conv1"
        ],
        "sparsity": 0.2822888432580426},
    # {'sparsity': 0.703125, 'op_types': ['Conv2d'], 'op_names': ['layer4.1.conv2']}, 
    # {'sparsity': 0.703125, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.conv2']}, 
    # {'sparsity': 0.15625*4, 'op_types': ['Conv2d'], 'op_names': ['layer1.0.conv2']}, 
    # {'sparsity': 0.15625*4, 'op_types': ['Conv2d'], 'op_names': ['layer1.1.conv2']}
    ]

# config_list = [
#     {
#         "op_types": [
#             "Conv2d"
#         ],
#         "op_names": [
#             "layer3.1.conv2"
#         ],
#         "sparsity": 0.2866829931972791
#     },
#     {
#         "op_types": [
#             "Conv2d"
#         ],
#         "op_names": [
#             "layer4.0.conv1"
#         ],
#         "sparsity": 0.2818956307355158
#     },
#     {
#         "op_types": [
#             "Conv2d"
#         ],
#         "op_names": [
#             "layer4.0.conv2"
#         ],
#         "sparsity": 0.2774719898570899
#     },
#     {
#         "op_types": [
#             "Conv2d"
#         ],
#         "op_names": [
#             "layer4.1.conv1"
#         ],
#         "sparsity": 0.2822888432580426
#     },
#     {
#         "op_types": [
#             "Conv2d"
#         ],
#         "op_names": [
#             "layer4.1.conv2"
#         ],
#         "sparsity": 0.2822888432580426
#     }
# ]
#%%
# nni.algorithms.compression.pytorch.pruning
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.speedup import ModelSpeedup

dummy_input = torch.randn((1,3,32,32))

device = torch.device('cpu')
model = ResNet18()
model.load_state_dict(torch.load('/work/scripts/experiments/cifar10_model_300.pth'))
model = model.to(device).eval()

pruner = PRUNER_DICT['l1'](copy.deepcopy(model), config_list=target_config_list, dependency_aware=True, dummy_input=torch.rand(1, 3, 32, 32).to(device))
prunemodel = pruner.compress()
pruner.export_model('model.pth', 'mask.pth')
model.load_state_dict(torch.load('model.pth'))
m_speedup = ModelSpeedup(model, dummy_input, 'mask.pth', device)
m_speedup.speedup_model()
model.eval()
#%%
os.remove('model.pth')
os.remove('mask.pth')
#%%
model
#%%
from tvm import relay, auto_scheduler
import tvm
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor

device = torch.device('cpu')
model = ResNet18()
model.load_state_dict(torch.load('/work/scripts/experiments/cifar10_model_300.pth'))
model = model.to(device).eval()

input_shape = [1,3,32,32]
# dummy_input2 = torch.randn(input_shape)
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
    
t = mod['main'].astext(show_meta_data=False)

#%%
device = torch.device('cpu')
model = ResNet18()
model.load_state_dict(torch.load('/work/scripts/experiments/cifar10_model_300.pth'))
model = model.to(device).eval()

pruner = PRUNER_DICT['l1'](copy.deepcopy(model), config_list=config_list, dependency_aware=True, dummy_input=torch.rand(1, 3, 32, 32).to(device))
prunemodel = pruner.compress()
pruner.export_model('model.pth', 'mask.pth')
model.load_state_dict(torch.load('model.pth'))
m_speedup = ModelSpeedup(model, dummy_input, 'mask.pth', device)
m_speedup.speedup_model()
model.eval()
os.remove('model.pth')
os.remove('mask.pth')
input_shape = [1,3,32,32]
# dummy_input2 = torch.randn(input_shape)
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
t2 = mod['main'].astext(show_meta_data=False)
#%%

d = tv.models.resnet18(pretrained=False)
print(count_flops_params(d, tuple(input_shape)))

#%%
print(t == t2)
#%%
import difflib

# 두 텍스트를 줄 단위로 나눕니다.
text1_lines = t.splitlines()
text2_lines = t2.splitlines()

# Differ 객체를 생성하여 차이점을 계산합니다.
d = difflib.Differ()
diff = d.compare(text1_lines, text2_lines)

# 결과를 출력합니다.
print('\n'.join(diff))

#%%

# 1. Speed UP 
# 2. TIR / Relay IR Lower Compare
# 3. 
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')

#%%
build_mod = relay.build_module.BuildModule()

with auto_scheduler.ApplyHistoryBest('long_cifar_rock.log'):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = build_mod.build(mod, params=params, target='llvm -mtriple=aarch64-linux-none')
#%%
# print(lib.ir_mod.astext)
# print(lib.get_graph_json())
# print(build_mod.get_function_metadata())
# print(build_mod.get_module())
sch, args = tasks[5].apply_best('long_cifar_rock.log')
tir_module = tvm.lower(sch, args, simple_mode=True)
print(tir_module)

#%%
mod
#%%


import torch.onnx
dummy_input = torch.randn((1,3,32,32))
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "resnet18_pr2une.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=12,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input0'],   # the model's input names 
        output_names = ['output0'], # the model's output names 
        ) 













#%%
pruner.show_pruned_weights()
pruner._unwrap_model()
ModelSpeedup(compact_model, dummy_input, pruner_generated_masks).speedup_model()
# torch.save(compact_model, 'model.pth')
# torch.save(pruner_generated_masks, 'mask.pth')
print(compact_model)
# compact_model.export_model('model.pth', 'mask.pth')
#%%
compact_model
#%%
model, pth = model_dict['resnet18']
model = model(True,True)
_, _, temp_results = count_flops_params(model, (1,3,224,224))
dummy_input = torch.randn((1,3,224,224))
pruner = PRUNER_DICT['l1'](copy.deepcopy(model), config_list, dependency_aware=True, dummy_input=dummy_input)
masked = pruner.compress()
pruner.export_model('model.pth', 'mask.pth')
model.load_state_dict(torch.load('model.pth'))
m_speedup = ModelSpeedup(model, dummy_input, 'mask.pth', torch.device('cpu'))
m_speedup.speedup_model()
model.eval()
model = model.to(torch.device('cpu')).eval()

# %%
# model.load_state_dict(torch.load('/work/experiments/imagenet_resnet18/tvm/003_000000_model.pth'))
# m_speedup = ModelSpeedup(model, dummy_input, '/work/experiments/imagenet_resnet18/tvm/003_000000_mask.pth', torch.device('cpu'))
# m_speedup.speedup_model()
model
# _, _, temp_results = count_flops_params(model, (1,3,224,224))

#%%
# model, pth = model_dict['alexnet']
# model = model(True,True)
dummy_input = torch.randn((1,3,224,224))
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "resnet18_pr2une.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=12,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input0'],   # the model's input names 
        output_names = ['output0'], # the model's output names 
        ) 

# for w in wrapper:
        # if w.name == wrapper.name:
        # masks = {'weight_mask': w.weight_mask,
                #  'bias_mask': w.bias_mask}
        # break
# #%%
# config_list = [
#                 {'sparsity': 0.5, 'op_types': ['Conv2d'], 'op_names': ['conv1']}, 
#                ]

# model = LeNet()
# _, _, temp_results = count_flops_params(model, (1,1,28,28))
# dummy_input = torch.randn((1,1,28,28))
# pruner = PRUNER_DICT['l1'](copy.deepcopy(model), config_list, dependency_aware=True, dummy_input=dummy_input)
# masked = pruner.compress()
# pruner.export_model('model.pth', 'mask.pth')
# model.load_state_dict(torch.load('model.pth'))
# m_speedup = ModelSpeedup(model, dummy_input, 'mask.pth', torch.device('cpu'))
# m_speedup.speedup_model()
# model.eval()
# model = model.to(torch.device('cpu')).eval()

# # %%
# # model.load_state_dict(torch.load('/work/experiments/imagenet_resnet18/tvm/003_000000_model.pth'))
# # m_speedup = ModelSpeedup(model, dummy_input, '/work/experiments/imagenet_resnet18/tvm/003_000000_mask.pth', torch.device('cpu'))
# m_speedup.speedup_model()

# %%


# %%
import torch
from utils import *
from nnia.compression.pytorch import ModelSpeedup

input_size = get_input_size('imagenet')
dummy_input = get_dummy_input(input_size, 1)
model.load_state_dict(torch.load('/work/experiments/imagenet_resnet18/tvm/001_000000_model.pth'))
masks_file = '/work/experiments/imagenet_resnet18/tvm/001_000000_mask.pth'
m_speedup = ModelSpeedup(model, dummy_input, masks_file, torch.device('cpu'))
m_speedup.speedup_model()

torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "resnet18_.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=12,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input0'],   # the model's input names 
        output_names = ['output0'], # the model's output names 
        ) 
# %%


from tvm.relay import backend
backend.lower
