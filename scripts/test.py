#%%
# from nnia.compression.pytorch import ModelSpeedup
import os
import torchvision as tv
from nnia.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
import copy
import torch
from nnia.compression.pytorch.utils.counter import count_flops_params
from models.implements.cnn.mnist.lenet import LeNet 
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
}
# config_list = [{'sparsity': 0.0625, 'op_types': ['Conv2d','bias'], 'op_names': ['layer4.1.conv2']}, {'sparsity': 0.0625, 'op_types': ['Conv2d','bias'], 'op_names': ['layer4.0.conv2']}]
# config_list = [{'sparsity': 0.125, 'op_types': ['Conv2d','ReLU','BatchNorm2d'], 'op_names': ['layer4.1.conv2','layer4.1.relu','layer4.1.bn']}, 
#                {'sparsity': 0.125, 'op_types': ['Conv2d','ReLU','BatchNorm2d'], 'op_names': ['layer4.0.conv2']},
#                ]
config_list = [
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer3.1.conv2"
        ],
        "total_sparsity": 0.2866829931972791
    },
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer4.0.conv1"
        ],
        "total_sparsity": 0.2818956307355158
    },
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer4.0.conv2"
        ],
        "total_sparsity": 0.2774719898570899
    },
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer4.0.shortcut.0"
        ],
        "total_sparsity": 0.7802925030381763
    },
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer4.1.conv1"
        ],
        "total_sparsity": 0.2822888432580426
    },
    {
        "op_types": [
            "Conv2d"
        ],
        "op_names": [
            "layer4.1.conv2"
        ],
        "total_sparsity": 0.2822888432580426
    }
]

# [
#         #         {'total_sparsity': 0.75, 'op_types': ['Conv2d'], 'op_names': ['layer4.1.conv2']}, 
#         #         {'total_sparsity': 0.75, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.conv2']}, 
#         # #        {'sparsity': 0.5, 'op_types': ['Conv2d'], 'op_names': ['layer4.0.conv2']},
#         #        {'sparsity': 0.125, 'op_types': ['Conv2d']}
#                ]

#%%
# nni.algorithms.compression.pytorch.pruning
from nni.algorithms.compression.v2.pytorch.pruning import L2NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup

dummy_input = torch.randn((1,3,224,224))

device = torch.device('cpu')
model, pth = model_dict['resnet18']
model = model(True,True).to(device).eval()
pruner = L2NormPruner(model=model, config_list=config_list, mode='dependency_aware', dummy_input=torch.rand(10, 3, 224, 224).to(device))
compact_model, pruner_generated_masks = pruner.compress()
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


