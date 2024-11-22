#%%
from nni.compression.pytorch import ModelSpeedup
import torch
from tvm_test import *

model = load_mode()

dummy_input = torch.randn((1,3,32,32))
model.load_state_dict(torch.load('./001_000000_model.pth'))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_speedup = ModelSpeedup(model, dummy_input, './001_000000_mask.pth', torch.device('cpu'))
m_speedup.speedup_model()
model.eval()

torch.onnx.export(model,         # model being run 
    dummy_input,       # model input (or a tuple for multiple inputs) 
    "resnet18.onnx",       # where to save the model  
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=12,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['input0'],   # the model's input names 
    output_names = ['output0'], # the model's output names 
    ) 

# %%
