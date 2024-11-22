#%%
import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))
import torch
torch.manual_seed(42)
from models.implements.cnn.cifar10.resnet import ResNet18
from tvm import relay, auto_scheduler
import tvm

# def optimizing_all(data: OptimizerTVMInput, load_log=None, at_least_trials = 500, num_per_round = 500, runner_number = 10, runner_repeat = 2, timeout=200, task_index=None, previous_file=None) -> OptimizerTVMOutput:
#%%
device = torch.device("cpu")
model = ResNet18()
model.load_state_dict(torch.load('./experiments/cifar10_model_300.pth'))
model.to(device)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%
# log_file = "%s.log" % ('long_cifar_resnet_rockpi')
log_file = "%s.log" % ('long_cifar_resnet_sd865')
dummy_input = torch.randn([1,3,32,32])
scripted_model = torch.jit.trace(model, dummy_input).eval()
input_name = "input0"
shape_list = [(input_name, [1,3,32,32])]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.ConvertLayout(desired_layouts),
                                relay.transform.InferType(),
                                relay.transform.FoldConstant(),
                                relay.transform.DeadCodeElimination()])
with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

#%%    
print("Extract tasks...")
# tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-android')

at_least_trials = 500
num_per_round= 60
tune_trials = 18000 #(conv2d_num + others_num)        

tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_trials,
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if True else "default"),
        runner=auto_scheduler.RPCRunner('ndroid', 
                                        host='127.0.0.1', 
                                        port=9190, 
                                        timeout=200, 
                                        number=10, 
                                        repeat=5,),
        
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
        #early_stopping=300,
        num_measures_per_round = num_per_round,
    )

start = time.time()
tuner.tune(tune_option)
end = time.time()
print(end - start)
with open('model_tune_tvm_sd865.txt', 'wt+') as f:
    f.write(f'{end-start} s\n')

# output2 = optimizer_tvm.optimizing_all(input2, tune_name, task_index=None, previous_file=prev_tune_name)
# %%
