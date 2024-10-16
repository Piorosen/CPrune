import os
from tvm import relay, auto_scheduler
import torch
import tvm
import numpy as np
import time
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
from .logs import Logger
from .structure import *

logger = Logger().get_logger()

def evaluate_tvm(mod, params, input_name, data: OptimizerTVMInput, log_file):
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            if data.DeviceType == DeviceType.CPU:
                lib = relay.build_module.build(mod, params=params, target=data.TVM_Target)
            else:
                lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=data.TVM_Target)
    
    tmp = utils.tempdir()
    if data.UseAndroid:
        lib_fname = tmp.relpath("net.so")
        lib.export_library(lib_fname, ndk.create_shared)
        remote = auto_scheduler.utils.request_remote(data.TVM_DeviceKey, data.TVM_TrackerHost, data.TVM_TrackerPort, timeout=200)
        remote.upload(lib_fname)
        rlib = remote.load_module("net.so")
    else:
        lib_fname = tmp.relpath("net.tar")
        lib.export_library(lib_fname)
        remote = auto_scheduler.utils.request_remote(data.TVM_DeviceKey, data.TVM_TrackerHost, data.TVM_TrackerPort, timeout=200)
        remote.upload(lib_fname)
        rlib = remote.load_module("net.tar")

    # Create graph executor
    if data.DeviceType == DeviceType.CPU:
        ctx = remote.cpu()
    else:
        ctx = remote.cl(0)
    module = graph_executor.GraphModule(rlib["default"](ctx))

    data_tvm = tvm.nd.array((np.random.uniform(size=data.InputSize)).astype(data.TVM_DataType))
    module.set_input(input_name, data_tvm)
    ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=2)
    prof_res = np.array(ftimer().results) * 1e3
    current_latency = np.mean(prof_res)
    logger.info('ftimer_latency: ' + str(current_latency))
    
    return prof_res

def optimizing(data: OptimizerTVMInput) -> OptimizerTVMOutput:
    _acc_requirement = None
    data.TVM_DeviceKey = os.getenv("ID_OPTIMIZATION_HARDWARE")
    data.TVM_TrackerHost = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    data.TVM_TrackerPort = int(os.environ["TVM_TRACKER_PORT"])
    
    log_file = "%s.log" % (data.TVM_DeviceKey)
    
    scripted_model = torch.jit.trace(data.Model, data.InputData).eval()
    input_name = "input0"
    shape_list = [(input_name, data.InputSize)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts),
                                    relay.transform.InferType(),
                                    relay.transform.FoldConstant(),
                                    relay.transform.DeadCodeElimination()])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    
    #################### Extract search tasks ###################
    print("Extract tasks...")
    if data.DeviceType == DeviceType.CPU:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, data.TVM_Target)
    else:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl -device=mali", target_host=data.TVM_Target)

    subgraph_tasks = [-1 for i in range(data.Subgraph.NumConv2d)]
    task_times = [-1 for i in range(data.Subgraph.NumConv2d)]
    pos_idx = 0
    downsample_idx = 0
    for idx, task in enumerate(tasks):
        if idx < data.Subgraph.NumOthers:
            continue
        if len(task.workload_key) < 80:
            continue
        for i in range(task_weights[idx]):
            subgraph_tasks[data.Subgraph.Pos[pos_idx]] = idx
            pos_idx += 1

    
    pruning_times = [0.0 for i in range(data.Subgraph.NumConv2d)]
    real_pruning_times = [0.0 for i in range(data.Subgraph.NumConv2d2d_num)]
    
    at_least_trials = 20
    num_per_round = 60
    runner_number = 10 # 10
    runner_repeat = 2  # 2
    tune_trials = (at_least_trials + num_per_round) * len(tasks) #(conv2d_num + others_num)        
    
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tune_trials,
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if data.UseAndroid else "default"),
        runner=auto_scheduler.RPCRunner(data.TVM_DeviceKey, 
                                        host=data.TVM_TrackerHost, 
                                        port=data.TVM_TrackerPort, 
                                        timeout=200, 
                                        number=runner_number, 
                                        repeat=runner_repeat,),
        
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
        #early_stopping=300,
        num_measures_per_round = num_per_round,
    )
    tuner.tune(tune_option)        
    total_estimated_latency = 0
    for i in range(data.Subgraph.NumConv2d):
        task_times[i] = tuner.best_costs[subgraph_tasks[i]] * task_weights[subgraph_tasks[i]]
        total_estimated_latency += tuner.best_costs[subgraph_tasks[i]] * 1000
    
    task_times_rank = np.argsort(task_times)
    task_times_rank = np.flip(task_times_rank)

    logger.info('=============== task_times ===============\n')
    logger.info(str(task_times))
    logger.info('\n')
    logger.info(str(task_times_rank))
    logger.info('\n')
    logger.info(str(np.argsort(task_times_rank) + 1))
    logger.info('\n\n')

    #################### Compile ####################
    current_latency = evaluate_tvm(mod, params, input_name, data, log_file)
    time.sleep(250)
    #################################################        
    budget = 0.1 * current_latency
    logger.info('Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}'.format(current_latency.mean(), total_estimated_latency))
    logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}\n'.format(budget, current_latency, total_estimated_latency))
    
    return OptimizerTVMOutput(tuner, 
                             pruning_times, 
                             real_pruning_times, 
                             current_latency,
                             total_estimated_latency,
                             subgraph_tasks)

