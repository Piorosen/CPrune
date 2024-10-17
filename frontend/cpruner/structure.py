from dataclasses import dataclass 
from typing import List, Union, Any, Tuple
from tvm import relay, auto_scheduler
import numpy as np
import torch
from enum import Enum

@dataclass 
class ExtractSubgraph: 
    def __init__(self):
        self.Pos = []
        self.SubgraphConv2d = []
        self.NumConv2d = 0 
        self.NumOthers = 0
      
    Pos: List[int]
    SubgraphConv2d: List[int]
    NumConv2d: int
    NumOthers: int

class DeviceType(Enum):
    CPU = 0
    GPU = 1


@dataclass 
class OptimizerTVMInput:
    def __init__(self):
        self.UseAndroid: bool = False
        self.TVM_TrackerHost: str = "0.0.0.0"
        self.TVM_Archtecture: str = "x86"
        self.TVM_DataType: str = "float32"
        self.TVM_Target: str = "llvm -mtriple=x86_64-linux-gnu"
        self.Subgraph = ExtractSubgraph()
      
    Model: Any
    InputData: torch.Tensor
    InputSize: Tuple
    DeviceType: DeviceType
    Subgraph: ExtractSubgraph
    
    UseAndroid: bool
    TVM_DeviceKey: str 
    TVM_TrackerHost: str
    TVM_TrackerPort: int
    TVM_Archtecture: str
    TVM_DataType: str
    TVM_Target: str
    # target = "llvm -mtriple=%s-linux-android" % arch        
    # target = "llvm -mtriple=%s-linux-none" % arch
    
@dataclass 
class OptimizerTVMOutput: 
    def __init__(self, task_times, task_times_rank, tune_trials=0, current_latency=np.array([]), total_estimated_latency=0, subgraph_tasks=[], prune_num={}):
        self.TuneTrials = tune_trials
        self.CurrentLatency = current_latency
        self.TotalEstimatedLatency = total_estimated_latency
        self.SubgraphTasks = subgraph_tasks
        self.PruneNum = prune_num
        self.TaskTimes = task_times
        self.TaskTimesRank = task_times_rank
    # TaskTimes
    # TaskTimesRank
    PruneNum: dict
    TuneTrials: int
    CurrentLatency: Union[np.ndarray, float] # Tuning 중에 얻은 값
    TotalEstimatedLatency: float             # 실제 수행하였을 때 얻은 값.
    SubgraphTasks: List[int]
    