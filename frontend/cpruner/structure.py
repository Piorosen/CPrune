from dataclasses import dataclass 
from typing import List, Union, Any, Tuple
from tvm import relay, auto_scheduler
import numpy as np
import torch
from enum import Enum

@dataclass 
class ExtractSubgraph: 
    Pos: List[int] = []
    SubgraphConv2d: List[int] = []
    NumConv2d: int = 0
    NumOthers: int = 0

class DeviceType(Enum):
    CPU = 0
    GPU = 1


@dataclass 
class OptimizerTVMInput:
    Model: Any
    InputData: torch.Tensor
    InputSize: Tuple
    DeviceType: DeviceType
    Subgraph: ExtractSubgraph
    
    UseAndroid: bool = False
    TVM_DeviceKey: str 
    TVM_TrackerHost: str = "0.0.0.0"
    TVM_TrackerPort: int
    TVM_Archtecture: str = "arm64"
    TVM_DataType: str = "float32"
    TVM_Target: str = "llvm"
    # target = "llvm -mtriple=%s-linux-android" % arch        
    # target = "llvm -mtriple=%s-linux-none" % arch
    
@dataclass 
class OptimizerTVMOutput: 
    Tuner: auto_scheduler.TaskScheduler
    PruningTime: List[float] = []
    RealPruningTime: List[float] = []
    CurrentLatency: Union[np.ndarray, float] # Tuning 중에 얻은 값
    TotalEstimatedLatency: float             # 실제 수행하였을 때 얻은 값.
    SubgraphTasks: List[int]
    