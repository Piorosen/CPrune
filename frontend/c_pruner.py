import pickle
from multiprocessing import Process
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import torch

from nni.compression.pytorch.compressor import Pruner
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency, ReshapeDependency, InputChannelDependency, AttentionWeightDependency

################### TVM build part addition ###############
import torchvision.models as models
import time
import sys 

import os
import torchvision as tv

def get_model_zoo():
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
    return model_dict

import tvm
from tvm import relay, auto_scheduler
import numpy as np
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner
from tvm import rpc
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch import ModelSpeedup
from torch.optim.lr_scheduler import MultiStepLR

from utils import get_dummy_input, write_log
from cpruner import DeviceType
from cpruner import optimizer_tvm
from cpruner import Logger 

logger = Logger().get_logger()
###########################################################

def safe_int(value):
    try:
        return int(value)
    except ValueError:
        return None  # 혹은 원하는 값을 반환
    
class CPruner(Pruner):
    '''
    Pruning the pre-trained model by utilizing measured latency from executable tuning
    
    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    short_term_trainer : function
        function to short-term train the masked model
    evaluator : function
        function to evaluate the masked model
    '''
    def __init__(self, model, config_list, short_term_trainer, evaluator, base_algo='l1', experiment_data_dir='./', cpu_or_gpu=DeviceType.CPU, input_size=(1, 3, 224, 224), acc_requirement=0.85):
        # models used for iterative pruning and evaluation
        self._original_model = copy.deepcopy(model)
        self._base_algo = base_algo
        self._cpu_or_gpu = cpu_or_gpu

        super().__init__(model, config_list)

        self._short_term_trainer = short_term_trainer
        self._evaluator = evaluator

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        # addition
        self._input_size = input_size
        self._dummy_input = get_dummy_input(input_size, 4)
        self._acc_requirement = acc_requirement

    def _update_config_list(self, config_list, op_name, sparsity):
        '''
        update sparsity of op_name in config_list
        '''
        config_list_updated = copy.deepcopy(config_list)
        if not op_name:
            return config_list_updated

        for idx, item in enumerate(config_list):
            if op_name in item['op_names']:
                config_list_updated[idx]['sparsity'] = sparsity
                return config_list_updated

        # if op_name is not in self._config_list_generated, create a new json item
        if self._base_algo in ['l1', 'l2', 'fpgm']:
            config_list_updated.append(
                {'sparsity': sparsity, 'op_types': ['Conv2d'], 'op_names': [op_name]})
        elif self._base_algo == 'level':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_names': [op_name]})

        return config_list_updated
    
    def task_to_layer(self, model, task_id, pos, subgraph_tasks):
        input_shape = [1,3,32,32]
        _,_,bs = count_flops_params(model, tuple(input_shape), verbose=True)
        f = list(filter(lambda x: x['module_type'] == 'Conv2d', bs))
        f = list(map(lambda x: x['name'], f))

        dummy_input = torch.randn(input_shape)
        depen = ChannelDependency(model, dummy_input=dummy_input).dependency
        task_to_nni_index = np.where(np.array(subgraph_tasks) == task_id)[0]
        nni_index_to_layer = np.array(f)[task_to_nni_index]

        list_depen = []
        list_index = []
        for i in nni_index_to_layer:
            if not i in depen:
                list_depen.append(i)
                list_index.append(f.index(i))
            else:
                for j in list(depen[i]):
                    list_depen.append(j)
                    list_index.append(f.index(j))

        return np.unique(list_depen), np.unique(list_index)


    def _get_extract_subgraph(self, model) -> optimizer_tvm.ExtractSubgraph:
        _input_size = self._input_size
        _, _, temp_results = count_flops_params(model, tuple(_input_size))
        
        conv2d_num = 0
        others_num = 0
        downsample_subgraphs = []
        temp_results_len = len(temp_results)
        for idx in range(temp_results_len):
            if 'downsample' in temp_results[idx].get('name'):
                downsample_subgraphs.append(idx)
            elif 'shortcut' in temp_results[idx].get('name'):
                downsample_subgraphs.append(idx)
            if temp_results[idx].get('module_type') == 'Conv2d':
                conv2d_num+=1
            else:
                others_num+=1
        conv2d_subgraph_chs = [-1 for i in range(conv2d_num)]
        temp_idx = 0
        for idx in range(temp_results_len):
            if temp_results[idx].get('module_type') == 'Conv2d':
                conv2d_subgraph_chs[temp_idx] = temp_results[idx].get('weight_shape')[0]
                temp_idx += 1

        ##################### subgraph_task connection #######################
        pos = []
        last_idx = conv2d_num - 1
        list_filled = [0 for i in range(conv2d_num)]
        for idx in range(conv2d_num):
            n = conv2d_num - 1 - idx
            if list_filled[n] == 1:
                continue
            elif 'downsample' in temp_results[n].get('name'):
                continue
            elif 'shortcut' in temp_results[n].get('name'):
                continue
            else:
                pos.append(n)
                list_filled[n] = 1
            split_name = temp_results[n].get('name').split('.')
            for i in range(conv2d_num):
                if i == n: break
                temp_split = temp_results[i].get('name').split('.')
                if split_name[0] == temp_split[0] and \
                   split_name[len(split_name)-1] == temp_split[len(temp_split)-1] and \
                   temp_results[n].get('weight_shape') == temp_results[i].get('weight_shape') and \
                   temp_results[n].get('flops') == temp_results[i].get('flops') and \
                   temp_results[n].get('params') == temp_results[i].get('params'):
                    pos.append(i)
                    list_filled[i] = 1
        pos = pos + downsample_subgraphs
        
        result = optimizer_tvm.ExtractSubgraph()
        
        result.Pos = pos
        result.SubgraphConv2d = conv2d_subgraph_chs
        result.NumConv2d = conv2d_num
        result.NumOthers = others_num
        
        return result
    
    def _pre_prunning(self, model):
        real_pruning_times = [0]
        subgraph_idx = 0
        for wrapper in self.get_modules_wrapper():
            if real_pruning_times[subgraph_idx] > 0:
                target_op_sparsity = real_pruning_times[subgraph_idx]
                self._config_list_generated = self._update_config_list(
                    self._config_list_generated, wrapper.name, target_op_sparsity)
                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(model), self._config_list_generated, dependency_aware=True, dummy_input=self._dummy_input)
                model_masked = pruner.compress()
                masks = {}
                for w in pruner.get_modules_wrapper():
                    if w.name == wrapper.name:
                        masks = {'weight_mask': w.weight_mask,
                                    'bias_mask': w.bias_mask}
                        break
                for k in masks:
                    setattr(wrapper, k, masks[k])
            subgraph_idx += 1
        pruning_times = [0]
        pruning_iteration = 12
        return 
    
    def __pruning_layer(self, cnt, task_times, task_times_rank, pruning_times, subgraph_tasks, conv2d_subgraph_chs, PruneNum, model,
                        output_mask, output_model, Pos):
        init_cnt = cnt
        overlap_num = 1
        while True:
            if cnt + 1 == len(task_times_rank):
                break
            if task_times[task_times_rank[cnt]] == task_times[task_times_rank[cnt+1]]:
                overlap_num += 1
                cnt += 1
            else:
                break
        cnt += 1
        extract_task_id_by_cnt = subgraph_tasks[task_times_rank[init_cnt]]
        
        target_layers, target_index = self.task_to_layer(model, extract_task_id_by_cnt, Pos, subgraph_tasks)
        import math
        from functools import reduce
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        def lcm_multiple(numbers):
            return reduce(lcm, numbers)

        lcm_chs = lcm_multiple(np.array(conv2d_subgraph_chs)[np.array(target_index)])
        sub = np.array(subgraph_tasks)[np.array(target_index)]
        lcm_prune = []
        for i in sub:
            lcm_prune.append(PruneNum[i])
        lcm_num = lcm_multiple(lcm_prune)
        
        for overlap_cnt in target_index:
            # pruning_times[overlap_cnt] += float(PruneNum[subgraph_tasks[overlap_cnt]]) * float(1/conv2d_subgraph_chs[overlap_cnt])
            pruning_times[overlap_cnt] += float(lcm_num)*float(1.0/lcm_chs)
        target_op_sparsity = pruning_times[task_times_rank[init_cnt]]
        ch_num = int(conv2d_subgraph_chs[task_times_rank[init_cnt]] * (1 - target_op_sparsity))
        
        if target_op_sparsity > 0.65:
            logger.info('Improper Subgraph')
            wrapper = self.get_modules_wrapper()[task_times_rank[init_cnt]]
            logger.info('Improper Subgraph: ' + wrapper.name + ', Total: ' + str(overlap_num) + ' subgraphs\n')
            # file_object = open('./record_tvm.txt', 'a')      
            # file_object.close()
            return cnt, None, ch_num, wrapper, target_op_sparsity, overlap_num, None, target_index

        config_list = copy.deepcopy(self._config_list_generated)
        for wrapper_idx in target_index:
            wrapper = self.get_modules_wrapper()[wrapper_idx]
            config_list = self._update_config_list(config_list, wrapper.name, target_op_sparsity)

            # logger.info('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num))
        # file_object = open('./record_tvm.txt', 'a')
            logger.info('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num) + '\n')
        logger.info('Temp_pruning_times:' + str(pruning_times) + '\n')
        # file_object.close()
        pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(model), config_list, dependency_aware=True, dummy_input=self._dummy_input)
        model_masked = pruner.compress()
        
        # if not (os.path.exists(output_mask) or os.path.exists(output_mask)):
        # added 0: speed_up
        pruner.export_model(output_model, output_mask)
        
        return cnt, pruner, ch_num, wrapper, target_op_sparsity, overlap_num, model_masked, target_index

    def compress(self, tune_mode, short_num=5):
        """
        Compress the model.

        Return
        -------
        torch.nn.Module : the final pruned model
        """
        prev_tune_name = ''
        device = torch.device('cpu')
        # target = "llvm -mtriple=%s-linux-android" % arch        
        # target = "llvm -mtriple=%s-linux-none" % arch
        use_android = True
        model_to_Prune = copy.deepcopy(self._original_model)
        
        model_to_Prune.eval()

        input_data = torch.randn(self._input_size).to(device)
        
        ######################################################################
        write_log(0,0, 'start', '_get_extract_subgraph', self._experiment_data_dir)
        subgraph = self._get_extract_subgraph(model_to_Prune)
        write_log(0,0, 'end', '_get_extract_subgraph', self._experiment_data_dir)
        pruning_times = [0.0 for _ in range(subgraph.NumConv2d)]
        real_pruning_times = [0.0 for _ in range(subgraph.NumConv2d)]
        
        input = optimizer_tvm.OptimizerTVMInput()
        input.Model = model_to_Prune
        input.InputData = input_data
        input.InputSize = self._input_size
        input.DeviceType = self._cpu_or_gpu
        input.Subgraph = subgraph
        
        input.TVM_DeviceKey = os.getenv("ID_OPTIMIZATION_HARDWARE")
        input.TVM_TrackerHost = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
        input.TVM_TrackerPort = int(os.environ["TVM_TRACKER_PORT"])
        
        tune_first = os.path.join(self._experiment_data_dir, 'tvm')
        os.makedirs(tune_first, exist_ok=True)
        tune_first = os.path.join(tune_first, "baseline")
        write_log(0,0, 'start', 'optimizer_tvm', self._experiment_data_dir)
        output = optimizer_tvm.optimizing_all(input, tune_first, previous_file='')
        write_log(0,0, 'end', 'optimizer_tvm', self._experiment_data_dir)
        prev_tune_name = tune_first

        pass_target_latency = 0
        # init_short_acc = 0
        # performance = 0
        minimum_acc_requirement = self._acc_requirement
        # alpha = 0.995  # target_accuracy = alpha * prev_best_accuracy
        alpha = 0.95  # target_accuracy = alpha * prev_best_accuracy
        beta = 0.99  # target_latency = beta * current_best_latency
        max_iter = 100
        pruning_iteration = 1
        budget = 0.1 * np.sort(output.CurrentLatency)[1:-1].mean()
        
        #################################################        
        
        # if self._dataset == 'cifar10':
        #     current_accuracy = self._evaluator(self._model_to_prune)                
        # elif self._dataset == 'imagenet':
        
        # setting default value. so initailize value of default before tunning.
        # Compute Accuracy for what?, Not yet prunned.
        # tune_name = os.path.join(self._experiment_data_dir, 'tvm', f'{str(pruning_iteration).zfill(3)}_{str(cnt).zfill(6)}')
        file_namess = os.path.join(self._experiment_data_dir, 'tvm', 'baseline_eval.pkl')
        top1, current_accuracy = self._evaluator(model_to_Prune, file_namess)
                
        # for what target latency?
        current_latency = np.sort(output.CurrentLatency)[1:-1].mean()
        target_latency = current_latency.mean() * beta
        output_model = ""
        output_mask = ""
        
        # pruning_iteration, output_name = self._get_last_epoch()
        # if pruning_iteration != 0:
        #     tune_name = os.path.join(self._experiment_data_dir, 'tvm', output_name)
        #     with open(tune_name + '_config.pkl', 'rb') as f:
        #         self._config_list_generated = pickle.load(f) # 단 한줄씩 읽어옴
        #     with open(tune_name + '_pruner.pkl', 'wb') as f:
        #         PRUNER_DICT = pickle.load(f)
                        
        #     output_model = tune_name + "_model.pth"
        #     output_mask = tune_name + "_mask.pth"
        #     model_to_Prune.load_state_dict(torch.load(output_model))
        #     # self.load_model_state_dict(torch.load(output_model))
            
        # pruning_iteration += 1
        
        # stop condition
        write_log(0, 0, 'start', 'pruning', self._experiment_data_dir)
        while pruning_iteration <= max_iter and current_latency > budget:
            # Print the message
            
            logger.info('=======================')
            logger.info(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                    'current_latency = {:>8.4f}, target_latency = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(
                        pruning_iteration, 
                        current_accuracy, 
                        current_latency, 
                        target_latency, 
                        output.TotalEstimatedLatency, 
                        output.TuneTrials))
            # file_object = open('./record_tvm.txt', 'a')            
            logger.info(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                   'current_latency = {:>8.4f}, target_resource = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency, output.TotalEstimatedLatency, output.TuneTrials))
            logger.info('Current pruning_times: ' + str(pruning_times) + '\n')
            logger.info('Real pruning_times: ' + str(real_pruning_times) + '\n')

            # variable to store the info of the best subgraph found in this iteration
            
            ########################### Pre-pruning (if it is necessary) ##########################
            if False:
                self._pre_prunning(model_to_Prune)
            ######################################################################
            # calculate target sparsity of this iteration
            if pass_target_latency == 1:
                target_latency = current_latency * beta
                pass_target_latency = 0
                
            best_op = {}
            cnt = 0
            tune_name = ''
            while cnt < len(output.TaskTimesRank):
                tune_name = os.path.join(self._experiment_data_dir, 'tvm', f'{str(pruning_iteration).zfill(3)}_{str(cnt).zfill(6)}')
                output_model = tune_name + '_model.pth'
                output_mask = tune_name + '_mask.pth'
                output_model_train = tune_name + '_model_train.pth'
                output_mask_train = tune_name + '_mask_train.pth'
                output_evals = tune_name + '_eval.pkl'
                init_cnt = cnt
                write_log(pruning_iteration,cnt, 'start', 'sequence_pruning', self._experiment_data_dir)
                write_log(pruning_iteration,cnt, 'start', 'layer_pruning', self._experiment_data_dir)
                try:
                    cnt, pruner, ch_num, wrapper, target_op_sparsity, overlap_num, model_masked, target_index = self.__pruning_layer(cnt,
                                            output.TaskTimes, 
                                            output.TaskTimesRank,
                                            pruning_times,
                                            output.SubgraphTasks,
                                            subgraph.SubgraphConv2d,
                                            output.PruneNum, 
                                            model_to_Prune,
                                            output_mask,
                                            output_model,
                                            subgraph.Pos)
                    if pruner == None:
                        continue
                    
                except:
                    logger.warning(f'this layer is not more sparsity.')
                    continue
                
                # Get Flops from Previous Model
                # added 1: Autotune + TVM build
                if True:
                    model = copy.deepcopy(model_to_Prune)
                    try: 
                        _, _ = self._get_last_epoch(pruning_iteration + 1)
                        _, epoch = self._get_last_epoch(pruning_iteration)
                        prev_tune = os.path.join(self._experiment_data_dir, 'tvm', epoch)
                        prev_model = prev_tune + '_model_train.pth'
                        model.load_state_dict(torch.load(prev_model))
                        prev_mask = prev_tune + '_mask_train.pth'
                        m_speedup = ModelSpeedup(model, self._dummy_input, prev_mask, device)
                        # m_speedup = ModelSpeedup(model, self._dummy_input, prev_mask, device)
                        m_speedup.speedup_model()
                    except:
                        model.load_state_dict(torch.load(output_model))
                        m_speedup = ModelSpeedup(model, self._dummy_input, output_mask, device)
                        m_speedup.speedup_model()
                        # added 1: Autotune + TVM build
                    model.eval()

                flop, param, _ = count_flops_params(model.eval(), self._dummy_input)
                    
                # model = copy.deepcopy(model_to_Prune)
                
                if False:
                    oflop, oparam, _ = count_flops_params(copy.deepcopy(model), self._dummy_input)
                    if flop == oflop and param == oparam:
                        # this is equally operation.
                        logger.warning(f'Warning! : this layer is only work that spasity layer. {self.get_modules_wrapper()[output.TaskTimesRank[init_cnt]]}')
                        continue
                write_log(pruning_iteration,cnt, 'end', 'layer_pruning', self._experiment_data_dir)
                    
                input_data = torch.randn(self._input_size).to(device)
                subgraph = self._get_extract_subgraph(model)
                input2 = optimizer_tvm.OptimizerTVMInput()
                input2.Model = model
                input2.InputData = input_data
                input2.InputSize = self._input_size
                input2.DeviceType = self._cpu_or_gpu
                input2.Subgraph = subgraph
                input2.TVM_DeviceKey = os.getenv("ID_OPTIMIZATION_HARDWARE")
                input2.TVM_TrackerHost = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
                input2.TVM_TrackerPort = int(os.environ["TVM_TRACKER_PORT"])
                
                task_index = np.array(output.TaskTimesRank[init_cnt: init_cnt + overlap_num])
                task_index = np.unique(task_index)
                print(task_index)

                write_log(pruning_iteration,cnt, 'start', f'optimizer_tvm (tune_mode : {tune_mode})', self._experiment_data_dir)
                
                if tune_mode == 0:
                    output2 = optimizer_tvm.optimizing_task_index(input2, tune_name, task_index=task_index, previous_file=prev_tune_name)
                elif tune_mode == 1:
                    output2 = optimizer_tvm.optimizing_all(input2, tune_name, task_index=None, previous_file=prev_tune_name)
                elif tune_mode == 2:
                    output2 = optimizer_tvm.optimizing_error(input2, tune_name, task_index=True, previous_file=prev_tune_name)
    
    
                prev_tune_name = tune_name
                write_log(pruning_iteration,cnt, 'end', 'optimizer_tvm', self._experiment_data_dir)
                
                ch_num = int(subgraph.SubgraphConv2d[output.TaskTimesRank[init_cnt]] * (1 - target_op_sparsity))
                #################################################
                logger.info('Subgraph: {}, Temp latency: {:>8.4f}, Total estimated latency: {:>8.4f}, Channel: {:4d}, Next trials: {:4d}'
                            .format(wrapper.name, 
                                    np.sort(output2.CurrentLatency)[1:-1].mean(), # temp_latency, 
                                    output2.TotalEstimatedLatency, 
                                    ch_num, 
                                    output2.TuneTrials))
                temp_latency = np.sort(output2.CurrentLatency)[1:-1].mean()
                # ################# Added part to prune the slow subgraph quickly ##################
                # if temp_latency > target_latency:
                #     # ('./record_tvm.txt', 'a')
                #     logger.info('Higher than target latency! Pruning_ratio of Subgraph {} increases one time more!\n'.format(wrapper.name))
                    
                #     # file_object.close()
                # ###############################################################################

                # if temp_latency <= target_latency:
                if True:
                    logger.info('Subgraph: {}, Temp latency: {:>8.4f}, Channel: {:4d}\n'.format(wrapper.name, temp_latency, ch_num))
                    # file_object.close()
                    # Short-term fine tune the pruned model
                    optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)                    
                    best_acc = 0
                    best_acc_5 = 0
                    
                    # short_num = 5 # Training Epoch
                    print(output_model_train)
                    id, epoch = self._get_last_epoch(pruning_iteration)
                    
                    if epoch == None:
                        now_tune = tune_name
                    else:
                        now_tune = os.path.join(self._experiment_data_dir, 'tvm', epoch)
                    write_log(pruning_iteration,cnt, 'start', 'fine_tune_train', self._experiment_data_dir)
                        
                    now_model = now_tune + '_model_train.pth'
                    if not os.path.exists(now_model):
                        self._short_term_trainer(model_masked, optimizer, epochs=short_num)
                    acc, acc_5 = self._evaluator(model_masked, output_evals)
                    # acc, acc_5 = 100, 100
                    # 82.91 base
                    
                    if acc_5 > best_acc_5:
                        best_acc_5 = acc_5
                    if acc > best_acc:
                        best_acc = acc
                    write_log(pruning_iteration,cnt, 'end', 'fine_tune_train', self._experiment_data_dir)

                    print('Subgraph: {}, Short_tune - Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                    logger.info('Subgraph: {}, Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                    ################ Added part to avoid excessive accuracy decrement ###############
                    # temp_acc = best_acc_5 if self._dataset == 'imagenet' else best_acc
                    temp_acc = best_acc_5
                    if temp_acc < alpha * current_accuracy: 
                        logger.info('Too low short-term accuracy! Improper subgraph: {}\n'.format(wrapper.name))
                        for cnt in target_index:
                            wrapper_idx = output2.TaskTimesRank[cnt]
                            # for wrapper_idx in output2.TaskTimesRank[init_cnt: init_cnt + overlap_num]:
                            pruning_times[wrapper_idx] = 1
                        continue
                    #################################################################################
                    
                    for wrapper_idx in target_index:
                        real_pruning_times[wrapper_idx] = pruning_times[wrapper_idx]
                    pass_target_latency = 1
                    # find weight mask of this subgraph
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                        
                    best_op = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'ch_num': ch_num,
                        'latency': temp_latency,
                        'performance': temp_acc,
                        'masks': masks
                    }

                    current_latency = temp_latency
                    target_latency = current_latency * beta
                    prev_task_times_rank = output.TaskTimesRank

                    # save model weights after train
                    output = output2
                    # subgraph_tasks = output2.SubgraphTasks
                    # output.TaskTimesRank = output2.TaskTimesRank
                    task_times =  output2.TaskTimes
                    with open(tune_name + '_best_op.pkl', 'wb') as f:
                        pickle.dump(best_op, f)
                        
                    if not os.path.exists(output_model_train):
                        pruner.export_model(output_model_train, output_mask_train)
                    logger.info('=============== task_times ===============\n')
                    logger.info(str(task_times))
                    logger.info('\n')
                    logger.info(str(output.TaskTimesRank))
                    logger.info('\n')
                    logger.info(str(np.argsort(output.TaskTimesRank) + 1))
                    logger.info('\n\n')
                    break
                else:
                    time.sleep(10)

            # Check the minimum accuracy requirement
            if best_op['performance'] < minimum_acc_requirement:
                break

            if pass_target_latency == 1:
                for cnt in target_index:
                    wrapper_idx = cnt
                # for wrapper_idx in prev_task_times_rank[init_cnt: init_cnt + overlap_num]:
                    wrapper = self.get_modules_wrapper()[wrapper_idx]
                    self._config_list_generated = self._update_config_list(
                        self._config_list_generated, wrapper.name, target_op_sparsity)
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask, 'bias_mask': w.bias_mask}
                            break
                    for k in masks:
                        setattr(wrapper, k, masks[k])
                
                with open(tune_name + '_config.pkl', 'wb') as f:
                    pickle.dump(self._config_list_generated, f)
                    
                # update weights parameters
                model_to_Prune.load_state_dict(torch.load(output_model_train))
                logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f}'.format(budget, best_op['latency']))
                logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f} \n'.format(budget, best_op['latency']))

                current_accuracy = temp_acc
                #########################
                logger.info('Subgraph {} selected with {:4d} channels, latency {:>8.4f}, accuracy {:>8.4f} \n'.format(best_op['op_name'], best_op['ch_num'], best_op['latency'], best_op['performance']))
            
            write_log(pruning_iteration,cnt, 'end', 'sequence_pruning', self._experiment_data_dir)
            pruning_iteration += 1

        write_log(-1, -1, 'end', 'pruning', self._experiment_data_dir)

        # load weights parameters
        self.load_model_state_dict(torch.load(output_model_train))

        model = copy.deepcopy(self._original_model)
        # model.load_state_dict(torch.load(output_model))
        # m_speedup = ModelSpeedup(model, self._dummy_input, output_mask, device)
        # m_speedup.speedup_model()
        
        return model
    
    def _get_latest_iter(self):
        dirs = os.path.join(self._experiment_data_dir, 'tvm')
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

    def _get_last_epoch(self, cnt=-1):
        if cnt == -1:
            pk_max = self._get_latest_iter()
        else:
            pk_max = cnt
        if pk_max == 0:
            return 0, None
        
        iter = str(pk_max).zfill(3)
        dirs = os.path.join(self._experiment_data_dir, 'tvm')
        dirs = os.listdir(dirs)
        dirs.sort()
        dd = list(filter(lambda x: x[:3] == iter, dirs))
        epoch = dd[-1].split('.')[0].split('_')[:2]
        return pk_max, '_'.join(epoch)
#%%