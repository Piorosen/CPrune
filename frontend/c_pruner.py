from multiprocessing import Process
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import json
import torch

from nni.compression.pytorch.compressor import Pruner
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT

################### TVM build part addition ###############
import torchvision.models as models
import time
import sys 

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

from .cpruner import optimizer_tvm
from .cpruner import Logger 

logger = Logger().get_logger()
###########################################################

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
    def __init__(self, model, config_list, short_term_trainer, evaluator, dummy_input, base_algo='l1', experiment_data_dir='./', cpu_or_gpu=1, input_size=(1, 3, 224, 224), acc_requirement=0.85):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
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

        self._tmp_model_path = './tmp_model.pth'

        # addition
        self._dummy_input = dummy_input
        self._input_size = input_size
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

    def _get_extract_subgraph(_model_to_prune, _input_size):
        _, _, temp_results = count_flops_params(_model_to_prune, _input_size)
        
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
        
        return (pos, conv2d_subgraph_chs, conv2d_num, others_num)
    
    def _pre_prunning(run: bool):
        if run:
            real_pruning_times = [0]
            subgraph_idx = 0
            for wrapper in self.get_modules_wrapper():
                if real_pruning_times[subgraph_idx] > 0:
                    target_op_sparsity = real_pruning_times[subgraph_idx]
                    self._config_list_generated = self._update_config_list(
                        self._config_list_generated, wrapper.name, target_op_sparsity)
                    pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), self._config_list_generated, dependency_aware=True, dummy_input=self._dummy_input)
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
        else:
            return 
    
    def compress(self):
        """
        Compress the model.

        Return
        -------
        torch.nn.Module : the final pruned model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # target = "llvm -mtriple=%s-linux-android" % arch        
        # target = "llvm -mtriple=%s-linux-none" % arch
        use_android = False
        self._model_to_prune.eval()

        input_data = torch.randn(self._input_size).to(device)
        
        ######################################################################
        
        pos, conv2d_subgraph_chs, conv2d_num, others_num = self._get_extract_subgraph(self._model_to_prune, self._input_size)
        pruning_times, real_pruning_times, current_latency, total_estimated_latency = optimizer_tvm.optimizing(self._model_to_prune, 
                                 input_data, 
                                 self._input_size, 
                                 self._cpu_or_gpu,
                                 conv2d_num,
                                 others_num,
                                 pos,
                                 self._acc_requirement)

        pass_target_latency = 0
        init_short_acc = 0
        performance = 0
        minimum_acc_requirement = self._acc_requirement
        alpha = 0.995  # target_accuracy = alpha * prev_best_accuracy
        beta = 0.99  # target_latency = beta * current_best_latency
        max_iter = 1
        pruning_iteration = 1
        budget = 0.1 * current_latency
        
        #################################################        
        
        
        logger.info('Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}'.format(current_latency, total_estimated_latency))
        logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f}, Total estimated latency: {:>8.4f}\n'.format(budget, current_latency, total_estimated_latency))
        
        # if self._dataset == 'cifar10':
        #     current_accuracy = self._evaluator(self._model_to_prune)                
        # elif self._dataset == 'imagenet':
        _, current_accuracy = self._evaluator(self._model_to_prune)
        target_latency = current_latency * beta

        # stop condition
        while pruning_iteration < max_iter and current_latency > budget:
            # calculate target sparsity of this iteration
            if pass_target_latency == 1:
                target_latency = current_latency * beta
                pass_target_latency = 0

            # Print the message
            logger.info('=======================')
            logger.info(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                    'current_latency = {:>8.4f}, target_latency = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency, total_estimated_latency, tune_trials))
            # file_object = open('./record_tvm.txt', 'a')            
            logger.info(('Process iteration {:>3}: current_accuracy = {:>8.4f}, '
                   'current_latency = {:>8.4f}, target_resource = {:>8.4f}, total_estimated_latency = {:>8.4f}, tune_trials = {:4d} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency, total_estimated_latency, tune_trials))
            logger.info('Current pruning_times: ' + str(pruning_times) + '\n')
            logger.info('Real pruning_times: ' + str(real_pruning_times) + '\n')

            # variable to store the info of the best subgraph found in this iteration
            best_op = {}
            
            ########################### Pre-pruning (if it is necessary) ##########################
            self._pre_prunning(False)
            ######################################################################
            
            cnt = 0
            while cnt < len(task_times_rank):
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
                for overlap_cnt in task_times_rank[init_cnt: init_cnt + overlap_num]:
                    pruning_times[overlap_cnt] += float(tuner.prune_num[subgraph_tasks[overlap_cnt]]) * float(1/conv2d_subgraph_chs[overlap_cnt])
                target_op_sparsity = pruning_times[task_times_rank[init_cnt]]
                ch_num = int(conv2d_subgraph_chs[task_times_rank[init_cnt]] * (1 - target_op_sparsity))
                
                if target_op_sparsity > 0.8:
                    logger.info('Improper Subgraph')
                    wrapper = self.get_modules_wrapper()[task_times_rank[init_cnt]]
                    # file_object = open('./record_tvm.txt', 'a')      
                    logger.info('Improper Subgraph: ' + wrapper.name + ', Total: ' + str(overlap_num) + ' subgraphs\n')
                    # file_object.close()
                    continue

                config_list = copy.deepcopy(self._config_list_generated)
                for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
                    wrapper = self.get_modules_wrapper()[wrapper_idx]
                    config_list = self._update_config_list(config_list, wrapper.name, target_op_sparsity)

                wrapper = self.get_modules_wrapper()[task_times_rank[init_cnt]]
                logger.info('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num))
                # file_object = open('./record_tvm.txt', 'a')
                logger.info('Subgraph: ' + wrapper.name + ', overlap_num: ' + str(overlap_num) + ', ch_num: ' + str(ch_num) + '\n')
                logger.info('Temp_pruning_times:' + str(pruning_times) + '\n')
                # file_object.close()

                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list, dependency_aware=True, dummy_input=self._dummy_input)
                model_masked = pruner.compress()

                # added 0: speed_up
                pruner.export_model('./model_masked.pth', './mask.pth')
                model = copy.deepcopy(self._original_model)
                model.load_state_dict(torch.load('./model_masked.pth'))
                masks_file = './mask.pth'
                m_speedup = ModelSpeedup(model, self._dummy_input, masks_file, device)
                m_speedup.speedup_model()
                # added 1: Autotune + TVM build
                model.eval()
                
                input_data = torch.randn(self._input_size).to(device)
                pos, conv2d_subgraph_chs, conv2d_num, others_num = self._get_extract_subgraph(model, self._input_size)
                pruning_times, real_pruning_times, temp_latency = optimizer_tvm.optimizing(model, 
                                        input_data, 
                                        self._input_size, 
                                        self._cpu_or_gpu,
                                        conv2d_num,
                                        others_num,
                                        pos,
                                        self._acc_requirement)
     
                #################################################
                logger.info('Subgraph: {}, Temp latency: {:>8.4f}, Total estimated latency: {:>8.4f}, Channel: {:4d}, Next trials: {:4d}'.format(wrapper.name, temp_latency, total_estimated_latency, ch_num, tune_trials))
                # file_object = open('./record_tvm.txt', 'a')
                logger.info('Subgraph: {}, Temp latency: {:>8.4f}, Total estimated latency: {:>8.4f}, Channel: {:4d}, Next trials: {:4d}\n'.format(wrapper.name, temp_latency, total_estimated_latency, ch_num, tune_trials))
                # file_object.close()
                ################# Added part to prune the slow subgraph quickly ##################
                if temp_latency > target_latency:
                    # ('./record_tvm.txt', 'a')
                    logger.info('Higher than target latency! Pruning_ratio of Subgraph {} increases one time more!\n'.format(wrapper.name))
                    # file_object.close()
                ###############################################################################

                if temp_latency <= target_latency:
                    logger.info('./train_epoch.txt', 'a')
                    logger.info('Subgraph: {}, Temp latency: {:>8.4f}, Channel: {:4d}\n'.format(wrapper.name, temp_latency, ch_num))
                    # file_object.close()
                    # Short-term fine tune the pruned model
                    optimizer = torch.optim.SGD(model_masked.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)                    
                    best_acc = 0
                    
                    best_acc_5 = 0
                    short_num = 5 # Training Epoch
                    
                    for epoch in range(short_num):
                        self._short_term_trainer(model_masked, optimizer, epochs=epoch)
                        acc, acc_5 = self._evaluator(model_masked)
                        if acc_5 > best_acc_5:
                            best_acc_5 = acc_5
                        if acc > best_acc:
                            best_acc = acc
                    print('Subgraph: {}, Short_tune - Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                    logger.info('Subgraph: {}, Top-1 Accuracy: {:>8.5f}, Top-5 Accuracy: {:>8.5f}'.format(wrapper.name, best_acc, best_acc_5))
                    ################ Added part to avoid excessive accuracy decrement ###############
                    # temp_acc = best_acc_5 if self._dataset == 'imagenet' else best_acc
                    temp_acc = best_acc_5
                    if temp_acc < alpha * current_accuracy: 
                        logger.info('Too low short-term accuracy! Improper subgraph: {}\n'.format(wrapper.name))
                        for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
                            pruning_times[wrapper_idx] = 1
                        continue
                    #################################################################################

                    for wrapper_idx in task_times_rank[init_cnt: init_cnt + overlap_num]:
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
                    prev_task_times_rank = task_times_rank

                    # save model weights
                    pruner.export_model(self._tmp_model_path, './tmp_mask.pth')
                    subgraph_tasks = subgraph_tasks_temp
                    task_times = task_times_temp
                    task_times_rank = task_times_rank_temp
                    
                    logger.info('=============== task_times ===============\n')
                    logger.info(str(task_times))
                    logger.info('\n')
                    logger.info(str(task_times_rank))
                    logger.info('\n')
                    logger.info(str(np.argsort(task_times_rank) + 1))
                    logger.info('\n\n')
                    break
                else:
                    time.sleep(250)

            # Check the minimum accuracy requirement
            if alpha * best_op['performance'] < minimum_acc_requirement:
                break

            if pass_target_latency == 1:
                for wrapper_idx in prev_task_times_rank[init_cnt: init_cnt + overlap_num]:
                    wrapper = self.get_modules_wrapper()[wrapper_idx]
                    self._config_list_generated = self._update_config_list(
                        self._config_list_generated, wrapper.name, target_op_sparsity)
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask, 'bias_mask': w.bias_mask}
                            break
                    for k in masks:
                        setattr(wrapper, k, masks[k])

                # update weights parameters
                self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))
                logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f}'.format(budget, best_op['latency']))
                logger.info('Budget: {:>8.4f}, Current latency: {:>8.4f} \n'.format(budget, best_op['latency']))

                current_accuracy = temp_acc
                #########################
                logger.info('Subgraph {} selected with {:4d} channels, latency {:>8.4f}, accuracy {:>8.4f} \n'.format(best_op['op_name'], best_op['ch_num'], best_op['latency'], best_op['performance']))
            pruning_iteration += 1

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))

        return self._model_to_prune
