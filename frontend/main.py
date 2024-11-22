#%%

from dotenv import load_dotenv

import os
import torch.utils.data
import torchvision.models as models
from c_pruner import CPruner
from nni.compression.pytorch import ModelSpeedup
import pickle

from cpruner import Logger, DeviceType
from utils import *
from models.implements import get_model_zoo
from models.implements.cnn.cifar10 import ResNet18

logger = Logger()

###########################################################
def main(args):
    cpu_or_gpu = DeviceType.CPU
    torch.manual_seed(42)

    # For Training
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    train_loader, val_loader, criterion = get_data_dataset(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    # model = LeNet().to(device)
    model = ResNet18().to(device)
    model.load_state_dict(torch.load('./cifar10_model_300.pth'))
    
    # models = get_model_zoo()
    # model, file_name = models[args.model]
    # model = model(True, True)
    
    os.makedirs(args.experiment_data_dir, exist_ok=True)
    
    input_size = get_input_size(args.dataset)
    dummy_input = get_dummy_input(input_size, args.batch_size).to(device)
    acc_requirement = args.accuracy_requirement
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    
    def short_term_trainer(model, optimizer=optimizer, epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for e in range(epochs):
            train(args, model, device, train_loader, criterion, optimizer, e)
        model = model.to(torch.device('cpu'))
        
    def evaluator(model, cache_file):
        top1, current_accuracy = None, None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                top1, current_accuracy = pickle.load(f)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            top1, current_accuracy = test(model, device, criterion, val_loader)
            model = model.to(torch.device('cpu'))
            with open(cache_file, 'wb') as f:
                pickle.dump([top1, current_accuracy], f)
                
        return top1, current_accuracy

    def evaluator_top1(model, cache_file):
        if cache_file != None and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                result, _ = pickle.load(f)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            result, _ = test_top1(model, device, criterion, val_loader)
            model = model.to(torch.device('cpu'))
            if cache_file != None:
                with open(cache_file, 'wb') as f:
                    pickle.dump([result, result], f)
        return result, result
    
    # ImageNet
    if args.dataset == 'imagenet':
        # ResNet-18
        accuracy = 0.69758
        accuracy_5 = 0.89078
        ## MnasNet1_0
        #accuracy = 0.73456
        # accuracy_5 = 0.91510
        # accuracy, accuracy_5 = evaluator(model)
        print('Original model - Top-1 Accuracy: %s, Top-5 Accuracy: %s' %(accuracy, accuracy_5))
    # CIFAR-10
    elif args.dataset == 'cifar10' or args.dataset == 'mnist':
        # pass
        # _, accuracy = evaluator_top1(model, None)
        # print('Original model - Top-1 Accuracy: %s' %(accuracy))
        
        print('Original model - Top-1 Accuracy: %s' %(82.91))
        
    # module types to prune, only "Conv2d" supported for channel pruning
    if args.base_algo in ['l1', 'l2', 'fpgm']:
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]
    
    pruner = CPruner(model, 
                     config_list, 
                     short_term_trainer=short_term_trainer, 
                     evaluator=evaluator if args.dataset == 'imagenet' else evaluator_top1, 
                     base_algo=args.base_algo, 
                     experiment_data_dir=args.experiment_data_dir, 
                     cpu_or_gpu=cpu_or_gpu, 
                     input_size=input_size, 
                     acc_requirement=acc_requirement)
    
    # # Pruner.compress() returns the masked model
    model = pruner.compress(args.tune_mode, short_num=args.fine_tune_epochs)
    
    # # model speed up
    # if args.speed_up:
    #     model.load_state_dict(torch.load('/work/experiments/mnist_lenet/tvm/001_000000_model.pth'))
    #     masks_file = '/work/experiments/mnist_lenet/tvm/001_000000_mask.pth'
    #     m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
    #     m_speedup.speedup_model()

    # model.eval() +
    # torch.onnx.export(model,         # model being run 
    #      dummy_input,       # model input (or a tuple for multiple inputs) 
    #      "LeNet.onnx",       # where to save the model  
    #      export_params=True,  # store the trained parameter weights inside the model file 
    #      opset_version=16,    # the ONNX version to export the model to 
    #      do_constant_folding=True,  # whether to execute constant folding for optimization 
    #      input_names = ['input0'],   # the model's input names 
    #      output_names = ['output0'], # the model's output names 
    #      ) 
    
    # export_model('./export')
    

#%%

from types import SimpleNamespace
 
if __name__ == '__main__':
    load_dotenv()
    args = SimpleNamespace(
    accuracy_requirement=82.91 * 0.93 / 100,
    dataset='cifar10',
    data_dir='/work/dataset',
    model='resnet18',
    batch_size=512,
    test_batch_size=512,  # 64
    fine_tune=True,
    fine_tune_epochs=5,
    experiment_data_dir='/work/experiments/manytime_rockpi_resnet18_error',
    base_algo='l1',
    sparsity=0.5,
    log_interval=1000,  # 200
    speed_up=True,
    tune_mode=2 # 0 : task, 1 : all, 2 : error
    )
    print(args)
    main(args)

# # %%

# # %%
# import torch
# from utils import *   
# from models.implements.cnn.mnist import LeNet
# from nni.compression.pytorch import ModelSpeedup

# model = LeNet()
# input_size = get_input_size('mnist')
# dummy_input = get_dummy_input(input_size, 1)
# model.load_state_dict(torch.load('/work/experiments/mnist_lenet/tvm/006_000000_model.pth'))
# masks_file = '/work/experiments/mnist_lenet/tvm/006_000000_mask.pth'
# m_speedup = ModelSpeedup(model, dummy_input, masks_file, torch.device('cpu'))
# m_speedup.speedup_model()

# torch.onnx.export(model,         # model being run 
#         dummy_input,       # model input (or a tuple for multiple inputs) 
#         "6.onnx",       # where to save the model  
#         export_params=True,  # store the trained parameter weights inside the model file 
#         opset_version=12,    # the ONNX version to export the model to 
#         do_constant_folding=True,  # whether to execute constant folding for optimization 
#         input_names = ['input0'],   # the model's input names 
#         output_names = ['output0'], # the model's output names 
#         ) 

# # %%
# from dotenv import load_dotenv
# import torch
# from models.implements import get_model_zoo
# from utils import *

# def evaluator(model):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     acc1, acc5 = test(model, device, criterion, val_loader)
#     model = model.to(torch.device('cpu'))
#     return acc1, acc5
# # %%
# model, pth = get_model_zoo()['alexnet']
# model = model(True)

# torch.manual_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_loader, val_loader, criterion = get_data_dataset('imagenet', '/work/dataset', 512, 64)

# # %%
# acc1, acc5 = evaluator(model)
# print(acc1, acc5)

# # %%

# #%%
# val_loader[0]
# # %%

# # %%

# # %%

# %%
