#%%

from dotenv import load_dotenv

import os
import torch.utils.data
import torchvision.models as models
from c_pruner import CPruner
from nni.compression.pytorch import ModelSpeedup

from cpruner import Logger, DeviceType
from utils import *
from models.implements.cnn.mnist import LeNet
logger = Logger()

###########################################################
def main(args):
    cpu_or_gpu = DeviceType.CPU
    torch.manual_seed(42)

    # For Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data_dataset(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    model = LeNet().to(device)
    
    # model = models.resnet18(pretrained=True).to(device)

    input_size = get_input_size(args.dataset)
    dummy_input = get_dummy_input(input_size, args.batch_size).to(device)
    acc_requirement = args.accuracy_requirement
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    
    def short_term_trainer(model, optimizer=optimizer, epochs=1):
        train(args, model, device, train_loader, criterion, optimizer, epochs)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    def evaluator_top1(model):
        return test_top1(model, device, criterion, val_loader)
    
    # If you need a training model.
    file_name = os.path.join(args.experiment_data_dir, f'{args.model}.pth')
    if os.path.exists(file_name):
        model.load_state_dict(torch.load(file_name))
    else:
        os.makedirs(args.experiment_data_dir)
        os.makedirs(os.path.join(args.experiment_data_dir, 'tvm'))
        short_term_trainer(model, epochs=10)
        torch.save(model.state_dict(), file_name)
    
    # ImageNet
    if args.dataset == 'imagenet':
        # ResNet-18
        accuracy = 0.69758
        accuracy_5 = 0.89078
        ## MnasNet1_0
        #accuracy = 0.73456
        #accuracy_5 = 0.91510
        #accuracy, accuracy_5 = evaluator(model)
        print('Original model - Top-1 Accuracy: %s, Top-5 Accuracy: %s' %(accuracy, accuracy_5))
    # CIFAR-10
    elif args.dataset == 'cifar10' or args.dataset == 'mnist':
        pass
        # _, accuracy = evaluator_top1(model)
        # print('Original model - Top-1 Accuracy: %s' %(accuracy))
        
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
    
    # Pruner.compress() returns the masked model
    model = pruner.compress()

    # model speed up
    if args.speed_up:
        model.load_state_dict(torch.load('./tmp_model.pth'))
        masks_file = './tmp_mask.pth'
        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
    
    export_model('./export')
 #%%
from types import SimpleNamespace
 
if __name__ == '__main__':
    
    load_dotenv()
    args = SimpleNamespace(
    accuracy_requirement=0.85,
    dataset='mnist',
    data_dir='/work/dataset',
    model='LeNet',
    batch_size=64,
    test_batch_size=1,  # 64
    fine_tune=True,
    fine_tune_epochs=3,
    experiment_data_dir='/work/experiments/mnist_lenet',
    base_algo='l1',
    sparsity=0.1,
    log_interval=1000,  # 200
    speed_up=True
    )
    main(args)

# %%
