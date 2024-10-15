import argparse
import os

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
  
def get_parser():
    parser = argparse.ArgumentParser(description='CPruner arguments')

    # dataset and model
    parser.add_argument('--accuracy-requirement', type=float, default=0.85,
                        help='the minimum accuracy requirement')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset to use, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./imagenet',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use, resnet18, mobilenetv2, mnasnet1_0')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=3,
                        help='epochs to fine-tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./',
                        help='Directory for saving experiment data')

    # TVM settings
    parser.add_argument('--tvm_tracker_host', type=str, default='127.0.0.1', 
                        help='TVM Tracker host address')
    parser.add_argument('--tvm_tracker_port', type=int, default=9190, 
                        help='TVM Tracker port')
    parser.add_argument('--id_optimization_hardware', type=str, default='rasp4b-64', 
                        help='ID for optimization hardware')
    parser.add_argument('--pydevd_disable_file_validation', type=int, default=1, 
                        help='Disable PyDev file validation (1: disable, 0: enable)')

    # pruner settings
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm (level, l1, l2, or fpgm)')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='target overall sparsity')

    # others
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--speed-up', type=str2bool, default=True,
                        help='whether to speed-up the pruned model')

    args = parser.parse_args()

    # Ensure experiment data directory exists
    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    return args
  
  
# def get_parser():
#     parser = argparse.ArgumentParser(description='CPruner arguments')
#     # dataset and model
#     parser.add_argument('--accuracy-requirement', type=float, default=0.85,
#                         help='the minimum accuracy requirement')
#     parser.add_argument('--dataset', type=str, default= 'imagenet',
#                         help='dataset to use, cifar10 or imagenet')
#     parser.add_argument('--data-dir', type=str, default='./data_fast/',
#                         help='dataset directory')
#     parser.add_argument('--model', type=str, default='resnet18',
#                         help='model to use, resnet18, mobilenetv2, mnasnet1_0')
#     parser.add_argument('--batch-size', type=int, default=64,
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1, #64
#                         help='input batch size for testing (default: 64)')
#     parser.add_argument('--fine-tune', type=str2bool, default=True,
#                         help='whether to fine-tune the pruned model')
#     parser.add_argument('--fine-tune-epochs', type=int, default=20,
#                         help='epochs to fine tune')
#     parser.add_argument('--experiment-data-dir', type=str, default='./',
#                         help='For saving experiment data')
#     # For TVM
#     parser.add_argument('--tvm_tracker_host', type=str, default='127.0.0.1', 
#                         help='TVM Tracker host address.')
#     parser.add_argument('--tvm_tracker_port', type=int, default=9190, 
#                         help='TVM Tracker port.')
#     parser.add_argument('--id_optimization_hardware', type=str, default='rasp4b-64', 
#                         help='ID for optimization hardware.')
#     parser.add_argument('--pydevd_disable_file_validation', type=int, default=1, 
#                         help='Disable PyDev file validation (1: disable, 0: enable).')
#     # pruner
#     parser.add_argument('--base-algo', type=str, default='l1',
#                         help='base pruning algorithm. level, l1, l2, or fpgm')
#     parser.add_argument('--sparsity', type=float, default=0.1,
#                         help='target overall target sparsity')
#     # others
#     parser.add_argument('--log-interval', type=int, default=1000, #200,
#                         help='how many batches to wait before logging training status')
#     # speed-up
#     parser.add_argument('--speed-up', type=str2bool, default=True,
#                         help='Whether to speed-up the pruned model')
#     args = parser.parse_args()
#     if not os.path.exists(args.experiment_data_dir):
#         os.makedirs(args.experiment_data_dir)
