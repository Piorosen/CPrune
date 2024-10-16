#%%
import os
import torch.utils.data
import torchvision.models as models
from c_pruner import CPruner
from nni.compression.pytorch import ModelSpeedup

from .cpruner import Logger, DeviceType
from .utils import *
from .models.implements.cnn.mnist.
logger = Logger()

###########################################################
def main(args):
    cpu_or_gpu = DeviceType.CPU
    torch.manual_seed(42)

    # For Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data_dataset(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    
    # model = models.resnet18(pretrained=True).to(device)

    dummy_input = get_dummy_input()
    input_size = get_input_size()
    acc_requirement = args.accuracy_requirement
    
    def short_term_trainer(model, optimizer=optimizer, epochs=1):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        train(args, model, device, train_loader, criterion, optimizer, epochs)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    def evaluator_top1(model):
        return test_top1(model, device, criterion, val_loader)
    
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
        accuracy = evaluator_top1(model)
        print('Original model - Top-1 Accuracy: %s' %(accuracy))
        
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
                     dummy_input=dummy_input, 
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
    
    ################ Long-term training ################
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    best_acc = 0
    if args.dataset == 'imagenet':
        best_acc_5 = 0
    for epoch in range(args.fine_tune_epochs): # imagenet: 20, cifar10: 100
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        if args.dataset == 'imagenet':
            acc, acc_5 = evaluator(model)
            if acc_5 > best_acc_5:
                best_acc_5 = acc_5
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))
            if acc > best_acc:
                best_acc = acc
        elif args.dataset == 'cifar10':
            acc = evaluator_top1(model)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    if args.dataset == 'imagenet':
        print('Evaluation result (Long-term): %s %s' %(best_acc, best_acc_5))
    elif args.dataset == 'cifar10':
        print('Evaluation result (Long-term): %s' %(best_acc))
    ####################################################    
    ################ Long-term tuning and compile ################
    if os.path.isfile('./model_fine_tuned.pth'):
        model.load_state_dict(torch.load(os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth')))
        
        
    # arch = "arm64"
    # # target = "llvm -mtriple=%s-linux-android" % arch
    # # target = "llvm -mtriple=%s-linux-none" % arch
    # target = "llvm"
    # device_key = os.getenv("ID_OPTIMIZATION_HARDWARE")
    # log_file = "%s.log" % (device_key)
    # dtype = "float32"
    # use_android = False
    # at_least_trials = 1
    # num_per_round = 1
    # model.eval()
    # _, _, temp_results = count_flops_params(model, get_input_size(args.dataset))
    # input_shape = get_input_size(args.dataset)
    # input_data = torch.randn(input_shape).to(device)
    # scripted_model = torch.jit.trace(model, input_data).eval()
    # input_name = "input0"
    # shape_list = [(input_name, input_shape)]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # ########### NCHW -> NHWC ############
    # desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
    # seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
    #                       relay.transform.ConvertLayout(desired_layouts)])
    # with tvm.transform.PassContext(opt_level=3):
    #     mod = seq(mod)
    # #####################################
    # tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    # tracker_port = int(os.environ["TVM_TRACKER_PORT"])
    # ########### Extract search tasks ###########
    # print("Extract tasks...")
    # if cpu_or_gpu == 1:
    #     tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    # else:
    #     tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl -device=mali", target_host=target)
    # tune_trials = 1 * (at_least_trials + num_per_round) * len(tasks)
    # print("tune_trials: " + str(tune_trials))
    # ########### Tuning ###########
    # print("Begin tuning...")
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=tune_trials,
    #     builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
    #     runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=20, number=10, repeat=2,),
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #     num_measures_per_round = num_per_round,
    # )
    # tuner.tune(tune_option)
    # ########### Compile ###########
    # print("Compile")
    # with auto_scheduler.ApplyHistoryBest(log_file):
    #     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
    #         if cpu_or_gpu == 1:
    #             lib = relay.build(mod, target=target, params=params)
    #         else:
    #             lib = relay.build(mod, params=params, target="opencl -device=mali", target_host=target)

    # tmp = utils.tempdir()
    # if use_android:
    #     lib_fname = tmp.relpath("net.so")
    #     lib.export_library(lib_fname, ndk.create_shared)
    #     remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=200)
    #     remote.upload(lib_fname)
    #     rlib = remote.load_module("net.so")
    # else:
    #     lib_fname = tmp.relpath("net.tar")
    #     lib.export_library(lib_fname)
    #     remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=200)
    #     remote.upload(lib_fname)
    #     rlib = remote.load_module("net.tar")
        
    # if cpu_or_gpu == 1:
    #     ctx = remote.cpu()
    # else:
    #     ctx = remote.cl()
    # module = graph_executor.GraphModule(rlib["default"](ctx))

    # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    # module.set_input(input_name, data_tvm)
    # ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=2)
    # prof_res = np.array(ftimer().results) * 1e3
    # temp_latency = np.mean(prof_res)
    # print('ftimer_latency: ' + str(temp_latency))
    ##############################################################
    
 #%%
from types import SimpleNamespace
 
if __name__ == '__main__':
    args = SimpleNamespace(
    accuracy_requirement=0.85,
    dataset='mnist',
    data_dir='/work/dataset',
    model='resnet18',
    batch_size=64,
    test_batch_size=1,  # 64
    fine_tune=True,
    fine_tune_epochs=3,
    experiment_data_dir='./',
    base_algo='l1',
    sparsity=0.1,
    log_interval=1000,  # 200
    speed_up=True
    )
    main(args)
