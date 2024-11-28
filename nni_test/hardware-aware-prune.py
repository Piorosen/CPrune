#%%
import pickle
from tvm_test import *
import os
#%%
def get_changed_task(model, log):
    scripted_model = torch.jit.trace(model, torch.rand(1, 3, 32, 32)).eval()
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

    if os.path.exists('/work/tmp_get_error_from_tvm.txt'):
        os.remove('/work/tmp_get_error_from_tvm.txt')
    with auto_scheduler.ApplyHistoryBest(log):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            _ = relay.build_module.build(mod, params=params, target='llvm -mtriple=aarch64-linux-none')
    error_list = []
    if os.path.exists('/work/tmp_get_error_from_tvm.txt'):
        with open('/work/tmp_get_error_from_tvm.txt') as f:
            error_list = f.readlines()    
    if os.path.exists('/work/tmp_get_error_from_tvm.txt'):
        os.remove('/work/tmp_get_error_from_tvm.txt')
    
    return error_list

#%%
dir = '/work/experiments/manytime_sd865_resnet18_error'
_, file = get_last_epoch(dir, 29)
p = os.path.join(dir, 'tvm', file)
base_log = os.path.join(dir, 'tvm', 'baseline.log')
log = p+'.log'
config = p + '_config.pkl'

with open(config, 'rb') as f:
    config = pickle.load(f)

# %%
_, file = get_last_epoch(dir, 29)
p = os.path.join(dir, 'tvm', file)
base_log = os.path.join(dir, 'tvm', 'baseline.log')
log = p+'.log'
config = p + '_config.pkl'
with open(config, 'rb') as f:
    config = pickle.load(f)
    
model = load_mode()
new_model = prune_model(model, config)
mod, params = torch_to_tvm(model)
tasks2, task_weights2 = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
pos2, subgraph2 = extract_task_layer(model, tasks2, task_weights2)
print(lower_tensor_ir(log, tasks2[subgraph2[4]]))
#%%

_, file = get_last_epoch(dir, 30)
p = os.path.join(dir, 'tvm', file)
base_log = os.path.join(dir, 'tvm', 'baseline.log')
log = p+'.log'
config = p + '_config.pkl'
with open(config, 'rb') as f:
    config = pickle.load(f)
    
model = load_mode()
new_model = prune_model(model, config)
mod, params = torch_to_tvm(model)
tasks1, task_weights1 = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
pos1, subgraph1 = extract_task_layer(model, tasks1, task_weights1)
print(lower_tensor_ir(log, tasks1[subgraph1[4]]))

# %%
model = load_mode()
mod, params = torch_to_tvm(model)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-none')
pos, subgraph = extract_task_layer(model, tasks, task_weights)
print(lower_tensor_ir(base_log, tasks[subgraph[19]]))
#%%
tasks[11].compute_dag
print(lower_tensor_ir(log, tasks[subgraph[4]]))


# %%
print_task_info(tasks2[11])
print_task_info(tasks1[11])
# %%
p = [{'sparsity':0,'op_types':['Conv2d'],'op_names':['layer4.0.conv2', 'layer4.0.shortcut.0', 'layer4.1.conv2']}]
data = [[0.0625, 450, 0], [0.09375, 420, 1], [0.12823275862068967, 389, 3], [0.1349441680166628, 383, 5], [0.13720150435977793, 381, 6], [0.141726391237606, 377, 7], [0.18718093669215147, 338, 8], [0.19437518129646802, 332, 9], [0.21132433383884092, 318, 10], [0.22122532393785083, 310, 11], [0.22874412093033203, 304, 14], [0.2414023487784333, 295, 15], [0.2439730428658369, 293, 16], [0.25428232121635236, 284, 17], [0.25690012226347275, 283, 18], [0.26477413801150423, 277, 20], [0.2992568966321939, 251, 22], [0.3020424119525282, 249, 23], [0.3076290041312991, 245, 24], [0.3217135111735526, 236, 25], [0.3561962697942423, 212, 30], [0.3743780879760605, 200, 32], [0.38372388236858385, 194, 33]]

dir = '/work/experiments/manytime_sd865_resnet18_error'
base_log = os.path.join(dir, 'tvm', 'baseline.log')
r = None
tasks = None
for d in data:
    model = load_mode()
    p[0]['sparsity'] = d[0]
    print(d[1], d[2], p)
    new_model = prune_model(model, p)
    mod, params = torch_to_tvm(model)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=aarch64-linux-android')
    pos, subgraph = extract_task_layer(model, tasks, task_weights)
    
    r = get_changed_task(model, base_log)
    print(r)
    break

# %%
with auto_scheduler.ApplyHistoryBest(log):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build_module.build(mod, params=params, target='llvm -mtriple=x86_64-linux-gnu')
#%%
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
lib.export_library(lib_fname)
ctx = tvm.cpu()
#%%y
module = graph_executor.GraphModule(lib["default"](ctx))

data_tvm = tvm.nd.array((np.random.uniform(size=[1,3,32,32])).astype('float32'))
module.set_input('input0', data_tvm)
ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=10)

# %%
ftimer()
# %%
module.module.get_source()
# %%
print(mod)
# %%
import tvm
from tvm import relay

# 변수 정의
x = relay.var("x", dtype="float32")
y = relay.var("y", dtype="float32")

# 덧셈 연산 정의
add_op = relay.add(x, y)

# 함수 정의
add_func = relay.Function([x, y], add_op)

# IRModule 생성
mod = tvm.IRModule.from_expr(add_func)
# tvm.build(add_func)
#%%
# Relay IR 출력
print(mod)
# %%
model = load_mode()
p = [{'sparsity':0,'op_types':['Conv2d'],'op_names':['layer4.0.conv2', 'layer4.0.shortcut.0', 'layer4.1.conv2']}]
p[0]['sparsity'] = data[-1][0]
new_model = prune_model(model, p)
mod, params = torch_to_tvm(model)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, 'llvm -mtriple=x86_64-linux-gnu')
pos, subgraph = extract_task_layer(model, tasks, task_weights)

# %%
_, file = get_last_epoch(dir, 34)
log = p = os.path.join(dir, 'tvm', file) + '.log'
sch, arg = tasks[0].apply_best(log)
print(tvm.lower(sch, arg, simple_mode=True))
func = tvm.build(sch, arg, 'llvm -mtriple=x86_64-linux-gnu')
# %%
dev = tvm.cpu()
t = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)

# %%
b3 = tvm.nd.array(np.random.randn(1, 316).astype('float32'))
b4 = tvm.nd.array(np.random.randn(1,1,1,1,158,10,2,1).astype('float32'))
b2 = tvm.nd.array(np.random.randn(10).astype('float32'))
b1 = tvm.nd.array(np.random.randn(1,10).astype('float32'))

t(b3,b4,b2,b1)
# %%
func
# %%
