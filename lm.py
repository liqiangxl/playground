import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch
from functorch.compile import memory_efficient_fusion
import time
import functools
import random



import torch.cuda.nvtx as nvtx
import torch.optim as optim
from apex.normalization import FusedLayerNorm
from apex.contrib.layer_norm import FastLayerNorm 
import time

torch.backends.cudnn.benchmark = True
#Turning on NVFuser Knobs
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_autocast_mode(True)
torch._C._jit_set_nvfuser_single_node_mode(True)
torch._C._debug_set_autodiff_subgraph_inlining(False)

random.seed(42)



# Utility to profile the workload
def profile_workload(forward_func, warm_count=50, iteration_count=100, label=""):
    # Perform warm-up iterations
    for _ in range(warm_count):
        output = forward_func()
    # Synchronize the GPU before starting the timer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iteration_count):
        output = forward_func()
    # Synchronize the GPU before stopping the timer
    torch.cuda.synchronize()
    stop = time.perf_counter()
    return (stop - start)/iteration_count


def getSpeed(dim0=2048, dim1=20480):
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    sizeof_input_dtype = 2
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')


    net = nn.LayerNorm(dim1, dtype= input_dtype).cuda();
    apx = FastLayerNorm(dim1).half().cuda();
    # Run and profile eager mode    
    func_eager = functools.partial(net, x,)
    func_fuser = functools.partial(torch.jit.script(net), x,)
    func_apex  = functools.partial(apx, x,)
    dataProcessed = (2*torch.numel(x) + dim1*4)*sizeof_input_dtype / 1e9;
    bd_eager = dataProcessed / profile_workload(func_eager)
    bd_fuser = dataProcessed / profile_workload(func_fuser)
    bd_apex  = dataProcessed / profile_workload(func_apex)
    print("shape= {:} x {:} bw_eager= {:.1f} bw_apex= {:.1f} bw_fuser= {:.1f} GB/s, ratio= {:.2f} {:.2f}".format(dim0,dim1,bd_eager,bd_apex, bd_fuser,bd_fuser/bd_eager,bd_fuser/bd_apex))

#getSpeed(4096,40960)
#popular = [768,1024,2048,4096,8192,10240,20480,40960]
popular = [4096]
for d1 in popular:
    getSpeed(dim0=512, dim1=d1)
