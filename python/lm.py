import time
import functools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import memory_efficient_fusion
import torch.cuda.nvtx as nvtx
import torch.optim as optim

from apex.normalization import FusedLayerNorm
from apex.contrib.layer_norm import FastLayerNorm 


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
bytes_per_GB = 1024*1024*1024

# clear L2 cache to avoid speedup of gmem in repeated runs
def clearL2Cache():
  l2_cache_size = 40*1024*1024 #at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  nele = int(l2_cache_size / 4)
  x = torch.empty(nele, dtype=torch.float32, device='cuda', requires_grad=False)
  y = torch.clone(x)

  


# Utility to profile the workload
def profile_workload_wallclock(forward_func, warm_count=50, iteration_count=1000, label=""):
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

# https://github.com/kevinstephano/simple_dl_models/blob/main/execution/execution_loop.py#L104-L138
def profile_workload(forward_func, warm_count=5, iteration_count=10, label=""):
    # Perform warm-up iterations
    for _ in range(warm_count):
        output = forward_func()
    # Synchronize the GPU before starting the timer
    beg_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    exec_time_ms = 0.0
    torch.cuda.synchronize()
    
    beg_event.record()

    for _ in range(iteration_count):
        output = forward_func()

    end_event.record()
    beg_event.synchronize()
    end_event.synchronize()
    
    exec_time_ms = beg_event.elapsed_time(end_event) / iteration_count
    print("exec_time_ms= {:} ms".format(exec_time_ms))
    return exec_time_ms / 1000.0 

# clear L2
def profile_workload_clearl2(forward_func, warm_count=5, iteration_count=10, label=""):
    for _ in range(warm_count):
        output = forward_func()

    beg_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    exec_time_ms = 0.0
    for _ in range(iteration_count):
        #clearL2Cache()
        beg_event.record()
        output = forward_func()
        end_event.record()
        end_event.synchronize()

        this_time = beg_event.elapsed_time(end_event)
        exec_time_ms += this_time
        #print("this_time= {:} ms".format(this_time))

    return exec_time_ms / 1000.0 / iteration_count


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
    dataProcessed = (2*torch.numel(x) + dim1*4)*sizeof_input_dtype / bytes_per_GB

    #print("-------running eager ------------")
    #torch.cuda.empty_cache()
    #bd_eager = dataProcessed / profile_workload(func_eager)

    print("\n-------running fuser ------------")
    torch.cuda.empty_cache()
    bd_fuser = dataProcessed / profile_workload(func_fuser)

    #print("\n-------running apex ------------")
    #torch.cuda.empty_cache()
    #bd_apex  = dataProcessed / profile_workload(func_apex)

    print("shape= {:} x {:} bw_eager= {:.1f} bw_apex= {:.1f} bw_fuser= {:.1f} GB/s, ratio= {:.2f} {:.2f}".format(dim0,dim1,bd_eager,bd_apex, bd_fuser,bd_fuser/bd_eager,bd_fuser/bd_apex))

    print("shape= {:} x {:} bw_eager= {:.4f} bw_apex= {:.4f} bw_fuser= {:.4f} ms".format(dim0,dim1,dataProcessed/bd_eager*1e3,dataProcessed/bd_apex*1e3, dataProcessed/bd_fuser*1e3))

def debugEager(dim0=2048, dim1=20480):
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    sizeof_input_dtype = 2
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    net = nn.LayerNorm(dim1, dtype= input_dtype).cuda();
    func_eager = functools.partial(net, x,)
    dataProcessed = (2*torch.numel(x) + dim1*4)*sizeof_input_dtype / bytes_per_GB

    torch.cuda.empty_cache()
    bd_eager = dataProcessed / profile_workload(func_eager, warm_count=100, iteration_count=10)

    torch.cuda.empty_cache()
    bd_eager_l2 = dataProcessed / profile_workload_clearl2(func_eager, warm_count=100, iteration_count=10)

    print("shape= {:} x {:}, before= {:} bytes, bd_eager= {:.3f} GB/s, after add code to clear L2= {:.3f} GB/s".format(dim0,dim1,dataProcessed*bytes_per_GB, bd_eager, bd_eager_l2))
    print("shape= {:} x {:} before= {:.4f} micro-s, after add code to clear L2= {:.4f} micro-s".format(dim0,dim1,dataProcessed/bd_eager*1e6,dataProcessed/bd_eager_l2*1e6))


#getSpeed(4096,40960)
#popular = [768,1024,2048,4096,8192,10240,20480,40960]
popular = [4096]
for d1 in popular:
    getSpeed(dim0=2048, dim1=d1)
