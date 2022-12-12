import time
import functools
import random
import torch
import torch.nn as nn


random.seed(42)
bytes_per_GB = 1024*1024*1024

# https://github.com/kevinstephano/simple_dl_models/blob/main/execution/execution_loop.py#L104-L138
def profile_workload(forward_func, warm_count=500, iteration_count=1000, label=""):
    for _ in range(warm_count):
        output = forward_func()

    torch.cuda.synchronize()
    beg_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    exec_time_ms = 0.0    
    
    beg_event.record()
    for _ in range(iteration_count):
        output = forward_func()
    end_event.record()
    beg_event.synchronize()
    end_event.synchronize()
    
    exec_time_ms = beg_event.elapsed_time(end_event) / iteration_count
    print("exec_time_ms= {:} ms".format(exec_time_ms))
    return exec_time_ms / 1000.0 


def profile_workload_acc(forward_func, warm_count=500, iteration_count=1000, label=""):
    for _ in range(warm_count):
        output = forward_func()

    torch.cuda.synchronize()
    beg_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    exec_time_ms = 0.0    
    
    
    for _ in range(iteration_count):
        beg_event.record()
        output = forward_func()
        end_event.record()
        beg_event.synchronize()
        end_event.synchronize()
        exec_time_ms += beg_event.elapsed_time(end_event)
    
    exec_time_ms = exec_time_ms / iteration_count
    print("exec_time_ms= {:} ms".format(exec_time_ms))
    return exec_time_ms / 1000.0 

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
    bd_eager_l2 = dataProcessed / profile_workload_acc(func_eager, warm_count=100, iteration_count=10)

    print("shape= {:} x {:} out_loop= {:.4f} micro-s, in_loop= {:.4f} micro-s".format(dim0,dim1,dataProcessed/bd_eager*1e6,dataProcessed/bd_eager_l2*1e6))


#getSpeed(4096,40960)
#popular = [768,1024,2048,4096,8192,10240,20480,40960]
popular = [4096]
for d1 in popular:
    debugEager(dim0=2048, dim1=d1)
