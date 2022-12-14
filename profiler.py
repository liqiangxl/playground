import torch
import random
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from apex.contrib.layer_norm import FastLayerNorm

torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_autocast_mode(True)
torch._C._jit_set_nvfuser_single_node_mode(True)
torch._C._debug_set_autodiff_subgraph_inlining(False)

# clear L2 cache to avoid speedup of gmem in repeated runs
def clearL2Cache():
  l2_cache_size = 40*1024*1024 #at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  nele = int(l2_cache_size / 4)
  x = torch.empty(nele, dtype=torch.float32, device='cuda', requires_grad=False)
  y = torch.clone(x)

def profileCode(model, inputs):
    for _ in range(3):
        output = model(inputs)
    clearL2Cache()
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule  = torch.profiler.schedule(wait=0, warmup=0, active=1), # just one iter, avoid L2 cache 
            with_stack = False,
            profile_memory=False,
            record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(inputs)
    res = (prof.key_averages().table(sort_by="cuda_time_total", row_limit=1))
    last_line = res.splitlines()[-1]
    cuda_time_us = last_line.split(" ")[-1] # xxx.000us
    cuda_time_val= float(cuda_time_us[:len(cuda_time_us)-2]) #remove tailing us
    return cuda_time_val # return micro-second

def getSpeed(dim0=2048, dim1=20480):
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    sizeof_input_dtype = 2
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    bytes_per_GB = 1024*1024*1024
    dataProcessed = (2*torch.numel(x) + dim1*4)*sizeof_input_dtype / bytes_per_GB


    net = nn.LayerNorm(dim1, dtype= input_dtype).cuda();
    apx = FastLayerNorm(dim1).half().cuda();

    micro_to_second = 1.0e6
    #print("===================== eager ============")
    torch.cuda.empty_cache()
    eager_us = profileCode(net, x)
    eager = dataProcessed / eager_us * micro_to_second
    
    # apex
    #print("===================== apex ============")
    torch.cuda.empty_cache()
    apex = dataProcessed / profileCode(apx, x) * micro_to_second

    # fuser
    #print("===================== fuser ============")
    torch.cuda.empty_cache()
    fuser_us = profileCode(torch.jit.script(net), x)
    fuser = dataProcessed / fuser_us * micro_to_second

    print("shape= {:d} x {:d} bw_eager= {:.1f} bw_apex= {:.1f} bw_fuser= {:.1f} GB/s, ratio= {:.2f} {:.2f}, fuser_time= {:.0f} micro-sec"
    .format(dim0,dim1,eager,apex,fuser,fuser/eager,fuser/apex, fuser_us))

    #print("shape= {:5d} x {:5d} bw_eager= {:5.1f} bw_apex= {:6.1f} bw_fuser= {:6.1f} GB/s, ratio= {:.2f} {:.2f}, fuser_time= {:4.0f} micro-sec"
    #.format(dim0,dim1,eager,apex,fuser,fuser/eager,fuser/apex, fuser_us))

def run_all():
    # bert
    dim0_list = [8192, 16384, 32768]
    for d0 in dim0_list:
        getSpeed(dim0=d0, dim1=1024) 

    popular = [768,1024,2048,4096,8192,10240,20480,40960]
    #popular = [768,1024,2048,2304,3072,4096,8192,10240,12288,20480,40960]
    for d1 in popular:
        getSpeed(dim0=108, dim1=d1)

   

def run_apex():
    dim0 = 8192
    dim1 = 10240
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    apx = FastLayerNorm(dim1).half().cuda();
    out = apx(x)
    
def run_fuser():
    dim0 = 8192
    dim1 = 10240
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    net_eager = nn.LayerNorm(dim1, dtype= input_dtype).cuda();
    net_fuser = torch.jit.script(net_eager)
    for _ in range(3):
        output = net_fuser(x)
    clearL2Cache()
    output = net_fuser(x)

run_all()
#run_apex()
#run_fuser()
