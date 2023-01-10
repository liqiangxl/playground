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

def profileCode_backward(model, inputs, verbose=1):
    bwkTensor = torch.zeros_like(inputs,requires_grad=False, device='cuda')
    for _ in range(3):
        output = model(inputs)
        output.backward(bwkTensor)
    clearL2Cache()
    
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule  = torch.profiler.schedule(wait=0, warmup=0, active=1), # just one iter, avoid L2 cache 
            with_stack = False,
            profile_memory=False,
            record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(inputs)            
            output.backward(bwkTensor)
            
        #clearL2Cache()
    res = (prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    if verbose:
        print(res)
    last_line = res.splitlines()[-1]
    cuda_time_us = last_line.split(" ")[-1] # xxx.000us
    cuda_time_val= float(cuda_time_us[:len(cuda_time_us)-2]) #remove tailing us
    return cuda_time_val # return micro-second

def profileCode(model, inputs, verbose=0):
    for _ in range(3):
        output = model(inputs)
    clearL2Cache()
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule  = torch.profiler.schedule(wait=0, warmup=0, active=10), # just one iter, avoid L2 cache 
            with_stack = False,
            profile_memory=False,
            record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(inputs)
        #clearL2Cache()
    res = (prof.key_averages().table(sort_by="cuda_time_total", row_limit=1))
    if verbose:
        print(res)
    last_line = res.splitlines()[-1]
    cuda_time_us = last_line.split(" ")[-1] # xxx.000us
    cuda_time_val= float(cuda_time_us[:len(cuda_time_us)-2]) #remove tailing us
    return cuda_time_val # return micro-second

def getSpeed(dim0=2048, dim1=20480, eager=True, apex=True, fuser=True):
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    sizeof_input_dtype = 2
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    bytes_per_GB = 1.0e9
    dataProcessed = (2*torch.numel(x)*sizeof_input_dtype + dim1*2*sizeof_input_dtype + dim0*2*4) / bytes_per_GB
    #print(torch.numel(x), dim0*dim1, dataProcessed)

    net = nn.LayerNorm(dim1, dtype= input_dtype, elementwise_affine=True).cuda();
    apx = FastLayerNorm(dim1).half().cuda();
    #apx = FastLayerNorm(dim1).cuda();
    eager=1.0
    fuser=1.0
    apex=1.0
    fuser_us=0.0
    apex_us = 0.0

    micro_to_second = 1.0e6
    #print("===================== eager ============")
    if eager:
        torch.cuda.empty_cache()
        eager_us = profileCode_backward(net, x)
        eager = dataProcessed / eager_us * micro_to_second
    
    # apex
    #print("===================== apex ============")
    if apex:
        torch.cuda.empty_cache()
        apex_us = profileCode_backward(apx, x)
        apex = dataProcessed / apex_us * micro_to_second

    # fuser
    #print("===================== fuser ============")
    if fuser:
        torch.cuda.empty_cache()
        fuser_us = profileCode_backward(torch.jit.script(net), x)
        fuser = dataProcessed / fuser_us * micro_to_second

    print("shape= {:d} x {:d} bw_eager= {:.1f} bw_apex= {:.1f} bw_fuser= {:.1f} GB/s, ratio= {:.2f} {:.2f}, fuser_time= {:.0f}, apex_time= {:.0f} micro-sec"
    .format(dim0,dim1,eager,apex,fuser,fuser/eager,fuser/apex, fuser_us, apex_us))

    #print("shape= {:5d} x {:5d} bw_eager= {:5.1f} bw_apex= {:6.1f} bw_fuser= {:6.1f} GB/s, ratio= {:.2f} {:.2f}, fuser_time= {:4.0f} micro-sec"
    #.format(dim0,dim1,eager,apex,fuser,fuser/eager,fuser/apex, fuser_us))

def run_all(eager=True, apex=True, fuser=True):

    popular = [768,1024,2048,4096,8192,10240,20480,40960]
    popular = [10240]
    for d1 in popular:
        getSpeed(2048, d1, eager, apex, fuser)

   

def run_apex():
    dim0 = 2048 
    dim1 = 10240
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    apx = FastLayerNorm(dim1).half().cuda();
    for _ in range(10):
        clearL2Cache()
        output = apx(x)
    
def run_fuser():
    dim0 = 2048
    dim1 = 10240
    input_shape = [dim0, dim1]
    input_dtype = torch.float16
    x = torch.randn(input_shape, dtype=input_dtype, device='cuda')
    net_eager = nn.LayerNorm(dim1, dtype= input_dtype, elementwise_affine=True).cuda();
    net_fuser = torch.jit.script(net_eager)
    for _ in range(3):
        output = net_fuser(x)
    clearL2Cache()
    output = net_fuser(x)

run_all(True, True, True)
#run_apex()
#run_fuser()
