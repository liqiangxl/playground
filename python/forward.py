import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import torch.optim as optim
import pprint

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

tensor_dtype = torch.float16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln = nn.LayerNorm(2048)

    def forward(self, x):
        out = self.ln(x)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.cuda()

target_shape = [2048, 1, 2048]
input_shape  = [2048, 1, 2048]

def generate_inp_res(generate_tensor=''):
        if generate_tensor == 'input':
            tensor = torch.rand(input_shape, dtype=tensor_dtype, requires_grad=True, device=device)
        elif generate_tensor == 'target':
            tensor = torch.rand(target_shape, dtype=tensor_dtype, requires_grad=False, device=device)
        else:
            assert False, "Generate_tensor must be either input or target. It was {}".format(generate_tensor)
                
        return tensor

with torch.autograd.profiler.emit_nvtx(record_shapes=True):
    net = net.half()
    scripted_fn = torch.jit.script(net)

    network_fn = scripted_fn
    #network_fn = net

    bench_iters = 8
    profile_iter = 5
    
    for idx in range(bench_iters):
    
        input_tensor = generate_inp_res('input')
        target_tensor = generate_inp_res('target')

        input_tensor.grad = None
        network_fn.zero_grad(set_to_none=True)

        if idx == profile_iter:
            torch.cuda.profiler.start()
            torch.cuda.cudart().cudaProfilerStart()

        outputs = network_fn(input_tensor)
#        outputs.backward(target_tensor)
        
        if idx == profile_iter:
            torch.cuda.cudart().cudaProfilerStop()
