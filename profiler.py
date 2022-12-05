import torch
import random
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_set_autocast_mode(True)
torch._C._jit_set_nvfuser_single_node_mode(True)
torch._C._debug_set_autodiff_subgraph_inlining(False)

# Create data
random.seed(42)
nreduction = 20480
input_shape = [2048, nreduction]
input_dtype = torch.float16
sizeof_input_dtype = 2
inputs = torch.randn(input_shape, dtype=input_dtype, device='cuda')
dataProcessed = (2*torch.numel(inputs) + nreduction*2)*sizeof_input_dtype

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln = nn.LayerNorm(nreduction, dtype= input_dtype).cuda()

    def forward(self, x):
        out = self.ln(x)
        return out
model_eager = Net()
model = torch.jit.script(model_eager)
for _ in range(1):
    output = model(inputs)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack = True,
        profile_memory=True,
        record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")