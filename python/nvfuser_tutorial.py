import torch
import torch.nn.functional as F
import functorch
from functorch.compile import memory_efficient_fusion
from copy import deepcopy
from typing import List
import time
import functools
import random

random.seed(42)

if torch.__version__ < (1, 12, 0):
    raise RuntimeError(
        "PyTorch >= 1.12.0 required, but your environment uses torch=={}".format(
            torch.__version__
        )
    )

major, minor, _ = functorch.__version__.split(".")
if int(major) == 0 and int(minor) < 2:
    raise RuntimeError(
        "FuncTorch >= 0.2.0 required, but your environment uses functorch=={}".format(
            functorch.__version__
        )
    )

def composite_definition(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    normalization_axis: int,
    dropout_prob: float,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, dropout_prob, training=True)
    norm_input = dropout_out + input2
    norm_output = F.layer_norm(
        norm_input, (input1.size(normalization_axis),), weight, bias2
    )
    return norm_output


# Setup initial tensors and parameters
input_size = [64, 128, 1024]
device = "cuda"
dtype = torch.float32

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
input2 = torch.rand_like(input1).requires_grad_()

# Precompute a grad output tensor, for this example it's the same size
# as the inputs
grad_output = torch.rand_like(input1)

# Randomly initialize the model parameters
weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

parameters = [input1, input2, weight, bias1, bias2]

# Utility to profile the workload
def profile_workload(forward_func, grad_output, iteration_count=100, label=""):
    # Perform warm-up iterations
    for _ in range(3):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before starting the timer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iteration_count):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before stopping the timer
    torch.cuda.synchronize()
    stop = time.perf_counter()
    iters_per_second = iteration_count / (stop - start)
    if label:
        print(label)
    print("Average iterations per second: {:.2f}".format(iters_per_second))

# Run and profile eager mode execution on the composite definition of our
# operations.
func = functools.partial(
    composite_definition,
    input1,
    input2,
    weight,
    bias1,
    bias2,
    normalization_axis=2,
    dropout_prob=0.1,
)
profile_workload(
    func, grad_output, iteration_count=100, label="Eager Mode - Composite definition"
)    

# TorchScript & nvFuser
scripted_composite_definition = torch.jit.script(composite_definition)
func = functools.partial(
    scripted_composite_definition,
    input1,
    input2,
    weight,
    bias1,
    bias2,
    normalization_axis=2,
    dropout_prob=0.1,
)
profile_workload(
    func, grad_output, iteration_count=100, label="TorchScript - Composite definition"
)