#include <third_party/nvfuser/csrc/executor.h>
#include <third_party/nvfuser/csrc/fusion.h>
#include <third_party/nvfuser/csrc/ir_all_nodes.h>
#include <third_party/nvfuser/csrc/ir_builder.h>
#include <third_party/nvfuser/csrc/ir_utils.h>
#include <third_party/nvfuser/csrc/lower2device.h>
#include <third_party/nvfuser/csrc/ops/all_ops.h>
#include <third_party/nvfuser/csrc/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupLayerNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const int kReductionAxis = 1;
  const float kEps = 1e-5;

  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  // setup fusion
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto layer_norm_results = layer_norm(input, 1, weight, bias, eps_ptr);

  auto output = layer_norm_results.output;

  if (dtype != DataType::Float) {
    output = castOp(dtype, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};
  const float kEps = 1e-5;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  std::vector<c10::IValue> aten_inputs({input, weight, bias});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      int64_t(dataTypeSize(dtype)));
}

static void NvFuserScheduler_TIMM_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  // NHWC, norm on C
  std::vector<int64_t> input_shape{
      benchmark_state.range(0) * benchmark_state.range(2) *
          benchmark_state.range(2),
      benchmark_state.range(1)};
  const float kEps = 1e-5;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  std::vector<c10::IValue> aten_inputs({input, weight, bias});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void Baseline_TIMM_LayerNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  // NHWC, norm on C
  std::vector<int64_t> input_shape{
      benchmark_state.range(0) * benchmark_state.range(2) *
          benchmark_state.range(2),
      benchmark_state.range(1)};
  const size_t kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (auto idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_LayerNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};
  const size_t kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (auto idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_LayerNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Float);
}

static void Baseline_LayerNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_fp32,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_fp16,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_LayerNorm_fp16,
    setupLayerNorm,
    NvFuserScheduler_TIMM_LayerNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{8, 16, 32, 64, 128, 256},
         {24, 40, 48, 56, 72, 152, 184, 200, 368},
         {7, 14, 28, 56, 112}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{128, 256, 512, 1024, 2048},
         {24, 40, 48, 56, 72, 152},
         {7, 14, 28, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();


static void Baseline_TIMM_LayerNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_TIMM_LayerNorm(benchmark_state, DataType::Half);
}

BENCHMARK(Baseline_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{8, 16, 32, 64, 128, 256},
         {24, 40, 48, 56, 72, 152, 184, 200, 368},
         {7, 14, 28, 56, 112}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{128, 256, 512, 1024, 2048},
         {24, 40, 48, 56, 72, 152},
         {7, 14, 28, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();