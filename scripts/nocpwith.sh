PYTORCH_NVFUSER_OUTER_REDUCTION_BENCHMARK=1 COMPUTE_WITH=0 PYTORCH_NVFUSER_DUMP=dump_eff_bandwidth ~/nf/pytorch/build/bin/test_jit  --gtest_filter=NVFuserTest.FusionGridPersistentBatchNormChannelsLastHalf256x28x512_CUDA