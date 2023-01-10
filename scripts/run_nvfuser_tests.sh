PYTORCH_NVFUSER_DUMP=ptxas_verbose,dump_eff_bandwidth,cuda_to_file,scheduler_params ./bin/nvfuser_tests --gtest_filter=NVFuserTest.$1
#PYTORCH_NVFUSER_DUMP=ptxas_verbose,dump_eff_bandwidth,cuda_to_file,scheduler_params ./bin/nvfuser_tests --gtest_filter=NVFuserTest.$1
#PYTORCH_NVFUSER_DUMP=ptxas_verbose,dump_eff_bandwidth,cuda_to_file ./bin/nvfuser_tests --gtest_filter=NVFuserTest.FusionMagicSchedulerLayerNormalization_CUDA
