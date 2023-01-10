file=/workspace/playground/debug/all_benchmarks/${1}${2}
./bin/nvfuser_bench --benchmark_min_time=0.01 --benchmark_out_format=csv --benchmark_out=${file}.csv  2>&1 |tee ${file}.log
#./bin/nvfuser_bench --benchmark_min_time=0.01 --benchmark_out_format=csv --benchmark_out=${file}.csv --benchmark_filter=${1} 2>&1 |tee ${file}.log
#./bin/nvfuser_bench --benchmark_min_time=0.1 --benchmark_out_format=csv --benchmark_out=${file}.csv --benchmark_filter=${1} 2>&1 |tee ${file}.log
#./bin/nvfuser_bench --benchmark_min_time=0.1 --benchmark_out_format=csv --benchmark_out=/workspace/playground/debug/ln_forward/${1}${2}.csv --benchmark_filter=${1} 2>&1 |tee /workspace/playground/debug/ln_forward/${1}${2}.log
