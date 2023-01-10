#nsys nvprof --print-gpu-trace python $2
#nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o $1 python $2 
#nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column --output .,- ${1}.nsys-rep
if [ $# -eq 0 ]
then
    python profiler.py
else
    nsys profile --output=$1.nsys-rep --force-overwrite=true python profiler.py
    nsys stats --report=kernexecsum --force-overwrite=true --format=csv,column --output .,- $1.nsys-rep
    python process_nsys_kernexecsum.py $1
fi
