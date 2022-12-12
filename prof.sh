#nsys nvprof --print-gpu-trace python $2
#nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o $1 python $2 
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o $1 python $2 
nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column --output .,- ${1}.nsys-rep
