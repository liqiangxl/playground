
pip uninstall -y torch
CUDA_HOME="/usr/local/cuda" \
CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
NCCL_INCLUDE_DIR="/usr/include/" \
NCCL_LIB_DIR="/usr/lib/" \
USE_SYSTEM_NCCL=0 \
USE_KINETO=1 \
CFLAGS=-DNO_CUDNN_DESTROY_HANDLE \
python setup.py develop
