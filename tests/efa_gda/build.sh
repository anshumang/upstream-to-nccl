#!/bin/bash
set -e

# Build the gin_putvalue_gpu functional test.
#
# gin_putvalue_gpu — small-value RDMA write via NCCL device API
#                   (gin.putValue<T>). Exercises the dedicated PutValue
#                   endpoint + GPU source slot pool added in
#                   "nccl_device/efa_gda: implement PutValue via
#                   dedicated endpoint".
#
# Other gin_*_gpu tests (gin_put_gpu, gin_signal_gdaki_gpu) live in
# this directory too once the GDAKI baseline series adds them; their
# build commands will be added back at that point.

NCCL_INSTALL=${NCCL_INSTALL:-$HOME/projects/proj_gin_dev/install/nccl}
MPI_HOME=${MPI_HOME:-/opt/amazon/openmpi}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

$CUDA_HOME/bin/nvcc -std=c++17 \
    -gencode=arch=compute_100,code=sm_100 \
    -I${NCCL_INSTALL}/include \
    -I${MPI_HOME}/include \
    -L${NCCL_INSTALL}/lib \
    -L${MPI_HOME}/lib \
    -L${CUDA_HOME}/lib64 \
    -o gin_putvalue_gpu \
    gin_putvalue_gpu.cu \
    -lnccl -lmpi -lcuda -lcudart -ldl
echo "Built: gin_putvalue_gpu"
