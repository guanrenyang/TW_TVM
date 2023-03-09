cmake_minimum_required(VERSION 3.10)

project(matrixMul CXX CUDA)

# Set path to nvcc compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)

find_package(CUDAToolkit REQUIRED)

add_executable(cublas_dense cublas_dense.cu)
set_target_properties(cublas_dense PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cublas_dense PRIVATE CUDA::cudart CUDA::cublas)

add_executable(cublas_sparse cublas_sparse.cu)
set_target_properties(cublas_sparse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cublas_sparse PRIVATE CUDA::cudart CUDA::cublas)


