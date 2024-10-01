#!/bin/bash

git submodule init && git submodule update && \
   git submodule foreach git submodule update --init --recursive

# https://tvm.apache.org/docs/v0.8.0/install/from_source.html

cd /work/3rdparty/tvm && \ 
   rm -rf build && mkdir build && cd build && \
   cp ../cmake/config.cmake . && \
   echo "set(CMAKE_BUILD_TYPE RELEASE)" >> config.cmake && \
   echo "set(USE_CUDA   OFF)" >> config.cmake && \
   echo "set(USE_METAL  OFF)" >> config.cmake && \
   echo "set(USE_VULKAN OFF)" >> config.cmake && \
   echo "set(USE_OPENCL OFF)" >> config.cmake && \
   echo "set(USE_CUBLAS OFF)" >> config.cmake && \
   echo "set(USE_CUDNN  OFF)" >> config.cmake && \
   echo "set(USE_CUTLASS OFF)" >> config.cmake && \
   echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake && \
   echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
   echo "set(CMAKE_C_COMPILER /usr/bin/clang)" >> config.cmake && \
   echo "set(CMAKE_CXX_COMPILER /usr/bin/clang++)" >> config.cmake && \
   cmake .. && cmake --build . --parallel $(nproc)

pip install -e /work/3rdparty/tvm/python