#!/bin/bash
chown -R root:root /work

git submodule init && git submodule update && \
   git submodule foreach git submodule update --init --recursive

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
   echo "set(USE_LLVM \"llvm-config-17 --ignore-libllvm --link-static\")" >> config.cmake && \
   echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
   echo "set(CMAKE_C_COMPILER /usr/bin/clang-17)" >> config.cmake && \
   echo "set(CMAKE_CXX_COMPILER /usr/bin/clang++-17)" >> config.cmake && \
   cmake .. && cmake --build . --parallel $(nproc)

echo 'export TVM_HOME=/work/3rdparty/tvm' >> ~/.bashrc && \
echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc && \
echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc && source ~/.bashrc

pip install -e /work/3rdparty/tvm/python
pip install setuptools nni==3.0 pytest python-dotenv xgboost
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
