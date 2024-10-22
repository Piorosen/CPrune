#!/bin/bash
# chown -R root:root /work

git config --global --add safe.directory /work
git config --global --add safe.directory /work/3rdparty/tvm
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/cutlass
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/dlpack
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/dmlc-core
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/libbacktrace
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/rang
git config --global --add safe.directory /work/3rdparty/tvm/3rdparty/vta-hw

# git submodule init && git submodule update && \
#    git submodule foreach git submodule update --init --recursive
# cd /work/3rdparty/tvm && \
#    rm -rf build && mkdir build && cd build && \
#    cp ../cmake/config.cmake . && \
#    echo "set(CMAKE_BUILD_TYPE RELEASE)" >> config.cmake && \
#    echo "set(USE_CUDA   OFF)" >> config.cmake && \
#    echo "set(USE_METAL  OFF)" >> config.cmake && \
#    echo "set(USE_VULKAN OFF)" >> config.cmake && \
#    echo "set(USE_OPENCL OFF)" >> config.cmake && \
#    echo "set(USE_CUBLAS OFF)" >> config.cmake && \
#    echo "set(USE_CUDNN  OFF)" >> config.cmake && \
#    echo "set(USE_CUTLASS OFF)" >> config.cmake && \
#    echo "set(USE_LLVM \"llvm-config-12 --ignore-libllvm --link-static\")" >> config.cmake && \
#    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
#    echo "set(CMAKE_C_COMPILER /usr/bin/clang-12)" >> config.cmake && \
#    echo "set(CMAKE_CXX_COMPILER /usr/bin/clang++-12)" >> config.cmake && \
#    CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12 \
#    cmake .. && cmake --build . --parallel $(nproc)

echo 'export TVM_HOME=/work/3rdparty/tvm' >> ~/.bashrc && \
echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc && \
echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc && source ~/.bashrc
echo 'export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' >> /root/.bashrc && source ~/.bashrc

pip install --upgrade pip
pip install -e /work/3rdparty/tvm/python
pip install --ignore-installed PyYAML
python3.8 -m pip install setuptools nni==2.10 pytest python-dotenv numpy==1.19.5 xgboost==1.4.2 tensorboard
# python3.8 -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu121
cd /work
