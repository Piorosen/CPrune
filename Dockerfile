FROM python:3.11.10-bullseye

WORKDIR /work

RUN apt-get update && apt-get install -y openssh-server libtinfo5 mak

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
# RUN git submodule foreach git submodule update --init --recursive

# RUN pip install nni

# RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
#     tar -xf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz -C ./3rdparty && \
#     mv ./3rdparty/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04 ./3rdparty/llvm-18 && \
#     cd ./3rdparty/llvm-18 && \
#     cp -R * /usr/local && \
#     rm -rf ./clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz

# RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
#     tar -zxf cmake-3.30.4-linux-x86_64.tar.gz -C ./3rdparty && \
#     mv ./3rdparty/cmake-3.30.4-linux-x86_64 ./3rdparty/cmake-30 && \
#     cd ./3rdparty/cmake-30 && \
#     cp -R . * /usr/local && \
#     rm -rf cmake-3.30.4-linux-x86_64.tar.gz

# RUN echo 'export PATH=/work/3rdparty/llvm-18/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
# RUN echo 'export PATH=/work/3rdparty/cmake-30/bin:$PATH' >> ~/.bashrc && source ~/.bashrc

# RUN cd /work/3rdparty/tvm && \
#     rm -rf build && mkdir build && cd build && \
#     cp ../cmake/config.cmake . && \
#     echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake && \
#     echo "set(USE_CUDA   OFF)" >> config.cmake && \
#     echo "set(USE_METAL  OFF)" >> config.cmake && \
#     echo "set(USE_VULKAN OFF)" >> config.cmake && \
#     echo "set(USE_OPENCL OFF)" >> config.cmake && \
#     echo "set(USE_CUBLAS OFF)" >> config.cmake && \
#     echo "set(USE_CUDNN  OFF)" >> config.cmake && \
#     echo "set(USE_CUTLASS OFF)" >> config.cmake && \
#     echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake && \
#     echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
#     cmake .. && cmake --build . --parallel $(nproc)

# RUN echo 'export TVM_HOME=/work/3rdparty/tvm' >> ~/.bashrc && source ~/.bashrc
# RUN echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc && source ~/.bashrc
# RUN echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc && source ~/.bashrc
# RUN pip install -e /work/3rdparty/tvm/python


CMD ["/usr/sbin/sshd", "-D"]
