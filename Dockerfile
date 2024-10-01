FROM python:3.12.6-bookworm

WORKDIR /work

RUN apt-get update && apt-get install -y openssh-server 

RUN mkdir /var/run/sshd
RUN echo 'chacha:chacha' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

# Install CMake 3.30
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
    tar -zxf cmake-3.30.4-linux-x86_64.tar.gz -C ./3rdparty && \
    mv ./3rdparty/cmake-3.30.4-linux-x86_64 ./3rdparty/cmake-30 && \
    cd ./3rdparty/cmake-30 && \
    cp -r . /usr && \
    cd .. && rm -rf ./cmake-30 && rm -rf ../cmake-3.30.4-linux-x86_64.tar.gz

# Install LLVM 17
RUN apt install -y lsb-release wget software-properties-common gnupg libzstd-dev libtinfo5
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && ./llvm.sh 17 all

RUN git submodule init && git submodule update && \
    git submodule foreach git submodule update --init --recursive

RUN cd /work/3rdparty/tvm && \
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

RUN echo 'export TVM_HOME=/work/3rdparty/tvm' >> ~/.bashrc && \
echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc && \
echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc && source ~/.bashrc

RUN pip install -e /work/3rdparty/tvm/python

CMD ["/usr/sbin/sshd", "-D"]
