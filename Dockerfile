FROM python:3.6.15-bullseye

WORKDIR /work

RUN apt-get update && apt-get install -y openssh-server 
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22

# Install CMake 3.30
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.tar.gz && \
    tar -zxf cmake-3.30.4-linux-x86_64.tar.gz && \
    mv ./cmake-3.30.4-linux-x86_64 ./cmake-30 && \
    cd ./cmake-30 && \
    cp -r . /usr && \
    cd .. && rm -rf ./cmake-30 && rm -rf ./cmake-3.30.4-linux-x86_64.tar.gz

# Install LLVM 17
# RUN apt install -y lsb-release wget software-properties-common gnupg libzstd-dev libtinfo5 && \
#     wget https://apt.llvm.org/llvm.sh && \
#     chmod +x llvm.sh && ./llvm.sh 16 all

# pip install nni=2.10
# https://pytorch.org/get-started/previous-versions/#v180
# pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# cmake -DUSE_LLVM=llvm-config-16 -DUSE_RPC=ON -DUSE_VULKAN=OFF -DUSE_GRAPH_EXECUTOR=ON -DUSE_LIBBACKTRACE=OFF  ..

# Install LLVM 8
RUN apt install -y lsb-release wget software-properties-common gnupg libzstd-dev libtinfo5
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/clang+llvm-8.0.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz && \
    tar -xf clang+llvm-8.0.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz -C . && \
    mv ./clang+llvm-8.0.1-x86_64-linux-gnu-ubuntu-14.04 ./llvm-8 && \
    cd ./llvm-8 && \
    cp -R * /usr && cd .. && \
    rm -rf /work/llvm-8
    # rm -rf ./clang+llvm-8.0.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz && \

CMD ["/usr/sbin/sshd", "-D"]
