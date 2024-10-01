FROM python:3.12.6-bookworm

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
RUN apt install -y lsb-release wget software-properties-common gnupg libzstd-dev libtinfo5 && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && ./llvm.sh 17 all


CMD ["/usr/sbin/sshd", "-D"]
