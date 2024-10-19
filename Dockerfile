# FROM python:3.12.6-bookworm
# FROM python:3.8.17-buster
# FROM nvcr.io/nvidia/pytorch:22.10-py3
# FROM nvcr.io/nvidia/cuda:11.1-base-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:21.02-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

WORKDIR /work

RUN apt-get update
RUN apt-get update && apt-get install -y openssh-server 
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#Port 22$/Port 5911/' /etc/ssh/sshd_config
EXPOSE 5911 

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
    chmod +x llvm.sh && ./llvm.sh 12 all


RUN apt install -y curl wget python3.8 python3.8-distutils && \
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.8 get-pip.py && \
        git config --global user.email "chacha@udon.party" && \
        git config --global user.name "Piorosen"
        
COPY setup.sh setup.sh 
RUN echo 'export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' >> /root/.bashrc
RUN apt install git make zlib1g-dev

# RUN ./setup.sh


CMD ["/usr/sbin/sshd", "-D"]
