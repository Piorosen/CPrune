#!/bin/bash

# sudo apt install -y lsb-release wget software-properties-common gnupg libzstd-dev libtinfo5 && \
# wget https://apt.llvm.org/llvm.sh && \
# chmod +x llvm.sh && ./llvm.sh 17 all

git clone --recursive https://github.com/apache/tvm tvm && cd tvm && git checkout v0.8.0
make runtime -j4
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/python" >> ~/.bashrc
python -m pip install -e $(pwd)/python
python -m tvm.exec.rpc_server --tracker=129.254.196.234:9190 --key=rasp4b-64 --custom-addr=129.254.74.132 --no-fork
