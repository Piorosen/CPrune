#!/bin/bash

commands=("make" "cmake" "git" "gcc" "g++" "python3" "nproc" "pip3")

for cmd in "${commands[@]}"; do
    if [ -z "$(which $cmd)" ]; then
        echo "Not Found, $cmd"
        exit
    fi
done

echo "Dependency Check: Ok"

if [ ! -d "$folder_path" ]; then
    git clone --recursive https://github.com/Piorosen/tvm-cprune/ -b v0.8 tvm-runtime
fi

cd tvm-runtime

half=$(($(nproc) / 2))
if [ "$half" -lt 1 ]; then
    half=1
fi

make runtime -j$half
python3 -m pip install -e $(pwd)/tvm-runtime/python
