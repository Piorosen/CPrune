#!/bin/bash

function show_help() {
    echo "Usage: tvm_runtime.sh [option] [arguments]"
    echo ""
    echo "Options:"
    echo "  install                Install dependencies and build the runtime."
    echo "  --tracker=ADDR:PORT     Set the address and port of the RPC tracker (for execute)."
    echo "  --key=KEY               Set the key for the RPC server (for execute)."
    echo "  --custom-addr=ADDR      Set the custom address for the RPC server (for execute)."
    echo "  --help                  Show this help message."
    echo ""
    echo "Examples:"
    echo "  tvm_runtime.sh install"
    echo "  tvm_runtime.sh --tracker=127.0.0.1:9190 --key=rasp4b --custom-addr=127.0.0.1"
}

function execute() {
    echo "Sample : tvm_runtime.sh --tracker=127.0.0.1:9190 --key=rasp4b --custom-addr=127.0.0.1"
    echo "python3 -m tvm.exec.rpc_server $@ --no-fork"
    python3 -m tvm.exec.rpc_server $@ --no-fork
}

function install() {
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
    python3 -m pip install -e $(pwd)/python
}

if [ "$1" == "install" ]; then
    shift 1
    install
    exit
elif  [ "$1" == "exec" ]; then
    shift 1
    execute $@
    exit
fi

show_help
