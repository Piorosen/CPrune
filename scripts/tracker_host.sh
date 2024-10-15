#!/bin/bash

# HOST
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# Device
python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=rasp4b-64 --custom-addr=127.0.0.1 --no-fork

# Query
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
