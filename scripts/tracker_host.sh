#!/bin/bash

# HOST
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# Device
python -m tvm.exec.rpc_server --tracker=129.254.196.234:9190 --key=rasp4b-64

# Query
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
