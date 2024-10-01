#!/bin/bash

# HOST
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# Device
python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rasp4b-64

