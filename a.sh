#!/bin/bash

SESSION_NAME="gpu_monitor"

# 세션이 이미 존재하는지 확인
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Session $SESSION_NAME already exists. Attaching..."
  tmux attach-session -t $SESSION_NAME
  exit 0
fi

# 새 tmux 세션 생성
tmux new-session -d -s $SESSION_NAME -n "main"

# 좌상 (Pane 0): watch -n 1 nvidia-smi
tmux send-keys "watch -n 1 nvidia-smi" C-m

# 좌하 (Pane 1): python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
tmux split-window -v
tmux send-keys "python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190" C-m

# 우상 (Pane 2): watch -n 1 python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
tmux select-pane -t 0
tmux split-window -h
tmux send-keys "watch -n 1 python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190" C-m

# 우하 (Pane 3): /bin/bash
tmux select-pane -t 2
tmux split-window -v

# 레이아웃 정리
tmux select-layout tiled

# 세션 연결
tmux attach-session -t $SESSION_NAME
