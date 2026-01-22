#!/bin/bash
# 后台启动vLLM服务器

nohup bash start_vllm_server.sh > vllm.log 2>&1 &
echo "vLLM服务器已在后台启动"
echo "查看日志: tail -f vllm.log"
echo "检查状态: curl http://127.0.0.1:8000/v1/models"
