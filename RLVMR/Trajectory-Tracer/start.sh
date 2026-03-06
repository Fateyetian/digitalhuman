#!/bin/bash
echo "============================================"
echo "  Trajectory Viewer - Docker 一键部署"
echo "============================================"
echo

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "[错误] Docker 未运行，请先启动 Docker"
    exit 1
fi

echo "[1/3] 停止旧容器..."
docker-compose down 2>/dev/null

echo "[2/3] 构建并启动服务..."
docker-compose up -d --build

if [ $? -ne 0 ]; then
    echo "[错误] 启动失败，请检查日志"
    docker-compose logs
    exit 1
fi

echo "[3/3] 等待服务就绪..."
sleep 5

echo
echo "============================================"
echo "  启动成功！"
echo "============================================"
echo "  前端地址: http://localhost"
echo "  后端API:  http://localhost:8000"
echo "  API文档:  http://localhost:8000/docs"
echo "============================================"
echo
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"
