@echo off
chcp 65001 >nul
echo ============================================
echo   Trajectory Viewer - Docker 一键部署
echo ============================================
echo.

REM 检查 Docker 是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)

echo [1/3] 停止旧容器...
docker-compose down 2>nul

echo [2/3] 构建并启动服务...
docker-compose up -d --build

if %errorlevel% neq 0 (
    echo [错误] 启动失败，请检查日志
    docker-compose logs
    pause
    exit /b 1
)

echo [3/3] 等待服务就绪...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   启动成功！
echo ============================================
echo   前端地址: http://localhost
echo   后端API:  http://localhost:8000
echo   API文档:  http://localhost:8000/docs
echo ============================================
echo.
echo 按任意键查看日志，Ctrl+C 退出...
pause >nul
docker-compose logs -f