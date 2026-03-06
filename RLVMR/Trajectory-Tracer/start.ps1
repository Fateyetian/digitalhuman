# Trajectory Viewer - Docker 一键部署 (PowerShell 版本)
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Trajectory Viewer - Docker 一键部署" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# 检查 Docker 是否运行
try {
    $dockerInfo = docker info 2>$null
    if (-not $dockerInfo) {
        Write-Host "[错误] Docker 未运行，请先启动 Docker Desktop" -ForegroundColor Red
        Read-Host "按任意键退出"
        exit 1
    }
}
catch {
    Write-Host "[错误] Docker 未运行，请先启动 Docker Desktop" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Write-Host "[1/3] 停止旧容器..." -ForegroundColor Yellow
docker-compose down 2>$null

Write-Host "[2/3] 构建并启动服务..." -ForegroundColor Yellow
docker-compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 启动失败，请检查日志" -ForegroundColor Red
    docker-compose logs
    Read-Host "按任意键退出"
    exit 1
}

Write-Host "[3/3] 等待服务就绪..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  启动成功！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  前端地址: http://localhost" -ForegroundColor Cyan
Write-Host "  后端API:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API文档:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

Write-Host "按任意键查看日志，Ctrl+C 退出..." -ForegroundColor Yellow
Read-Host ""
docker-compose logs -f