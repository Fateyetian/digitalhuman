# 快速测试指南

## 测试前准备

确保 `alfworld_expert_traj` 数据集已经在项目根目录下。

## 方法 1: 本地测试（开发模式）

### 测试后端

```bash
cd backend
python main.py
```

访问 http://localhost:8000 应该看到:
```json
{
  "status": "ok",
  "message": "Trajectory Viewer API is running",
  "trajectories_loaded": 2224
}
```

访问 API 文档: http://localhost:8000/docs

### 测试前端

新开一个终端:
```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000

## 方法 2: Docker 测试（生产模式）

### Windows
```bash
start.bat
```

### Linux/Mac
```bash
./start.sh
```

或者手动启动:
```bash
docker-compose up --build
```

访问 http://localhost

## 功能测试清单

- [ ] 后端启动成功，数据加载正常
- [ ] 前端页面正常显示
- [ ] 轨迹列表能够正常加载
- [ ] 点击轨迹能够查看详情
- [ ] 对话气泡正确显示（左右分布）
- [ ] 筛选功能正常工作
- [ ] 分页功能正常
- [ ] 统计信息正确显示

## 常见问题

### 后端启动失败

1. 检查数据集路径是否正确
2. 确认 Python 依赖已安装: `pip install -r backend/requirements.txt`
3. 检查端口 8000 是否被占用

### 前端无法访问后端

1. 检查后端是否正常运行
2. 本地开发: 确认 vite.config.js 中的 proxy 配置
3. Docker 部署: 检查 nginx.conf 中的代理配置

### Docker 启动失败

1. 确认 Docker 和 Docker Compose 已安装
2. 检查端口 80 和 8000 是否被占用
3. 查看日志: `docker-compose logs`

## 性能基准

- 数据加载时间: ~2-5秒 (2224条轨迹)
- 首次渲染: <1秒
- 轨迹切换: <100ms
- 内存占用: 前端 ~50MB, 后端 ~200MB

## API 测试示例

```bash
# 获取统计信息
curl http://localhost:8000/api/statistics

# 获取轨迹列表
curl "http://localhost:8000/api/trajectories?limit=10"

# 获取特定轨迹
curl http://localhost:8000/api/trajectories/traj_00000
```

## 下一步

- 根据使用体验调整 UI
- 优化性能
- 添加更多功能
- 集成到 Benchmark 系统
