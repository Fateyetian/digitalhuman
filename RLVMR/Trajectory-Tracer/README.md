# Trajectory Viewer

> 专为 ALFWorld 等交互式环境的智能体轨迹数据设计的可视化工具

一个美观、高效的 Web 应用，用于可视化和分析大规模智能体轨迹数据，将复杂的对话式轨迹转化为易于理解的交互界面。

## ✨ 特性

- 🎯 **对话式展示**: 左右对话气泡形式，清晰展示智能体思考和环境反馈
- 🔍 **高级筛选**: 支持按状态、任务类型、步数范围等多维度筛选
- 📊 **统计面板**: 实时展示轨迹统计信息（成功率、平均步数、数据源等）
- 🎨 **精美 UI**: 基于 TailwindCSS，现代化设计，响应式布局
- ⚡ **高性能**: 虚拟滚动技术，轻松处理数万条轨迹
- 🐳 **一键部署**: Docker Compose 快速部署到服务器
- 🔌 **多格式支持**: 适配器模式支持多种轨迹数据格式（HuggingFace、REBEL JSON等）

## 🏗️ 技术栈

### 后端
- **FastAPI**: 高性能 Python Web 框架
- **Datasets**: HuggingFace 数据集处理
- **Uvicorn**: ASGI 服务器

### 前端
- **React 18**: UI 框架
- **Zustand**: 轻量级状态管理
- **TailwindCSS**: 原子化 CSS 框架
- **Vite**: 快速构建工具

### 部署
- **Docker**: 容器化
- **Nginx**: 反向代理和静态文件服务

## 📦 项目结构

```
Trajectory-Tracer/
├── backend/                    # 后端服务
│   ├── main.py                # FastAPI 主应用
│   ├── trajectory_adapters.py # 轨迹格式适配器
│   ├── data_sources.json      # 数据源配置
│   ├── requirements.txt       # Python 依赖
│   └── Dockerfile            # 后端 Docker 配置
├── frontend/                  # 前端应用
│   ├── src/
│   │   ├── components/       # React 组件
│   │   │   ├── Header.jsx
│   │   │   ├── FilterPanel.jsx
│   │   │   ├── TrajectoryList.jsx
│   │   │   ├── TrajectoryViewer.jsx
│   │   │   └── MessageBubble.jsx
│   │   ├── App.jsx           # 主应用
│   │   ├── store.js          # Zustand 状态管理
│   │   └── main.jsx          # 入口文件
│   ├── package.json
│   ├── vite.config.js
│   ├── nginx.conf            # Nginx 配置
│   └── Dockerfile           # 前端 Docker 配置
├── alfworld_expert_traj/     # 轨迹数据目录
├── docker-compose.yml        # Docker Compose 配置
├── README.md                 # 项目说明
├── ADDING_NEW_TRAJECTORY_TYPES.md  # 添加新格式指南
└── TESTING.md               # 测试文档
```

## 🚀 快速开始

### 方式 1: Docker 部署（推荐）

1. **确保已安装 Docker 和 Docker Compose**

2. **启动服务**
```bash
docker-compose up -d
```

3. **访问应用**
打开浏览器访问: `http://localhost`

4. **查看日志**
```bash
# 查看所有服务日志
docker-compose logs -f

# 只查看后端日志
docker-compose logs -f backend

# 只查看前端日志
docker-compose logs -f frontend
```

5. **停止服务**
```bash
docker-compose down
```

### 方式 2: 本地开发

#### 后端启动

```bash
cd backend

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

后端将在 `http://localhost:8000` 运行

#### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端将在 `http://localhost:3000` 运行

## 📖 使用说明

### 1. 浏览轨迹列表

左侧面板显示所有轨迹：
- **绿色勾**: 成功的轨迹
- **红色叉**: 失败的轨迹
- **问号**: 状态未知的轨迹

### 2. 应用筛选器

使用左侧筛选面板：
- **状态**: 成功/失败/未知
- **任务类型**: put/clean/heat/cool/examine
- **步数范围**: 设置最小和最大步数

点击"应用筛选"按钮生效

### 3. 查看轨迹详情

点击列表中的任意轨迹，右侧显示详细对话：

- **任务卡片**: 显示任务目标、环境描述、统计信息
- **对话气泡**:
  - **左侧灰色**: 环境反馈
  - **右侧蓝色**: 智能体的思考和动作
- **思考与动作**: 清晰区分智能体的推理过程和实际行动

### 4. 分页浏览

列表底部提供分页功能，每页显示 50 条轨迹

## 🔌 API 接口

### 获取轨迹列表
```
GET /api/trajectories?skip=0&limit=50&status=success&task_type=put&min_steps=5&max_steps=20
```

### 获取轨迹详情
```
GET /api/trajectories/{trajectory_id}
```

### 获取统计信息
```
GET /api/statistics
```

返回包含数据源统计的信息

### 获取数据源信息
```
GET /api/data-sources
```

返回已加载的所有数据源及其格式信息

## 🎨 界面预览

### 主界面布局
- **顶部**: 标题栏 + 统计面板
- **左侧**: 筛选器 + 轨迹列表
- **右侧**: 任务信息 + 对话详情

### 对话展示风格
- 清晰的视觉层次
- 智能识别任务、思考、动作
- 彩色标签区分不同信息类型
- 优雅的渐变色和阴影效果

## 🔧 配置

### 后端配置

编辑 `backend/main.py`:
- 数据集路径: 默认为 `../alfworld_expert_traj`
- 端口: 默认 8000

### 前端配置

编辑 `frontend/.env.production`:
```env
VITE_API_BASE=/api
```

### Docker 端口配置

编辑 `docker-compose.yml`:
```yaml
services:
  frontend:
    ports:
      - "80:80"  # 修改左侧端口号更改访问端口
  backend:
    ports:
      - "8000:8000"
```

## 📊 数据格式

本项目支持多种轨迹数据格式，通过适配器模式实现。

### 支持的格式

#### 1. HuggingFace Datasets 格式

```json
{
  "conversations": [
    {
      "from": "human" | "gpt",
      "value": "消息内容",
      "loss": true | false | null
    }
  ],
  "item_id": "唯一标识"
}
```

#### 2. REBEL JSON 格式

```json
{
  "task": "任务描述",
  "done": "True" | "False",
  "data": [
    {
      "step": 1,
      "obs": "环境观察",
      "prompt": "提示词",
      "response": "<belief>...</belief><reasoning>...</reasoning><action>...</action>"
    }
  ]
}
```

### 添加新格式

如需支持其他轨迹格式，请参考详细指南：[如何添加新的轨迹类型](./ADDING_NEW_TRAJECTORY_TYPES.md)

简要步骤：
1. 在 `backend/trajectory_adapters.py` 中创建新的适配器类
2. 继承 `TrajectoryAdapter` 并实现 `load()` 和 `parse()` 方法
3. 在 `TrajectoryLoader` 中注册适配器
4. 在 `backend/data_sources.json` 中配置数据源

## 🚀 生产部署建议

### 1. 使用反向代理

使用 Nginx 或 Traefik 作为入口点：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:80;
    }
}
```

### 2. 启用 HTTPS

使用 Let's Encrypt 免费证书：
```bash
certbot --nginx -d your-domain.com
```

### 3. 性能优化

- 启用 gzip 压缩（已在 nginx.conf 中配置）
- 使用 CDN 加速静态资源
- 配置 Redis 缓存（可选）

### 4. 监控和日志

```bash
# 持续监控容器状态
docker-compose ps

# 查看资源使用
docker stats

# 导出日志
docker-compose logs > app.log
```

## 🔮 后续计划

- [ ] 添加轨迹对比功能
- [ ] 支持导出为 PDF/图片
- [ ] 实现轨迹播放模式（动画展示）
- [x] 支持多种数据格式（适配器模式）
- [ ] 添加全文搜索功能
- [ ] 接入 Benchmark 系统
- [ ] 支持更多轨迹格式（CSV、数据库等）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License

## 💬 联系方式

如有问题或建议，请提交 Issue。

---

**享受轨迹可视化！** 🎉
