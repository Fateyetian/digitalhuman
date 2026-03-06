# 更新日志 v1.1

## 🐛 修复问题

### 1. 任务类型识别错误修复
**问题**: 之前使用关键词包含的方式识别任务类型，导致所有任务都被识别为 "put"

**修复**: 改为使用任务描述的第一个单词作为类型识别

**结果**: 现在可以正确识别所有任务类型

### 2. 实际任务类型分布

根据 2224 条轨迹的分析：

| 任务类型 | 数量 | 占比 |
|---------|------|------|
| **put** | 1445 | 65.0% |
| **clean** | 238 | 10.7% |
| **cool** | 217 | 9.8% |
| **heat** | 163 | 7.3% |
| **find** | 161 | 7.2% |

### 3. 新增功能：任务类型分布面板

在顶部 Header 新增任务类型分布可视化：
- 🎨 彩色标签显示各类型任务数量和占比
- 📊 自动按数量降序排列
- 🔍 一目了然的数据概览

### 4. 优化筛选器

更新任务类型筛选器，支持所有类型：
- ✅ Put (放置)
- ✅ Clean (清洁)
- ✅ Cool (冷却)
- ✅ Heat (加热)
- ✅ Find (寻找)
- ✅ Examine (检查)
- ✅ Use (使用)
- ✅ Other (其他)

## 🎨 界面改进

### Header 统计卡片美化
- 添加渐变背景色
- 优化边框和阴影
- 更清晰的视觉层次

### 任务类型色彩系统
- **Put**: 蓝色
- **Clean**: 紫色
- **Cool**: 青色
- **Heat**: 橙色
- **Find**: 粉色
- **Examine**: 绿色
- **Use**: 黄色
- **Other**: 灰色

## 📝 技术细节

### 后端更改
**文件**: `backend/main.py:79-85`

```python
# 提取任务类型（使用第一个单词作为类型）
first_word = task.split()[0].lower() if task else 'unknown'
# 支持的任务类型: put, clean, heat, cool, find, examine, use
if first_word in ['put', 'clean', 'heat', 'cool', 'find', 'examine', 'use']:
    task_type = first_word
else:
    task_type = 'other'
```

### 前端更改
1. **Header.jsx**: 新增任务类型分布面板
2. **FilterPanel.jsx**: 更新任务类型选项
3. **TrajectoryList.jsx**: 优化任务类型徽章显示

## 🚀 如何更新

### Docker 部署
```bash
docker-compose down
docker-compose up -d --build
```

### 本地开发
```bash
# 后端无需重装依赖，直接重启
cd backend
python main.py

# 前端无需重装依赖，自动热更新
cd frontend
npm run dev
```

## ✨ 效果预览

**之前**:
- ❌ 所有任务显示为 "put"
- ❌ 筛选其他类型无结果
- ❌ 看不到任务类型分布

**现在**:
- ✅ 正确显示 5 种主要任务类型
- ✅ 筛选功能正常工作
- ✅ 顶部显示彩色任务类型分布
- ✅ 一目了然的数据统计

---

**更新时间**: 2024-12-24
**版本**: v1.1
