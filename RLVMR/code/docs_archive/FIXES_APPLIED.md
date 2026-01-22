# 方案A快速修复 - 已完成的修改

## 问题诊断
- **成功率**: 0% (0/64)
- **无效动作率**: 66.2% (1237/1868)
- **根本原因**: 
  1. 训练数据使用BDRS格式 (`<PLAN>/<EXECUTE>/<EXPLORE>/<VERIFY>`)
  2. 评测使用`<think>`格式
  3. Action提取时有方括号污染 (如 `go to drawer 3]`)

## 已应用的修复

### 1. 添加BDRS Projection函数
**文件**: `agent_system/environments/env_package/alfworld/projection.py`
- 新增 `alfworld_projection_bdrs()` 函数
- 支持 `<PLAN>/<EXECUTE>/<EXPLORE>/<VERIFY>` 标签
- 自动清理方括号错误: `act_candidate.rstrip(']').rstrip("']")`

### 2. 导出BDRS Projection
**文件**: `agent_system/environments/env_package/alfworld/__init__.py`
- 添加 `alfworld_projection_bdrs` 到导出列表

### 3. 环境管理器支持BDRS
**文件**: `agent_system/environments/env_manager.py` (行654, 694-708)
- 导入 `alfworld_projection_bdrs`
- 添加BDRS检测逻辑: 检查 `config.algorithm.bdrs.enable`
- 根据配置选择正确的projection函数

### 4. 修改Rollout脚本支持BDRS
**文件**: `examples/bdrs_trainer/rollout/run_alfworld_rollout.py`
- 修改 `build_env()` 函数接受 `use_bdrs` 参数 (行25)
- 添加命令行参数 `--use_bdrs` (默认True) (行135)
- 创建带有 `algorithm.bdrs.enable` 的配置 (行44-47)
- 传递 `use_bdrs` 到环境构建 (行267)

## 如何运行评测

### 步骤1: 启动vLLM服务器 (终端1)
```bash
cd /root/digitalhuman/RLVMR/code
bash start_vllm.sh
# 等待看到: "Uvicorn running on http://0.0.0.0:8000"
```

### 步骤2: 运行评测 (终端2)
```bash
cd /root/digitalhuman/RLVMR/code
bash run_eval_only.sh
```

## 预期改进
- **成功率**: 从 0% → 15-25%
- **无效动作率**: 从 66.2% → <20%
- **主要提升**: Prompt格式匹配，action解析更robust

## 下一步
如果成功率仍然较低,考虑:
1. 方案B: 重新生成冷启动数据（统一格式）
2. 方案C: 优化SFT训练参数
3. 方案D: 改进标注流程
