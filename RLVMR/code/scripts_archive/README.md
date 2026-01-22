# 脚本归档目录

本目录存放开发和测试过程中产生的辅助脚本。

## 归档内容

### 测试脚本
- `test_*.sh` - 各类测试脚本
- `test_*.py` - Python测试脚本

### 数据生成脚本（旧版本）
- `generate_rebel_golden_*.py` - Golden数据生成（已被hindsight方法替代）
- `generate_rebel_cold_start_data.py` - Cold-start数据生成（旧版）
- `convert_*.py` - 格式转换脚本

### 分析和验证脚本
- `analyze_*.py` - 结果分析脚本
- `verify_*.py` - 数据验证脚本（部分）
- `view_*.py` - 数据查看工具

### 数据处理脚本
- `merge_*.py` - 数据集合并脚本

### 评测和快速启动脚本
- `eval_*.sh` - 各类评测脚本
- `quick_start_*.sh` - 快速启动脚本
- `run_eval_only.sh` - 纯评测脚本

### vLLM辅助脚本
- `start_vllm.sh` - vLLM启动脚本（前台）
- `start_vllm_background.sh` - vLLM后台启动
- `test_vllm_load.py` - vLLM测试

## 核心脚本（在主目录）

主目录保留的核心脚本：

### Shell脚本
- `run_rebel_annotation.sh` - **数据标注主脚本**
- `run_sft_coldstart_426.sh` - **SFT训练主脚本**
- `run_rebel_evaluation.sh` - **评测主脚本**
- `start_vllm_server.sh` - **vLLM服务器**

### Python脚本
- `generate_rebel_hindsight.py` - **Hindsight标注核心**
- `regenerate_coldstart_clean.py` - **生成clean cold-start数据**
- `verify_annotation_quality.py` - **质量验证**

## 使用说明

如需使用归档的脚本，可以从本目录直接运行，或复制到主目录。

大部分归档脚本是开发过程中的辅助工具，已被更完善的主流程脚本替代。

---

归档时间: 2025-12-24
