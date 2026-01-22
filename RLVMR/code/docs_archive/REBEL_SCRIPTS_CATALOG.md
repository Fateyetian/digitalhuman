# ReBel 完整脚本目录与使用指南

**更新日期**: 2025-12-24
**版本**: v1.0
**目的**: 整理ReBel方法的所有脚本，按功能分类，并提供使用流程指导

---

## 📑 目录

1. [数据标注脚本](#1-数据标注脚本)
2. [SFT训练脚本](#2-sft训练脚本)
3. [RL训练脚本](#3-rl训练脚本)
4. [评测脚本](#4-评测脚本)
5. [全流程脚本](#5-全流程脚本)
6. [辅助工具脚本](#6-辅助工具脚本)
7. [完整工作流程](#7-完整工作流程)

---

## 1. 数据标注脚本

### 1.1 ReBel Hindsight标注

**脚本位置**: `run_rebel_annotation.sh`
**主要功能**: 使用Teacher LLM（Claude Opus）对专家轨迹进行Hindsight标注

**关键参数**:
```bash
# API配置
export OPENAI_API_KEY="your-api-key"
MODEL_NAME="aws:claude-3-5-sonnet-20241022"  # Teacher LLM

# 数据配置
EXPERT_DATA="data/alfworld_expert_traj"      # 专家轨迹数据
OUTPUT_DIR="data/alfworld_rebel_250_new"      # 输出目录
NUM_SAMPLES=250                               # 标注样本数量
```

**使用方法**:
```bash
# 基础运行
bash run_rebel_annotation.sh

# 自定义参数
export OPENAI_API_KEY="your-key"
bash run_rebel_annotation.sh
```

**输出文件**:
- `rebel_hindsight.jsonl`: Hindsight标注数据（包含Available Actions）
- `rebel_coldstart.json`: Cold-start SFT数据（应删除Available Actions）
- `annotation_report.json`: 标注质量报告

**相关Python脚本**:
- `generate_rebel_hindsight.py`: 核心标注逻辑
- `verify_annotation_quality.py`: 质量验证

---

## 2. SFT训练脚本

### 2.1 Cold-start SFT训练（主脚本）

**脚本位置**: `run_sft_coldstart_426.sh`
**主要功能**: 使用426条ReBel轨迹进行Cold-start SFT训练

**关键配置**:
```bash
# 数据配置
DATA_SOURCE="data/alfworld_rebel_merged_final/rebel_coldstart.json"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 训练配置
CHECKPOINT_DIR="./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426"
EXPERIMENT_NAME="qwen1.5b_rebel_cold-start_426"
```

**使用方法**:
```bash
# 默认配置（1.5B模型）
bash run_sft_coldstart_426.sh

# 使用7B模型（需修改脚本中的MODEL_NAME）
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

**训练特点**:
- ❌ **No Available Actions**: 训练数据不包含admissible actions
- 💪 **高难度**: 模型需要从纯observation推理动作
- 🎯 **目标**: 学习强大的belief state维护能力

### 2.2 Cold-start SFT训练（Quick Start版）

**脚本位置**: `quick_start_sft_426.sh`
**主要功能**: 快速启动版本，用于测试和调试

**与主脚本的区别**:
- 更少的训练epoch
- 更小的batch size
- 适合快速验证

### 2.3 示例SFT脚本（examples目录）

**脚本位置**: `examples/sft/cold_start/`

| 脚本名称 | 模型 | 环境 | 说明 |
|---------|------|------|-----|
| `run_alfworld_qwen2.5-1.5b.sh` | Qwen2.5-1.5B | ALFWorld | 1.5B模型训练 |
| `run_alfworld_qwen2.5-7b.sh` | Qwen2.5-7B | ALFWorld | 7B模型训练 |
| `run_sciworld_qwen2.5-1.5b.sh` | Qwen2.5-1.5B | SciWorld | SciWorld环境 |
| `run_sciworld_qwen2.5-7b.sh` | Qwen2.5-7B | SciWorld | SciWorld环境 |

---

## 3. RL训练脚本

### 3.1 ReBel RL训练（主脚本）

**脚本位置**: `examples/rebel_trainer/run_alfworld.sh`
**主要功能**: 使用ReBel算法进行强化学习训练

**关键配置**:
```bash
# 算法配置
algorithm.adv_estimator=rebel
algorithm.rebel.enable=True
algorithm.rebel.belief_granularity='subgoal'
algorithm.rebel.step_advantage_w=1.0
algorithm.rebel.mode='mean_norm'

# 数据配置
train_data_size=16
val_data_size=128
group_size=64  # ReBel需要更大的group size用于belief分组

# 奖励配置
algorithm.rebel.alpha=0.3  # belief一致性权重
algorithm.rebel.beta=0.5   # belief质量权重
algorithm.rebel.gamma=0.2  # belief完整性权重
algorithm.rebel.delta=0.1  # belief进度权重
```

**使用方法**:
```bash
# 使用vLLM后端
bash examples/rebel_trainer/run_alfworld.sh vllm

# 使用Megatron后端
bash examples/rebel_trainer/run_alfworld.sh megatron
```

**训练特点**:
- ✅ **With Available Actions**: RL阶段提供admissible actions
- 📊 **Belief-guided**: 使用belief state进行分组和奖励计算
- 🎯 **渐进式学习**: 从Cold-start SFT模型继续训练

### 3.2 BDRS训练脚本（对比方法）

**脚本位置**: `examples/bdrs_trainer/run_alfworld.sh`
**主要功能**: BDRS方法（与ReBel对比）

**关键差异**:
```bash
# BDRS使用BDRS prompt
# ReBel使用ReBel prompt（带belief state）
```

### 3.3 其他训练脚本

| 脚本路径 | 算法 | 说明 |
|---------|------|-----|
| `examples/ppo_trainer/run_alfworld.sh` | PPO | 标准PPO方法 |
| `examples/grpo_trainer/run_alfworld.sh` | GRPO | Group-based PPO |
| `examples/gigpo_trainer/run_alfworld.sh` | GiGPO | 分组强化学习 |
| `examples/rlvmr_trainer/run_alfworld.sh` | RLVMR | RLVMR方法 |

---

## 4. 评测脚本

### 4.1 SFT模型评测

**脚本位置**: `eval_sft_coldstart_426.sh`
**主要功能**: 评测SFT训练后的模型性能

**使用方法**:
```bash
# 评测最新的SFT checkpoint
bash eval_sft_coldstart_426.sh

# 评测特定checkpoint
CHECKPOINT_PATH="checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_500" \
bash eval_sft_coldstart_426.sh
```

**评测指标**:
- Success Rate（成功率）
- Average Steps（平均步数）
- Belief Consistency（belief一致性）

### 4.2 ReBel评测脚本

**脚本位置**: `run_rebel_evaluation.sh`
**主要功能**: ReBel方法的完整评测

**评测内容**:
1. Base Model评测（SFT后）
2. RL训练后评测
3. Belief质量分析
4. 与baseline对比

### 4.3 独立评测脚本

**脚本位置**: `run_eval_only.sh`
**主要功能**: 纯评测脚本，不包含训练

**使用场景**:
- 评测已训练好的模型
- 快速验证模型性能
- 生成评测报告

### 4.4 BDRS评测脚本

**脚本位置**: `examples/bdrs_trainer/eval_*.sh`

| 脚本名称 | 功能 |
|---------|------|
| `eval_alfworld.sh` | ALFWorld环境评测 |
| `eval_alfworld_cold_start.sh` | Cold-start模型评测 |
| `eval_cold_start.sh` | 通用Cold-start评测 |
| `eval_cold_start_minimal.sh` | 最小化评测配置 |
| `eval_cold_start_with_testset.sh` | 使用测试集评测 |
| `eval_rebel.sh` | ReBel方法评测 |
| `eval_sciworld.sh` | SciWorld环境评测 |

### 4.5 Rollout评测脚本

**脚本位置**: `examples/bdrs_trainer/rollout/eval_rebel.sh`
**主要功能**: ReBel rollout数据生成和评测

---

## 5. 全流程脚本

### 5.1 完整Pipeline（推荐）

**脚本位置**: `scripts/run_full_pipeline.sh`
**主要功能**: 一键运行完整流程（数据标注 → SFT → RL → 评测）

**流程步骤**:
```
1. Hindsight数据标注
   ↓
2. Cold-start SFT训练
   ↓
3. SFT模型评测
   ↓
4. ReBel RL训练
   ↓
5. RL模型评测
   ↓
6. 生成最终报告
```

**使用方法**:
```bash
# 完整运行（默认配置）
bash scripts/run_full_pipeline.sh

# 跳过数据生成（使用现有数据）
SKIP_DATA_GENERATION=true bash scripts/run_full_pipeline.sh

# 仅运行SFT训练和评测
SKIP_RL=true SKIP_RL_EVAL=true bash scripts/run_full_pipeline.sh

# 自定义配置
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export NUM_GPUS=8
export SFT_EPOCHS=10
bash scripts/run_full_pipeline.sh
```

**配置参数**:
```bash
# 基础配置
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
NUM_GPUS=8
EXPERIMENT_NAME="bdrs_full_pipeline_$(date +%Y%m%d_%H%M%S)"

# API配置
OPENAI_API_KEY=""
LLM_MODEL="aws:claude-3-5-sonnet-20241022"
NUM_TRAJS=300

# SFT配置
SFT_EPOCHS=5
SFT_LR=1e-5
SFT_BATCH_SIZE=16

# RL配置
RL_EPOCHS=100
RL_LR=1e-6
RL_GROUP_SIZE=8

# 流程控制
SKIP_DATA_GENERATION=false
SKIP_SFT=false
SKIP_SFT_EVAL=false
SKIP_RL=false
SKIP_RL_EVAL=false
```

### 5.2 Quick Start（2 GPU版）

**脚本位置**: `quick_start_2gpu.sh`
**主要功能**: 适合2 GPU环境的快速启动脚本

**特点**:
- 更小的batch size
- 更少的并行度
- 适合资源受限环境

---

## 6. 辅助工具脚本

### 6.1 数据处理脚本

| Python脚本 | 功能 |
|-----------|------|
| `generate_rebel_hindsight.py` | Hindsight标注核心逻辑 |
| `convert_hindsight_to_coldstart.py` | Hindsight→Cold-start转换 |
| `merge_rebel_datasets.py` | 合并多个ReBel数据集 |
| `merge_coldstart_datasets.py` | 合并Cold-start数据集 |
| `view_arrow_data.py` | 查看Arrow格式数据 |
| `convert_arrow_to_json.py` | Arrow→JSON转换 |

### 6.2 分析验证脚本

| Python脚本 | 功能 |
|-----------|------|
| `verify_annotation_quality.py` | 验证标注质量 |
| `verify_base_model.py` | 验证base model |
| `verify_bdrs_config.py` | 验证BDRS配置 |
| `analyze_rebel_results.py` | 分析ReBel结果 |
| `analyze_trajectory.py` | 分析轨迹数据 |

### 6.3 测试脚本

| Bash脚本 | 功能 |
|---------|------|
| `test_rebel_base_model.sh` | 测试base model |
| `test_rebel_base_model_fixed.sh` | 测试修复后的base model |
| `test_rebel_hindsight.sh` | 测试hindsight标注 |
| `test_rebel_improvements.sh` | 测试改进效果 |
| `test_rebel_small.sh` | 小规模测试 |
| `test_10samples.sh` | 10样本快速测试 |

### 6.4 vLLM服务脚本

| Bash脚本 | 功能 |
|---------|------|
| `start_vllm.sh` | 启动vLLM服务（前台） |
| `start_vllm_background.sh` | 启动vLLM服务（后台） |
| `start_vllm_server.sh` | 启动vLLM API服务器 |

---

## 7. 完整工作流程

### 7.1 从零开始的完整流程

```bash
# ============================================================================
# 步骤1: 数据标注（Hindsight Annotation）
# ============================================================================
# 使用Teacher LLM标注专家轨迹

export OPENAI_API_KEY="your-api-key"
bash run_rebel_annotation.sh

# 输出:
# - data/alfworld_rebel_250_new/rebel_hindsight.jsonl
# - data/alfworld_rebel_250_new/rebel_coldstart.json

# 验证数据质量
python verify_annotation_quality.py \
    --data_path data/alfworld_rebel_250_new/rebel_hindsight.jsonl

# ============================================================================
# 步骤2: Cold-start SFT训练
# ============================================================================
# 使用cold-start数据训练base model（无Available Actions）

bash run_sft_coldstart_426.sh

# 输出:
# - checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_*

# ============================================================================
# 步骤3: SFT模型评测
# ============================================================================
# 评测SFT训练后的模型

bash eval_sft_coldstart_426.sh

# 输出:
# - results/sft_evaluation_report.json
# - 预期成功率: 40-50% (cold-start很困难)

# ============================================================================
# 步骤4: ReBel RL训练
# ============================================================================
# 使用ReBel算法进行强化学习（有Available Actions）

# 4.1 准备SFT模型作为初始化
SFT_MODEL_PATH="checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_500"

# 4.2 启动RL训练
bash examples/rebel_trainer/run_alfworld.sh vllm

# 输出:
# - checkpoints/rl_training/alfworld/rebel_qwen1.5b/epoch_*

# ============================================================================
# 步骤5: RL模型评测
# ============================================================================
# 评测RL训练后的模型

bash run_rebel_evaluation.sh

# 输出:
# - results/rl_evaluation_report.json
# - 预期成功率: 70-85% (RL显著提升)

# ============================================================================
# 步骤6: 分析结果
# ============================================================================
# 分析belief state质量和性能提升

python analyze_rebel_results.py \
    --sft_results results/sft_evaluation_report.json \
    --rl_results results/rl_evaluation_report.json \
    --output results/rebel_analysis_report.json
```

### 7.2 使用一键脚本

```bash
# 方案A: 完整运行（推荐用于正式实验）
export OPENAI_API_KEY="your-api-key"
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export NUM_GPUS=8

bash scripts/run_full_pipeline.sh

# 方案B: 跳过数据生成（使用现有数据）
export SKIP_DATA_GENERATION=true
bash scripts/run_full_pipeline.sh

# 方案C: 仅SFT训练（快速验证）
export SKIP_RL=true
export SKIP_RL_EVAL=true
bash scripts/run_full_pipeline.sh
```

### 7.3 快速测试流程（2 GPU环境）

```bash
# 使用2 GPU快速验证整个pipeline

# 1. 使用小数据集测试标注
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj \
    --output_dir data/rebel_test \
    --num_samples 10

# 2. 快速SFT训练
bash quick_start_2gpu.sh

# 3. 快速评测
bash test_10samples.sh

# 预期: 10分钟内完成端到端测试
```

---

## 8. 脚本选择指南

### 8.1 按需求选择脚本

| 需求 | 推荐脚本 | 预估时间 |
|------|---------|---------|
| **完整实验（从头开始）** | `scripts/run_full_pipeline.sh` | 2-3天 |
| **仅数据标注** | `run_rebel_annotation.sh` | 2-4小时 |
| **仅SFT训练** | `run_sft_coldstart_426.sh` | 4-8小时 |
| **仅RL训练** | `examples/rebel_trainer/run_alfworld.sh` | 12-24小时 |
| **快速测试** | `test_10samples.sh` | 5-10分钟 |
| **评测现有模型** | `run_eval_only.sh` | 30-60分钟 |
| **2 GPU环境** | `quick_start_2gpu.sh` | 视配置而定 |

### 8.2 按环境选择配置

| 环境 | GPU数量 | 推荐模型 | 推荐脚本 |
|------|--------|---------|---------|
| **生产环境** | 8 GPU | Qwen2.5-7B | `scripts/run_full_pipeline.sh` |
| **开发环境** | 4 GPU | Qwen2.5-1.5B | `quick_start_sft_426.sh` |
| **测试环境** | 2 GPU | Qwen2.5-1.5B | `quick_start_2gpu.sh` |
| **调试环境** | 1 GPU | Qwen2.5-1.5B | `test_rebel_small.sh` |

---

## 9. 常见问题

### 9.1 数据格式问题

**Q**: Cold-start数据包含Available Actions怎么办？
**A**: 这是已知问题，参见`REBEL_DATA_PROMPT_CONSISTENCY_REPORT.md`

**解决方案**:
```bash
# 方案1: 重新生成数据
python generate_rebel_hindsight.py --regenerate

# 方案2: 手动清理数据
python clean_coldstart_data.py \
    --input data/rebel_coldstart.json \
    --output data/rebel_coldstart_clean.json
```

### 9.2 vLLM服务问题

**Q**: RL训练需要vLLM服务吗？
**A**: 是的，使用vLLM引擎时需要先启动服务

```bash
# 启动vLLM服务
bash start_vllm_background.sh

# 检查服务状态
curl http://localhost:8000/health

# 运行RL训练
bash examples/rebel_trainer/run_alfworld.sh vllm
```

### 9.3 OOM（内存溢出）问题

**Q**: 训练时出现OOM怎么办？
**A**: 减小batch size和group size

```bash
# 修改配置
export SFT_BATCH_SIZE=8  # 从16降到8
export RL_GROUP_SIZE=32  # 从64降到32

# 或使用gradient checkpointing
export SFT_GRADIENT_CHECKPOINTING=True
```

---

## 10. 脚本维护建议

### 10.1 命名规范

```
run_*        : 主要运行脚本
test_*       : 测试脚本
eval_*       : 评测脚本
quick_start_*: 快速启动脚本
verify_*     : 验证脚本
analyze_*    : 分析脚本
generate_*   : 数据生成脚本
convert_*    : 格式转换脚本
```

### 10.2 版本控制

建议在关键节点创建tag:
```bash
git tag -a v1.0-rebel-annotation -m "ReBel hindsight annotation完成"
git tag -a v1.1-rebel-sft -m "ReBel SFT训练完成"
git tag -a v1.2-rebel-rl -m "ReBel RL训练完成"
```

### 10.3 文档更新

每次修改脚本后，同步更新：
1. 本文档（`REBEL_SCRIPTS_CATALOG.md`）
2. 脚本内注释
3. README.md

---

## 附录：脚本文件树

```
/root/testttt/RLVMR/code/
│
├── 数据标注相关
│   ├── run_rebel_annotation.sh              # 主标注脚本
│   ├── run_claude_opus_annotation.sh        # Claude Opus标注
│   ├── generate_rebel_hindsight.py          # Hindsight标注核心
│   ├── generate_rebel_golden_v2.py          # Golden数据生成v2
│   ├── generate_rebel_golden_with_expert.py # 使用专家数据生成
│   └── generate_rebel_golden_env.py         # 环境数据生成
│
├── SFT训练相关
│   ├── run_sft_coldstart_426.sh             # 主SFT训练脚本
│   ├── quick_start_sft_426.sh               # 快速启动版
│   ├── quick_start_2gpu.sh                  # 2GPU版本
│   └── examples/sft/cold_start/
│       ├── run_alfworld_qwen2.5-1.5b.sh
│       └── run_alfworld_qwen2.5-7b.sh
│
├── RL训练相关
│   ├── examples/rebel_trainer/
│   │   └── run_alfworld.sh                  # ReBel RL训练
│   ├── examples/bdrs_trainer/
│   │   └── run_alfworld.sh                  # BDRS训练
│   ├── examples/ppo_trainer/
│   │   └── run_alfworld.sh                  # PPO训练
│   └── examples/rlvmr_trainer/
│       └── run_alfworld.sh                  # RLVMR训练
│
├── 评测相关
│   ├── eval_sft_coldstart_426.sh            # SFT评测
│   ├── run_rebel_evaluation.sh              # ReBel评测
│   ├── run_eval_only.sh                     # 纯评测
│   └── examples/bdrs_trainer/
│       ├── eval_alfworld.sh
│       ├── eval_alfworld_cold_start.sh
│       ├── eval_cold_start.sh
│       ├── eval_cold_start_minimal.sh
│       ├── eval_cold_start_with_testset.sh
│       ├── eval_rebel.sh
│       └── rollout/eval_rebel.sh
│
├── 全流程脚本
│   └── scripts/
│       ├── run_full_pipeline.sh             # 完整pipeline
│       └── pipeline_config.example.sh       # 配置示例
│
├── 测试脚本
│   ├── test_rebel_base_model.sh
│   ├── test_rebel_base_model_fixed.sh
│   ├── test_rebel_hindsight.sh
│   ├── test_rebel_improvements.sh
│   ├── test_rebel_small.sh
│   └── test_10samples.sh
│
├── 辅助工具
│   ├── verify_annotation_quality.py         # 验证标注
│   ├── verify_base_model.py                 # 验证模型
│   ├── analyze_rebel_results.py             # 分析结果
│   ├── merge_rebel_datasets.py              # 合并数据集
│   ├── convert_hindsight_to_coldstart.py    # 格式转换
│   └── vLLM服务
│       ├── start_vllm.sh
│       ├── start_vllm_background.sh
│       └── start_vllm_server.sh
│
└── 文档
    ├── REBEL_DATA_PROMPT_CONSISTENCY_REPORT.md
    ├── REBEL_SCRIPTS_CATALOG.md (本文档)
    ├── REBEL_QUICK_START.md
    └── REBEL_README.md
```

---

## 总结

本文档整理了ReBel方法的所有关键脚本，按功能分为：
1. ✅ **数据标注脚本** - Hindsight annotation
2. ✅ **SFT训练脚本** - Cold-start training
3. ✅ **RL训练脚本** - ReBel RL
4. ✅ **评测脚本** - Evaluation
5. ✅ **全流程脚本** - End-to-end pipeline
6. ✅ **辅助工具** - Utilities

**推荐使用流程**:
- 🟢 **新手**: 使用`scripts/run_full_pipeline.sh`一键运行
- 🟡 **进阶**: 分步骤运行各个脚本，逐步理解流程
- 🔴 **专家**: 自定义脚本参数，优化训练配置

**下一步**:
1. 检查并修复cold-start数据格式问题（见`REBEL_DATA_PROMPT_CONSISTENCY_REPORT.md`）
2. 重新生成符合规范的cold-start数据
3. 使用正确数据重新训练SFT模型
4. 继续RL训练并评测最终性能
