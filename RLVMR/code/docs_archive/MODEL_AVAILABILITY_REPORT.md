# 当前环境可用模型报告

## ✅ 已有模型

### 1. BDRS训练后的Qwen2-1.5B模型

**模型位置:** `/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/`

**可用的检查点:**

| Checkpoint | 路径 | 大小 | 用途 |
|-----------|------|------|------|
| global_step_15 | `global_step_15/` | ~6.7GB | 早期checkpoint |
| global_step_30 | `global_step_30/` | ~6.7GB | 中期checkpoint |
| global_step_45 | `global_step_45/` | ~6.7GB | 中期checkpoint |
| global_step_60 | `global_step_60/` | ~6.7GB | 后期checkpoint |
| **global_step_75** ⭐ | `global_step_75/` | **~6.7GB** | **最新checkpoint（推荐）** |

**模型详情:**
- **架构:** Qwen2ForCausalLM
- **参数量:** ~1.5B (1536 hidden_size, 28 layers)
- **最大长度:** 32,768 tokens
- **词汇表:** 151,936 tokens
- **训练方法:** BDRS (Belief-Driven Reward Shaping)
- **训练任务:** ALFWorld embodied AI tasks

**模型文件内容:**
```
global_step_75/
├── model-00001-of-00002.safetensors  (4.7GB)
├── model-00002-of-00002.safetensors  (2.0GB)
├── model.safetensors.index.json
├── config.json
├── tokenizer.json (11MB)
├── vocab.json (2.7MB)
├── merges.txt (1.6MB)
├── generation_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── chat_template.jinja
└── added_tokens.json
```

**✅ 模型完整性:** 完整（包含所有必要文件）

---

## 🚀 可以立即使用

### 推荐使用的模型

**路径:** `/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75`

**使用方法:**

#### 1. 运行ReBel小规模测试

```bash
cd /root/testttt/RLVMR/code

# 使用最新checkpoint运行测试
bash test_rebel_small.sh vllm ./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75
```

#### 2. 运行ReBel完整训练

```bash
cd /root/testttt/RLVMR/code

# 修改训练脚本中的模型路径
bash examples/rebel_trainer/run_alfworld.sh
# 需要修改脚本中的: actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75
```

#### 3. 直接在代码中使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

---

## ⚠️ 缺失的模型

### 原始基础模型（非必需）

**名称:** Qwen2.5-1.5B-Instruct (原始预训练模型)

**状态:** ❌ 本地未找到

**是否需要:**
- ✅ **不需要** - 如果只是继续训练或测试ReBel
  - 已有的BDRS checkpoint可以直接使用
  - BDRS模型已经是在ALFWorld上训练过的

- ⚠️ **可能需要** - 如果要从头训练或对比实验
  - 需要从HuggingFace下载
  - 用于对比cold-start vs 训练后的效果

**如何获取（如果需要）:**
```bash
# 方法1: 使用huggingface-cli下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct

# 方法2: 使用Python下载
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model.save_pretrained("~/models/Qwen2.5-1.5B-Instruct")
tokenizer.save_pretrained("~/models/Qwen2.5-1.5B-Instruct")
```

**下载大小:** ~3.5GB

---

## 📊 模型状态总结

| 模型 | 状态 | 大小 | 用途 | 可用性 |
|------|------|------|------|--------|
| **BDRS Qwen2-1.5B (step 75)** | ✅ 已有 | 6.7GB | ReBel训练/测试 | **立即可用** ⭐ |
| BDRS Qwen2-1.5B (step 60) | ✅ 已有 | 6.7GB | 对比实验 | 可用 |
| BDRS Qwen2-1.5B (step 45) | ✅ 已有 | 6.7GB | 对比实验 | 可用 |
| BDRS Qwen2-1.5B (step 30) | ✅ 已有 | 6.7GB | 对比实验 | 可用 |
| BDRS Qwen2-1.5B (step 15) | ✅ 已有 | 6.7GB | 对比实验 | 可用 |
| Qwen2.5-1.5B-Instruct (原始) | ❌ 未找到 | ~3.5GB | 从头训练 | 需下载 |

---

## 🎯 推荐操作

### 方案1: 直接测试ReBel（推荐）⭐

**优点:**
- ✅ 无需下载额外模型
- ✅ 立即可以运行
- ✅ 使用已训练好的模型

**命令:**
```bash
cd /root/testttt/RLVMR/code
bash test_rebel_small.sh vllm ./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75
```

**预计时间:** 10-30分钟

**输出:**
- 成功率、信念组统计
- 完整的实验数据
- 论文用LaTeX表格

### 方案2: 对比不同checkpoint

**用途:** 研究训练过程中的性能变化

**命令:**
```bash
# 测试step 15 (早期)
bash test_rebel_small.sh vllm ./checkpoints/.../global_step_15

# 测试step 45 (中期)
bash test_rebel_small.sh vllm ./checkpoints/.../global_step_45

# 测试step 75 (最新)
bash test_rebel_small.sh vllm ./checkpoints/.../global_step_75
```

### 方案3: 完整训练实验

**命令:**
```bash
# 修改run_alfworld.sh中的模型路径为:
# actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75

bash examples/rebel_trainer/run_alfworld.sh
```

---

## 🔍 验证模型可用性

### 快速检查脚本

```bash
# 检查模型文件是否完整
MODEL_PATH="/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"

echo "检查模型文件..."
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "✅ config.json 存在"
else
    echo "❌ config.json 缺失"
fi

if [ -f "$MODEL_PATH/model-00001-of-00002.safetensors" ]; then
    echo "✅ 模型权重存在"
else
    echo "❌ 模型权重缺失"
fi

if [ -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "✅ tokenizer 存在"
else
    echo "❌ tokenizer 缺失"
fi

echo ""
echo "模型大小:"
du -sh $MODEL_PATH
```

### Python验证脚本

```python
# 验证模型可以正常加载
import sys
sys.path.insert(0, '/root/testttt/RLVMR/code')

model_path = "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"

try:
    from transformers import AutoConfig, AutoTokenizer

    # 加载配置
    config = AutoConfig.from_pretrained(model_path)
    print(f"✅ 配置加载成功: {config.model_type}")
    print(f"   隐藏层大小: {config.hidden_size}")
    print(f"   层数: {config.num_hidden_layers}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✅ Tokenizer加载成功: {len(tokenizer)} tokens")

    print("\n🟢 模型完整且可用！")

except Exception as e:
    print(f"❌ 错误: {e}")
```

---

## 📝 总结

**当前状态:**
- ✅ **有可用模型** - BDRS训练的Qwen2-1.5B (5个checkpoint)
- ✅ **模型完整** - 包含所有必要文件
- ✅ **立即可用** - 可以直接运行ReBel测试
- ⚠️ **原始模型缺失** - 如需从头训练需下载

**推荐行动:**
1. **立即运行测试** - 使用global_step_75
2. **获取实验数据** - 用于论文
3. **可选下载原始模型** - 如需对比或从头训练

**测试命令:**
```bash
bash test_rebel_small.sh vllm ./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75
```

---

**生成时间:** 2025-12-20
**环境:** /root/testttt/RLVMR/code
**状态:** ✅ 模型ready，可以开始实验
