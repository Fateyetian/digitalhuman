# BDRS冷启动模型评测问题修复报告

## 📊 问题分析总结

### 根本原因
**训练-推理格式不匹配**导致100%失败率：

| 组件 | 预期格式 | 实际格式 | 状态 |
|------|---------|---------|------|
| 训练数据 | BDRS (100%) | BDRS | ✅ |
| Projection函数 | alfworld_projection_bdrs | 已配置 | ✅ |
| 评测Prompt | BDRS格式 | `<think>`格式 | ❌ |
| 模型输出 | BDRS标签 | `<think>`标签 | ❌ |
| 动作验证 | 需要BDRS标签 | 缺失→拒绝 | ❌ |

### 影响
```
训练数据(BDRS) → 模型学习BDRS
                   ↓
评测用<think>prompt → 模型输出<think>
                   ↓
projection验证 → valid=0 (100%无效)
                   ↓
环境返回"Nothing happens" → 重复失败
                   ↓
成功率: 0% (0/64)
```

## 🔧 已执行的修复步骤

### ✅ 步骤1: 修复评测脚本
**文件**: `examples/bdrs_trainer/rollout/run_local_eval.sh`

**修改**:
```diff
  python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
      --env_name alfworld \
      --batch_size $BATCH_SIZE \
      --total_envs $NUM_ENVS \
      --max_steps $MAX_STEPS \
      --model "$MODEL" \
      --base_url "http://localhost:8000/v1" \
      --temperature 0.4 \
      --concurrency 4 \
      --dump_path "$OUTPUT_DIR/trajectory.jsonl" \
      --chat_root "$OUTPUT_DIR/chats" \
      --unique_envs \
+     --use_bdrs
```

**效果**: 强制使用BDRS格式的prompt模板

### ✅ 步骤2: 配置验证
**验证工具**: `verify_bdrs_config.py`

**当前状态**:
- ✅ 训练数据格式: 300/300样本使用BDRS (100%)
- ✅ Projection函数: alfworld_projection_bdrs已配置
- ⏳ 评测prompt: 待重新运行评测后验证

## 📋 下一步操作指南

### 1. 运行评测
```bash
cd /root/digitalhuman/RLVMR/code

# 确保vLLM正在运行
# 如果没有运行，先执行：bash start_vllm.sh

# 运行评测
bash run_eval_only.sh
```

### 2. 验证修复效果
评测完成后，运行验证脚本：
```bash
python3 verify_bdrs_config.py
```

**预期看到**:
```
Prompt格式:
  - 包含BDRS标签: ✓
  - 包含belief modules: ✓
  - 要求<think>标签: ✗

模型输出:
  - 使用BDRS标签: ✓
  - 使用<think>标签: ✗
  - 动作是否有效: ✓

✅ 配置正确，模型正常工作
```

### 3. 检查关键指标

查看新生成的trajectory文件：
```bash
# 找到最新的评测结果
ls -lt results/

# 快速统计成功率
python3 << 'PYEOF'
import json
from collections import defaultdict

# 替换为实际的trajectory路径
traj_file = "results/eval_YYYYMMDD_HHMMSS/trajectory.jsonl"

episodes = defaultdict(lambda: {'won': False})
with open(traj_file, 'r') as f:
    for line in f:
        d = json.loads(line)
        if d.get('done'):
            episodes[d['env_id']]['won'] = d.get('won', False)

total = len(episodes)
success = sum(1 for e in episodes.values() if e['won'])
print(f"成功率: {success}/{total} = {success/total*100:.1f}%")
PYEOF
```

## 📈 预期改进效果

| 指标 | 修复前 | 修复后预期 | 目标 |
|------|--------|-----------|------|
| 成功率 | 0% | 15-25% | >20% |
| 动作有效率 | 0% | 85-95% | >90% |
| 平均步数 | 29.5 | 15-20 | <20 |
| BDRS标签使用 | 0% | 100% | 100% |

### 如果成功率仍然<15%

可能需要进一步优化：

1. **检查BDRS标签分布**:
   ```bash
   python3 << 'PYEOF'
   import json
   from collections import Counter
   
   with open('results/latest/trajectory.jsonl', 'r') as f:
       modes = []
       for line in f:
           d = json.loads(line)
           action = d.get('action', '')
           for mode in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
               if f'<{mode}>' in action:
                   modes.append(mode)
                   break
   
   print(Counter(modes))
   # 期望分布: EXECUTE最多(40-50%), EXPLORE(30-40%), PLAN(10-15%), VERIFY(5-10%)
   PYEOF
   ```

2. **优化BDRS Runtime Prompt** (如果模式选择不合理)
   - 文件: `agent_system/environments/prompts/alfworld.py`
   - 增强模式选择指导
   - 添加更多示例

3. **增强冷启动训练数据**
   - 增加样本数量 (300 → 500+)
   - 平衡任务类型分布
   - 添加失败案例的标注

## 🔍 故障排查

如果修复后仍有问题：

1. **检查prompt是否真的使用了BDRS格式**:
   ```bash
   python3 << 'PYEOF'
   import json
   with open('results/latest/trajectory.jsonl', 'r') as f:
       first = json.loads(f.readline())
       prompt = first['prompt']
       print("Has <PLAN>:", '<PLAN>' in prompt)
       print("Has belief modules:", 'M_t' in prompt)
       print("\nPrompt sample:")
       print(prompt[:500])
   PYEOF
   ```

2. **检查模型是否正确加载**:
   ```bash
   curl -s http://localhost:8000/v1/models | python3 -m json.tool
   # 应该看到你的checkpoint路径
   ```

3. **查看详细日志**:
   ```bash
   tail -f logs/alfworld/*.log
   ```

## 📞 联系信息

如果遇到问题，检查：
1. vLLM是否正常运行 (`curl http://localhost:8000/health`)
2. 模型checkpoint路径是否正确
3. verify_bdrs_config.py的输出

---

**生成时间**: $(date)
**修复人**: Claude Code
**预期完成时间**: 运行评测约需5-15分钟（取决于环境数量）
