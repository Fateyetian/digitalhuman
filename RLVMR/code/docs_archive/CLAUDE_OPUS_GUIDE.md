# Claude Opus 4.5 标注配置指南

## 快速开始

### 方法1: 使用专用脚本（推荐）

```bash
cd /root/testttt/RLVMR/code
bash run_claude_opus_annotation.sh
```

这个脚本会：
1. ✅ 自动测试API连接
2. ✅ 引导设置参数
3. ✅ 运行标注
4. ✅ 自动质量检验
5. ✅ 显示质量摘要

---

### 方法2: 手动运行

```bash
# 1. 设置API密钥
export OPENAI_API_KEY="sk-sIY1HNPxgl4liDRw5zZ6ivUlvzBKLL9mtkhBOwulBarG9LKV"

# 2. 运行标注
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_claude_opus4.5 \
    --num_samples 100 \
    --teacher_model_url https://api.yourapi.cn/v1 \
    --teacher_model_name claude-opus-4-5-20251101 \
    --temperature 0.2

# 3. 质量检验
python verify_annotation_quality.py \
    --annotated_data data/rebel_claude_opus4.5 \
    --output_report data/rebel_claude_opus4.5/quality_report.txt

# 4. 查看报告
cat data/rebel_claude_opus4.5/quality_report.txt
```

---

## Claude Opus 4.5 性能预期

### 质量指标

| 指标 | 预期值 | 备注 |
|------|--------|------|
| 标注成功率 | 95-98% | 最高水平 |
| Belief完整性 | 95%+ | 字段齐全 |
| Inventory准确性 | 98%+ | 逻辑严密 |
| Cleared覆盖率 | 90%+ | 模式识别强 |
| Reasoning质量 | 90%+ | 清晰深入 |
| **最终得分** | **88-95%** | 顶级质量 |

### 成本估算

- **定价**: 约$15-30/百万tokens（具体看API提供商）
- **每样本token消耗**:
  - 输入: ~2000-3000 tokens (Prompt + Context)
  - 输出: ~300-500 tokens (Belief + Reasoning)
  - 总计: ~2500-3500 tokens/样本
- **100样本总成本**:
  - Token总量: ~300,000 tokens
  - 预估费用: **$4.5-9** (约30-60 RMB)

### 速度

- API延迟: 2-5秒/样本
- 100样本总时间: **约5-10分钟**

---

## 推荐配置

### 最佳参数

```bash
--temperature 0.2          # 低温度确保稳定输出
--num_samples 100          # 建议批量处理
```

**为什么temperature=0.2?**
- Claude Opus已经非常强大
- 低温度确保输出格式稳定
- 减少JSON解析错误
- 保持推理一致性

---

## 质量检验要点

### 预期达标标准

由于Claude Opus 4.5质量极高，应该轻松达到：

✅ **标注成功率**: ≥95%
✅ **Belief完整性**: ≥95%
✅ **Inventory准确性**: ≥95%
✅ **Cleared覆盖率**: ≥85%
✅ **Reasoning质量**: ≥90%
✅ **格式正确率**: 100%
✅ **最终得分**: ≥90%

### 如果得分低于预期

如果最终得分<85%，可能原因：

1. **API兼容性问题**
   - 检查API端点是否完全兼容OpenAI格式
   - 尝试调整temperature

2. **Prompt不匹配**
   - Claude可能对某些Prompt模式有偏好
   - 可以微调prompt措辞

3. **数据格式问题**
   - 检查专家数据格式是否正确
   - 确认AVAILABLE ACTIONS字段存在

---

## 常见问题

### Q1: API返回错误

```bash
# 测试API连接
curl -X POST "https://api.yourapi.cn/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-sIY1HNPxgl4liDRw5zZ6ivUlvzBKLL9mtkhBOwulBarG9LKV" \
    -d '{
        "model": "claude-opus-4-5-20251101",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100
    }'
```

如果返回错误，检查：
- ✅ API key是否有效
- ✅ 模型名称是否正确
- ✅ API余额是否充足

### Q2: JSON解析错误频繁

尝试：
```bash
# 降低temperature
--temperature 0.1

# 或在prompt中强调JSON格式
```

### Q3: 速度太慢

Claude Opus 4.5是最大的模型，每次调用2-5秒是正常的。

优化建议：
- 使用批量API（如果API提供商支持）
- 分批处理，避免超时

### Q4: 成本控制

```bash
# 先测试小批量
--num_samples 10

# 查看质量和成本
# 如果满意再扩大到100+
```

---

## 实际运行示例

```bash
# 完整流程
cd /root/testttt/RLVMR/code

# 方式1: 使用专用脚本（推荐）
bash run_claude_opus_annotation.sh

# 按提示操作：
# - 确认样本数量: 100
# - 确认输出目录: 默认
# - 确认专家数据: 默认
# - 确认开始: y

# 等待5-10分钟...

# 查看结果
cat data/rebel_claude_opus4.5/quality_report.txt

# 如果最终得分≥90%，恭喜！这是顶级质量的数据
```

---

## 质量示例

### 预期的Belief输出

```json
{
  "world_model_update": {
    "found_objects": {
      "alarmclock 1": "sidetable 1",
      "cd 1": "sidetable 1"
    },
    "inventory": null,
    "state_changes": {
      "drawer 3": "open"
    },
    "cleared_receptacles": ["drawer 1", "drawer 2", "drawer 3"]
  },
  "task_progress_update": {
    "subgoal_status": "in_progress",
    "evidence": "Checked all 3 drawers without finding the alarm clock",
    "updated_subgoal": "Search shelves for alarm clock"
  },
  "exploration_map_update": {
    "newly_visited": ["drawer 3"],
    "next_priority": ["shelf 1", "shelf 2"]
  }
}
```

### 预期的Reasoning

```
Based on the observation, I opened drawer 3 and only found a pencil,
which is not my target object (alarm clock). Since I have now checked
all three drawers (1, 2, and 3) without finding the alarm clock, I
should expand my search to other furniture types. The shelves are the
next logical location to check, as alarm clocks are commonly placed
on shelves in bedrooms.
```

这种质量的输出应该是Claude Opus的常态。

---

## 安全提醒

⚠️ **API Key安全**:

1. **不要分享**: API key已在代码中，请勿将这些文件分享给他人
2. **环境变量**: 使用完后可以unset
   ```bash
   unset OPENAI_API_KEY
   ```
3. **定期轮换**: 建议定期更新API key
4. **监控使用**: 定期检查API使用量和账单

---

## 下一步

标注完成后：

1. **验证质量**
   ```bash
   cat data/rebel_claude_opus4.5/quality_report.txt
   ```

2. **查看示例**
   ```bash
   head -200 data/rebel_claude_opus4.5/rebel_hindsight.jsonl | less
   ```

3. **如果质量满意（得分≥85%）**
   - 可以扩大到更多样本（500-1000）
   - 直接用于SFT训练
   - 预期模型性能会有显著提升

4. **SFT训练**
   - 使用生成的数据集
   - 参考 `examples/rebel_trainer/` 中的训练脚本
   - 预期在ALFWorld上的成功率会大幅提升

---

**准备好了吗？运行：**

```bash
bash run_claude_opus_annotation.sh
```

标注过程中可以实时看到进度和任何警告信息。完成后会自动生成质量报告！
