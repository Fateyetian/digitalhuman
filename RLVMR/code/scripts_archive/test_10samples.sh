#!/bin/bash
# Claude Opus 4.5 - 测试标注（10条样本）
# 输出JSON格式，便于人工检查

set -e

echo "=========================================="
echo "ReBel测试标注 - Claude Opus 4.5"
echo "=========================================="
echo "样本数量: 10条"
echo "输出格式: JSON"
echo "目的: 质量验证"
echo ""

# API配置
export OPENAI_API_KEY="sk-sIY1HNPxgl4liDRw5zZ6ivUlvzBKLL9mtkhBOwulBarG9LKV"
API_BASE="https://api.yourapi.cn/v1"
MODEL_NAME="claude-opus-4-5-20251101"

# 测试输出目录
OUTPUT_DIR="data/rebel_test_10samples"
EXPERT_DATA="data/alfworld_expert_traj.json"

# 检查专家数据
if [ ! -f "$EXPERT_DATA" ] && [ ! -d "$EXPERT_DATA" ]; then
    echo "❌ 未找到专家数据: $EXPERT_DATA"
    echo "   请确认路径是否正确"
    exit 1
fi

echo "✅ 专家数据: $EXPERT_DATA"
echo ""

# 清理旧的测试数据
if [ -d "$OUTPUT_DIR" ]; then
    echo "发现旧的测试数据，清理中..."
    rm -rf "$OUTPUT_DIR"
fi

echo "=========================================="
echo "开始标注10条测试样本..."
echo "=========================================="
echo ""

# 运行标注
python generate_rebel_hindsight.py \
    --expert_data "$EXPERT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 10 \
    --teacher_model_url "$API_BASE" \
    --teacher_model_name "$MODEL_NAME" \
    --temperature 0.2

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 标注失败，请检查上方错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "标注完成！运行质量检验..."
echo "=========================================="
echo ""

# 质量检验
python verify_annotation_quality.py \
    --annotated_data "$OUTPUT_DIR" \
    --output_report "${OUTPUT_DIR}/quality_report.txt"

echo ""
echo "=========================================="
echo "生成易读的JSON文件..."
echo "=========================================="

# 将JSONL转换为格式化的JSON数组（便于查看）
python3 << 'PYEOF'
import json
import os

input_file = "data/rebel_test_10samples/rebel_hindsight.jsonl"
output_file = "data/rebel_test_10samples/rebel_hindsight_formatted.json"

if os.path.exists(input_file):
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # 保存为格式化的JSON数组
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成格式化JSON: {output_file}")
    print(f"   包含 {len(samples)} 个样本")
else:
    print(f"❌ 未找到: {input_file}")
PYEOF

echo ""
echo "=========================================="
echo "生成第一个样本的详细示例..."
echo "=========================================="

# 提取第一个样本的前两个对话轮次作为示例
python3 << 'PYEOF'
import json

try:
    with open("data/rebel_test_10samples/rebel_hindsight.jsonl", 'r') as f:
        first_sample = json.loads(f.readline())

    # 保存第一个样本到单独文件
    with open("data/rebel_test_10samples/sample_1_full.json", 'w', encoding='utf-8') as f:
        json.dump(first_sample, f, indent=2, ensure_ascii=False)

    # 提取前4个对话轮次（系统消息 + 第一轮对话）
    preview = {
        'item_id': first_sample.get('item_id', 'unknown'),
        'task': first_sample.get('task', 'unknown'),
        'num_steps': first_sample.get('num_steps', 0),
        'annotation_success_rate': first_sample.get('annotation_success_rate', 0),
        'conversations_preview': first_sample['conversations'][:4]
    }

    with open("data/rebel_test_10samples/sample_1_preview.json", 'w', encoding='utf-8') as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print("✅ 已生成示例文件:")
    print("   - sample_1_full.json (完整样本)")
    print("   - sample_1_preview.json (前4轮对话预览)")

except Exception as e:
    print(f"❌ 生成示例时出错: {e}")
PYEOF

echo ""
echo "=========================================="
echo "✅ 测试标注完成！"
echo "=========================================="
echo ""
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "生成的文件："
echo "  1. rebel_hindsight.jsonl          - 原始JSONL格式"
echo "  2. rebel_hindsight_formatted.json - 格式化JSON数组"
echo "  3. sample_1_full.json             - 第1个样本（完整）"
echo "  4. sample_1_preview.json          - 第1个样本（前4轮预览）"
echo "  5. quality_report.txt             - 质量检验报告"
echo ""

# 显示质量摘要
echo "=========================================="
echo "质量检验摘要"
echo "=========================================="
if [ -f "${OUTPUT_DIR}/quality_report.txt" ]; then
    grep -E "标注成功率|最终质量得分|推荐" "${OUTPUT_DIR}/quality_report.txt" || true
else
    echo "未找到质量报告"
fi
echo ""

# 显示第一个样本的基本信息
echo "=========================================="
echo "第一个样本信息"
echo "=========================================="
python3 << 'PYEOF'
import json
try:
    with open("data/rebel_test_10samples/sample_1_preview.json", 'r') as f:
        preview = json.load(f)

    print(f"任务ID: {preview['item_id']}")
    print(f"任务: {preview['task']}")
    print(f"步数: {preview['num_steps']}")
    print(f"标注成功率: {preview['annotation_success_rate']:.1%}")
    print()
    print("前4轮对话:")
    for i, turn in enumerate(preview['conversations_preview']):
        print(f"\n[{i}] {turn['from']}:")
        value = turn['value']
        if len(value) > 300:
            print(value[:300] + "...")
        else:
            print(value)
except:
    pass
PYEOF

echo ""
echo "=========================================="
echo "请检查以下内容："
echo "=========================================="
echo ""
echo "1. 查看质量报告:"
echo "   cat data/rebel_test_10samples/quality_report.txt"
echo ""
echo "2. 查看第一个样本（完整）:"
echo "   cat data/rebel_test_10samples/sample_1_full.json | less"
echo ""
echo "3. 查看第一个样本（预览）:"
echo "   cat data/rebel_test_10samples/sample_1_preview.json"
echo ""
echo "4. 查看所有样本（格式化）:"
echo "   cat data/rebel_test_10samples/rebel_hindsight_formatted.json | less"
echo ""
echo "=========================================="
echo "检查要点："
echo "=========================================="
echo ""
echo "✅ 质量报告中最终得分应 ≥85%"
echo "✅ Human Turn包含: Task + Observation + Belief + Actions"
echo "✅ GPT Turn包含: <belief> + <reasoning> + <action>"
echo "✅ Belief JSON格式正确，包含inventory字段"
echo "✅ Reasoning逻辑清晰，长度适中（100-300字符）"
echo "✅ Action在Available Actions中"
echo ""
echo "如果以上都满足，可以运行完整标注:"
echo "   bash run_claude_opus_annotation.sh"
echo ""
