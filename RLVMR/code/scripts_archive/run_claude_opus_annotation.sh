#!/bin/bash
# 使用Claude Opus 4.5进行ReBel标注
# 自动配置和运行脚本

set -e

echo "=========================================="
echo "ReBel标注 - Claude Opus 4.5配置"
echo "=========================================="

# API配置
export OPENAI_API_KEY="sk-sIY1HNPxgl4liDRw5zZ6ivUlvzBKLL9mtkhBOwulBarG9LKV"
API_BASE="https://api.yourapi.cn/v1"
MODEL_NAME="claude-opus-4-5-20251101"

echo "✅ API配置完成"
echo "   模型: $MODEL_NAME"
echo "   端点: $API_BASE"
echo ""

# 测试连接
echo "测试API连接..."
response=$(curl -s -X POST "${API_BASE}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }' 2>&1)

if echo "$response" | grep -q "error"; then
    echo "❌ API连接测试失败"
    echo "响应: $response"
    echo ""
    echo "请检查："
    echo "  1. API key是否正确"
    echo "  2. API端点是否可访问"
    echo "  3. 模型名称是否正确"
    exit 1
fi

echo "✅ API连接成功"
echo ""

# 询问参数
read -p "要标注的样本数量 [100]: " NUM_SAMPLES
NUM_SAMPLES=${NUM_SAMPLES:-100}

read -p "输出目录 [data/rebel_claude_opus4.5]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-data/rebel_claude_opus4.5}

read -p "专家数据路径 [data/alfworld_expert_traj.json]: " EXPERT_DATA
EXPERT_DATA=${EXPERT_DATA:-data/alfworld_expert_traj.json}

# 检查专家数据
if [ ! -f "$EXPERT_DATA" ] && [ ! -d "$EXPERT_DATA" ]; then
    echo "❌ 未找到专家数据: $EXPERT_DATA"
    exit 1
fi

echo ""
echo "=========================================="
echo "配置摘要"
echo "=========================================="
echo "Teacher模型: Claude Opus 4.5"
echo "样本数量: $NUM_SAMPLES"
echo "输出目录: $OUTPUT_DIR"
echo "专家数据: $EXPERT_DATA"
echo "预期质量: 95-98%标注成功率"
echo "预期时间: 约 $((NUM_SAMPLES / 10)) 分钟"
echo ""
read -p "确认开始标注? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始标注..."
echo "=========================================="

# 运行标注
python generate_rebel_hindsight.py \
    --expert_data "$EXPERT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
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
echo "运行质量检验..."
echo "=========================================="

# 质量检验
python verify_annotation_quality.py \
    --annotated_data "$OUTPUT_DIR" \
    --output_report "${OUTPUT_DIR}/quality_report.txt"

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
echo ""
echo "输出位置: $OUTPUT_DIR"
echo "质量报告: ${OUTPUT_DIR}/quality_report.txt"
echo ""

# 显示质量摘要
if [ -f "${OUTPUT_DIR}/quality_report.txt" ]; then
    echo "质量摘要:"
    echo "---"
    grep -E "标注成功率|最终质量得分|推荐" "${OUTPUT_DIR}/quality_report.txt" || true
    echo ""
fi

echo "查看完整报告:"
echo "  cat ${OUTPUT_DIR}/quality_report.txt"
echo ""
echo "查看示例数据:"
echo "  head -100 ${OUTPUT_DIR}/rebel_hindsight.jsonl | less"
