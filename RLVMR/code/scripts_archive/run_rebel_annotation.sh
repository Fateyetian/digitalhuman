#!/bin/bash
# ReBel Hindsight Annotation - 完整运行指南
#
# 本脚本将引导您完成整个标注流程

set -e  # 遇到错误立即退出

echo "=================================="
echo "ReBel Hindsight Annotation 运行向导"
echo "=================================="

# ============================================================================
# 步骤 1: 环境检查
# ============================================================================

echo ""
echo "[步骤 1/5] 环境检查..."

# 检查Python依赖
echo "检查Python依赖..."
python -c "import openai, datasets, tqdm" 2>/dev/null || {
    echo "❌ 缺少必要的Python包"
    echo "   请安装: pip install openai datasets tqdm"
    exit 1
}
echo "✅ Python依赖已安装"

# 检查专家数据
echo "检查专家轨迹数据..."
if [ ! -f "data/alfworld_expert_traj.json" ] && [ ! -d "data/alfworld_expert_traj" ]; then
    echo "❌ 未找到专家轨迹数据"
    echo "   期望位置: data/alfworld_expert_traj.json 或 data/alfworld_expert_traj/"
    exit 1
fi
echo "✅ 专家数据已找到"

# ============================================================================
# 步骤 2: 选择Teacher模型
# ============================================================================

echo ""
echo "[步骤 2/5] 选择Teacher LLM..."
echo ""
echo "推荐模型列表（按质量排序）："
echo ""
echo "【商业模型 - 最高质量】"
echo "  1. GPT-4 (OpenAI)"
echo "     - 质量: ⭐⭐⭐⭐⭐"
echo "     - 成本: 约$30-50/100样本"
echo "     - 标注成功率: 95-98%"
echo "     - 适用场景: 生成最终黄金数据集"
echo ""
echo "  2. Claude-3.5-Sonnet (Anthropic)"
echo "     - 质量: ⭐⭐⭐⭐⭐"
echo "     - 成本: 约$25-40/100样本"
echo "     - 标注成功率: 94-97%"
echo "     - 适用场景: GPT-4的高质量替代"
echo ""
echo "【开源模型 - 本地部署】"
echo "  3. Qwen2.5-72B-Instruct"
echo "     - 质量: ⭐⭐⭐⭐"
echo "     - GPU需求: 2x A100 (80GB) 或 4x A6000"
echo "     - 标注成功率: 85-92%"
echo "     - 适用场景: 高质量本地部署"
echo ""
echo "  4. Qwen2.5-32B-Instruct"
echo "     - 质量: ⭐⭐⭐⭐"
echo "     - GPU需求: 1x A100 (80GB) 或 2x A6000"
echo "     - 标注成功率: 80-88%"
echo "     - 适用场景: 性价比最佳"
echo ""
echo "  5. Qwen2.5-14B-Instruct"
echo "     - 质量: ⭐⭐⭐"
echo "     - GPU需求: 1x A6000 (48GB)"
echo "     - 标注成功率: 75-85%"
echo "     - 适用场景: 中等规模数据集"
echo ""
echo "  6. Qwen2.5-7B-Instruct"
echo "     - 质量: ⭐⭐⭐"
echo "     - GPU需求: 1x RTX 4090 或 A10"
echo "     - 标注成功率: 70-80%"
echo "     - 适用场景: 快速原型/测试"
echo ""
echo "  7. Llama-3.1-70B-Instruct"
echo "     - 质量: ⭐⭐⭐⭐"
echo "     - GPU需求: 2x A100 (80GB)"
echo "     - 标注成功率: 82-90%"
echo "     - 适用场景: Qwen的替代方案"
echo ""

echo "请选择Teacher模型："
echo "  [1] GPT-4 (需要API密钥)"
echo "  [2] Claude-3.5-Sonnet (需要API密钥)"
echo "  [3] 本地vLLM - Qwen2.5系列"
echo "  [4] 本地vLLM - Llama3.1系列"
echo "  [5] 自定义API端点"
read -p "输入选项 (1-5): " model_choice

case $model_choice in
    1)
        echo "选择: GPT-4"
        if [ -z "$OPENAI_API_KEY" ]; then
            read -p "请输入OpenAI API Key: " OPENAI_API_KEY
            export OPENAI_API_KEY
        fi
        SERVER_URL="https://api.openai.com/v1"
        MODEL_NAME="gpt-4"
        TEMPERATURE=0.2
        ;;
    2)
        echo "选择: Claude-3.5-Sonnet"
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            read -p "请输入Anthropic API Key: " ANTHROPIC_API_KEY
            export ANTHROPIC_API_KEY
        fi
        echo "⚠️  注意: 需要使用OpenAI兼容的Claude代理"
        read -p "Claude API代理地址 (例如 https://api.claude-proxy.com/v1): " SERVER_URL
        MODEL_NAME="claude-3-5-sonnet-20241022"
        TEMPERATURE=0.2
        ;;
    3)
        echo "选择: 本地vLLM - Qwen系列"
        echo "请先启动vLLM服务器："
        echo ""
        echo "Qwen2.5-72B (需要 2x A100):"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen2.5-72B-Instruct \\"
        echo "    --port 8000 \\"
        echo "    --tensor-parallel-size 2"
        echo ""
        echo "Qwen2.5-32B (需要 1x A100):"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen2.5-32B-Instruct \\"
        echo "    --port 8000 \\"
        echo "    --tensor-parallel-size 1"
        echo ""
        echo "Qwen2.5-14B (需要 1x A6000):"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen2.5-14B-Instruct \\"
        echo "    --port 8000"
        echo ""
        read -p "服务器端口 [8000]: " port
        port=${port:-8000}
        SERVER_URL="http://127.0.0.1:${port}/v1"
        read -p "模型大小 (7B/14B/32B/72B) [32B]: " size
        size=${size:-32B}
        MODEL_NAME="Qwen2.5-${size}-Instruct"
        TEMPERATURE=0.3
        ;;
    4)
        echo "选择: 本地vLLM - Llama系列"
        echo "启动命令示例:"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model meta-llama/Meta-Llama-3.1-70B-Instruct \\"
        echo "    --port 8000 \\"
        echo "    --tensor-parallel-size 2"
        read -p "服务器端口 [8000]: " port
        port=${port:-8000}
        SERVER_URL="http://127.0.0.1:${port}/v1"
        MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"
        TEMPERATURE=0.3
        ;;
    5)
        echo "选择: 自定义端点"
        read -p "API端点URL: " SERVER_URL
        read -p "模型名称: " MODEL_NAME
        read -p "Temperature [0.3]: " TEMPERATURE
        TEMPERATURE=${TEMPERATURE:-0.3}
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

# 检查服务器连接
echo ""
echo "检查Teacher LLM连接..."
if [[ "$SERVER_URL" == *"127.0.0.1"* ]] || [[ "$SERVER_URL" == *"localhost"* ]]; then
    if ! curl -s "${SERVER_URL}/models" > /dev/null 2>&1; then
        echo "❌ 无法连接到本地服务器: $SERVER_URL"
        echo "   请确保vLLM服务器正在运行"
        exit 1
    fi
fi
echo "✅ Teacher LLM配置完成"

# ============================================================================
# 步骤 3: 设置标注参数
# ============================================================================

echo ""
echo "[步骤 3/5] 设置标注参数..."

read -p "要标注的样本数量 [100]: " NUM_SAMPLES
NUM_SAMPLES=${NUM_SAMPLES:-100}

read -p "输出目录 [data/alfworld_rebel_hindsight]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-data/alfworld_rebel_hindsight}

read -p "专家数据路径 [data/alfworld_expert_traj.json]: " EXPERT_DATA
EXPERT_DATA=${EXPERT_DATA:-data/alfworld_expert_traj.json}

echo ""
echo "配置摘要:"
echo "  Teacher模型: $MODEL_NAME"
echo "  API端点: $SERVER_URL"
echo "  Temperature: $TEMPERATURE"
echo "  样本数量: $NUM_SAMPLES"
echo "  输出目录: $OUTPUT_DIR"
echo "  专家数据: $EXPERT_DATA"
echo ""
read -p "确认开始标注? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ============================================================================
# 步骤 4: 运行标注
# ============================================================================

echo ""
echo "[步骤 4/5] 开始标注..."
echo "=================================="

python generate_rebel_hindsight.py \
    --expert_data "$EXPERT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --teacher_model_url "$SERVER_URL" \
    --teacher_model_name "$MODEL_NAME" \
    --temperature $TEMPERATURE

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 标注失败，请检查上方错误信息"
    exit 1
fi

# ============================================================================
# 步骤 5: 质量检验
# ============================================================================

echo ""
echo "[步骤 5/5] 质量检验..."
echo "=================================="

# 运行质量检验脚本
python verify_annotation_quality.py \
    --annotated_data "$OUTPUT_DIR" \
    --output_report "${OUTPUT_DIR}/quality_report.txt"

echo ""
echo "✅ 标注完成！"
echo ""
echo "输出位置: $OUTPUT_DIR"
echo "质量报告: ${OUTPUT_DIR}/quality_report.txt"
echo ""
echo "下一步："
echo "  1. 查看质量报告: cat ${OUTPUT_DIR}/quality_report.txt"
echo "  2. 检查示例数据: head -50 ${OUTPUT_DIR}/rebel_hindsight.jsonl"
echo "  3. 如果质量满意，可用于SFT训练"
