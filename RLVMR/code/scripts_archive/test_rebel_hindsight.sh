#!/bin/bash
# Test script for ReBel Hindsight Annotation System

echo "=========================================="
echo "ReBel Hindsight Annotation - Quick Test"
echo "=========================================="

# Check if expert data exists
if [ ! -f "data/alfworld_expert_traj.json" ] && [ ! -d "data/alfworld_expert_traj" ]; then
    echo "❌ Error: Expert trajectory data not found!"
    echo "   Expected: data/alfworld_expert_traj.json or data/alfworld_expert_traj/"
    exit 1
fi

echo "✅ Expert data found"

# Check if vLLM server is running
echo ""
echo "Checking for Teacher LLM server..."
if curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    echo "✅ Teacher LLM server is running at http://127.0.0.1:8000"
    SERVER_URL="http://127.0.0.1:8000/v1"
    MODEL_NAME="Qwen2.5-7B-Instruct"
else
    echo "⚠️  Local vLLM server not detected"
    echo "   You can either:"
    echo "   1. Start a local vLLM server (recommended for testing)"
    echo "   2. Use OpenAI API (requires API key)"
    echo ""
    read -p "Use OpenAI API? (y/N): " use_openai

    if [[ "$use_openai" =~ ^[Yy]$ ]]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "❌ Error: OPENAI_API_KEY not set"
            exit 1
        fi
        SERVER_URL="https://api.openai.com/v1"
        MODEL_NAME="gpt-4"
        echo "✅ Using OpenAI API with GPT-4"
    else
        echo "❌ No Teacher LLM available. Exiting."
        echo ""
        echo "To start a local vLLM server:"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
        echo "    --port 8000"
        exit 1
    fi
fi

# Run test with 2 samples
echo ""
echo "=========================================="
echo "Running annotation on 2 test samples..."
echo "=========================================="

python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/test_rebel_hindsight \
    --num_samples 2 \
    --teacher_model_url "$SERVER_URL" \
    --teacher_model_name "$MODEL_NAME" \
    --temperature 0.2

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Test completed successfully!"
    echo "=========================================="

    if [ -d "data/test_rebel_hindsight" ]; then
        echo ""
        echo "Output location: data/test_rebel_hindsight/"
        echo ""
        echo "Files generated:"
        ls -lh data/test_rebel_hindsight/

        echo ""
        echo "Sample data preview:"
        echo "---"
        head -30 data/test_rebel_hindsight/rebel_hindsight.jsonl 2>/dev/null || echo "(JSONL not found)"
    fi
else
    echo ""
    echo "❌ Test failed. Check errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Review the generated data in data/test_rebel_hindsight/"
echo "2. Check annotation quality and success rate"
echo "3. If quality is good, run on full dataset:"
echo "   python generate_rebel_hindsight.py --num_samples 100"
echo ""
