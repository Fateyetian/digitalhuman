#!/usr/bin/env python3
"""
测试 vLLM 是否能独立加载模型

用途：当评测脚本卡住时，使用此脚本验证问题是否出在 vLLM 模型加载上

预期结果：
  - 如果成功：会打印 "✓ vLLM 模型加载成功！" 和推理结果
  - 如果失败：会显示具体错误信息，帮助定位问题

运行方法：
  python test_vllm_load.py
"""

import sys
import torch

def main():
    print("=" * 60)
    print("vLLM 模型加载测试")
    print("=" * 60)

    # 检查 CUDA 环境
    print(f"\n1. CUDA 环境检查:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("   ✗ CUDA 不可用，无法运行 vLLM")
        sys.exit(1)

    # 模型路径（修改为你的实际路径）
    MODEL_PATH = "/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75"

    print(f"\n2. 开始加载模型: {MODEL_PATH}")

    try:
        from vllm import LLM, SamplingParams

        print("   正在初始化 vLLM...")
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=2048,  # 限制序列长度，减少显存
        )
        print("   ✓ vLLM 模型加载成功！")

        # 简单推理测试
        print("\n3. 推理测试:")
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            top_p=1.0,
        )

        print("   正在生成...")
        outputs = llm.generate(prompts, sampling_params)

        result = outputs[0].outputs[0].text
        print(f"   输入: {prompts[0]}")
        print(f"   输出: {result}")
        print("   ✓ 推理测试成功！")

        print("\n" + "=" * 60)
        print("✓ 所有测试通过，vLLM 工作正常")
        print("=" * 60)

    except ImportError as e:
        print(f"   ✗ vLLM 未安装或导入失败: {e}")
        print("\n   解决方案：")
        print("   pip install vllm")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"   ✗ 模型文件未找到: {e}")
        print("\n   解决方案：")
        print("   1. 检查模型路径是否正确")
        print("   2. 确认模型文件完整（config.json, model.safetensors等）")
        print(f"   3. 运行: ls -lh {MODEL_PATH}/")
        sys.exit(1)

    except torch.cuda.OutOfMemoryError as e:
        print(f"   ✗ GPU 显存不足: {e}")
        print("\n   解决方案：")
        print("   1. 降低 gpu_memory_utilization（当前0.4，可改为0.3）")
        print("   2. 减少 max_model_len")
        print("   3. 使用更大显存的 GPU")
        sys.exit(1)

    except Exception as e:
        print(f"   ✗ vLLM 加载失败: {e}")
        print("\n   详细错误信息：")
        import traceback
        traceback.print_exc()
        print("\n   可能的原因：")
        print("   1. 模型文件损坏或格式不兼容")
        print("   2. CUDA 版本与 vLLM 不匹配")
        print("   3. 显存不足")
        print("   4. 模型配置有问题（如tokenizer缺失）")
        sys.exit(1)

if __name__ == '__main__':
    main()
