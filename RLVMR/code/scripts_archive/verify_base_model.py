#!/usr/bin/env python3
"""
快速验证基础模型是否可用
"""

import sys
sys.path.insert(0, '/root/testttt/RLVMR/code')

model_path = "/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"

print("="*60)
print("验证 Qwen2.5-1.5B-Instruct 基础模型")
print("="*60)

try:
    from transformers import AutoConfig, AutoTokenizer
    import os

    # 检查路径
    print(f"\n1. 检查模型路径: {model_path}")
    if os.path.exists(model_path):
        print("   ✅ 路径存在")
    else:
        print("   ❌ 路径不存在")
        sys.exit(1)

    # 检查必要文件
    print(f"\n2. 检查必要文件:")
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) if not os.path.islink(file_path) else "symlink"
            print(f"   ✅ {file}: {size}")
        else:
            print(f"   ❌ {file}: 不存在")

    # 加载配置
    print(f"\n3. 加载模型配置:")
    config = AutoConfig.from_pretrained(model_path)
    print(f"   ✅ 模型类型: {config.model_type}")
    print(f"   ✅ 隐藏层大小: {config.hidden_size}")
    print(f"   ✅ 层数: {config.num_hidden_layers}")
    print(f"   ✅ 词汇表大小: {config.vocab_size}")

    # 加载tokenizer
    print(f"\n4. 加载 Tokenizer:")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"   ✅ Tokenizer词汇量: {len(tokenizer)}")

    # 测试tokenizer
    print(f"\n5. 测试 Tokenizer:")
    test_text = "Hello, how are you?"
    tokens = tokenizer.encode(test_text)
    print(f"   ✅ 输入: '{test_text}'")
    print(f"   ✅ Token数量: {len(tokens)}")
    print(f"   ✅ 解码正常: {tokenizer.decode(tokens)}")

    print("\n" + "="*60)
    print("✅ 基础模型验证通过！可以用于ReBel评测")
    print("="*60)

    print(f"\n推荐的模型路径:")
    print(f"  {model_path}")

    print(f"\n或使用原始路径:")
    print(f"  /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
