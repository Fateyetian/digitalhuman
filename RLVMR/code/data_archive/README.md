# 数据归档目录

本目录存放项目开发过程中产生的中间数据和测试数据。

## 归档内容

### 测试数据集
- `rebel_test_10samples/` - 10样本测试集
- `rebel_test_fixed/` - 修复后的测试集
- `rebel_test_improved/` - 改进后的测试集
- `alfworld_rebel_test_3samples_improved/` - 3样本改进测试集

### 中间版本数据集
- `alfworld_rebel_250_new/` - 250样本ReBel数据集
- `alfworld_rebel_cold_start/` - Cold-start数据（早期版本）
- `alfworld_rebel_full_improved/` - 改进版完整数据集
- `alfworld_rebel_golden/` - Golden轨迹数据集

## 使用说明

这些数据集是开发和调试过程中产生的中间版本，已被最终版本替代。

**最新可用数据集位于主data目录：**
- `data/alfworld_rebel_merged_final/` - 最终合并的ReBel数据集（390个成功样本）
  - 包含clean版本的cold-start数据: `rebel_coldstart_clean.json`

## 归档原因

- 保持主data目录简洁
- 便于版本追踪和对比
- 节省主目录空间

---

归档时间: 2025-12-24
