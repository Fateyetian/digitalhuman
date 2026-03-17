# ReBel 论文构建方案
**目标**：从现有代码、实验数据、草稿 → 一篇可提交 NeurIPS 2026 的完整论文
**当前状态**：main.tex 18页初稿已能编译，实验图表11张已生成，数据齐全
**截稿目标**：NeurIPS 2026 Abstract ~5/15，Full paper ~5/22

---

## 模型分工原则

| 任务类型 | 首选模型 | 备选 | 理由 |
|---------|---------|------|------|
| 长文逻辑推理、方法创新叙述 | `gemini-2.5-pro-thinking` | `o3-2025-04-16` | 超长上下文 + 深度推理 |
| 英文学术润色、句式重写 | `gpt-4.1` | `claude-opus-4-6` | 英文写作质量最高 |
| 去AI味、逻辑检查 | `gpt-4.1` | `gemini-2.5-flash` | 保守风格，不过度修改 |
| 数学公式推导验证 | `o3-2025-04-16` | `gemini-2.5-pro-thinking` | 数学严谨性最强 |
| 中文→英文翻译 | `gpt-4.1` | `claude-opus-4-6` | 翻译地道性 |
| 图表代码生成（Python/matplotlib） | `claude-opus-4-6` | `gpt-4.1` | 代码能力 |
| 主图（架构图）生成 | `gemini-2.5-pro` (image) | GPT-4o image | 视觉生成 |
| Reviewer 模拟审稿 | `gemini-2.5-pro-thinking` | `o3` | 批判性思维 |

**调用方式**：通过 dmxapi（`https://www.dmxapi.cn`），API KEY 已配置在环境变量
**辅助脚本**：`/tmp/dmx_refine.py`（可按需扩展模型参数）

---

## 全流程步骤（共8个阶段）

---

### 阶段 1：数据核查与图表审计 ✅（已完成）

**目标**：确认所有图表数据与论文叙述一致，无笔误
**人工介入点**：✋ 对照 PAPER_INDEX.md 中的数值，逐表核对

**检查清单**：
- [ ] Table 1 主结果（M1/M3/M4/M5 的 Peak/Final/Late-Avg SR）
- [ ] Table 2 V10消融（7行，峰值 + Late-Avg）
- [ ] Table 3 V11针对性消融（A1/A3/A4 vs ReBel Full）
- [ ] Table 4 Valid Action Ratio（step1/step100/mean）
- [ ] Table 5 WebShop 结果（3 seeds，含 valid_act_ratio）
- [ ] 所有图的 y 轴数值与 training.log 一致

**关键数据参考**：`/root/testttt/RLVMR/PAPER_INDEX.md`

---

### 阶段 2：实验图表质量提升 🔴（高优先级）

**目标**：将现有 `paper_figures/` 中的图（Python 脚本自动生成）提升到顶会视觉标准

**现有图表**：11张（fig1~fig11，PDF+PNG双格式）
**生成脚本**：`code/rebel_test_results/v11_final/paper_figures/generate_all_figures.py`

#### 2.1 图表质量评估（人工）
✋ **人工介入**：打开每张 PNG，对照以下标准评估：

| 图编号 | 内容 | 用于章节 | 需重绘？ |
|--------|------|---------|---------|
| fig1 | V10 训练曲线（7组）| §Experiments | 待检查 |
| fig2 | 主实验方法对比柱状图 | §Main Results | 待检查 |
| fig3 | Per-Task SR 热力图/分组柱 | §Analysis | 待检查 |
| fig4 | 因素贡献瀑布图 | §Analysis | 待检查 |
| fig5 | HiBO 覆盖率分析 | §Method | 待检查 |
| fig6 | 自适应衰减曲线 | §Method | 待检查 |
| fig7 | 消融实验对比 | §Experiments | 待检查 |
| fig8 | 学习速度对比 | §Analysis | 待检查 |
| fig9 | 算法框架图（主图）| §Method | ⚠️ 需重绘 |
| fig10 | 版本演进曲线 | §Intro/附录 | 待检查 |
| fig11 | GiGPO单例问题示意 | §Motivation | ⚠️ 需重绘 |

#### 2.2 数据图重绘（Claude Code 执行）
**工具**：修改 `generate_all_figures.py`，参考写作Prompt中"实验绘图推荐"
**配色规范**：
- 主色系：蓝`#2E86AB`、橙`#E8871E`、绿`#4CAF50`、红`#E53935`
- 背景：白色，无网格或浅灰网格
- 字体：Arial/Helvetica，标题12pt，轴标10pt
- 线宽：2pt，误差棒：标准差

```bash
# 执行重绘
cd /root/testttt/RLVMR/code/rebel_test_results/v11_final/paper_figures/
python generate_all_figures.py
```

#### 2.3 主架构图（fig9）重绘 ✋（人工+AI）
**方法**：使用写作Prompt中"论文架构图"模板，调用 `gemini-2.5-pro` 图像生成
**内容要素**：
1. 左：SFT冷启动 → RL训练循环
2. 中：ReBel三创新（HiBO分组 / 自适应Curriculum / Belief Prompt）
3. 右：ALFWorld + WebShop 双环境评测结果

**参考样式**：DeepMind/OpenAI论文中的方法概览图风格

#### 2.4 图11单例问题示意图重绘
**要求**：左图显示GiGPO的singleton group（大量0 advantage），右图显示HiBO恢复后的分布
可使用 Python matplotlib 绘制，无需图像生成模型

---

### 阶段 3：各章节精细化写作 🔴

当前草稿各章节完成度评估：

| 章节 | 当前状态 | 主要问题 | 优先级 |
|------|---------|---------|--------|
| Abstract | 初稿 | 结构可能不符合"问题→挑战→方法→结果→意义"五段式 | 高 |
| §1 Introduction | 初稿 | 开篇冲击力、贡献点表达需优化 | 高 |
| §2 Related Work | 初稿 | 分组逻辑、与本文的对比角度需锐化 | 中 |
| §3 Background | 初稿 | Singleton问题的量化论证需加强 | 中 |
| §4 Method | 初稿 | 公式推导过渡、Algorithm 1伪代码需检查 | 高 |
| §5 Experiments | **已完成** | WebShop baseline结果待填入 | 低（等数据）|
| §6 Conclusion | 初稿 | 局限性和未来工作需具体化 | 低 |

#### 3.1 Abstract 优化流程
1. **自动**：用写作Prompt §"摘要优化"模板发送给 `gpt-4.1`
2. ✋ **人工**：检查5个结构要素是否齐全，数值是否准确
3. **自动**：用"去AI味"Prompt二次处理

#### 3.2 Introduction 优化流程
1. **自动**：用"Introduction逻辑优化"Prompt发送给 `gemini-2.5-pro-thinking`（需要长上下文看全文）
2. ✋ **人工**：确认3点贡献与实验结果一一对应
3. **自动**：用"学术英文润色"Prompt发送给 `gpt-4.1`

#### 3.3 Method 章节公式检查
1. **自动**：将所有公式块发送给 `o3-2025-04-16`，验证推导一致性
2. ✋ **人工**：检查 Algorithm 1 伪代码行号逻辑
3. **自动**：润色叙述文字

#### 3.4 逐节标准化润色流程（通用）
每节按此顺序处理：
```
Step 1: [逻辑检查 Prompt] → gpt-4.1 → 确认无致命逻辑问题
Step 2: [学术英文润色 Prompt] → gpt-4.1 → 提升语言质量
Step 3: [去AI味 Prompt] → gpt-4.1 → 消除机械感
Step 4: ✋ 人工核对数值、引用、LaTeX命令无误
Step 5: 合并修改，重新编译 pdflatex
```

---

### 阶段 4：公式与符号体系规范化 🟡

**目标**：全文符号使用一致，无歧义

**检查项**：
- [ ] 所有数学符号首次出现时给出定义
- [ ] 公式编号连贯，正文引用格式统一（`Eq.~\ref{}`）
- [ ] 矩阵/向量区分（粗体向量、大写矩阵）
- [ ] 算法伪代码变量与正文符号一致
- [ ] `\epsilon`, `\lambda`, `\gamma`, `\tau`, `\alpha` 在不同上下文无歧义

**执行**：将全文公式块整理后发送 `o3-2025-04-16` 做符号一致性检查

---

### 阶段 5：参考文献完善 🟡

**当前状态**：`refs.bib` 包含13条引用，覆盖核心方法

**待补充**（根据 Related Work 内容）：
- [ ] PPO (Schulman 2017) —— 已有
- [ ] RLOO (Ahmadian 2024) —— 已有
- [ ] ReAct / Reflexion / RAGEN —— 已有
- [ ] 其他 LLM Agent RL 工作（如 AgentQ, TWOSOME 等）—— 需检查是否引用
- [ ] Reward shaping 理论背景（Ng 1999）—— 已有
- [ ] 运行环境说明（ALFWorld, WebShop原始论文）—— 已有

**执行**：
```bash
# 检查所有 \cite 是否在 refs.bib 中有对应条目
grep -o '\\cite[tp]*{[^}]*}' sections/*.tex main.tex | \
  sed 's/.*{\(.*\)}/\1/' | tr ',' '\n' | sort | uniq | \
  while read key; do grep -q "$key" refs.bib || echo "MISSING: $key"; done
```

**格式规范**：bibtex warning（`dyna1991sutton` volume+number冲突）需修复：
```bibtex
# 删除 number 字段保留 volume
```

---

### 阶段 6：WebShop 基线结果填入 ⏳（等待实验完成）

**阻塞项**：M1(GRPO) + M3(GiGPO) WebShop baseline 正在运行

**完成后执行**：
1. 从 training.log 提取 Peak/Final/Late-Avg SR
2. 更新 `sections/experiments.tex` 中 `tab:webshop_results` 表格
3. 补充 WebShop 分析段落（§Generalization to WebShop 中的 baseline 对比）
4. 更新 Abstract 中的 WebShop 数值（如结论有变化）

**参考脚本**：
```bash
grep "val_sr\|success_rate" .../v11_final_webshop/M1_*/training.log | \
  awk '{print $1, $NF}' | sort -k1 -n
```

---

### 阶段 7：整体审稿与质量闭环 🟡

#### 7.1 自动化审稿（AI Reviewer）
**工具**：写作Prompt中"论文整体以Reviewer视角审视"模板
**模型**：`gemini-2.5-pro-thinking`（批判性思维最强）
**输入**：编译好的 main.pdf
**产出**：Critical Weaknesses 列表 + 改稿建议

✋ **人工介入**：
- 判断每条 weakness 的严重程度（致命 / 中等 / 无需处理）
- 决定是否补充实验、是否重写某段论证

#### 7.2 Overstatement 检查
**模型**：`gpt-4.1`，用"语气检查"Prompt
**重点扫描**：Abstract、Introduction、Conclusion 中的声称句

#### 7.3 最终编译检查
```bash
cd /root/testttt/RLVMR/paper_draft
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "Warning|Error|^!"
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# 检查最终页数和文件大小
ls -lh main.pdf
```

**NeurIPS 2026 格式要求**（预计）：
- 正文 ≤ 9 页（不含参考文献）
- 附录无页数限制
- 双栏格式（neurips_2025.sty 已处理）

---

### 阶段 8：提交准备 🟢

**检查清单**：
- [ ] 页数符合要求（当前18页需拆分正文/附录）
- [ ] 所有图为矢量格式（PDF）
- [ ] 盲审版本（移除作者信息，main.tex 用 `\usepackage[final]{neurips_2025}` 或 `\usepackage{neurips_2025}`）
- [ ] arXiv 预印本版本（用 `[preprint]` 选项）
- [ ] 补充材料（appendix）单独打包 or 合并
- [ ] 代码链接（GitHub + README）
- [ ] 论文标题最终确认

---

## 执行时间线（建议）

```
3月16日 ~ 3月31日（2周）：
  ├─ 阶段2：图表审计 + 重绘（重点fig9主架构图）
  ├─ 阶段3：Abstract + Introduction + Method 精细润色
  └─ 阶段4：公式符号规范化

4月1日 ~ 4月15日（2周）：
  ├─ 阶段5：参考文献完善
  ├─ 阶段6：填入 WebShop baseline 结果（等实验）
  └─ 阶段3续：Related Work + Conclusion 润色

4月16日 ~ 5月10日（约3.5周）：
  ├─ 阶段7：AI Reviewer 审稿 + 修改迭代（2~3轮）
  └─ 阶段8：格式检查 + 提交准备

5月15日：Abstract Due（NeurIPS 2026 预估）
5月22日：Full Paper Due
```

---

## 快速参考：关键文件路径

```
论文文件：
  /root/testttt/RLVMR/paper_draft/main.tex          ← 主文件
  /root/testttt/RLVMR/paper_draft/refs.bib           ← 参考文献
  /root/testttt/RLVMR/paper_draft/sections/          ← 各章节
  /root/testttt/RLVMR/paper_draft/main.pdf           ← 编译输出

实验数据：
  /root/testttt/RLVMR/PAPER_INDEX.md                 ← 数据总索引
  /root/testttt/RLVMR/PAPER_GAP_ANALYSIS.md          ← 实验缺口

图表：
  /root/testttt/RLVMR/code/rebel_test_results/v11_final/paper_figures/

写作辅助：
  /root/testttt/RLVMR/paper_draft/写作常用Prompt.md  ← Prompt库
  /tmp/dmx_refine.py                                  ← dmxapi调用脚本

编译命令：
  cd /root/testttt/RLVMR/paper_draft
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 各阶段调用模型速查

```python
# 学术润色 / 去AI味 / 逻辑检查
model = "gpt-4.1"

# 深度逻辑推理 / 全文审稿 / 方法章节
model = "gemini-2.5-pro-thinking"

# 公式推导验证
model = "o3-2025-04-16"

# 主图架构图生成
model = "gemini-2.5-pro"  # 图像版本

# 快速迭代 / 表格标题 / 短文本
model = "gemini-2.5-flash"
```
