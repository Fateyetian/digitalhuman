# ReBel Paper — Auto Review Log

## Round 1 (2026-03-16)

### Assessment Summary
- **Score: 6/10**
- **Verdict: Almost** (not yet "ready" for top-tier submission)
- Key criticisms:
  - WebShop baselines (GRPO, GiGPO) missing — largest hole
  - Most ALFWorld results are single-seed (seed=42); gains of +0.8pp over GiGPO+Belief are hard to interpret without variance
  - HiBO's own contribution is modest (+0.8pp peak); framing overstates it as the central driver
  - Reward shaping claims invoke Ng et al. 1999 but are not actually policy-invariant — language was too strong
  - Grouping gains vs reward shaping gains not cleanly disentangled

### Reviewer Raw Response

<details>
<summary>Click to expand full GPT-5.4 reviewer response</summary>

## 1. Score

**6/10**

Borderline accept on ideas, but **not yet fully convincing as a top-tier empirical paper** in current form.

Reason for 6:
- The core idea is genuinely interesting and plausibly useful.
- The paper identifies a real pathology in step-level credit assignment and proposes a targeted fix.
- ALFWorld results are strong.

But:
- **Key empirical support is incomplete**, especially **missing WebShop baselines**.
- **Most ALFWorld results are single-seed**, which is risky for RL.
- The gains over the strongest relevant baseline on ALFWorld appear **modest** for the full method (**95.3 vs 94.5**, +0.8pp over GiGPO+Belief), so the paper needs stronger evidence that each component matters robustly.

If baselines and multi-seed robustness were added, this could move to 7–8.

## 2. Strengths

1. **Clear identification of a real bottleneck in step-level credit assignment** — quantifying 63–85% singleton rate is a strong motivation.
2. **Method is well-structured and conceptually coherent** — HiBO, belief reward, and structured prompting fit together naturally.
3. **Coverage improvement is substantial** — ~20% → ~76% is meaningful, not a minor tweak.
4. **ALFWorld performance is strong in absolute terms** — 95.3% peak SR, broader ablation than most agent-training papers.
5. **Engages with reward shaping concerns** — adaptive decay is more careful than simply adding dense rewards.

## 3. Weaknesses (ranked by severity)

1. **Missing WebShop baselines** — the second benchmark has no comparative numbers.
2. **Single-seed ALFWorld** — +0.8pp over GiGPO+Belief is below statistical noise threshold without multi-seed.
3. **HiBO's narrative vs empirical contribution** — paper emphasizes HiBO but ablation shows belief reward/adaptive decay contribute more (+3.1pp and +2.3pp vs +0.8pp).
4. **Hand-designed belief abstraction** limits generality claims.
5. **Reward shaping claims are overstated** — adaptive decay is a heuristic, not a formal guarantee.
6. **SFT cold-start dependence** reduces appeal as a plug-in improvement.
7. **Narrow evaluation** — only 2 environments, 1.5B model only.

## 4. Missing Experiments

### A. WebShop baselines [CRITICAL]
- What: Full WebShop training for GRPO, GiGPO, GiGPO+Belief under identical setup
- Why: Without these, the "cross-environment generalization" claim is not comparative
- Cost: Medium–High (several GPU-days per method)

### B. Multi-seed ALFWorld [CRITICAL]
- What: 3–5 seeds for GRPO, GiGPO, GiGPO+Belief, ReBel
- Why: +0.8pp gain from HiBO is within plausible noise without variance estimates
- Cost: Medium (3–5× current ALFWorld compute)

### C. Multi-seed targeted ablations
- What: 3 seeds for V11 ablations (full, w/o HiBO, w/o belief reward, w/o adaptive decay)
- Why: Single-seed differences are over-interpreted
- Cost: Medium

### D. Coverage → learning signal correlation [LOW COST if logs exist]
- What: Plot fraction of non-singleton steps, step-advantage variance/magnitude, policy improvement over training; compare exact-hash only vs HiBO
- Why: Prove coverage increase translates to better optimization, not just descriptive stats

### E. Factorial ablation: grouping vs reward shaping
- What: (1) exact grouping, no belief reward; (2) HiBO only, no belief reward; (3) exact + belief reward; (4) HiBO + belief reward; (5) belief reward only, no step-level advantage
- Why: Disentangle whether gains come from rescued credit assignment or dense reward engineering
- Cost: Medium

### F. Belief abstraction sensitivity
- What: Compare full 4D belief, coarser versions, noisy labels, wrong fields
- Why: Show robustness and quantify task-specific engineering cost

### G. No-JSON prompting ablation [LOW COST]
- What: Lighter natural-language belief vs structured JSON belief
- Why: Separate scaffold benefit from algorithmic benefit of HiBO/reward

### H. Output-length and token-cost analysis [LOW COST]
- What: Training/inference token counts, wall-clock time, SR-per-token efficiency
- Why: Practical overhead assessment of ~30% longer outputs

### I. Larger model test
- What: At least one 3B or 7B backbone
- Why: Confirm method is not 1.5B-specific

## 5. Writing Issues

1. Narrative overstates HiBO; ablation shows belief reward/decay contribute more.
2. WebShop section should be labeled "preliminary" not "generalization results."
3. Reward shaping claims use formal-sounding language without formal guarantees.
4. No confidence intervals/error bars in any tables or figures.
5. Failure mode analysis of invalid actions/belief formatting would help.

## 6. Verdict

**Almost Ready**

Most urgently needs: (1) WebShop baselines, (2) multi-seed ALFWorld, (3) cleaner HiBO evidence, (4) disentangled grouping vs reward shaping gains.

</details>

### Actions Taken (Round 1 — writing fixes only, no experiments)
1. **Rebalanced Contributions section** — added per-component ablation numbers (+0.8pp HiBO, +3.1pp belief reward, +2.3pp adaptive decay); split into 4 bullet points, clarified WebShop is preliminary.
2. **Softened reward-shaping claims** — removed "provably converging to unbiased optimal" in Conclusion; replaced with empirical framing.
3. **Softened Method section** — differential decay described as "principled heuristic rather than formal guarantee."
4. **WebShop table caption** — retitled as "preliminary, three seeds" and removed speculative language about baselines.
5. **WebShop section text** — added explicit "Preliminary assessment" paragraph acknowledging absence of comparative baseline numbers.
6. **Limitations expanded** — added single-seed caveat, SFT non-drop-in note, and acknowledged the +0.8pp HiBO gain needs multi-seed validation.
7. **HiBO analysis** — added "Coverage and learning signal" paragraph explaining the +0.8pp peak vs Late-Avg framing, linking coverage to variance/stability rather than peak alone.

### Required Experiments (for author to run)
See Section 4 above. **Priority order: A → B → C → E → D → G → H → F → I**

### Status
- Score: 6/10 — threshold met but verdict is "Almost", not "ready"
- Continuing to Round 2 after writing fixes applied

## Round 2 (2026-03-16)

### Assessment Summary
- **Score: 7/10**
- **Verdict: Almost**
- Remaining weaknesses:
  1. HiBO evidence still weak (+0.8pp single-seed) — needs reframing as stability device
  2. WebShop still lacks baseline comparison — ok as preliminary, but no superiority claims
  3. No factorial disentanglement of grouping vs reward shaping — needs explicit caveat
  4. SFT/structured prompting as non-drop-in requirement — plug-and-play language to remove
  5. Scope too broad — constrain to "small-scale embodied and web navigation settings"

### Reviewer Raw Response

<details>
<summary>Click to expand full GPT-5.4 Round 2 response</summary>

**Score: 7/10**

The writing changes substantially improved honesty and defensibility.

Remaining weaknesses ranked:
1. HiBO evidence still weak — reframe as modest stabilization, not core proven driver
2. WebShop still weak — must avoid superiority claims, frame as feasibility only
3. No factorial disentanglement — add explicit caveat where ablations discussed
4. SFT requirement — remove drop-in/plug-and-play language
5. Narrow scope — constrain generalization claims to tested settings

Minimum fixes per weakness:
- HiBO: reposition in abstract/introduction/conclusion
- WebShop: "preliminary feasibility" only, no comparative claims
- Factorial: add explicit caveat in ablation section
- SFT: remove general plug-and-play language
- Scope: "evidence in small-scale embodied/web settings" not broad LLM RL

**Verdict: Almost**
Consistently modest positioning would make this defensible enough to submit.

</details>

### Actions Taken (Round 2 — writing fixes)
1. **Abstract rewritten** — belief reward curriculum now listed as main contributor (largest gains), HiBO listed last with explicit "+0.8pp peak" quantification; WebShop framed as "preliminary feasibility result"
2. **Introduction "Our approach" reordered** — structured prompting first (foundation), curriculum second (largest gains, +3.1pp+2.3pp), HiBO third (modest +0.8pp, primary benefit = stability)
3. **Ablation section** — added "Interaction effects and causal attribution" paragraph with explicit factorial design caveat
4. **Conclusion rewritten** — scoped to "small-scale embodied and web navigation settings"; removed "powerful lever for multi-step agent learning" generalization; HiBO framed as stability improvement
5. **Related Work** — softened "first to leverage" to "first to our knowledge... in text-based agent benchmarks"

### Required Experiments (unchanged from Round 1)
Priority: A (WebShop baselines) → B (multi-seed ALFWorld) → C (multi-seed ablations) → E (factorial) → D/G/H/F/I

### Status
- Score: 7/10 — improved from Round 1
- Verdict still "Almost" — writing is now honest and calibrated
- Proceeding to Round 3

---

## Round 3 (2026-03-16)

### Assessment Summary
- **Score: 8/10**
- **Verdict: Almost** — "writing near-maxed, bottleneck is experiments"
- Remaining issues: "necessary precondition" language; HiBO framing consistency; causal language audit

### Actions Taken (Round 3 — small precision fixes)
1. **"necessary precondition" → "practical foundation"** everywhere (Method, Intro, Contributions)
2. **HiBO framing consistent** — "+0.8pp peak, primary benefit = late-training stability" in all sections
3. **WebShop "preliminary"** confirmed in all locations
4. **Causal language audit** — no strong causal terms remain
5. **Contributions bullet 3** — "serves as practical foundation" not "necessary precondition"

### Status
- Score: 8/10
- Proceeding to Round 4 (final)

---

## Round 4 — FINAL (2026-03-16)

### Assessment Summary
- **Score: 8.5/10**
- **Verdict: Almost**
- Writing quality: 9.0–9.5/10
- Experimental completeness: 7.5–8.0/10

### Reviewer Raw Response

<details>
<summary>Click to expand full GPT-5.4 Round 4 final response</summary>

Score: 8.5/10

Writing/positioning: 9.0–9.5/10. Experimental completeness: 7.5–8.0/10.

Remaining issues that could cause rejection:
1. Single-seed evidence (BIGGEST risk) — +0.8pp HiBO and other gains could be noise
2. No factorial ablation — component attribution partial
3. WebShop too preliminary — won't rescue the paper
4. SFT prerequisite limits generality
5. Hand-designed abstraction — domain-specific engineering concern
6. HiBO modest gain — reviewers may conclude curriculum is the real contribution

Delta to "Ready":
- MINIMUM: (1) Multi-seed ALFWorld (3-5 seeds, mean±std), (2) compact near-factorial ablation
- Nice-to-have: WebShop baseline comparisons

Final verdict: Almost. With multi-seed ALFWorld + compact factorial → would upgrade to READY.

</details>

### Actions Taken (Round 4)
Writing is at ceiling. No further modifications made.

---

## ═══════════════════════════════════════════
## FINAL SUMMARY — 4 ROUNDS COMPLETE
## ═══════════════════════════════════════════

### Score Trajectory
| Round | Score | Verdict | Main action |
|-------|-------|---------|-------------|
| 1 | 6/10 | Almost | Rebalanced contributions, softened reward shaping, fixed WebShop framing |
| 2 | 7/10 | Almost | Reordered components by evidence strength, factorial caveat, scoped conclusions |
| 3 | 8/10 | Almost | Removed "necessary precondition", causal language audit |
| 4 | **8.5/10** | **Almost** | Writing at ceiling — no further changes needed |

### Writing Changes Made (complete list)
1. Abstract rewritten — curriculum first (+3.1pp+2.3pp), HiBO last (+0.8pp); WebShop = preliminary feasibility
2. Introduction component order reversed — prompting (foundation) → curriculum (main gains) → HiBO (stability)
3. Contributions: per-component numbers; WebShop explicitly preliminary; 4 bullets not 2
4. Method: "practical foundation" not "necessary precondition"; reward shaping framed as heuristic
5. Ablation: "Interaction effects and causal attribution" paragraph with explicit factorial caveat
6. HiBO analysis: "Coverage and learning signal" paragraph linking ~76% coverage to stability, not just peak
7. WebShop table caption: "preliminary, three seeds — no comparative baselines"
8. WebShop section text: explicit "Preliminary assessment" paragraph
9. Limitations: single-seed caveat, SFT non-drop-in, +0.8pp HiBO needs multi-seed, 1.5B only
10. Conclusion: scoped to tested settings; component gains stated with numbers
11. Related Work: softened "first to leverage" → "first to our knowledge in text-based agent benchmarks"

---

## 🧪 EXPERIMENTS NEEDED TO REACH "READY"

### 🔴 CRITICAL — Blocking acceptance

**Exp-A: Multi-seed ALFWorld main comparison**
- What to run: GRPO, GiGPO (think), GiGPO+Belief, ReBel — each with ≥3 seeds (recommend: 42, 123, 456)
- What to report: mean ± std for Peak SR, Final SR, Late-Avg SR in main Table 1
- Why critical: Single-seed is the #1 rejection risk. +0.8pp HiBO gain and even larger gains are unverifiable without variance. Reviewers will discount single-seed RL results.
- Estimated cost: ~3× current ALFWorld compute per method

**Exp-B: Compact component ablation (near-factorial)**
- What to run (option 1 — sequential build-up, 3 seeds each):
  1. GiGPO + Belief (current C1 baseline)
  2. + adaptive belief reward curriculum (no HiBO)
  3. + HiBO = full ReBel
- What to run (option 2 — 2×2 factorial, 3 seeds each):
  - {exact-hash grouping, HiBO} × {no belief reward, belief reward+adaptive decay}
- Why critical: Current sequential ablation confounds HiBO and curriculum (both depend on belief quality). Reviewers want clean component attribution.
- Estimated cost: 4–6 additional ALFWorld runs × 3 seeds

### 🟡 IMPORTANT — Strongly recommended

**Exp-C: Multi-seed targeted ablations (V11)**
- What to run: V11 configs (full, w/o HiBO, w/o belief reward, w/o adaptive decay) × 3 seeds
- Why: Current single-seed gaps (92.2 / 94.5 / 95.3%) need variance bounds to be interpretable
- Estimated cost: ~3× current V11 compute

**Exp-D: WebShop baselines**
- What to run: GRPO, GiGPO (think), GiGPO+Belief on WebShop, 3 seeds each
- Why: "cross-environment transfer" currently has no comparative data; WebShop section is preliminary-only
- Estimated cost: medium-high (several GPU-days per method × 3 seeds)

### 🟢 USEFUL — If compute allows

**Exp-E: Coverage → learning signal analysis (likely zero extra compute)**
- What to run: Re-analyze existing training logs to plot:
  (a) fraction of non-singleton/covered steps per epoch
  (b) step-advantage variance and magnitude per epoch
  (c) policy improvement (val SR delta) per epoch
  Compare exact-hash only vs HiBO
- Why: Proves the 76% coverage increase translates to better optimization, not just a descriptive statistic

**Exp-F: No-JSON prompting ablation**
- What to run: Replace structured JSON belief with natural-language belief summary; otherwise identical setup
- Why: Separates scaffold benefit from the algorithmic mechanisms (HiBO, reward)

**Exp-G: Output-length / token-cost report**
- What to measure: Training tokens/step, inference tokens/step, wall-clock time per epoch
- Why: ~30% longer outputs need quantified practical overhead assessment
