from datasets import load_dataset, load_from_disk
import logging
import numpy as np
import time
import json
from agent_system.environments.prompts import *
import re
import random
import openai
import argparse
import os
from pathlib import Path


# ============================================================
# Configuration Section - Modify these values
# ============================================================
DEFAULT_API_KEY = "YOUR_OPENAI_API_KEY"  # Or set via command line: --api_key
DEFAULT_MODEL = "gpt-4o"  # Options: "gpt-4o", "gpt-4o-mini"
DEFAULT_NUM_TRAJS = 300
DEFAULT_SAVE_PATH = "data/alfworld_cold-start.json"
DEFAULT_DATASET_PATH = "data/alfworld_expert_traj"
DEFAULT_RESUME_PATH = "data/alfworld_cold-start_progress.json"  # For resume


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BDRS cold start data for ALFWorld")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (default: read from env OPENAI_API_KEY or use DEFAULT_API_KEY)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--num_trajs", type=int, default=DEFAULT_NUM_TRAJS,
                        help=f"Number of trajectories to annotate (default: {DEFAULT_NUM_TRAJS})")
    parser.add_argument("--output_path", type=str, default=DEFAULT_SAVE_PATH,
                        help=f"Output JSON file path (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Expert trajectory dataset path (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress")
    parser.add_argument("--resume_path", type=str, default=DEFAULT_RESUME_PATH,
                        help=f"Progress file for resume (default: {DEFAULT_RESUME_PATH})")
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Set API key priority: command line > environment variable > default
if args.api_key:
    openai.api_key = args.api_key
elif os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    openai.api_key = DEFAULT_API_KEY

# Validate API key
if openai.api_key == "YOUR_OPENAI_API_KEY" or not openai.api_key.startswith("sk-"):
    print("\n" + "="*60)
    print("ERROR: Invalid OpenAI API Key")
    print("="*60)
    print("Please set your API key using one of these methods:")
    print("1. Command line: --api_key sk-your-key")
    print("2. Environment variable: export OPENAI_API_KEY=sk-your-key")
    print("3. Edit script: DEFAULT_API_KEY = 'sk-your-key'")
    print("="*60 + "\n")
    exit(1)

MODEL = args.model
NUM_TRAJS = args.num_trajs
SAVE_PATH = args.output_path
DATASET_PATH = args.dataset_path
RESUME_PATH = args.resume_path

# Create output directory if not exists
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Load dataset
try:
    alfworld_dataset = load_from_disk(DATASET_PATH)
except Exception as e:
    print(f"\nERROR: Cannot load dataset from {DATASET_PATH}")
    print(f"Error: {e}\n")
    exit(1)

trajs = []
for traj in alfworld_dataset:
    conversations = traj["conversations"][2:]
    traj_log = []
    task = conversations[0]['value'].split("Your task is to:")[-1].split("AVAILABLE ACTIONS:")[0].strip()
    for i in range(0, len(conversations), 2):
        obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]
        llm_action = conversations[i + 1]['value']
        # Extracting the action from the llm_action: Action: xxx
        action = llm_action.split("Action:")[-1].strip()
        traj_log.append({
            "observation": obs,
            "action": action
        })
    if len(traj_log) < 30:
        trajs.append({
            "task": task,
            "traj": traj_log
        })

random.shuffle(trajs)

def llm(prompt, model, temperature=0.0, max_tokens=1024, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def llm_json(prompt, model, temperature=0.0, max_tokens=1024, retries=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    
    return []


def merge(res, traj):
    if (len(res) != len(traj)):
        print(f"Length mismatch: {len(res)} vs {len(traj)}")
        return False, None

    if "action" not in res[0] or "reason" not in res[0]:
        print("Missing 'action' or 'reason' in the response structure.")
        return False, None
    
    output = []
    for a, b in zip(res, traj):
        if a["action"] != b["action"]:
            print(f"Action mismatch: {a['action']} vs {b['action']}")
            return False, None
        if not a["reason"].startswith("<"):
            print(f"Invalid reason format: {a['reason']}")
            return False, None
        output.append({
            "obs": b["observation"],
            "reason": a["reason"],
            "action": a["action"],
        })
    return True, output

meta_traj = []
sft_data = []

# Load progress if resume
completed_indices = set()
if args.resume and os.path.exists(RESUME_PATH):
    try:
        with open(RESUME_PATH, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            sft_data = progress.get('sft_data', [])
            completed_indices = set(progress.get('completed_indices', []))
        print(f"\n[RESUME] Loaded progress: {len(completed_indices)} trajectories already completed")
        print(f"[RESUME] Starting from trajectory {len(completed_indices)+1}/{NUM_TRAJS}\n")
    except Exception as e:
        print(f"\n[WARNING] Cannot load progress file: {e}")
        print("[WARNING] Starting from scratch\n")

print(f"\n{'='*60}")
print(f"Starting data generation for {NUM_TRAJS} trajectories")
print(f"Model: {MODEL}")
print(f"Output: {SAVE_PATH}")
if args.resume:
    print(f"Resume: Enabled (progress file: {RESUME_PATH})")
print(f"{'='*60}\n")

# Function to save progress
def save_progress(idx, sft_data, completed_indices):
    try:
        with open(RESUME_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'completed_indices': list(completed_indices),
                'sft_data': sft_data,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n[WARNING] Cannot save progress: {e}")

for idx, traj in enumerate(trajs[:NUM_TRAJS]):
    # Skip if already completed
    if idx in completed_indices:
        print(f"[{idx+1}/{NUM_TRAJS}] Skipping (already completed)")
        continue

    print(f"[{idx+1}/{NUM_TRAJS}] Processing trajectory (task: {traj['task'][:50]}...)", end="", flush=True)

    # 选择使用 BDRS 的标注模板（如需切换回 RLVMR，可改为 ALFWORLD_TAGGING_TEMPLATE）
    base_prompt = ALFWORLD_TAGGING_TEMPLATE_BDRS.format(traj=json.dumps(traj["traj"], ensure_ascii=False))

    # 方案A优化：添加防止hindsight bias的提示
    enhanced_prompt = f"""{base_prompt}

IMPORTANT INSTRUCTIONS:
1. **Avoid Hindsight Bias**: Annotate AS IF you don't know the future steps. For each step, only consider what the agent would know at that moment based on:
   - Current and previous observations
   - Actions taken so far
   - Current belief state (M_t, P_t, E_t)

2. **Include Brief Belief Context**: In each reasoning, briefly mention the relevant belief state. Examples:
   - "<EXPLORE>M_t lacks information about microwave location. E_t shows living room unexplored. Need to search there.</EXPLORE>"
   - "<EXECUTE>M_t confirms apple 1 is on countertop 1. P_t current subgoal is to obtain apple. Take it now.</EXECUTE>"
   - "<VERIFY>M_t assumed fridge was open, but observation suggests it's closed. Need to verify.</VERIFY>"

3. **Mode Selection Guidelines**:
   - Use <PLAN> when: Starting task, or replanning after failures
   - Use <EXECUTE> when: M_t has all needed information to complete current subgoal
   - Use <EXPLORE> when: M_t lacks key facts OR E_t shows unexplored areas
   - Use <VERIFY> when: Observations contradict M_t beliefs

Now annotate the trajectory:"""

    res = llm_json(enhanced_prompt, MODEL)
    valid, res = merge(res, traj["traj"])

    if not valid:
        print(f" [FAILED]")
        print(f"    Reason: Invalid response for task: {traj['task']}")
        print(f"    Response: {res}")
        continue

    if valid:
        print(f" [OK] ({len(res)} steps)")
        meta_traj.append({
            "task": traj["task"],
            "traj": res
        })

        step_level_data = []
        latest_planning = "No plan."
        for i, item in enumerate(res):
            if i == 0:
                # 使用 BDRS 无历史冷启动模板
                prompt = ALFWORLD_TEMPLATE_NO_HIS_BDRS_CS.format(
                    current_observation=item["obs"],
                )
            else:
                action_history = "\n".join([f"[Observation {j + 1}: '{res[j]['obs']}', Action {j + 1}: '{res[j]['action']}']" for j in range(i)])
                history_think_length = min(3, i)
                action_history += "\n- recent reasoning process: \n"
                for j in range(i - history_think_length, i):
                    action_history += f"[Observation {j + 1}: {res[j]['obs']}, output: '{res[j]['reason']} <action>{res[j]['action']}</action>']\n"
                # 使用 BDRS 冷启动模板（含历史）
                prompt = ALFWORLD_TEMPLATE_BDRS_CS.format(
                    task_description=traj["task"],
                    step_count=i,
                    history_length=i,
                    action_history=action_history,
                    current_step=i + 1,
                    current_observation=item["obs"],
                    planning=latest_planning
                )

            # update current planning
            if '<planning>' in item["reason"]:
                current_planning = re.search(r'<planning>(.*?)</planning>', item["reason"], re.DOTALL)
                if current_planning:
                    latest_planning = current_planning.group(1).strip()
                else:
                    pass
            response = f"{item['reason']}\n<action>{item['action']}</action>\n"

            step_level_data.append({
                "step": i + 1,
                "obs": item["obs"],
                "prompt": prompt,
                "response": response,
            })

        sft_data.append({
            "task": traj["task"],
            "done": "True",
            "data": step_level_data
        })

        # Mark as completed and save progress
        completed_indices.add(idx)
        if args.resume:
            save_progress(idx, sft_data, completed_indices)

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=4)

print(f"\n{'='*60}")
print("Data generation completed!")
print(f"{'='*60}")
print(f"Data saved to: {SAVE_PATH}")
print(f"Total trajectories: {len(sft_data)}")

# 数据质量检查报告
print(f"\n{'='*60}")
print("Data Quality Report")
print(f"{'='*60}")

# 1. 统计总步数和模式分布
total_steps = 0
mode_counts = {"PLAN": 0, "EXECUTE": 0, "EXPLORE": 0, "VERIFY": 0, "UNKNOWN": 0}

for item in sft_data:
    for step in item["data"]:
        total_steps += 1
        response = step["response"]
        if "<PLAN>" in response:
            mode_counts["PLAN"] += 1
        elif "<EXECUTE>" in response:
            mode_counts["EXECUTE"] += 1
        elif "<EXPLORE>" in response:
            mode_counts["EXPLORE"] += 1
        elif "<VERIFY>" in response:
            mode_counts["VERIFY"] += 1
        else:
            mode_counts["UNKNOWN"] += 1

print(f"\nTotal steps: {total_steps}")
print("\nMode Distribution:")
for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
    if count > 0:
        percentage = (count / total_steps) * 100
        print(f"  {mode:8s}: {count:4d} ({percentage:5.1f}%)")

# 2. 轨迹长度统计
traj_lengths = [len(item["data"]) for item in sft_data]
avg_length = sum(traj_lengths) / len(traj_lengths)
min_length = min(traj_lengths)
max_length = max(traj_lengths)

print(f"\nTrajectory Length Statistics:")
print(f"  Average: {avg_length:.1f} steps")
print(f"  Min: {min_length} steps")
print(f"  Max: {max_length} steps")

# 3. 任务类型覆盖
task_types = set()
for item in sft_data:
    task = item["task"]
    # 提取任务类型（简单分类）
    if "heat" in task.lower() and "place" in task.lower():
        task_types.add("pick_heat_then_place")
    elif "cool" in task.lower() and "place" in task.lower():
        task_types.add("pick_cool_then_place")
    elif "clean" in task.lower() and "place" in task.lower():
        task_types.add("pick_clean_then_place")
    elif "pick" in task.lower() and "place" in task.lower():
        task_types.add("pick_and_place")
    elif "two" in task.lower():
        task_types.add("pick_two_obj_and_place")
    elif "examine" in task.lower():
        task_types.add("look_at_obj")

print(f"\nTask Type Coverage: {len(task_types)} types")
for task_type in sorted(task_types):
    print(f"  - {task_type}")

# 4. 格式检查
format_errors = 0
for item in sft_data:
    for step in item["data"]:
        response = step["response"]
        # 检查是否包含必要的标签
        if not ("<" in response and ">" in response and "<action>" in response):
            format_errors += 1

error_rate = (format_errors / total_steps) * 100 if total_steps > 0 else 0
print(f"\nFormat Errors: {format_errors} / {total_steps} ({error_rate:.2f}%)")

# 5. 质量建议
print(f"\n{'='*60}")
print("Quality Assessment")
print(f"{'='*60}")

# 检查模式分布是否合理
execute_ratio = mode_counts["EXECUTE"] / total_steps if total_steps > 0 else 0
explore_ratio = mode_counts["EXPLORE"] / total_steps if total_steps > 0 else 0
verify_ratio = mode_counts["VERIFY"] / total_steps if total_steps > 0 else 0

issues = []
if execute_ratio > 0.65:
    issues.append(f"⚠️  EXECUTE ratio too high ({execute_ratio*100:.1f}% > 65%)")
if explore_ratio < 0.15:
    issues.append(f"⚠️  EXPLORE ratio too low ({explore_ratio*100:.1f}% < 15%)")
if verify_ratio < 0.05:
    issues.append(f"⚠️  VERIFY ratio too low ({verify_ratio*100:.1f}% < 5%)")
if format_errors > 0:
    issues.append(f"⚠️  Format errors detected: {format_errors}")

if issues:
    print("\n⚠️  Issues detected:")
    for issue in issues:
        print(f"  {issue}")
    print("\n💡 Suggestions:")
    if execute_ratio > 0.65:
        print("  - Consider implementing balanced sampling (Plan B/C)")
    if explore_ratio < 0.15 or verify_ratio < 0.05:
        print("  - Consider adding failure-recovery samples (Plan C)")
    if format_errors > 0:
        print("  - Check LLM responses and adjust temperature/prompts")
else:
    print("\n✅ Data quality looks good!")
    print("   - Mode distribution is balanced")
    print("   - No format errors detected")
    print("   - Ready for cold start training")

print(f"\n{'='*60}")
print("Next Steps")
print(f"{'='*60}")
print("1. Convert to parquet format:")
print(f"   python -m examples.data_preprocess.cold_start_data \\")
print(f"       --local_dir=$HOME/data/alfworld \\")
print(f"       --data_source={SAVE_PATH}")
print("\n2. Train cold start model:")
print("   See BDRS_EXPERIMENT_PLAN.md Section 2.4")
print("\n3. Evaluate cold start model:")
print("   bash examples/bdrs_trainer/eval_alfworld.sh")
print(f"{'='*60}\n")