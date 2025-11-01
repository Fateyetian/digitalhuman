# Server Deployment Guide - Cold Start Data Generation

## Prerequisites

### 1. Pull Latest Code

```bash
cd /path/to/RLVMR
git pull origin improvement_1
```

### 2. Activate Conda Environment

```bash
conda activate rlvmr-alfworld
```

### 3. Install Required Dependencies

```bash
# Install OpenAI library
pip install openai

# Install datasets library (if not already installed)
pip install datasets
```

### 4. Verify Expert Trajectory Data

```bash
cd code
python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/alfworld_expert_traj')
print(f'✓ Found {len(dataset)} expert trajectories')
"
```

## Running the Script

### Basic Usage

```bash
cd code

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Run the script
python scripts/alfworld_prepare.py
```

### Advanced Usage with Custom Parameters

```bash
# Use GPT-4o-mini (cheaper, ~10x less cost)
python scripts/alfworld_prepare.py \
    --model gpt-4o-mini \
    --num_trajs 300 \
    --output_path data/alfworld_cold-start.json

# Or specify API key directly in command
python scripts/alfworld_prepare.py \
    --api_key sk-your-key-here \
    --model gpt-4o \
    --num_trajs 300

# Resume from interrupted run
python scripts/alfworld_prepare.py \
    --resume \
    --resume_path data/alfworld_cold-start_progress.json
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--api_key` | env var or script default | OpenAI API key |
| `--model` | `gpt-4o` | Model to use (`gpt-4o` or `gpt-4o-mini`) |
| `--num_trajs` | `300` | Number of trajectories to annotate |
| `--output_path` | `data/alfworld_cold-start.json` | Output file path |
| `--dataset_path` | `data/alfworld_expert_traj` | Expert trajectory dataset path |
| `--resume` | `False` | Resume from previous progress |
| `--resume_path` | `data/alfworld_cold-start_progress.json` | Progress file path |

## Running in Background

For long-running processes, use `nohup` or `screen`:

### Option 1: Using nohup

```bash
nohup python scripts/alfworld_prepare.py \
    --model gpt-4o-mini \
    --num_trajs 300 \
    --resume \
    > logs/cold_start_generation.log 2>&1 &

# Monitor progress
tail -f logs/cold_start_generation.log
```

### Option 2: Using screen

```bash
# Start screen session
screen -S cold_start

# Run script
python scripts/alfworld_prepare.py --resume

# Detach: Press Ctrl+A then D
# Reattach: screen -r cold_start
```

## Expected Runtime and Cost

### GPT-4o
- **Time**: 60-90 minutes for 300 trajectories
- **Cost**: ~$15-30
- **Quality**: Highest

### GPT-4o-mini
- **Time**: 45-60 minutes for 300 trajectories
- **Cost**: ~$1.5-3
- **Quality**: Good (recommended for testing)

## Output Files

After successful completion, you'll have:

1. **Main output**: `data/alfworld_cold-start.json`
   - Contains all annotated trajectories in SFT format
   - Ready for training after parquet conversion

2. **Progress file** (if `--resume` used): `data/alfworld_cold-start_progress.json`
   - Tracks completed trajectory indices
   - Contains intermediate results
   - Allows resuming if interrupted

## Monitoring Progress

The script outputs real-time progress:

```
[1/300] Processing trajectory (task: heat apple and put in fridge...) [OK] (15 steps)
[2/300] Processing trajectory (task: clean plate and put in cabinet...) [OK] (12 steps)
[3/300] Processing trajectory (task: cool tomato and put on counter...) [FAILED]
    Reason: Invalid response for task: ...
[4/300] Processing trajectory (task: put two apple in fridge...) [OK] (18 steps)
...
```

## Quality Report

After completion, the script automatically generates a quality report:

```
============================================================
Data Quality Report
============================================================

Total steps: 4523

Mode Distribution:
  EXECUTE : 2461 ( 54.4%)
  EXPLORE : 1131 ( 25.0%)
  PLAN    :  678 ( 15.0%)
  VERIFY  :  253 (  5.6%)

Trajectory Length Statistics:
  Average: 15.1 steps
  Min: 8 steps
  Max: 28 steps

Task Type Coverage: 6 types
  - look_at_obj
  - pick_and_place
  - pick_clean_then_place
  - pick_cool_then_place
  - pick_heat_then_place
  - pick_two_obj_and_place

Format Errors: 0 / 4523 (0.00%)

============================================================
Quality Assessment
============================================================

✅ Data quality looks good!
   - Mode distribution is balanced
   - No format errors detected
   - Ready for cold start training
```

## Next Steps

After data generation completes successfully:

### 1. Convert to Parquet Format

```bash
python -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/alfworld \
    --data_source=data/alfworld_cold-start.json
```

### 2. Train Cold Start Model

```bash
bash examples/sft/cold_start/run_alfworld_qwen2.5-7b.sh
# Or manually:
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/alfworld/train.parquet \
    data.val_files=$HOME/data/alfworld/val.parquet \
    # ... (see QUICK_START_COLD_START.md for full command)
```

### 3. Evaluate Cold Start Model

```bash
bash examples/bdrs_trainer/eval_alfworld.sh \
    actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/qwen7b_plan_a/default/epoch_5
```

## Troubleshooting

### API Key Issues

```
ERROR: Invalid OpenAI API Key
```

**Solution**: Check API key format (must start with `sk-`) and account balance

### Rate Limit Errors

```
Error: RateLimitError: You exceeded your current quota
```

**Solution**:
- Wait and retry (script has exponential backoff)
- Use `--resume` to continue from where it stopped
- Reduce concurrent requests by using sequential processing (already default)

### Out of Memory

```
Error: Cannot allocate memory
```

**Solution**: This shouldn't happen as we process one trajectory at a time. Check available memory.

### Dataset Not Found

```
ERROR: Cannot load dataset from data/alfworld_expert_traj
```

**Solution**:
- Verify dataset path: `ls data/alfworld_expert_traj`
- Use `--dataset_path` to specify correct path

## Important Files Modified

1. **code/agent_system/environments/prompts/alfworld.py**
   - Enhanced BDRS runtime prompts

2. **code/agent_system/environments/prompts/cold_start.py**
   - Enhanced BDRS cold start prompts with belief examples

3. **code/scripts/alfworld_prepare.py**
   - Main data generation script with resume capability

## Documentation

- **QUICK_START_COLD_START.md**: Quick start guide for Plan A
- **BDRS_COLD_START_PLAN_C.md**: Complete optimization guide (backup plan)
- **BDRS_EXPERIMENT_PLAN.md**: Overall BDRS experiment plan

## Support

If you encounter issues:
1. Check the quality report for data quality problems
2. Review `QUICK_START_COLD_START.md` troubleshooting section
3. Examine the progress file to see which trajectories failed
4. Use `--resume` to retry failed trajectories
