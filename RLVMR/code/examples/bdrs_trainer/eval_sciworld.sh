set -x
ENGINE=${1:-vllm}

python3 -m verl.trainer.main_generation \
  model.path=COLD_START_MODEL_PATH \
  rollout.name=$ENGINE \
  data.path=$HOME/data/verl-agent/text/test.parquet \
  data.prompt_key=prompt \
  data.batch_size=16 \
  rollout.prompt_length=6000 \
  rollout.response_length=512 \
  rollout.temperature=0 \
  trainer.project_name='BDRS' \
  trainer.experiment_name='bdrs_eval_sciworld' $@


