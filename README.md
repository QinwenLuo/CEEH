# CEEH

The code of our **KDD2026** paper [Compress the Easy, Explore the Hard: Difficulty-Aware Entropy Regularization for Efficient LLM Reasoning](https://arxiv.org/abs/2602.22642)。

The trained models of [7B-ME](https://huggingface.co/Qwenzzzz/CEEH_7B_ME) and [1.5B-EA](https://huggingface.co/Qwenzzzz/CEEH_1.5B_EA) are accessible.

## Requirements

The project is based on [verl](https://github.com/verl-project/verl).

Install `verl==0.5.0`.

Use the dependency versions required by `verl==0.5.0` as the baseline. This project keeps its
Python dependencies in `pyproject.toml` and `uv.lock`, but compatibility should be checked against
verl 0.5.0 first.

```bash
uv sync
```

Training also requires a CUDA environment that can run Ray, vLLM, FSDP/FSDP2, and flash-attn.

## Files

```text
main_edlao.py                 # training entrypoint
config/edlao_trainer.yaml     # default Hydra config
train.sh                      # minimal launch script
trainer/edlao_trainer.py      # active custom trainer
actor/edlao_dp_actor.py       # entropy-advantage actor logic
workers/edlao_fsdp_workers.py # custom FSDP workers
utils/deepmath_reward.py      # math reward function
```

`main_edlao.py` imports `trainer/edlao_trainer.py` of verl. The root-level `edlao_trainer.py`, if present,
is not used by the entrypoint.

## Run

Set the model and parquet paths, then run:

```bash
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/test.parquet \
bash train.sh
```

Common environment overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=4
PROJECT=CEEH
EXP=entropy-advantage
CHECKPOINT_DIR=./checkpoints/CEEH/entropy-advantage
LOG_DIR=./logs
```

You can also call Hydra directly:

```bash
uv run python main_edlao.py \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1
```

## Entropy Modes

Maximum-entropy Loss (ME) and [Entropy-based Advantage](https://arxiv.org/abs/2506.14758) (EA) are mutually exclusive. Choose one mode
for each run.

Use EA:

```bash
actor_rollout_ref.actor.use_entropy_advantage=True
actor_rollout_ref.actor.entropy_advantage_alpha=0.4
actor_rollout_ref.actor.entropy_advantage_kappa=2.0
actor_rollout_ref.actor.entropy_coeff=0.0
actor_rollout_ref.actor.entropy_coeff_annealing=constant
```

EA converts token entropy into an additional advantage term. It should not be combined with ME.

Use ME:

```bash
actor_rollout_ref.actor.use_entropy_advantage=False
actor_rollout_ref.actor.entropy_coeff=0.001
actor_rollout_ref.actor.entropy_coeff_annealing=cosine
```


## Other Common Switches

LoRA:

```bash
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.target_modules=all-linear
```

Disable LoRA:

```bash
actor_rollout_ref.model.lora_rank=0
```

Length reward:

```bash
length_rewards.use_length_reward=True
length_rewards.reward_scale=0.1
```
