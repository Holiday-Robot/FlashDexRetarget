# FlashDexRetarget: training options and evaluation

Setup is in the [README](../../README.md#setup); the data pipeline (release -> `motion.pt`) is in
[`kinematic_retargeting.md`](kinematic_retargeting.md).

## Training

`scripts/train_flashdexretarget.sh` sources `scripts/env.sh` and runs `train.py`. It takes the motion
file and the robot as environment variables; every other option is a Hydra override passed as an argument.

```sh
# XHAND, on a dataset prepared with scripts/retarget/<release>.sh
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh

# Sharpa Wave (clips retargeted to assets/robot/sharpa/bimanual.xml)
ROBOT=sharpa MOTION=<output_dataset_path>/sharpa/motion.pt bash scripts/train_flashdexretarget.sh

# any config key, e.g. a run name, fewer envs, a smaller buffer, no wandb upload
MOTION=<output_dataset_path>/xhand/motion.pt bash scripts/train_flashdexretarget.sh \
    logger.wandb.run_name=small num_envs=1024 runner.agent.buffer_max_length=20000000 logger.wandb.mode=offline
```

## Options

### Launcher

| variable | default | meaning |
|---|---|---|
| `MOTION` | – (required) | motion `.pt` to train on ([`kinematic_retargeting.md`](kinematic_retargeting.md)) |
| `ROBOT` | `xhand` | `xhand` or `sharpa`: robot config and robot USD |

The run is named `flashdexretarget-<dataset>-<robot>` and tagged `<dataset>`, `<robot>` in wandb, where
`<dataset>` is the directory of `MOTION` (`<dataset>/motion.pt` or `<dataset>/<robot>/motion.pt`);
`logger.wandb.run_name=<name>` replaces it.

### Hydra overrides

The defaults below come from `config/` (`envs/`, `runner/`, `eval/`, `logger/`); arguments to the launcher
are applied last.

**Run**

| key | default | meaning |
|---|---|---|
| `num_envs` | `2048` | parallel environments |
| `seed` | `42` | random seed |
| `device` | `cuda:0` | simulation and learner device |
| `episode_length_s` | `20.0` | episode time limit; an episode also ends with its clip |

**Learner** (`config/runner/flashsac.yaml`)

| key | default | meaning |
|---|---|---|
| `gamma` / `n_step` | `0.98` / `5` | discount and n-step return |
| `updates_per_interaction_step` | `2` | gradient updates per env step |
| `per_hand_actor_critic` | `true` | one actor and one critic per hand, each on its own hand's reward; `false` is a single actor-critic on the summed reward |
| `runner.agent.buffer_max_length` | `50000000` | replay buffer rows (1.76 GB per million rows) |
| `runner.agent.buffer_device_type` | `cuda` | `cpu` keeps the replay buffer in pinned host RAM ([GPU memory](#gpu-memory)) |
| `runner.agent.sample_batch_size` | `4096` | batch size per update |
| `runner.agent.critic_hidden_dim` / `critic_num_blocks` | `1024` / `2` | critic width / depth |
| `runner.agent.actor_hidden_dim` / `actor_num_blocks` | `128` / `2` | actor width / depth |
| `runner.agent.learning_rate_init` / `_peak` / `_end` | `3e-4` / `3e-4` / `1.5e-4` | learning-rate schedule |
| `runner.agent.traj_encoder.enable` | `true` | Conv1D encoder over the next 10 reference frames |
| `runner.agent.use_compile` / `use_amp` | `true` / `true` | `torch.compile` / mixed precision |

**Run length and checkpoints** (`config/runner/base.yaml`)

| key | default | meaning |
|---|---|---|
| `num_env_steps` | `500000000` | env-step budget |
| `save_checkpoint_per_env_step` | `20000000` | checkpoint stride in env steps |
| `save_replay_buffer` | `false` | also store the replay buffer in each `step<N>/` checkpoint; set `true` to resume from one |
| `runner.keep_last_checkpoint_only` | `true` | keep only the newest `step<N>/` directory |

**Evaluation** (`config/eval/motion_tracking.yaml`)

| key | default | meaning |
|---|---|---|
| `eval.interval` | `20000000` | env steps between evaluations |
| `eval.success_criterion` | `obj` | `obj` (object errors only) or `full` (adds the finger errors) for the success archive |
| `eval.stop_when_all_solved` | `true` | end the run once every clip has passed |
| `eval.save_success_data` | `true` | archive the first passing rollout of each clip |

**Environment**

| key | default | meaning |
|---|---|---|
| `rewards.reward_weights.<term>` | see `config/envs/rewards/flashdexretarget.yaml` | per-term reward weight; `0.0` turns a term off |
| `commands.sampling.mode` | `uniform` | episode start: `uniform` over clip frames, or `start` (first frame) |

**Viewer, logging and resume**

| key | default | meaning |
|---|---|---|
| `headless` | `true` | `false` opens the Isaac Sim viewport |
| `logger.wandb.project_name` / `group_name` / `run_name` | `FlashDexRetarget` / `flashdexretarget` / see above | wandb naming; the run name is also the log directory name |
| `logger.wandb.mode` | `online` | `offline` or `disabled` to run without uploading |
| `logger.wandb.entity` | `null` | wandb entity (`null`: `$WANDB_ENTITY` or your default) |
| `agent_load_path` / `buffer_load_path` | `null` | checkpoint directory to load networks / replay buffer from |
| `step_offset` | `0` | env step the run starts from (set on resume) |
| `logger.wandb.resume` / `resumeid` | `false` / `null` | continue an existing wandb run |

`python train.py --overrides commands.motion_file=<pt> --overrides <key>=<value> ...` runs the same
config without the launcher (XHAND; `--gui` opens the viewport).

## GPU memory

The replay buffer dominates: 50M rows take 88 GB.

| | GPU memory | host RAM |
|---|---|---|
| default (`runner.agent.buffer_device_type=cuda`) | ~110 GB | ~16 GB |
| `runner.agent.buffer_device_type=cpu` | ~16 GB | ~100 GB (88 GB of it the pinned buffer) |


On `cpu` the buffer stays in pinned host RAM and the GPU reads and writes it directly (no CPU-side
gather), so a GPU with less memory can run the same config. The buffer scales with
`runner.agent.buffer_max_length`, e.g. `runner.agent.buffer_max_length=20000000` for 35 GB.

## Stop / resume

`kill -USR1 <pid>` finishes the iteration, saves `stop<N>/` (networks, replay buffer,
`resume_state.json`) and exits. A periodic `step<N>/` holds the replay buffer only with
`save_replay_buffer=true`, so pass it when the run may need to resume from one.

Every checkpoint's `resume_state.json` records the env step, the wandb id and the curriculum state a resume
needs; resume by passing `agent_load_path=<ckpt> buffer_load_path=<ckpt> step_offset=<N>
logger.wandb.resume=true logger.wandb.resumeid=<id> ...` to the launcher. Eval and checkpoint points lie on
an absolute env-step grid, so a resume does not shift them.

## Other algorithms

The runner config picks the algorithm: `runner=flashsac` (the default). A new algorithm is
`src/agents/<name>.py` implementing `agents.Agent` (`src/agents/base.py`) plus `config/runner/<name>.yaml`
with `defaults: [base, _self_]`, `algorithm: <name>` (checkpoints go to `logs/<name>/`) and
`runner.agent.class_name`.

## Evaluation

Every `eval.interval` env steps (20M) the policy is rolled out deterministically from the first frame of
every clip, on the training envs. A clip is scored by its **time-mean** tracking error (worst hand):

| error | threshold |
|---|---|
| object position | 3 cm |
| object orientation | 30 deg |
| fingertips vs MANO tips | 6 cm |
| finger links vs the retarget | 8 cm |

`success_rate_obj_<k>` requires the two object errors within `k` x threshold, `success_rate_<k>` all four.
Each clip that passes SR@1.0 (`eval.success_criterion`, default `obj`) is archived once under
`<run>/success/` (`manifest.json` + one rollout `.npz` per motion), and `eval.stop_when_all_solved` ends
the run once every clip is archived.

## Outputs

```
logs/flashsac/<run_name>-<MMDD-HHMMSS>/
├── step<N>/         networks (+ replay_buffer.pt) + resume_state.json, every `save_checkpoint_per_env_step` env steps
├── stop<N>/         SIGUSR1 stop
├── final/           networks at the end of the run
└── success/         first passing rollout per motion
logs/rsl_rl/<YYYY-MM-DD_HH-MM-SS>_<run_name>/wandb/   wandb run directory
```

## Pipeline

```text
motion .pt (mocap retargeted to the robot) ──► MotionTrackingCommand (reference, sampling, object assist)
                                                       │
Isaac Sim (PhysX, N envs) ◄── scene: bimanual robot USD + object USDs (+ support disks)
        │
ManagerBasedRlEnv (vendored mjlab managers)
   obs · residual PD action (+ LPF) · rewards · terminations · curricula
        │  FlatObsVecEnv: flat obs, one reward column per hand
        ▼
OffPolicyRunner ──► agents.Agent (runner.agent.class_name) — FlashSAC: side actors, per-hand
        │           critic heads, compact fp16 replay buffer, Conv1D future-trajectory encoder
        │ every eval.interval (20M) env steps
        ▼
evaluation/evaluate.py: deterministic rollout over every clip ──► SR@k, ever-success,
                        success archive (first passing clip per motion)
```
