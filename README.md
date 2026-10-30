<h1 align="center">BlenDAgger: Blended Shared Control for Interactive Imitation Learning</h1>

<p align="center">
  Cailyn Smith, Geoffrey Sun, Henny Admoni<sup>†</sup>, Zackory Erickson<sup>†</sup><br>
  The Robotics Institute, School of Computer Science, Carnegie Mellon University<br>
  <sup>†</sup>Equal advising
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2609.37599">[Paper]</a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  [Website (coming soon)]
</h3>

---

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Simulation Experiments](#simulation-experiments)
- [Real Robot](#real-robot)
- [Citation](#citation)

---

## Overview

BlenDAgger collects corrections for fine-tuning a robot policy using blended shared control.
While the demonstrator intervenes, their action is blended with the policy's. The blend weight is
set per group of DoFs from three signals:

- **Action similarity** between the human and policy actions
- **Policy confidence**, from k-nearest-neighbor uncertainty on the policy's encoder embedding
- **Intervention magnitude**, how hard the demonstrator is pushing

This repo contains the arbitration, the RoboCasa pipeline (collect, fine-tune, evaluate), a
robot-agnostic real-robot collector, and the analysis code.

---

## Repository Structure

```
main.py        collect a round, then fine-tune
conf/          configs: simulation runs, analysis, xArm presets, training
arbitration/   BlenDAgger, HG-DAgger and the ablations
collection/    simulation and real-robot data collection, teleop, live blend-weight view
evaluation/    policy rollouts and success checks
training/      robomimic fine-tuning patches, and the robomimic patch setup_sim.sh applies
analysis/      collection metrics
scripts/       evaluation, dataset, scene and seed-retraining scripts
interfaces/    interfaces for running collection on another robot
fixed_envs/    saved RoboCasa scenes
```

---

## Installation

### 1) Clone the repo

```bash
git clone https://github.com/cailyn-s/BlenDAgger.git
cd BlenDAgger
```

### 2) Create the environment

```bash
bash setup_sim.sh          # creates the `blendagger` conda env
conda activate blendagger
```

This installs the pinned packages in `requirements-sim.txt`, then robosuite, RoboCasa and robomimic
at the commits used for the paper (in `third_party/`), and downloads the RoboCasa kitchen assets.

> Requires Linux and an NVIDIA GPU whose driver supports CUDA 12.8. The asset download is about 5 GB.
> Collecting new data also needs a 3Dconnexion SpaceMouse and a display.

### 3) Download the base policies and build the base demonstrations

Download the base policies from **<DATA_URL>** into `checkpoints/`:

```
checkpoints/
  coffee_base.pth
  thawing_base.pth
  pick_place_base.pth
```

Then build the base demonstrations from RoboCasa's own datasets (downloading those if needed) into
`data/`:

```bash
bash scripts/data/build_base_datasets.sh                 # all three tasks
bash scripts/data/build_base_datasets.sh PrepareCoffee   # or only some
```

### 4) (Optional) Keep data somewhere else

Skip this if you used the layout above. Otherwise, either set environment variables:

```bash
export SCD_DATASET_DIR=/path/to/data           # base demonstrations
export SCD_CHECKPOINT_DIR=/path/to/checkpoints # base policies
export SCD_SAVE_DIR=/path/to/storage           # where data/ and results/ are read and written
```

or copy [`conf/paths_local.example.yaml`](conf/paths_local.example.yaml) to `conf/local/paths.yaml`
(gitignored) and edit it; that file can also point at individual checkpoints. Defaults live in
[`conf/paths_db.yaml`](conf/paths_db.yaml).

---

## Simulation Experiments

| Task | RoboCasa env | BlenDAgger config | HG-DAgger config |
|---|---|---|---|
| Prepare Coffee | `PrepareCoffee` | `sim/coffee/blendagger` | `sim/coffee/hg_dagger` |
| Microwave Thawing | `MicrowaveThawing` | `sim/thawing/blendagger` | `sim/thawing/hg_dagger` |
| Pick and Place | `PnPCounterToCab` | `sim/pick_place/blendagger` | `sim/pick_place/hg_dagger` |

Run every command from the repo root. Each run writes its data to `data/<run_name>_<algo>_<env>/`
and its checkpoints to `results/<run_name>_<algo>_<env>/`. The configs use the run names from the
paper (`coffee_realmatch`, `thawing_realmatch`, `lemon_realmatch`, ...), which the evaluation, seed
and analysis scripts expect.

### 1) Collect and fine-tune

Each call collects one round of 10 episodes, then fine-tunes the base policy on the base
demonstrations plus every round so far. Run it five times for five rounds; it resumes from disk.

```bash
python main.py --config-name sim/coffee/blendagger
```

**SpaceMouse controls**
- Move the puck to intervene
- `Space` hands control back to the policy
- Left / right button closes / opens the gripper
- Hold both buttons for one second to reset the scene

### 2) Evaluate

100 headless rollouts per checkpoint. `EVAL_ROUND` picks the round.

```bash
# Prepare Coffee and Microwave Thawing -> eval_composite_results/
EVAL_ROUND=3 python scripts/eval/eval_composite.py coffee_realmatch_combined_blend_PrepareCoffee

# The base policy
python scripts/eval/eval_composite.py base:PrepareCoffee

# Pick and Place -> eval_realmatch_results/
EVAL_ROUND=3 python scripts/eval/eval_realmatch.py lemon_realmatch_combined_blend_PnPCounterToCab
```

### 3) Ablations (Prepare Coffee)

Collect and evaluate as above with the configs in `conf/sim/coffee/ablations/`: `no_uncertainty`,
`no_similarity`, `no_magnitude`, `only_uncertainty`, `only_similarity`, `only_magnitude` and
`fixed_alpha` (α = 0.5), e.g. `--config-name sim/coffee/ablations/no_uncertainty`.

Human action targets trains on the data collected with `sim/coffee/blendagger`, without collecting:

```bash
python main.py --config-name sim/coffee/ablations/human_targets
```

HG-DAgger and BlenDAgger are averaged over training seeds 123, 456 and 789. To retrain a round
with another seed:

```bash
scripts/train/train_seed_round.sh blend 3 2      # <blend|hg_dagger> <round> <seed 2|3>
```

The responsiveness column:

```bash
python analysis/ablation_responsiveness.py
```

### 4) Analysis metrics

Action discontinuity, OOD states and trajectory distance:

```bash
python analysis/run_analysis.py --config-name analysis/sim/coffee
python analysis/run_analysis.py --config-name analysis/sim/thawing
python analysis/run_analysis.py --config-name analysis/sim/pick_place
```

Results go to `analysis/results/<task>/`, including the per-episode export
`export/episodes.csv` used for the paper's statistics.

---

## Real Robot

### Hardware

- UFACTORY xArm6
- One external and one wrist camera (Orbbec)
- Meta Quest controller (expert demonstrators) or SpaceMouse (study participants)

The xArm driver, cameras, teleop, data processing and Diffusion Policy training live in the
companion repository **<XARM_REPO_URL>**, which includes this repo as a submodule. The commands
below run from its root.

### 1) Collect base demonstrations

```bash
python scripts/collect.py --teleop oculus --task-name cupboard_pretrain \
    --output-dir data/cupboard --save --save-images
```

### 2) Process for training

Relative end-effector actions, stationary frames trimmed, resized images (all defaults):

```bash
python scripts/data_processing.py --input <run>_full.hdf5 --output proc_<run>_full.hdf5
```

### 3) Train the policy

Base and fine-tuned policies use the Diffusion Policy fork in the companion repository, with the
`finetune_xarm_opno` workspace for fine-tuning and the hyperparameters in Table I of the paper.

### 4) Build the uncertainty index

Rebuild it for every new checkpoint:

```bash
python BlenDAgger/arbitration/build_knn_index.py --robot-type xarm \
    --ckpt-dir <checkpoint> --dataset data/cupboard \
    --output BlenDAgger/arbitration/knn_index_cupboard.pkl \
    --calib-out BlenDAgger/arbitration/uncertainty_knn_cupboard.json
```

### 5) Collect a round

The expert settings for each method are in [`conf/real/`](conf/real/). The presets expect the base
policies at `checkpoints/cupboard_base.ckpt` and `checkpoints/almond_base.ckpt`; any flag on the
command line, such as `--ckpt-dir`, overrides the preset:

```bash
python scripts/collect.py --config BlenDAgger/conf/real/cupboard_blend.yaml \
    --ckpt-dir <checkpoint> --task-name cupboard_blended_round1
```

> The commands used in the user study are in `scripts/user_study/<task>/` of the companion repository.

### 6) Evaluate

```bash
python scripts/collect.py --policy diffusion --ckpt-dir <checkpoint> --teleop none --num-episodes 20
```

### 7) Analysis metrics

```bash
python analysis/run_analysis_suite.py --config cupboard   # also almond, cupboard_user_study, almond_user_study
```

Rounds without a processed `proc_*.hdf5` are processed with `scripts/data_processing.py` from the
companion repository; set `SCD_PROC_SCRIPT` if it is not at `../scripts/data_processing.py`.

### Using another robot

[`collection/real_collector.py`](collection/real_collector.py) runs collection against the interfaces in
[`interfaces/`](interfaces/). Implement:

- `AbstractRobotEnv` — 7-D end-effector delta actions (xyz, roll/pitch/yaw, gripper)
- `AbstractPolicy`
- `AbstractTeleopDevice`
- `AbstractDataWriter`

Then pass an arbitrator from [`arbitration/`](arbitration/), such as `CombinedBlend`.
`scripts/collect.py` in the xArm repository is a complete example.

---

## Citation

```bibtex
@article{smith2026blendagger,
  title   = {BlenDAgger: Blended Shared Control for Interactive Imitation Learning},
  author  = {Smith, Cailyn and Sun, Geoffrey and Admoni, Henny and Erickson, Zackory},
  journal = {arXiv preprint arXiv:2609.37599},
  year    = {2026}
}
```
