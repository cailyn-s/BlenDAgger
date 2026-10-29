import hydra
from omegaconf import DictConfig, OmegaConf
import os
import glob
from math import ceil
import json
import argparse
import time
import sys
import subprocess
import random
import re
from datetime import datetime

# Suffix of the throwaway per-round training file built by merge_robomimic_datasets.
# Anything without this suffix is real data and is never deleted.
MERGED_DATASET_SUFFIX = "_merged_training_data.hdf5"

def run_isolated(target_func, *args):
    """
    Runs a function in a fully isolated subprocess by serializing arguments to JSON.
    """
    func_name = target_func.__name__
    
    # Serialize arguments to JSON so they can be safely passed to the subprocess
    args_json = json.dumps(args)
    
    # Dynamically get the module name from this file
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # Minimal bridge code to import and call function
    bridge_code = f"""
import sys, json
from {module_name} import {func_name}
args = json.loads(sys.argv[1])
{func_name}(*args)
"""
    
    # Ensure same objects are chosen per seed
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    # Run the function in a fresh, isolated Python shell
    result = subprocess.run([sys.executable, "-c", bridge_code, args_json])
    
    if result.returncode != 0:
        raise RuntimeError(f"Phase {func_name} failed with exit code {result.returncode}")

def phase_1_collect_data(pretrained_dataset, model_weights, env_name, train_all_steps, input_device, DATASET_PATH, INTERVENTION_DATASET_PATH, ROLLOUT_DATASET_PATH, algo, button_for_intervention, use_sequence, iteration, fixed_env, num_episodes, horizon, uncertainty_path, uncertainty_method="knn", knn_index_path=None, start_episode=0):
    import torch
    import numpy as np
    from robomimic.utils.file_utils import policy_from_checkpoint, get_env_metadata_from_dataset
    from robomimic.utils.env_utils import create_env_from_metadata
    from collection.data_writer import DataWriter
    from arbitration.HGDAgger import HGDagger
    from arbitration.ablations.LinearBlend import LinearBlend
    from arbitration.CombinedBlend import CombinedBlend
    from arbitration.ablations.CombinedBlend_NoMagnitude import CombinedBlend_NoMagnitude
    from arbitration.ablations.CombinedBlend_NoUncertainty import CombinedBlend_NoUncertainty
    from arbitration.ablations.CombinedBlend_NoSimilarity import CombinedBlend_NoSimilarity
    from arbitration.ablations.CombinedBlend_OnlySimilarity import CombinedBlend_OnlySimilarity
    from arbitration.ablations.CombinedBlend_OnlyUncertainty import CombinedBlend_OnlyUncertainty
    from arbitration.ablations.CombinedBlend_OnlyMagnitude import CombinedBlend_OnlyMagnitude
    from arbitration.utils import RobomimicKNNUncertaintyEstimator
    from collection.collector import Collector

    def _make_uncertainty_estimator():
        print(f"[UQ] Using k-NN uncertainty (index: {knn_index_path})")
        return RobomimicKNNUncertaintyEstimator(index_path=knn_index_path)  # cached embedding -> ~1ms, runs sync

    seed = 1000*(iteration + 1)
    metadata = get_env_metadata_from_dataset(pretrained_dataset)
    random.seed(seed)
    metadata["env_kwargs"]["layout_ids"] = [-1] # sample from all layouts
    metadata["env_kwargs"]["seed"] = seed
    metadata["env_kwargs"]["style_ids"] = [-1] # sample from all styles
    metadata["env_kwargs"]["control_freq"] = 20

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", torch_device)
    pretrained_policy, ckpt_dict = policy_from_checkpoint(
        ckpt_path=model_weights,
        device=torch_device
    )
    env = create_env_from_metadata(metadata, render=False, use_image_obs=True)
    print("Loaded pretrained model")

    human, teleop_env = make_teleop(env, input_device)

    raw_env_args = {}
    raw_env_args['env_kwargs'] = metadata['env_kwargs']
    raw_env_args["env_name"] = env_name
    raw_env_args["type"] = 1

    writer = DataWriter(
        path=DATASET_PATH,
        intervention_path=INTERVENTION_DATASET_PATH,
        rollout_path=ROLLOUT_DATASET_PATH,
        env_args=raw_env_args,
        resume=start_episode > 0
    )
    print("Started data writer")

    # Per-group SpaceMouse engagement threshold, same value as the real-robot SpaceMouse setup.
    SPACEMOUSE_HUMAN_ACTIVE_THRESHOLD = 0.15
    _blend_common = dict(train_all_steps=train_all_steps, alpha_smoothing=0.5,
                         onset_smoothing=1.0, human_active_threshold=SPACEMOUSE_HUMAN_ACTIVE_THRESHOLD)

    if algo == "hg_dagger":
        arbitrator = HGDagger(train_all_steps=train_all_steps, button_for_intervention=button_for_intervention)
    elif algo == "linear_blend":
        arbitrator = LinearBlend(alpha=0.5, train_all_steps=train_all_steps)
    elif algo == "combined_blend":
        arbitrator = CombinedBlend(uncertainty_estimator=_make_uncertainty_estimator(), uncertainty_path=uncertainty_path, **_blend_common)
    elif algo == "combined_blend_no_magnitude":
        arbitrator = CombinedBlend_NoMagnitude(uncertainty_estimator=_make_uncertainty_estimator(), uncertainty_path=uncertainty_path, **_blend_common)
    elif algo == "combined_blend_no_uncertainty":
        arbitrator = CombinedBlend_NoUncertainty(**_blend_common)
    elif algo == "combined_blend_no_similarity":
        arbitrator = CombinedBlend_NoSimilarity(uncertainty_estimator=_make_uncertainty_estimator(), uncertainty_path=uncertainty_path, **_blend_common)
    elif algo == "combined_blend_only_similarity":
        arbitrator = CombinedBlend_OnlySimilarity(**_blend_common)
    elif algo == "combined_blend_only_uncertainty":
        arbitrator = CombinedBlend_OnlyUncertainty(uncertainty_estimator=_make_uncertainty_estimator(), uncertainty_path=uncertainty_path, **_blend_common)
    elif algo == "combined_blend_only_magnitude":
        arbitrator = CombinedBlend_OnlyMagnitude(**_blend_common)
    else:
        raise ValueError(f"unknown algo {algo!r}")

    config_dict = json.loads(ckpt_dict["config"])
    if config_dict["train"].get("frame_stack") is not None:
        seq_length = config_dict["train"]["frame_stack"]
    elif use_sequence:
        seq_length = config_dict["train"]["seq_length"]

    collector = Collector(
        env=env,
        policy=pretrained_policy,
        task=metadata["env_name"],
        torch_device=torch_device,
        human=human,
        writer=writer,
        arbitrator=arbitrator,
        keep_if_unsuccessful=True,
        seed=seed,
        render=True,
        use_sequence=use_sequence,
        seq_length=seq_length if use_sequence else None,
        uncertainty_path=uncertainty_path
    )
    # Episode k always gets seed 1000*(iteration+1) + k + 1 (Collector increments per rollout), so resumes match.
    collector.seed = seed + start_episode
    print("Created data collector")
    print("Arbitration algorithm:", algo)

    if fixed_env is not None:
        from repo_paths import load_fixed_env
        state_dict = load_fixed_env(fixed_env)
    else:
        state_dict = None

    # finally: a Ctrl-C must still close the hdf5 files, or they are left unreadable.
    try:
        for episode in range(start_episode, num_episodes):
            print(f"Starting episode {episode}")
            create_grp = True
            while True:
                result = collector.rollout(max_horizon=horizon, create_grp=create_grp, state_dict=state_dict)
                if result != -1:
                    break
                create_grp = False
    except KeyboardInterrupt:
        print(f"\n[collect] interrupted during episode {episode}; closing data files cleanly.")
    finally:
        writer.close()
    env.close()
    if teleop_env is not None:
        teleop_env.close()
        if collector.human is not None:
            try:
                print("Closing teleop device...")
                collector.human.device.close()
            except Exception as e:
                print("Error closing teleop device. Exception:", e)
                pass

def phase_2_train(
        aggregate_data, TRAIN_SOURCE_DIR, FULL_DATASET_PATH, config_path, RESULTS_ROOT,
        aggregate_pretrain, pretrained_dataset, init_ckpt, num_epochs, RUN_NAME,
        name, freeze_encoder, use_iwr, training_mode, iteration=None,
        intervention_only=False, save_every_n_epochs=None, from_ckpt=False
):
    import h5py
    import glob
    from training.trainer import train

    # Round number first: copied or downloaded files do not keep their collection-order mtimes.
    def _round_order(path):
        m = re.search(r"_round(\d+)_", os.path.basename(path))
        return (int(m.group(1)) if m else float("inf"), os.path.getmtime(path))
    all_datasets_paths = sorted(glob.glob(os.path.join(TRAIN_SOURCE_DIR, "*dataset_full.hdf5")), key=_round_order)
    if iteration is not None:
        if from_ckpt:
            # Finetune prev round's ckpt on this round's data only (+ pretrain).
            all_datasets_paths = all_datasets_paths[iteration - 1:iteration]
        else:
            all_datasets_paths = all_datasets_paths[:iteration]
    print("All datasets found for training:", all_datasets_paths)
    all_datasets_converted_paths = [path.replace(".hdf5", "_im128.hdf5") for path in all_datasets_paths]
    print("All converted datasets found for training:", all_datasets_converted_paths)
    FULL_DATASET_CONVERTED_PATH = FULL_DATASET_PATH.replace(".hdf5", "_im128.hdf5")

    for full_path, im128_path in zip(all_datasets_paths, all_datasets_converted_paths):
        trimmed_path = full_path.replace(".hdf5", "_trimmed.hdf5")
        # Ensure a trimmed file exists (collection makes it; this backstops train-only / pre-existing data).
        if os.path.exists(full_path) and not os.path.exists(trimmed_path):
            subprocess.run([sys.executable, "collection/processing/trim_zeros.py", "--dataset", full_path, "--output", trimmed_path], check=True)
        # Render the TRIMMED data to the im128 training input (keep the expected _im128 name).
        if os.path.exists(trimmed_path) and not os.path.exists(im128_path):
            subprocess.run([sys.executable, "collection/processing/dataset_states_to_obs.py", "--dataset", trimmed_path,
                            "--output_name", os.path.basename(im128_path), "--no_abs_actions"], check=True)

    print("Starting training with robomimic BC...")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config["train"]["output_dir"] = RESULTS_ROOT
    config["train"]["data"] = [] # Clear existing data list
    valid_datasets = []

    if aggregate_data:
        print("Aggregating datasets...")
        for ds_path in all_datasets_converted_paths:
            print("Dataset: ", ds_path)
            with h5py.File(ds_path, "r") as f:
                if "data" in f and len(f["data"].keys()) > 0:
                    valid_datasets.append(ds_path)
                else:
                    print(f"Skipping empty dataset: {ds_path}")
    else:
        valid_datasets.append(FULL_DATASET_CONVERTED_PATH)

    # Aggregate with pretrain demos (as additional rollout data)
    if aggregate_pretrain and pretrained_dataset and os.path.exists(pretrained_dataset):
        print(f"Aggregating full pretraining dataset: {pretrained_dataset}")
        valid_datasets.append(pretrained_dataset)

    if len(valid_datasets) == 0:
        print("⚠️ No trainable datasets found. Skipping training.")
        return

    batch_size = config["train"]["batch_size"]
    num_train_steps = 0
    for p in valid_datasets:
        with h5py.File(p, "r") as f:
            if "data" not in f:
                continue
            for demo in f["data"].values():
                if "actions" in demo:
                    num_train_steps += demo["actions"].shape[0]
    if num_train_steps < batch_size:
        print("Too few trainable steps — skipping training iteration.")
        return
    print(f"Total training steps (merged dataset): {num_train_steps}")

    merged_dataset_path = os.path.join(TRAIN_SOURCE_DIR, f"{RUN_NAME}{MERGED_DATASET_SUFFIX}")
    merge_robomimic_datasets(valid_datasets, merged_dataset_path)
    config["train"]["data"] = [{"path": merged_dataset_path}]

    config["experiment"]["epoch_every_n_steps"] = ceil(num_train_steps / batch_size)
    
    config["experiment"]["ckpt_path"] = init_ckpt
    config["train"]["num_epochs"] = num_epochs
    if save_every_n_epochs is not None:
        config["experiment"]["save"]["every_n_epochs"] = save_every_n_epochs
    else:
        config["experiment"]["save"]["every_n_epochs"] = num_epochs if num_epochs < 5 else 25
    config["experiment"]["rollout"]["rate"] = num_epochs
    config["experiment"]["logging"]["wandb_proj_name"] = name
    # Per-run config so concurrent runs don't clobber the shared base file.
    run_config_path = os.path.join(RESULTS_ROOT, f"{RUN_NAME}_train_config.json")
    with open(run_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Starting isolated training process...")
    try:
        train(
            run_config_path, RUN_NAME, freeze_encoder=freeze_encoder, use_iwr=use_iwr, training_mode=training_mode,
            intervention_only=intervention_only
        )
    finally:
        # Reclaim the merge even if training failed; a retry rebuilds it.
        _remove_merged_dataset(merged_dataset_path, valid_datasets)
    print(f"Training complete.")

def phase_3_rollout(pretrained_dataset, model_weights_path, horizon, fixed_env, video_path, use_sequence):
    import torch
    import imageio
    import numpy as np
    from robomimic.utils.file_utils import policy_from_checkpoint, get_env_metadata_from_dataset
    from robomimic.utils.env_utils import create_env_from_metadata
    from evaluation.video_rollout import run_rollout

    metadata = get_env_metadata_from_dataset(pretrained_dataset)
    metadata["env_kwargs"]["layout_ids"] = [0]
    metadata["env_kwargs"]["seed"] = 0
    metadata["env_kwargs"]["style_ids"] = [0]
    metadata["env_kwargs"]["control_freq"] = 20

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env_from_metadata(metadata, render=True, use_image_obs=True)
    new_policy, _ = policy_from_checkpoint(
        ckpt_path=model_weights_path,
        device=torch_device
    )
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video_writer = imageio.get_writer(
        video_path,
        fps=20,
        codec="libx264",
        quality=8,
    )
    
    if fixed_env is not None:
        from repo_paths import load_fixed_env
        state_dict = load_fixed_env(fixed_env)
    else:
        state_dict = None

    _, ckpt_dict = policy_from_checkpoint(
        ckpt_path=model_weights_path,
        device=torch_device
    )
    config_dict = json.loads(ckpt_dict["config"])
    if config_dict["train"].get("frame_stack") is not None:
        seq_length = config_dict["train"]["frame_stack"]
    elif use_sequence:
        seq_length = config_dict["train"]["seq_length"]
    results = run_rollout(
        policy=new_policy,
        env=env,
        task=metadata["env_name"],
        horizon=horizon,
        state_dict=state_dict,
        video_writer=video_writer,
        terminate_on_success=True,
        use_sequence=use_sequence,
        seq_length=seq_length,
        device=torch_device
    )
    print("Rollout results with new model:", results)
    print("Rollout video saved at:", video_path)
    video_writer.close()
    env.close()

# Calibration flags for the auto-built KNN index, used when a config sets no arbitration.calib_* keys.
KNN_CALIB_DEFAULTS = {"loeo": True, "subsample": 4, "low_pct": 5.0, "high_pct": 99.0, "n_calib": 2000, "k": 10}


def _ensure_round_observations(data_root, skip_round=None):
    """Render *_im128.hdf5 for any completed round that lacks it.

    skip_round excludes the round being collected right now: its *_dataset_full.hdf5 exists
    on disk while it is still being topped up (a resumed round), and rendering it here would
    both waste a long render and freeze a stale _im128 that later rounds would treat as that
    round's data. Only rounds 1..iteration-1 are prior data for this checkpoint.

    Collected *_dataset_full.hdf5 stores sim states, not observations, so the KNN index
    (which embeds images) can only read the rendered _im128 sibling, so missing renders are
    made here; otherwise the index would silently cover only the pretrain data.
    """
    rendered = []
    for full in sorted(glob.glob(os.path.join(data_root, "**", "*dataset_full.hdf5"), recursive=True),
                       key=os.path.getmtime):
        if skip_round is not None and f"_round{skip_round}_" in os.path.basename(full):
            continue
        im128 = full[:-len(".hdf5")] + "_im128.hdf5"
        if os.path.exists(im128):
            continue
        print(f"[UQ auto-calib] rendering observations for {os.path.basename(full)} (missing _im128)")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "collection", "processing", "dataset_states_to_obs.py"), "--dataset", full, "--no_abs_actions"],
                       check=True)
        if not os.path.exists(im128):
            raise RuntimeError(f"dataset_states_to_obs.py did not produce {im128}; cannot build the KNN index.")
        rendered.append(os.path.basename(im128))
    if rendered:
        print(f"[UQ auto-calib] rendered {len(rendered)} missing observation file(s)")


def _prior_round_observations(data_root, iteration):
    """Rendered rounds that count as prior data for this checkpoint: 1..iteration-1, never iteration."""
    files = [f for f in glob.glob(os.path.join(data_root, "**", "*dataset_full_im128.hdf5"), recursive=True)
             if f"_round{iteration}_" not in os.path.basename(f)]
    return sorted(files, key=os.path.getmtime)[:max(iteration - 1, 0)]


def _expected_datasets(data_root, pretrained_dataset, iteration):
    """The dataset list a fresh build would use, for deciding whether a cached round is stale."""
    prior = _prior_round_observations(data_root, iteration)
    return [d for d in ([pretrained_dataset] + prior) if d and os.path.exists(d)]


def _build_knn_calibration(ckpt_path, index_path, calib_path, pretrained_dataset, data_root, iteration,
                           obs_horizon=2, calib=None):
    """Build (or reuse) the per-checkpoint KNN index + calibration; returns (index_path, calib_path) to use.

    Files are named per round so each checkpoint's calibration is preserved (never overwrites a prior round).
    A round file is reused only when it was built from THIS checkpoint AND with THESE calibration flags.
    Flags come from the config's `arbitration.calib_*` block (see KNN_CALIB_DEFAULTS)."""
    c = {**KNN_CALIB_DEFAULTS, **(calib or {})}
    # Render before anything else: both the reuse decision and the build depend on which
    # rounds have observations, so a missing render must not change either silently.
    _ensure_round_observations(data_root, skip_round=iteration)
    r_index = index_path.replace(".pkl", f"_round{iteration}.pkl")
    r_calib = calib_path.replace(".json", f"_round{iteration}.json")
    ckpt_key = f"{os.path.abspath(ckpt_path)}@{os.path.getmtime(ckpt_path):.0f}" if os.path.exists(ckpt_path) else None
    # Reuse needs the same checkpoint AND the same flags: changing calibration must force a rebuild,
    # not silently hand back bounds built under the old settings.
    calib_key = json.dumps(c, sort_keys=True)
    if os.path.exists(r_index) and os.path.exists(r_calib):
        try:
            _bm = json.load(open(r_calib)).get("_build_meta", {})
            _prev_ds = sorted(os.path.basename(x) for x in (_bm.get("args", {}) or {}).get("dataset", []) or [])
            _now_ds  = sorted(os.path.basename(x) for x in _expected_datasets(data_root, pretrained_dataset, iteration))
            if (_bm.get("ckpt_key") == ckpt_key and _bm.get("calib_key") == calib_key
                    and _prev_ds == _now_ds):
                print(f"[UQ auto-calib] round {iteration}: reusing existing calibration ({os.path.basename(r_calib)})")
                return r_index, r_calib
        except Exception:
            pass
    # Prior data this collection checkpoint saw: pretrain + rounds 1..iteration-1 (im128, needed for obs embeddings).
    want_prior = max(iteration - 1, 0)
    prior = _prior_round_observations(data_root, iteration)
    if len(prior) < want_prior:
        raise RuntimeError(
            f"KNN calibration for round {iteration} needs {want_prior} prior round(s) of rendered "
            f"observations but found {len(prior)} under {data_root}. The checkpoint trained on data the "
            f"index would not cover, which silently inflates uncertainty. Check that each round's "
            f"*_dataset_full.hdf5 rendered to *_dataset_full_im128.hdf5.")
    datasets = [d for d in ([pretrained_dataset] + prior) if d and os.path.exists(d)]
    if not datasets:
        raise RuntimeError(
            f"No datasets available to build the KNN index (pretrained_dataset={pretrained_dataset!r}). "
            f"Refusing to collect with an uncalibrated confidence signal.")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arbitration", "build_knn_index.py")
    cmd = [sys.executable, script, "--robot-type", "robomimic", "--ckpt-path", ckpt_path,
           "--dataset", *datasets, "--output", r_index, "--calib-out", r_calib, "--obs-horizon", str(obs_horizon),
           "--subsample", str(c["subsample"]), "--calib-high-pct", str(c["high_pct"]),
           "--calib-low-pct", str(c["low_pct"]), "--n-calib", str(c["n_calib"]), "--k", str(c["k"])]
    if not c["loeo"]:
        cmd.append("--no-loeo")
    print(f"[UQ auto-calib] round {iteration}: building KNN index+calibration from {len(datasets)} dataset(s) "
          f"[{'LOEO' if c['loeo'] else 'no-loeo'} sub={c['subsample']} "
          f"pct={c['low_pct']}/{c['high_pct']} n_calib={c['n_calib']} k={c['k']}]")
    subprocess.run(cmd, check=True)
    try:  # stamp the checkpoint identity so this round's file can be safely reused next run
        j = json.load(open(r_calib)); j.setdefault("_build_meta", {})["ckpt_key"] = ckpt_key
        j["_build_meta"]["calib_key"] = calib_key
        json.dump(j, open(r_calib, "w"))
    except Exception:
        pass
    return r_index, r_calib

def _clean_aborted_collections(data_root):
    """Detect empty/truncated *_dataset_full.hdf5 (+ siblings) from an aborted prior collection and set
    them aside BEFORE collecting, so an abort is caught here instead of silently corrupting later training."""
    import h5py, shutil
    aborted_dir = data_root.rstrip("/") + "_aborted"
    for full in sorted(glob.glob(os.path.join(data_root, "*_dataset_full.hdf5"))):
        try:
            with h5py.File(full, "r") as h: valid = "data" in h and len(h["data"].keys()) > 0
        except Exception: valid = False
        if valid: continue
        os.makedirs(aborted_dir, exist_ok=True)
        for sib in glob.glob(full[:-len("_dataset_full.hdf5")] + "_*"):
            shutil.move(sib, os.path.join(aborted_dir, os.path.basename(sib)))
        print(f"\n[collect] *** ABORTED/EMPTY COLLECTION DETECTED: {os.path.basename(full)} has 0 demos.")
        print(f"[collect] *** Moved its files to {aborted_dir}/ so they can't corrupt training.\n")

def _round_files(data_root, run_name):
    base = os.path.join(data_root, run_name)
    return {"full": f"{base}_dataset_full.hdf5", "intervention": f"{base}_intervention.hdf5",
            "rollout": f"{base}_rollout.hdf5", "meta": f"{base}_meta.json"}


def _completed_episodes(meta_path):
    """Episodes that count toward the round: aborted _restartN entries do not."""
    if not os.path.exists(meta_path):
        return 0
    with open(meta_path) as f:
        meta = json.load(f)
    return sum(1 for v in meta.values() if not v.get("summary", {}).get("aborted", False))


def _find_resumable_round(data_root, num_episodes):
    """Highest-numbered round that stopped short of num_episodes -> (run_name, iteration, done)."""
    best = None
    for full in glob.glob(os.path.join(data_root, "*_round*_*_dataset_full.hdf5")):
        run_name = os.path.basename(full)[:-len("_dataset_full.hdf5")]
        m = re.search(r"_round(\d+)_", run_name)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (run_name, int(m.group(1)))
    if best is None:
        return None
    run_name, rnd = best
    done = _completed_episodes(os.path.join(data_root, f"{run_name}_meta.json"))
    return (run_name, rnd, done) if 0 < done < num_episodes else None


def _trim_orphan_segments(files):
    """Drop intervention/rollout segments left by a Ctrl-C mid-episode.

    end_episode never runs on an interrupt, so that episode has no meta entry and no
    dataset_full group -- but segments are flushed as they close, so any segment indexed
    above the last full demo belongs to the episode that never finished."""
    import h5py
    def _idx(names):
        return [int(n.split("_")[1]) for n in names if n.startswith("demo_")]
    with h5py.File(files["full"], "r") as h:
        last = max(_idx(h["data"].keys()), default=-1)
    removed = 0
    for kind in ("intervention", "rollout"):
        if not os.path.exists(files[kind]):
            continue
        with h5py.File(files[kind], "a") as h:
            for name in [n for n in h["data"].keys() if n.startswith("demo_") and int(n.split("_")[1]) > last]:
                del h["data"][name]
                removed += 1
    return removed


def main(iteration, algo, name, train_all_steps, num_epochs, num_episodes, env_name, config_path,
         model_weights, pretrained_model, pretrained_dataset, use_sequence,
         horizon, save_rollout, fixed_env,
         aggregate_data, aggregate_pretrain,
         button_for_intervention, input_device,
         freeze_encoder, use_iwr,
         DIR_NAME, DATA_ROOT, RESULTS_ROOT, uncertainty_path,
         collect_only=False, train_only=False, training_mode="default",
         train_data_override=None,
         uncertainty_method="knn", knn_index_path=None, knn_calib=None,
         from_ckpt=False, save_every_n_epochs=None):
    # ---------------------------
    # Configuration
    # ---------------------------
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # An interrupted round is topped up in place: same file set, same round number.
    _resume = None if train_only else _find_resumable_round(DATA_ROOT, num_episodes)
    if _resume is not None and _resume[1] == iteration:
        RUN_NAME, _, start_episode = _resume
        print(f"[collect] resuming round {iteration}: {start_episode}/{num_episodes} episodes on disk")
    else:
        RUN_NAME = f"{DIR_NAME}_round{iteration}_{current_time}"
        start_episode = 0
    DATASET_PATH = os.path.join(DATA_ROOT, f"{RUN_NAME}_dataset.hdf5")
    INTERVENTION_DATASET_PATH = os.path.join(DATA_ROOT, f"{RUN_NAME}_intervention.hdf5")
    ROLLOUT_DATASET_PATH = os.path.join(DATA_ROOT, f"{RUN_NAME}_rollout.hdf5")
    FULL_DATASET_PATH = os.path.join(DATA_ROOT, f"{RUN_NAME}_dataset_full.hdf5")
    TRAIN_SOURCE_DIR = train_data_override if train_data_override is not None else DATA_ROOT
    print("TRAIN SOURCE DIR: ", TRAIN_SOURCE_DIR)

    if train_only:
        latest_dataset = _find_latest_dataset(TRAIN_SOURCE_DIR, ["*dataset_full.hdf5"])
        if latest_dataset:
            FULL_DATASET_PATH = latest_dataset
        print("Train-only mode: using datasets:")
        print("  full dataset: ", latest_dataset)
    else:
        # Catch aborted/empty leftovers from a prior collection HERE (at collection time), not later at training.
        _clean_aborted_collections(DATA_ROOT)
        if start_episode > 0:
            _n = _trim_orphan_segments(_round_files(DATA_ROOT, RUN_NAME))
            if _n:
                print(f"[collect] dropped {_n} orphan segment(s) from the interrupted episode")
        # Auto-calibrate KNN per checkpoint before collecting (detects the checkpoint's prior training data).
        _knn_blend = algo in ("combined_blend", "combined_blend_no_magnitude",
            "combined_blend_no_similarity", "combined_blend_only_uncertainty")
        if uncertainty_method == "knn" and _knn_blend and knn_index_path:
            knn_index_path, uncertainty_path = _build_knn_calibration(
                model_weights, knn_index_path, uncertainty_path, pretrained_dataset, DATA_ROOT, iteration,
                calib=knn_calib)
        run_isolated(
            phase_1_collect_data,
            pretrained_dataset, model_weights, env_name, train_all_steps, input_device,
            DATASET_PATH, INTERVENTION_DATASET_PATH, ROLLOUT_DATASET_PATH, algo,
            button_for_intervention, use_sequence, iteration, fixed_env, num_episodes, horizon,
            uncertainty_path, uncertainty_method, knn_index_path, start_episode
        )
        # Auto trim-zeros at end of collection -> separate *_trimmed.hdf5 (dataset_full kept for analysis).
        if os.path.exists(FULL_DATASET_PATH):
            subprocess.run([sys.executable, "collection/processing/trim_zeros.py", "--dataset", FULL_DATASET_PATH,
                            "--output", FULL_DATASET_PATH.replace(".hdf5", "_trimmed.hdf5")], check=True)

    if not collect_only:
        # from_ckpt inits from the prev round's ckpt; round 1 model_weights == base.
        init_ckpt = model_weights if from_ckpt else pretrained_model
        intervention_only = not train_all_steps
        run_isolated(
            phase_2_train,
            aggregate_data, TRAIN_SOURCE_DIR, FULL_DATASET_PATH, config_path, RESULTS_ROOT,
            aggregate_pretrain, pretrained_dataset, init_ckpt, num_epochs, RUN_NAME, name,
            freeze_encoder, use_iwr, training_mode,
            iteration, intervention_only, save_every_n_epochs, from_ckpt
        )

        time.sleep(0.5)
        exp_root = os.path.join(RESULTS_ROOT, RUN_NAME)
        
        if not os.path.exists(exp_root):
            print(f"Experiment dir not found: {exp_root}, training might have skipped.")
            return model_weights

        run_dirs = [
            os.path.join(exp_root, d)
            for d in os.listdir(exp_root)
            if os.path.isdir(os.path.join(exp_root, d))
        ]
        if not run_dirs:
            raise RuntimeError("No run directories found")
        latest_run = max(run_dirs, key=os.path.getmtime)
        model_dir = os.path.join(latest_run, "models")
        if not os.path.exists(model_dir):
            raise RuntimeError("Models directory not found")
        ckpts = glob.glob(os.path.join(model_dir, "*.pth"))
        if not ckpts:
            raise RuntimeError("No checkpoint files found")
        model_weights_path = max(ckpts, key=os.path.getmtime)
        print("Final model weights at: ", model_weights_path)
        
        if save_rollout:
            video_path = f"{model_dir}/../rollout.mp4"
            run_isolated(
                phase_3_rollout,
                pretrained_dataset, model_weights_path, horizon, fixed_env, video_path, use_sequence
            )
        
        return model_weights_path

def make_teleop(env, input_device="spacemouse"):
    from robosuite.wrappers import VisualizationWrapper
    from collection.teleop.teleop import Teleop
    from collection.teleop.KeyboardHoldGripper import KeyboardHoldGripper
    from collection.teleop.CustomSpaceMouse import CustomSpaceMouse

    teleop_env = VisualizationWrapper(env.env)

    # Print information about the robot action space
    robot = teleop_env.robots[0]
    cc = robot.composite_controller
    for name, ctrl in cc.part_controllers.items():
        print("\nController:", name)
        print("  action_dim:", ctrl.control_dim)
    if input_device == "spacemouse":
        teleop_device = CustomSpaceMouse(
            env=teleop_env,
            pos_sensitivity=2.0,
            rot_sensitivity=2.0
        )
    else:
        teleop_device = KeyboardHoldGripper(
            env=teleop_env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0
        )
    teleop = Teleop(env=teleop_env, device=teleop_device)
    print("Started teleop device")
    return teleop, env

def _count_completed_rounds(results_root, num_epochs):
    # Consecutive rounds (from 1) with model_epoch_<num_epochs>.pth, keyed by the
    # _round<i>_ tag so retried/duplicate attempts of a round count once.
    if not os.path.isdir(results_root):
        return 0, None
    rounds = {}     # round index -> newest final ckpt for that round
    untagged = []   # run dirs predating the _round<i>_ tag
    for c in glob.glob(os.path.join(results_root, "*", "*", "models", f"model_epoch_{num_epochs}.pth")):
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c))))
        m = re.findall(r"_round(\d+)_", run_dir)
        if not m:
            untagged.append(c)
            continue
        idx = int(m[-1])
        if idx not in rounds or os.path.getmtime(c) > os.path.getmtime(rounds[idx]):
            rounds[idx] = c

    # Pre-tag runs are rounds 1..n in completion order; tagged rounds win on collision.
    if untagged:
        legacy = {i: c for i, c in enumerate(sorted(untagged, key=os.path.getmtime), start=1)}
        legacy.update(rounds)
        rounds = legacy

    completed = 0
    while (completed + 1) in rounds:
        completed += 1
    return completed, (rounds[completed] if completed else None)

def _find_latest_dataset(data_root, patterns):
    """
    Search for files matching patterns (ordered list). Return the newest match or None.
    """
    for pattern in patterns:
        paths = glob.glob(os.path.join(data_root, pattern))
        if paths:
            return max(paths, key=os.path.getmtime)
    return None

def _remove_merged_dataset(merged_path, source_paths):
    """
    Delete the per-round merged file. It only duplicates datasets that still
    exist on disk, so re-running the round rebuilds it.

    Deletes only a path that carries MERGED_DATASET_SUFFIX, wasn't one of the
    merge inputs, and is a regular file. A failed delete never fails the round.
    """
    if not merged_path or not merged_path.endswith(MERGED_DATASET_SUFFIX):
        print(f"[Cleanup] Not a merged dataset, leaving in place: {merged_path}")
        return

    target = os.path.realpath(merged_path)
    if target in {os.path.realpath(p) for p in (source_paths or [])}:
        print(f"[Cleanup] Also a source dataset, leaving in place: {merged_path}")
        return

    if os.path.islink(merged_path) or not os.path.isfile(merged_path):
        return

    try:
        size_gb = os.path.getsize(merged_path) / 1e9
        os.remove(merged_path)
        print(f"[Cleanup] Removed merged dataset ({size_gb:.2f} GB): {merged_path}")
    except OSError as e:
        print(f"[Cleanup] Could not remove {merged_path}: {e}")

def merge_robomimic_datasets(dataset_paths, output_path):
    """
    Merges multiple Robomimic HDF5 datasets into a single file.
    """
    import h5py
    import os
    print(f"\n[Merging] Combining {len(dataset_paths)} datasets into {output_path}...")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    with h5py.File(output_path, 'w') as out_f:
        out_data = out_f.create_group('data')
        
        total_frames = 0
        demo_idx = 0
        metadata_copied = False
        
        for ds_path in dataset_paths:
            print(f"  -> Copying from: {ds_path}")
            with h5py.File(ds_path, 'r') as in_f:
                if "data" not in in_f:
                    continue
                    
                # Copy attributes (like env_args) from the first valid dataset
                if not metadata_copied:
                    for attr_name, attr_value in in_f["data"].attrs.items():
                        if attr_name != "total": # We will manually calculate and set 'total' later
                            out_data.attrs[attr_name] = attr_value
                    metadata_copied = True
                    
                # Find and sort demos to maintain chronological order
                demo_keys = [k for k in in_f['data'].keys() if k.startswith('demo_')]
                demo_keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
                
                for key in demo_keys:
                    new_demo_name = f"demo_{demo_idx}"
                    
                    # This safely copies the entire demo structure, including 
                    # obs, actions, train_mask, and intervention_mask
                    in_f.copy(f"data/{key}", out_data, name=new_demo_name)
                    
                    # Update global frame count
                    num_samples = out_data[new_demo_name].attrs.get('num_samples', 0)
                    total_frames += num_samples
                    
                    demo_idx += 1
                    
        # Robomimic requires a global 'total' attribute
        out_data.attrs['total'] = total_frames
        
    print(f"[Merging] Successfully merged {demo_idx} total demos ({total_frames} frames).\n")
    return output_path

@hydra.main(version_base=None, config_path="conf", config_name="sim/coffee/blendagger")
def run_pipeline(cfg: DictConfig):
    if OmegaConf.select(cfg, "pipeline.run_eval", default=False):
        raise ValueError("pipeline.run_eval is not supported; evaluate with scripts/eval/eval_composite.py or scripts/eval/eval_realmatch.py")
    os.makedirs(cfg.paths.save_path, exist_ok=True)
    
    collect_only = cfg.pipeline.run_collect and not cfg.pipeline.run_train
    train_only = cfg.pipeline.run_train and not cfg.pipeline.run_collect
    train_data_path = cfg.training.get("train_data", None)
    if cfg.training.get("training_mode", "default") not in ("default", "human_targets"):
        raise ValueError("training.training_mode must be 'default' or 'human_targets'")

    # Policy confidence comes from k-NN uncertainty on the policy's encoder embedding.
    uncertainty_method = OmegaConf.select(cfg, "arbitration.uncertainty_method", default="knn")
    if uncertainty_method != "knn":
        raise ValueError("arbitration.uncertainty_method must be 'knn'")
    knn_index_path = OmegaConf.select(cfg, "arbitration.knn_index_path", default=None)
    # Calibration flags for the per-round KNN index build; omitted keys fall back to KNN_CALIB_DEFAULTS.
    knn_calib = {k: v for k, v in (
        (name, OmegaConf.select(cfg, f"arbitration.calib_{name}", default=None))
        for name in KNN_CALIB_DEFAULTS) if v is not None}

    # UQ blend needs calibration + (knn) index, else uncertainty is uncalibrated clip(raw,0,1).
    _uses_uncertainty = cfg.algo in (
        "combined_blend", "combined_blend_no_magnitude", "combined_blend_no_similarity",
        "combined_blend_only_uncertainty")
    if cfg.pipeline.run_collect and _uses_uncertainty:
        if not OmegaConf.select(cfg, "paths.uncertainty_path", default=None):
            raise ValueError(
                f"algo '{cfg.algo}' uses uncertainty and requires paths.uncertainty_path "
                f"(calibration JSON from build_knn_index.py). Without it uncertainty falls back "
                f"to an uncalibrated clip(raw,0,1). Use algo=combined_blend_no_uncertainty for "
                f"similarity-only blending.")
        if not knn_index_path:
            raise ValueError("arbitration.knn_index_path is required for algos that use uncertainty")

    # from_ckpt: init from the previous round's ckpt on that round's data only (default: base ckpt, rounds 1..i).
    from_ckpt = OmegaConf.select(cfg, "training.from_ckpt", default=False)
    save_every_n_epochs = OmegaConf.select(cfg, "training.save_every_n_epochs", default=None)

    name = cfg.run_name
    dir_name_partial = f"{name}_{cfg.algo}"
    dir_name_partial = f"{dir_name_partial}_{cfg.training.training_mode}" if cfg.training.training_mode != "default" else dir_name_partial
    dir_name_partial = f"{dir_name_partial}_noIWR" if cfg.training.get("use_iwr", False) is False else dir_name_partial
    DIR_NAME = f"{dir_name_partial}_{f'only_intervention_' if cfg.training.train_only_intervention else ''}{cfg.env.name}"
    DATA_ROOT = os.path.join(cfg.paths.save_path, "data", DIR_NAME)
    RESULTS_ROOT = os.path.join(cfg.paths.save_path, "results", DIR_NAME)
    config_save_path = os.path.join(DATA_ROOT, "run_config.yaml")
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    starting_iter = 1
    model_weights = cfg.paths.pretrained_model

    # Determine the starting iter and model weights to use based on existing data and results
    print("DATA_ROOT:", DATA_ROOT)
    print("RESULTS_ROOT:", RESULTS_ROOT)
    if os.path.exists(DATA_ROOT):
        if cfg.pipeline.run_collect:
            _clean_aborted_collections(DATA_ROOT)   # before counting, so moved sets aren't counted
        meta_files = glob.glob(os.path.join(DATA_ROOT, "**", "*meta.json"), recursive=True)
        num_data_iters = len(meta_files)

        # A round interrupted before num_episodes is topped up in place, so it is not yet a
        # completed data iteration: it must not trip the count check below, and starting_iter
        # has to land back on it instead of opening the next round.
        if cfg.pipeline.run_collect and _find_resumable_round(DATA_ROOT, cfg.collection.num_episodes):
            num_data_iters -= 1

        if num_data_iters > 0:
            if os.path.exists(RESULTS_ROOT):
                run_dirs = [os.path.join(RESULTS_ROOT, d) for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
                num_train_runs = len(run_dirs)
                
                if cfg.pipeline.run_collect and num_train_runs != num_data_iters:
                    raise RuntimeError(f"Mismatch: Found {num_data_iters} data iterations but {num_train_runs} training run folders in {RESULTS_ROOT}. Expected {num_data_iters} training iterations to match data iterations.")
                
                if train_only and num_train_runs != (num_data_iters - 1):
                    raise RuntimeError(f"Mismatch: Found {num_data_iters} data iterations but {num_train_runs} training run folders in {RESULTS_ROOT}. Expected {num_data_iters - 1} training iterations for train-only mode.")
                
                # Find the most recent iteration folder
                if num_train_runs != 0:
                    latest_run = max(run_dirs, key=os.path.getmtime)
                
                    # Use a wildcard to bypass the single intermediate folder: latest_run -> [some_folder] -> models
                    model_dirs = glob.glob(os.path.join(latest_run, "*", "models"))
                    
                    if model_dirs:
                        model_dir = model_dirs[0] # Grab the first (and only) models directory found
                        ckpts = glob.glob(os.path.join(model_dir, "*.pth"))
                        if ckpts:
                            model_weights = max(ckpts, key=os.path.getmtime)
                            print(f"Resuming from checkpoint model: {model_weights}")
                        else:
                            raise RuntimeError(f"No checkpoint files found in latest run models dir: {model_dir}")
                    else:
                        raise RuntimeError(f"Models directory not found inside any subfolder of latest run: {latest_run}")
                if cfg.pipeline.run_collect:
                    starting_iter = num_data_iters + 1
                else:
                    starting_iter = num_data_iters
                print(f"Found {num_data_iters} previous data iterations. Setting starting_iter to {starting_iter}.")
            else:
                raise RuntimeError(f"Found {num_data_iters} data iterations but RESULTS_ROOT does not exist: {RESULTS_ROOT}")

    if cfg.pipeline.run_collect or cfg.pipeline.run_train:
        if train_data_path is not None:
            data_files = glob.glob(os.path.join(train_data_path, "*dataset_full.hdf5"))
            train_iters = len(data_files)
            if cfg.training.get("run_all_iters", True):
                # Resume: skip rounds already trained to num_epochs.
                completed, last_ckpt = _count_completed_rounds(RESULTS_ROOT, cfg.training.num_epochs)
                completed = min(completed, train_iters)
                starting_iter = completed + 1
                if completed > 0:
                    model_weights = last_ckpt  # continue the from_ckpt chain / next init
                    print(f"[Resume] {completed}/{train_iters} round(s) already trained to "
                          f"{cfg.training.num_epochs} epochs; resuming at round {starting_iter}.")
            else:
                starting_iter = train_iters
        else:
            train_iters = starting_iter
        
        save_rollout_freq = cfg.training.get("save_rollout_freq", 5)

        for i in range(starting_iter, train_iters + 1):
            save_rollout = (i % save_rollout_freq == 0)
            
            print(f"\n=== Training iteration {i} ===")
            print("saving rollout:", save_rollout)
            print("Data override: ", train_data_path)
            new_weights = main(
                iteration=i,
                algo=cfg.algo,
                name=name,
                train_all_steps=not cfg.training.train_only_intervention,
                num_epochs=cfg.training.num_epochs,
                num_episodes=cfg.collection.num_episodes,
                env_name=cfg.env.name,
                config_path=cfg.paths.config_path,
                model_weights=model_weights,
                pretrained_model=cfg.paths.pretrained_model,
                pretrained_dataset=cfg.paths.pretrained_dataset,
                use_sequence=(cfg.model in ["transformer", "diffusion"]),
                horizon=cfg.env.horizon,
                save_rollout=save_rollout,
                fixed_env=cfg.env.fixed_env,
                aggregate_data=cfg.training.aggregate_data,
                aggregate_pretrain=cfg.training.aggregate_pretrain,
                button_for_intervention=cfg.collection.button_for_intervention,
                input_device=cfg.collection.input_device,
                collect_only=collect_only,
                train_only=train_only,
                freeze_encoder=cfg.training.freeze_encoder,
                use_iwr=cfg.training.use_iwr,
                DIR_NAME=DIR_NAME,   
                DATA_ROOT=DATA_ROOT,
                RESULTS_ROOT=RESULTS_ROOT,
                uncertainty_path=cfg.paths.uncertainty_path,
                training_mode=cfg.training.training_mode,
                train_data_override=train_data_path,
                uncertainty_method=uncertainty_method,
                knn_index_path=knn_index_path,
                knn_calib=knn_calib,
                from_ckpt=from_ckpt,
                save_every_n_epochs=save_every_n_epochs
            )
            
            if new_weights is None:
                print("No new model weights returned from iteration, stopping training loop.")
                break
                
            model_weights = new_weights
            print("New pretrained weights for next iteration:", model_weights)


if __name__ == "__main__":
    run_pipeline()
