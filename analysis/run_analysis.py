"""
Main orchestrator for OOD and trajectory similarity analysis.

Reads a task config under conf/analysis/ (via Hydra), loads the run folders it
lists, calls ood_measures and traj_similarity, then prints a comparison table and
saves a JSON results file, the raw distributions and a per-episode export for
downstream plotting/statistics code. No figures are drawn here.

Usage
-----
    python analysis/run_analysis.py --config-name analysis/sim/coffee
    python analysis/run_analysis.py --config-name analysis/sim/thawing
    python analysis/run_analysis.py --config-name analysis/sim/pick_place

    # Cap pretrain demos used for trajectory distances (faster for quick checks)
    python analysis/run_analysis.py --config-name analysis/sim/coffee trajectory.max_pretrain_demos=5
"""

from __future__ import annotations
import collections
import glob
import json
import os
import re

import h5py
import hydra
import textwrap

import numpy as np
from metrics.alpha import alpha_is_constant, has_blend_fields
from omegaconf import DictConfig, OmegaConf
from scipy.stats import t as t_dist
from tqdm import tqdm

from metrics import episode_export as export
from metrics import intervention_response as iresp
from metrics import intervention_timing as itiming
from metrics import joint_jerk as jjerk
from metrics import ood_measures as ood
from metrics import task_time
from metrics import traj_similarity as traj

# Re-exported so callers can import these from this module.
from metrics import state                                        # noqa: E402
from metrics.state import set_angle_space, angle_space_for       # noqa: E402,F401
from metrics.stats import (                                    # noqa: E402,F401
    _EMPTY_POINTS, _ci95, _ci95_cluster, _episode_means,
)
# This file computes and saves; it draws nothing. Outputs go under <output_dir>/<category>/.


def _cat_path(output_dir: str, category: str, filename: str) -> str:
    """Path inside output_dir/<category>/, creating the subfolder if needed."""
    d = os.path.join(output_dir, category)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


from metrics import transition_jerk as tjerk


# ===========================================================================
# Common utilities  (data loading, stats formatting)
# ===========================================================================

# Recompute only a subset of the metric families. Empty = everything.
#   "trajsim"    DTW / Frechet family        "ood"  kNN indices
#   "collection" timing + action discontinuity
# Skipped families are copied forward from the existing distributions.npz, so a
# trajsim-only rerun does not have to redo the (unaffected) kNN pass.
ONLY_METRICS: set[str] = set()


def set_only_metrics(names) -> None:
    global ONLY_METRICS
    ONLY_METRICS = set(names or ())


def want_metric(name: str) -> bool:
    return not ONLY_METRICS or name in ONLY_METRICS


# Pauses this short are filled in on the mask itself, so every metric — timing, OOD, trajectory
# similarity, action change — reads one interrupted correction as one intervention. 0 leaves the
# recorded mask exactly as it is.
MERGE_MASK_GAP: int = 0


def set_merge_mask_gap(steps: int) -> None:
    global MERGE_MASK_GAP
    MERGE_MASK_GAP = int(steps or 0)


# Obs keys holding joint angles; expanded to (sin, cos) so distance respects wraparound.
ANGLE_KEYS: set[str] = set()


def set_angle_keys(keys) -> None:
    global ANGLE_KEYS
    ANGLE_KEYS = set(keys or ())


def _concat_obs(obs_grp: h5py.Group, keys: list[str]) -> np.ndarray:
    """Concatenate selected obs keys along the feature axis → (T, D).

    Keys named in ANGLE_KEYS expand to sin/cos so distance respects wraparound.
    """
    parts = []
    for k in keys:
        if k not in obs_grp:
            raise KeyError(f"obs key '{k}' not found; available: {list(obs_grp.keys())}")
        arr = obs_grp[k][:].astype(np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if k in ANGLE_KEYS:
            arr = np.concatenate([np.sin(arr), np.cos(arr)], axis=1)
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def joint_state_vel(states_sincos: np.ndarray, hz: float) -> np.ndarray:
    """[sin q, cos q, dq*hz] for ONE trajectory of sin/cos joint states.

    The joint-space counterpart of the obs+action vector. The recorded action is a
    CARTESIAN velocity fraction, so it cannot go into an angular metric directly;
    the realised joint delta can, needs no kinematic model, and is unaffected by
    the wrist singularities the driver damps. dq is wrapped to (-pi, pi] and scaled
    to rad/s so a configuration difference and a motion difference are both angles.
    Must be called PER TRAJECTORY — a diff across an episode boundary is meaningless.
    """
    J = states_sincos.shape[1] // 2
    th = np.arctan2(states_sincos[:, :J], states_sincos[:, J:])
    dq = np.zeros_like(th)
    if len(th) > 1:
        d = np.diff(th, axis=0)
        dq[1:] = np.arctan2(np.sin(d), np.cos(d))
    return np.concatenate([states_sincos, dq * hz], axis=1)


def load_pretrain_states(
    dataset_path: str, obs_keys: list[str]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Load all timesteps from a pretrain HDF5 dataset.

    Returns
    -------
    all_states   : (N, D) — every timestep stacked
    trajectories : list of (T_i, D) arrays, one per demo
    """
    all_states, trajectories, all_actions = [], [], []
    with h5py.File(dataset_path, "r") as f:
        demos = sorted(f["data"].keys())
        for demo in tqdm(demos, desc="  Loading pretrain demos", leave=False):
            traj_arr = _concat_obs(f["data"][demo]["obs"], obs_keys)
            all_states.append(traj_arr)
            trajectories.append(traj_arr)
            grp = f["data"][demo]
            if "actions" in grp:
                all_actions.append(np.asarray(grp["actions"][:], dtype=np.float64))
    actions = np.concatenate(all_actions, axis=0) if all_actions else None
    return np.concatenate(all_states, axis=0), trajectories, actions


def load_collected_states(
    hdf5_path: str,
    obs_keys: list[str],
    load_actions: bool = True,
    skip_aborted: bool = True,
) -> list[dict]:
    """
    Load all demos from a collected (HG-DAgger / BlenDAgger) HDF5 file.

    Aborted episodes (per-demo attr `aborted=True`) are skipped by default: every
    metric that consumes this (OOD, DTW, action-change, intervention timing) must
    exclude them — only the *_accumulated collection plots keep aborted attempts.

    Returns a list of dicts, one per demo:
        states            : (T, D) raw low-dim states
        intervention_mask : (T,) bool — True = human was intervening
        train_mask        : (T,) bool — True = timestep included in training
        actions           : (T, A) float — executed actions (if load_actions=True)
    """
    demos_out, empty = [], []
    with h5py.File(hdf5_path, "r") as f:
        for demo in sorted(f["data"].keys()):
            grp = f["data"][demo]
            if skip_aborted and bool(grp.attrs.get("aborted", False)):
                continue
            # A round can hold recordings that captured camera frames and nothing else — no
            # proprioception and no intervention mask. They are not episodes: the meta does not
            # list them either, and they can contribute to no metric.
            if "obs" not in grp or "intervention_mask" not in grp or \
                    any(k not in grp["obs"] for k in obs_keys):
                empty.append(demo)
                continue
            d = dict(
                demo=demo,
                states=_concat_obs(grp["obs"], obs_keys),
                intervention_mask=itiming.fill_short_gaps(
                    grp["intervention_mask"][:].astype(bool), MERGE_MASK_GAP),
                train_mask=grp["train_mask"][:].astype(bool),
            )
            if load_actions:
                if "actions" not in grp:
                    raise KeyError(f"'actions' not found in demo {demo} of {hdf5_path}")
                d["actions"] = grp["actions"][:].astype(np.float64)
            # Needed by intervention_response; when absent (e.g. sim) that module contributes nothing.
            for key, src in (("human_actions", grp), ("robot_actions", grp),
                             ("arm_pos", grp.get("obs", {})), ("arm_quat", grp.get("obs", {}))):
                d[key] = src[key][:].astype(np.float64) if key in src else None
            d["joint_acc"] = grp["joint_acc"][:].astype(np.float64) if "joint_acc" in grp else None
            d["alphas"] = grp["alphas"][:].astype(np.float64) if "alphas" in grp else None
            d["grip_window"] = task_time.grip_window(grp, len(d["intervention_mask"]))
            demos_out.append(d)
    if empty:
        print(f"    [demos] {os.path.basename(hdf5_path)}: skipped {len(empty)} recording(s) "
              f"with no robot state ({empty[:4]}) — camera frames only, not episodes")
    return demos_out


# ---------------------------------------------------------------------------
# The reference the deviation metrics are measured against
# ---------------------------------------------------------------------------
# For round N of method M the reference is the data the policy run at round N was actually
# trained on: the pretrain demos plus everything M collected in rounds 1..N-1. Round 1 is the
# pretrain set alone and is unchanged, so a five-round run fits ten kNN indices rather than one.
#
# Measured against a frozen pretrain set the distance drifts upward every round for both methods
# simply because both are learning, and says nothing about whether either policy was operating
# inside its own competence — which is what an intervention is correcting. Moving the reference
# with the policy is also what makes round a factor rather than a confound.
#
# Set ood.reference=pretrain to measure everything against the pretrain set instead.
#
# `ood.reference_sets` declares the reference explicitly instead, per run key and round:
#
#   ood.reference_sets: {dagger: {2: [.../cupboard_dagger_round1]}}
#
# for runs whose policies were not trained on their own earlier rounds (e.g. the user study).
REFERENCE_POLICY = "policy_training_set"
REFERENCE_PRETRAIN = "pretrain"
REFERENCE_DECLARED = "declared"
_TAU_PERCENTILES = (90, 95, 99)


# Percentile sweep, so the exceedance curve can be drawn rather than a single threshold.
_TAU_SWEEP = (50, 60, 70, 75, 80, 85, 90, 95, 97, 99)


class RoundReference:
    """kNN index, trajectory references and thresholds for one round's training set."""

    def __init__(self, states: np.ndarray, trajs: list[np.ndarray], k: int,
                 sources: list[str], episodes: list[np.ndarray] | None = None):
        # Which episode each state came from, so tau can be calibrated leave-one-episode-out.
        # `trajs` may be a capped random subset, so it is used only if it partitions `states`.
        parts = episodes if episodes is not None else trajs
        counted = sum(len(t) for t in parts)
        if counted != len(states):
            raise ValueError(
                f"cannot tell which episode each reference state came from: the {len(parts)} "
                f"episode(s) given cover {counted} states but the reference holds {len(states)}. "
                "Pass episodes= with the per-demo arrays these states were concatenated from.")
        self.groups = np.concatenate(
            [np.full(len(t), i) for i, t in enumerate(parts)]) if parts else np.zeros(0)

        if state.ANGLE_SPACE:
            # Pure joint-angle (sin, cos) space is already commensurate and bounded; z-scoring
            # would only re-weight joints by how much they happened to move.
            self.mean = np.zeros(states.shape[1])
            self.std = np.ones(states.shape[1])
        else:
            self.mean, self.std = ood.fit_normalizer(states)
        self.states_norm = ood.normalize(states, self.mean, self.std)
        self.nn = ood.fit_knn(self.states_norm, k=k)
        self.trajs_norm = [ood.normalize(t, self.mean, self.std) for t in trajs]
        # Leave-one-episode-out to avoid self-matches; see ood.knn_self_scores.
        self.self_scores = ood.knn_self_scores(self.states_norm, self.groups, k=k)
        self.n_states = int(len(states))
        self.n_trajs = len(trajs)
        self.sources = list(sources)
        # tau is a percentile of the REFERENCE's own distance distribution, not of the collected
        # data: the question is how far the robot went outside what its policy was trained on.
        # nan is a state whose reference held fewer than k out-of-episode neighbours, and there is
        # nothing to calibrate from it.
        scored = self.self_scores[~np.isnan(self.self_scores)]
        self.n_self_scored = int(len(scored))
        self.tau = {p: float(np.percentile(scored, p)) for p in _TAU_PERCENTILES}
        self.tau_sweep = {p: float(np.percentile(scored, p)) for p in _TAU_SWEEP}

    def summary(self) -> dict:
        return dict(sources=self.sources, n_states=self.n_states, n_trajectories=self.n_trajs,
                    tau={f"p{p}": v for p, v in self.tau.items()},
                    tau_sweep={f"p{p}": v for p, v in self.tau_sweep.items()},
                    knn_self=dict(mean=float(np.nanmean(self.self_scores)),
                                  std=float(np.nanstd(self.self_scores)),
                                  calibration="leave-one-episode-out",
                                  n_scored=self.n_self_scored))


def load_reference_source(path: str, obs_keys: list[str]) -> tuple[np.ndarray, list]:
    """(all states, per-demo trajectories) for one declared reference dataset.

    A folder resolves to its processed full file, which is what the finetune was fitted on.
    Aborted attempts are skipped: they were never training data.
    """
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "proc_*_full.hdf5")))
        if not found:
            found = [f for f in sorted(glob.glob(os.path.join(path, "*_full.hdf5")))
                     if not os.path.basename(f).startswith("proc_")]
        if not found:
            raise FileNotFoundError(f"no hdf5 to use as a reference in {path}")
        path = found[0]
    trajs = []
    with h5py.File(path, "r") as f:
        for demo in sorted(f["data"].keys()):
            grp = f["data"][demo]
            if bool(grp.attrs.get("aborted", False)):
                continue
            trajs.append(_concat_obs(grp["obs"], obs_keys))
    return np.concatenate(trajs, axis=0), trajs


def match_processed_demos(proc_path: str, full_path: str | None,
                         obs_keys: list[str]) -> dict[str, str]:
    """{proc demo name: the collection-file demo it was made from}.

    `scripts/data_processing.py` renumbers its output contiguously in h5py key order, so as soon
    as a round has more than ten episodes the proc file's `demo_2` is the collection file's
    `demo_10`, and any episode trimmed to nothing is dropped without leaving a gap. The names
    therefore cannot be joined. The FRAMES can: trimming removes rows without altering them, so a
    processed episode's observations are a subset of its source episode's.

    Returns {} when there is nothing to map (no collection file, or the two are the same file),
    which callers read as "the names already agree".
    """
    if not full_path or os.path.abspath(proc_path) == os.path.abspath(full_path):
        return {}

    def carries(handle, key):
        """True if ANY demo has this obs key — a round can hold empty placeholder demos."""
        return any("obs" in handle["data"][d] and key in handle["data"][d]["obs"]
                   for d in handle["data"])

    def rows(handle, key):
        """{demo: (T, D) obs} for the demos that actually carry the key."""
        out = {}
        for d in handle["data"]:
            grp = handle["data"][d]
            if "obs" not in grp or key not in grp["obs"]:
                continue
            arr = np.atleast_2d(np.asarray(grp["obs"][key][:]))
            if arr.size:
                out[d] = arr
        return out

    with h5py.File(full_path, "r") as ff, h5py.File(proc_path, "r") as fp:
        key = next((k for k in obs_keys if carries(ff, k) and carries(fp, k)), None)
        if key is None:
            return {}
        source = {d: {tuple(np.round(r, 5)) for r in a} for d, a in rows(ff, key).items()}
        mapping, taken = {}, set()
        for d, arr in rows(fp, key).items():
            probe = arr[np.linspace(0, len(arr) - 1, min(32, len(arr))).astype(int)]
            probe = {tuple(np.round(r, 5)) for r in probe}
            best, score = None, 0.0
            for cand in source:
                if cand in taken:
                    continue
                hit = len(probe & source[cand]) / len(probe)
                if hit > score:
                    best, score = cand, hit
            if best is not None and score == 1.0:
                mapping[d] = best
                taken.add(best)
    return mapping


def agg(values: np.ndarray) -> dict:
    """Summary statistics (mean, std, median, n) for a 1-D array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(mean=None, std=None, median=None, n=0)
    return dict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        n=int(len(values)),
    )


def _fmt(stats: dict) -> str:
    """Format a stats dict as 'mean ± std' for table printing."""
    if stats.get("mean") is None:
        return "     N/A     "
    return f"{stats['mean']:7.4f} ± {stats['std']:6.4f}"


# ===========================================================================
# Run discovery
# ===========================================================================

def meta_path_for(fpath: str) -> str | None:
    """Meta JSON beside a full hdf5: real-robot <stem>_meta.json, sim <stem>.hdf5_meta.json.

    The sim writes `<run>_<stamp>_dataset_full.hdf5` beside `<run>_<stamp>_meta.json`, so the
    `_dataset` part has to come off too; longest suffix first, and every candidate is tried.
    """
    for suffix in ("_dataset_full_im128.hdf5", "_dataset_full.hdf5",
                   "_full_im128.hdf5", "_full.hdf5"):
        if fpath.endswith(suffix):
            stem = fpath[:-len(suffix)]
            for cand in (stem + "_meta.json", stem + ".hdf5_meta.json"):
                if os.path.exists(cand):
                    return cand
    return None


def resolve_runs(cfg: DictConfig) -> dict[str, str]:
    """
    Return {algo_label: folder_path} for all algorithms to analyse.

    If cfg.runs is set explicitly, use it as-is.
    Otherwise auto-discover:
      data/{run_name}_hg_dagger_{env_name}
      data/{run_name}_combined_blend_{env_name}
    """
    if cfg.runs is not None:
        # Values may be a single folder (str) or a list of round folders that
        # are treated as successive iterations of one algorithm.
        runs = OmegaConf.to_container(cfg.runs, resolve=True)
        print(f"\n[Config] Using explicit runs override ({len(runs)} algorithms):")
        for label, folder in runs.items():
            if isinstance(folder, (list, tuple)):
                print(f"  {label:30s} → {len(folder)} round folders (iterations):")
                for fol in folder:
                    print(f"  {'':30s}    {fol}")
            else:
                print(f"  {label:30s} → {folder}")
        return runs

    default_algos = ["hg_dagger", "combined_blend"]
    runs = {}
    print(f"\n[Discovery] run_name='{cfg.run_name}', env_name='{cfg.env_name}'")
    print("  Looking for: data/{run_name}_{algo}_{env_name}")
    for algo in default_algos:
        folder = os.path.join("data", f"{cfg.run_name}_{algo}_{cfg.env_name}")
        if not os.path.isdir(folder):
            print(f"  [SKIP]  {folder}  (directory not found)")
            continue
        files = sorted(
            glob.glob(os.path.join(folder, "*_dataset_full_im128.hdf5")) +
            glob.glob(os.path.join(folder, "*_full.hdf5"))
        )
        if not files:
            print(f"  [SKIP]  {folder}  (no *_dataset_full_im128.hdf5 or *_full.hdf5 files)")
            continue
        runs[algo] = folder
        print(f"  [FOUND] {algo:25s} → {folder}  ({len(files)} iteration files)")

    if not runs:
        raise RuntimeError(
            f"No folders found for run_name='{cfg.run_name}', env_name='{cfg.env_name}'. "
            "Use 'runs=' to specify folders explicitly."
        )
    return runs


# ===========================================================================
# Per-algorithm processing
# ===========================================================================

def _summarize_collection(meta_rows: list[dict]) -> dict:
    """Aggregate per-demo collection stats read from *_meta.json files.

    These describe the collection episode itself (not the training data), so
    they are computed from the FULL files' metadata, never the processed files.
    """
    def col(key):
        return np.array([r[key] for r in meta_rows if key in r], dtype=np.float64)
    succ = np.array([bool(r.get("task_success", False)) for r in meta_rows])
    return dict(
        n_demos=len(meta_rows),
        success_rate=float(succ.mean()) if len(succ) else float("nan"),
        task_time_s=agg(col("task_time_seconds")),
        total_interv_mag=agg(col("total_intervention_magnitude")),
        mean_interv_mag=agg(col("mean_intervention_magnitude")),
        # mean magnitude over "moving" steps only; n=0 when the field is absent
        mean_interv_mag_moving=agg(col("mean_intervention_magnitude_moving")),
        num_interventions=agg(col("num_distinct_interventions")),
        num_interv_steps=agg(col("num_intervention_steps")),
    )


# ---------------------------------------------------------------------------
# Per-episode rows for the export
# ---------------------------------------------------------------------------
# Each of these reduces ONE episode's arrays to the exported columns. Nothing
# here redefines a metric: the timing rows are what intervention_timing.py already produced and
# the action rows are what transition_jerk.py already produced, grouped per episode instead of
# pooled. A value an episode does not have is left out entirely, so it lands as a blank cell
# rather than a zero.

def gap_timesteps(cfg: DictConfig, hz: float) -> int:
    """The intervention-merge window in timesteps, from a duration when the config gives one.

    A gap is a pause in the operator's correction, which is a duration, so `gap_seconds` keeps it
    consistent across control rates; `gap_threshold` is the raw-timestep fallback.
    """
    block = cfg.intervention_timing
    seconds = block.get("gap_seconds", None)
    if seconds is not None:
        return max(1, int(round(float(seconds) * float(hz))))
    return int(block.gap_threshold)


def _finite_mean(values):
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _seconds(steps, hz: float):
    return None if steps is None else float(steps) / float(hz)


def _first(values):
    """Earliest finite value in an episode's groups (they are produced in time order)."""
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else None


def _last(values):
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else None


def _timing_row(timing: dict, hz: float) -> dict:
    """Intervention onset, offset and duration, on both the episode-fraction and second scales.

    Both scales are required rather than either/or: the fraction says where in the task the
    correction happened, the seconds say how long the robot ran before it did.
    """
    return {
        "intervention_start_frac": _finite_mean(timing.get("group_start_fracs", [])),
        "intervention_end_frac": _finite_mean(timing.get("group_end_fracs", [])),
        "intervention_start_s": _seconds(_finite_mean(timing.get("group_starts", [])), hz),
        "intervention_end_s": _seconds(_finite_mean(timing.get("group_ends", [])), hz),
        "intervention_duration_s": _seconds(_finite_mean(timing.get("group_lengths", [])), hz),
        "intervention_group_length_s": _seconds(
            _finite_mean(timing.get("group_lengths", [])), hz),
        "num_intervention_groups": int(len(timing.get("group_starts", []))),
        # The FIRST onset and the LAST offset, which answer "when did the human step in" and
        # "when did they let go" rather than "where did their corrections sit on average". The
        # mean above is over an episode's merged groups, and the methods fragment differently.
        "intervention_first_start_frac": _first(timing.get("group_start_fracs", [])),
        "intervention_last_end_frac": _last(timing.get("group_end_fracs", [])),
        "intervention_first_start_s": _seconds(_first(timing.get("group_starts", [])), hz),
        "intervention_last_end_s": _seconds(_last(timing.get("group_ends", [])), hz),
        "intervention_start_frac_segments": _finite_mean(timing.get("segment_start_fracs", [])),
        "intervention_end_frac_segments": _finite_mean(timing.get("segment_end_fracs", [])),
        "intervention_duration_s_segments": _seconds(
            _finite_mean(timing.get("segment_lengths", [])), hz),
    }


def _action_row(jerk: dict, cmd: dict, joint_acc: dict | None, during: float) -> dict:
    """Action change at each handover, during the correction itself, and split by unit."""
    row = {
        "action_delta_ro_to_iv": _finite_mean(jerk.get("ro_to_iv", [])),
        "action_delta_iv_to_ro": _finite_mean(jerk.get("iv_to_ro", [])),
        "action_delta_during": float(during) if np.isfinite(during) else None,
        "action_delta_ro_to_iv_linear": _finite_mean(cmd.get("lin_ro_to_iv", [])),
        "action_delta_iv_to_ro_linear": _finite_mean(cmd.get("lin_iv_to_ro", [])),
        "action_delta_ro_to_iv_angular": _finite_mean(cmd.get("ang_ro_to_iv", [])),
        "action_delta_iv_to_ro_angular": _finite_mean(cmd.get("ang_iv_to_ro", [])),
    }
    if joint_acc:
        row["joint_jerk"] = _finite_mean(np.concatenate(
            [np.asarray(v, dtype=np.float64).ravel() for v in joint_acc.values()]))
        row["joint_jerk_ro_to_iv"] = _finite_mean(joint_acc.get("ro_to_iv", []))
        row["joint_jerk_iv_to_ro"] = _finite_mean(joint_acc.get("iv_to_ro", []))
    return row


# Smoothness metrics on the PROCESSED file (absolute-pose actions), exported beside the executed ones.
# `joint_jerk_training` is comparable to `joint_jerk`: proc does not re-encode the state.
TRAINING_SUFFIX = "_training"


def _training_action_row(demo: dict, cfg: DictConfig, hz: float) -> dict:
    """The action-change columns recomputed on one episode of the processed training file."""
    jerk = tjerk.compute_transition_jerk([demo])
    cmd = tjerk.compute_command_delta(
        [demo], action_scale=float(cfg.get("collection_action_scale", 0.8)))
    during = tjerk.compute_during_intervention_delta([demo])
    joint_acc = tjerk.compute_joint_accel([demo], hz=hz) if state.ANGLE_SPACE else None
    row = _action_row(jerk, cmd, joint_acc,
                      during[0] if len(during) else float("nan"))
    return {f"{k}{TRAINING_SUFFIX}": v for k, v in row.items()}


def _proc_demos(folder: str, obs_keys, full_path: str | None = None) -> dict:
    """{collection demo name: processed demo} for the folder's proc FULL file, or {}.

    Keyed on the COLLECTION file's name, not the processed one. data_processing renumbers its
    output contiguously, so a round whose full file does not start at demo_0 — an episode marked
    discard, or more than ten episodes — has proc demo_N built from a different collection demo_N.
    Matching by name there pairs each episode's collection metrics with another episode's training
    ones. The frames are what survive renumbering, which is what match_processed_demos joins on.

    Folders without a proc file return {}, so the training columns are absent there rather than
    duplicating the executed numbers.
    """
    found = sorted(glob.glob(os.path.join(folder, "proc_*_full.hdf5")))
    if not found:
        return {}
    demos = load_collected_states(found[0], obs_keys)
    names = match_processed_demos(found[0], full_path, obs_keys) if full_path else {}
    if names:
        unmatched = [d["demo"] for d in demos if d["demo"] not in names]
        if unmatched:
            print(f"    [warn] {len(unmatched)}/{len(demos)} processed episode(s) could not be "
                  f"matched to a collection episode ({unmatched[:4]}); they carry no training "
                  f"columns rather than landing on another episode's row")
        return {names[d["demo"]]: d for d in demos if d["demo"] in names}
    return {d["demo"]: d for d in demos}


def _config_name(cfg: DictConfig) -> str:
    """The Hydra config this run was launched with, so a result traces back to its definition."""
    try:
        from hydra.core.hydra_config import HydraConfig
        return str(HydraConfig.get().job.config_name)
    except Exception:
        return str(cfg.get("run_name", ""))


def _metric_notes(hz: float) -> dict:
    """What each exported column means, recorded beside the numbers rather than assumed."""
    return {
        "ood_mean_all": {"units": "kNN distance", "per": "episode mean over timesteps"},
        "ood_mean_intervention": {"units": "kNN distance",
                                  "per": "episode mean over interventions"},
        "ood_fraction_above_p95": {"units": "fraction of INTERVENED timesteps",
                                   "reference": "round reference kNN p95"},
        "ood_reference_n": {"units": "states", "note": "kNN distance falls as a reference grows"},
        "intervention_start_frac": {"units": "fraction of episode",
                                    "per": "episode mean over merged groups"},
        "intervention_start_s": {"units": "seconds", "per": "episode mean over merged groups",
                                 "note": f"timesteps / {hz:g} Hz"},
        "intervention_duration_s": {"units": "seconds",
                                    "per": "episode mean over merged groups"},
        "action_delta_ro_to_iv": {"units": "action units",
                                  "definition": "mean ||a[t]-a[t-1]|| at each handover to the "
                                                "human"},
        "action_delta_during": {"units": "action units",
                                "definition": "mean ||a[t]-a[t-1]|| with t-1 and t both inside "
                                              "one intervention segment",
                                "note": "uses intervention_mask, not train_mask as "
                                        "transition_jerk's train_non_transition does"},
        "action_delta_ro_to_iv_linear": {"units": "m/s"},
        "action_delta_ro_to_iv_angular": {"units": "rad/s"},
        "joint_jerk": {"units": "rad/s^2", "per": "episode mean over all timesteps",
                       "file": "executed (*_full.hdf5) -- the commands the robot actually ran"},
        # The same three, measured on the processed training file where a folder has one. Absent for
        # the sim tasks, which have no proc_* file, so a blank cell there means "not measured", not
        # "zero". See TRAINING_SUFFIX for why both are worth having.
        "action_delta_ro_to_iv_training": {
            "units": "action units",
            "file": "processed (proc_*_full.hdf5) -- the actions the policy is fit on",
            "note": "proc encodes actions as absolute pose, so values are small for every "
                    "method and not comparable to action_delta_ro_to_iv"},
        "action_delta_iv_to_ro_training": {"units": "action units", "file": "processed"},
        "action_delta_during_training": {"units": "action units", "file": "processed"},
        "joint_jerk_training": {
            "units": "rad/s^2", "per": "episode mean over all timesteps", "file": "processed",
            "note": "comparable to joint_jerk: a realised joint-velocity change from the state, "
                    "which proc does not re-encode"},
        "frechet_full": {"units": "normalised distance", "transform": "none"},
        "dtw_full": {"units": "normalised distance", "transform": "none"},
        "alpha_mean_during_intervention": {"units": "blend weight, 0 = human, 1 = robot",
                                           "per": "episode mean over intervened steps, "
                                                  "averaged across DoFs"},
    }


def _deviation_row(ep_knn, per_intervention, iv_scores, ref: RoundReference,
                   full_dtw, full_fre, seg_dtw, seg_fre, seg_of, d_idx) -> dict:
    """How far one episode sat from the data its policy was trained on.

    tau is a percentile of the REFERENCE's own leave-one-episode-out kNN distance distribution, so
    "beyond p95" means beyond where the policy's training data itself sat. The reference size
    travels on every row because kNN distance falls as a reference grows, so downstream analysis
    can account for it.

    The exceedance fractions are over the INTERVENED timesteps, not the whole episode: the
    question is how far outside its training data the robot was when it had to be corrected.
    `ood_mean_all` still covers the whole episode, so both views survive.
    """
    row = {
        "ood_mean_all": _finite_mean(ep_knn),
        "ood_mean_intervention": _finite_mean(per_intervention),
        "ood_reference_n": ref.n_states,
    }
    scores = np.asarray(iv_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    for p, tau in ref.tau.items():
        row[f"ood_reference_p{p}"] = tau
    # The exceedance at every sweep threshold, so the full exceedance curve can be drawn.
    for p, tau in ref.tau_sweep.items():
        if scores.size:
            row[f"ood_fraction_above_p{p}"] = float((scores > tau).mean())
    if d_idx < len(full_dtw):
        row["dtw_full"] = float(full_dtw[d_idx])
    if d_idx < len(full_fre):
        row["frechet_full"] = float(full_fre[d_idx])
    if len(seg_dtw):
        row["dtw_intervention"] = _finite_mean(seg_dtw[seg_of == d_idx])
    if len(seg_fre):
        row["frechet_intervention"] = _finite_mean(seg_fre[seg_of == d_idx])
    return row


def _alpha_row(demo: dict, blended: bool) -> dict:
    """Mean blend weight over the episode's intervened steps, overall and per DoF.

    Blended runs only. Blank for a run that logs no alphas, for an episode with no intervention,
    and for HG-DAgger, whose handover is a switch rather than an arbitration: blank means "not
    applicable", zero would mean the arbitration handed everything over, and they differ.
    """
    if not blended:
        return {}
    mean_alpha, per_dof = export.alpha_during_intervention(
        demo.get("alphas"), demo["intervention_mask"])
    if mean_alpha is None:
        return {}
    row = {"alpha_mean_during_intervention": mean_alpha}
    for i, value in enumerate(per_dof):
        row[f"alpha_mean_during_intervention_dof{i}"] = value
    return row


def process_algo(
    algo_label: str,
    folder,
    obs_keys: list[str],
    base_ref: RoundReference,
    pretrain_states: np.ndarray,
    pretrain_trajs: list[np.ndarray],
    cfg: DictConfig,
    oa_index: tuple = (None, None, None),
    nn_ja=None,
    control_hz: float = 30.0,
    registry=None,
    method: str | None = None,
    pretrain_label: str = "pretrain",
    reference_mode: str = REFERENCE_POLICY,
    declared_refs: dict | None = None,
) -> tuple[dict, dict, dict, dict]:
    """
    Process all iteration files for one algorithm.

    Calls ood_measures.compute_ood_stats() and traj_similarity.compute_demo_dtw()
    for each iteration, then aggregates across all iterations.

    Each iteration is measured against its own reference set — the data the policy that ran that
    round was trained on — which grows by one round each iteration (see RoundReference). When a
    registry is given, every episode is also recorded on it for the per-episode export.

    Returns
    -------
    iter_results   : {iter_label: per-iteration stats dict}
    iter_knn       : {iter_label: {"all": arr, "intervention": arr, "rollout": arr}}
    iter_dtw       : {iter_label: {"full": arr, "intervention_segs": arr, "rollout_segs": arr}}
    algo_aggregate : aggregate stats across all iterations combined
    references     : {f"{method}/{round}": reference summary} for the export manifest
    ep_rounds      : {episode id: collection round}, so a figure can restrict itself to one
    timing_by_round: {round: {timing key: array}}, the diagnostic figure's per-round split
    """
    def _is_proc(p):
        return os.path.basename(p).startswith("proc_")

    def _one_per_round(paths):
        """Sim writes *_full.hdf5 AND *_full_im128.hdf5 per round; only the latter has obs."""
        best = {}
        for p in paths:
            key = p.replace("_full_im128.hdf5", "_full.hdf5")
            if key not in best or p.endswith("_full_im128.hdf5"):
                best[key] = p
        return [best[k] for k in sorted(best)]

    # A single folder or a list of round folders (each round = one iteration).
    folder_list = [folder] if isinstance(folder, str) else list(folder)

    # Gather files folder-by-folder so iteration order follows round order. The folder each file
    # came from travels with it: round and session are read off it, never off an array index.
    full_files, proc_files = [], []
    full_folders, proc_folders = [], []
    for fol in folder_list:
        found = _one_per_round([
            f for f in (glob.glob(os.path.join(fol, "*_dataset_full_im128.hdf5")) +
                        glob.glob(os.path.join(fol, "*_full.hdf5")))
            if not _is_proc(f)
        ])
        full_files += found
        full_folders += [fol] * len(found)
        # Only the processed FULL file. Some rounds also carry
        # proc_*_intervention.hdf5 / proc_*_rollout.hdf5 splits (often empty),
        # and a bare proc_*.hdf5 glob would count each as another round.
        found = sorted(glob.glob(os.path.join(fol, "proc_*_full.hdf5")))
        proc_files += found
        proc_folders += [fol] * len(found)
    # Split by what the metric is ABOUT:
    #   proc_*  → training-data metrics (OOD, DTW): the states the policy is fit on.
    #   *_full  → collection metrics (intervention timing, action discontinuity,
    #             meta summary): what the operator and robot actually did.
    # proc re-encodes actions as 10-D absolute pose, which cannot jump at a handoff, while the
    # full file holds the executed 7-D relative commands; proc also drops non-training steps.
    train_files = proc_files if proc_files else full_files
    train_folders = proc_folders if proc_files else full_folders

    print(f"\n{'─'*60}")
    print(f"Algorithm : {algo_label}")
    print(f"Folder    : {folder_list if len(folder_list) > 1 else folder_list[0]}")
    print(f"  training-metric files (OOD/DTW/action-change): {len(train_files)} "
          f"{'[proc]' if proc_files else '[full — no proc found]'}")
    print(f"  collection-metric files (timing/summary):      {len(full_files)} [full]")
    print(f"{'─'*60}")

    # Raw data points for the paper figures, each tagged with the episode it came
    # from: the violin/sigma use the points, the CI resamples whole episodes.
    PAPER_KEYS = ("ood_intervention", "ood_all",
                  "ood_oa_intervention", "ood_oa_all",
                  "ood_ja_intervention", "ood_ja_all", "dtw_intervention", "dtw_full",
                  "frechet_intervention", "frechet_full",
                  "dtwfrechet_intervention", "dtwfrechet_full",
                  "dtwmean_intervention", "dtwmean_full",
                  "dtwwin_intervention", "dtwwin_full",
                  "frechetwin_intervention", "frechetwin_full",
                  "act_ro_to_iv", "act_iv_to_ro", "timing_starts", "timing_ends",
                  "jacc_ro_to_iv", "jacc_iv_to_ro",
                  "cmdlin_ro_to_iv", "cmdlin_iv_to_ro",
                  "cmdang_ro_to_iv", "cmdang_iv_to_ro",
                  "timing_start_fracs", "timing_end_fracs",
                  "timing_group_lengths", "timing_seg_start_fracs",
                  "timing_seg_end_fracs", "timing_seg_lengths", "act_during",
                  "timing_first_starts", "timing_last_ends",
                  "timing_first_start_fracs", "timing_last_end_fracs",
                  "resp_latency_cmd", "resp_latency_ach", "resp_stall_event_steps")
    pooled = {k: {"values": [], "episode": []} for k in PAPER_KEYS}

    # One id per (folder, demo), so the collection pass over the full files and the training pass
    # over the proc files land on the same episode rather than numbering independently.
    _local_ids: dict = {}

    def ep_id_for(fol, demo, fpath=None):
        if registry is not None:
            return registry.id_of(fol, demo, method or algo_label, fpath)
        key = (os.path.abspath(fol), demo)
        return _local_ids.setdefault(key, len(_local_ids))

    def record(ep_id, values):
        """Per-episode values for the export; a no-op when nothing is exporting."""
        if registry is not None:
            registry.update(ep_id, values)

    def add_points(key, values, episode_id):
        vals = np.asarray(values, dtype=np.float64).ravel()
        if vals.size:
            pooled[key]["values"].append(vals)
            pooled[key]["episode"].append(np.full(vals.size, episode_id, dtype=np.int64))

    iter_results, iter_knn, iter_dtw, iter_timing, iter_jerk, iter_jjerk = {}, {}, {}, {}, {}, {}
    pool_knn    = {cat: [] for cat in ("all", "intervention", "rollout")}
    pool_dtw    = {cat: [] for cat in ("full", "intervention_segs", "rollout_segs")}
    pool_timing = {k: [] for k in (
        "segment_starts", "segment_ends", "segment_lengths",
        "segment_start_fracs", "segment_end_fracs",
        "group_starts", "group_ends", "group_lengths",
        "group_n_segments", "group_start_fracs", "group_end_fracs",
    )}
    _jerk_cats = (
        "ro_to_iv", "iv_to_ro", "non_transition",
        "train_ro_to_iv", "train_iv_to_ro", "train_non_transition",
    )
    pool_jerk   = {cat: [] for cat in _jerk_cats}
    pool_jjerk  = {cat: [] for cat in _jerk_cats}
    # Keyed off whatever compute_response returns, so adding a metric there needs no edit here.
    pool_resp: dict[str, list] = {}

    # Arbitration is a Blended-family quantity: HG-DAgger's handover is a switch, so an alpha
    # for it would be a measurement of something the method does not do.
    is_blended = "blend" in f"{algo_label} {method or ''}".lower()

    # ---- Collection pass over FULL files: intervention timing + meta summary ----
    timing_per_iter = []
    jerk_per_iter = []
    meta_rows = []
    # Keyed by (folder, round): sim keeps every round in one folder, so the folder alone hands
    # rounds 2+ round 1's file and their OOD/DTW rows land on no episode.
    folder_full: dict[tuple, str] = {
        (os.path.abspath(fol), export.session_round(fol, f)[1]): f
        for fol, f in zip(full_folders, full_files)}
    if not want_metric("collection"):
        full_files = []           # only_metrics=ood/trajsim: skip timing + action change
        print("  [only_metrics] skipping the collection pass (timing + action change)")
    # {abspath(collection file): {hdf5 demo name: meta.json key}} — identical on the real robots,
    # a positional pairing in sim, where the two files number episodes differently.
    canon: dict[str, dict] = {}
    # Which round each episode belongs to, so a figure can restrict itself to one. Intervention
    # timing is only comparable within a round: it is measured against the episode itself, so
    # from round 2 on the only thing making rounds differ is that each method is running a policy
    # finetuned on its own data, and a pooled violin compares different policies.
    ep_rounds: dict[int, int] = {}
    # The same raw segment/group arrays kept per round, so the diagnostic timing figure can be
    # split the same way the paper one is instead of pooling policies.
    timing_by_round: dict[int, dict[str, list]] = {}
    for f_idx, fpath in enumerate(full_files):
        fol = full_folders[f_idx]
        _, file_round = export.session_round(fol, fpath)
        demos_full = load_collected_states(fpath, obs_keys)
        meta_path = meta_path_for(fpath)
        meta = {}
        if meta_path:
            with open(meta_path) as mf:
                meta = json.load(mf)
            meta, n_corr, n_tot = task_time.corrected_task_times(fpath, meta)
            if n_corr:
                print(f"  [task_time] recomputed from grip trigger for {n_corr}/{n_tot} demos")
        names = export.canonical_names([d["demo"] for d in demos_full], list(meta))
        canon[os.path.abspath(fpath)] = names
        renamed = sum(1 for k, v in names.items() if k != v)
        if renamed:
            print(f"    [demos] the collection file numbers {renamed} episode(s) differently from "
                  f"its meta.json — paired in order so both sets of metrics land on one row")
        gap = gap_timesteps(cfg, control_hz)
        print(f"\n  [collection] {os.path.basename(fpath)}: intervention timing "
              f"(gap {gap} steps = {gap / control_hz:.2f}s) ...", end=" ", flush=True)
        timing_results = (itiming.compute_timing(demos_full, gap_threshold=gap)
                          if want_metric("collection") else
                          {k: np.array([]) for k in pool_timing})
        print("done.")
        timing_per_iter.append(timing_results)
        iter_timing[f"iter{f_idx + 1}"] = timing_results
        for k in pool_timing:
            pool_timing[k].append(timing_results[k])
        if file_round is not None:
            bucket = timing_by_round.setdefault(file_round, {k: [] for k in pool_timing})
            for k in pool_timing:
                bucket[k].append(timing_results[k])
        # Intervention responsiveness: latency to follow the human, stalls, input magnitude.
        # Same gap and grip window as the timing above, so an "intervention" is the same object.
        resp_full = iresp.compute_response(
            demos_full, hz=control_hz, gap_threshold=gap,
            action_scale=float(cfg.get("collection_action_scale", 0.8)))
        for k, v in resp_full.items():
            pool_resp.setdefault(k, []).append(v)
        # Action discontinuity uses the full file's executed commands, not proc's absolute pose.
        jerk_full = tjerk.compute_transition_jerk(demos_full)
        jerk_per_iter.append(jerk_full)
        iter_jerk[f"iter{f_idx + 1}"] = jerk_full
        for cat in pool_jerk:
            pool_jerk[cat].append(jerk_full.get(cat, np.array([])))

        # One point per merged intervention group / transition, tagged with its episode.
        during = tjerk.compute_during_intervention_delta(demos_full)
        # Keyed on the collection file's demo names, so an episode's training columns are its own.
        # A name that is absent means that episode is not in the training set at all.
        proc_by_demo = _proc_demos(fol, obs_keys, fpath)
        if proc_by_demo:
            print(f"    [training file] {len(proc_by_demo)} episode(s) also measured on proc_*")
        for d_idx, d in enumerate(demos_full):
            ep_id = ep_id_for(fol, names.get(d["demo"], d["demo"]), fpath)
            if file_round is not None:
                ep_rounds[ep_id] = file_round
            one = itiming.compute_timing([d], gap_threshold=gap)
            add_points("timing_starts", one.get("group_starts", []), ep_id)
            add_points("timing_ends", one.get("group_ends", []), ep_id)
            add_points("timing_start_fracs", one.get("group_start_fracs", []), ep_id)
            add_points("timing_end_fracs", one.get("group_end_fracs", []), ep_id)
            add_points("timing_group_lengths", one.get("group_lengths", []), ep_id)
            add_points("timing_seg_start_fracs", one.get("segment_start_fracs", []), ep_id)
            add_points("timing_seg_end_fracs", one.get("segment_end_fracs", []), ep_id)
            add_points("timing_seg_lengths", one.get("segment_lengths", []), ep_id)
            for _k, _src, _pick in (("timing_first_starts", "group_starts", _first),
                                    ("timing_last_ends", "group_ends", _last),
                                    ("timing_first_start_fracs", "group_start_fracs", _first),
                                    ("timing_last_end_fracs", "group_end_fracs", _last)):
                _v = _pick(one.get(_src, []))
                if _v is not None:
                    add_points(_k, [_v], ep_id)
            ep_jerk = tjerk.compute_transition_jerk([d])
            add_points("act_ro_to_iv", ep_jerk.get("ro_to_iv", []), ep_id)
            add_points("act_iv_to_ro", ep_jerk.get("iv_to_ro", []), ep_id)
            add_points("act_during", [during[d_idx]], ep_id)
            ep_ja = None
            if state.ANGLE_SPACE:
                ep_ja = tjerk.compute_joint_accel([d], hz=float(cfg.get("control_hz", 30)))
                add_points("jacc_ro_to_iv", ep_ja.get("ro_to_iv", []), ep_id)
                add_points("jacc_iv_to_ro", ep_ja.get("iv_to_ro", []), ep_id)
            ep_cmd = tjerk.compute_command_delta(
                [d], action_scale=float(cfg.get("collection_action_scale", 0.8)))
            for pre_, src in (("cmdlin", "lin"), ("cmdang", "ang")):
                for suf in ("ro_to_iv", "iv_to_ro"):
                    add_points(f"{pre_}_{suf}", ep_cmd.get(f"{src}_{suf}", []), ep_id)
            record(ep_id, _timing_row(one, control_hz))
            record(ep_id, _action_row(ep_jerk, ep_cmd, ep_ja, during[d_idx]))
            proc_demo = proc_by_demo.get(d["demo"])
            if proc_demo is not None:
                record(ep_id, _training_action_row(proc_demo, cfg, control_hz))
            record(ep_id, _alpha_row(d, is_blended))
            ep_resp = iresp.compute_response(
                [d], hz=control_hz, gap_threshold=gap,
                action_scale=float(cfg.get("collection_action_scale", 0.8)))
            for _k, _src in (("resp_latency_cmd", "latency_cmd"),
                             ("resp_latency_ach", "latency_ach"),
                             ("resp_stall_event_steps", "stall_event_steps")):
                add_points(_k, ep_resp[_src], ep_id)
            record(ep_id, iresp.episode_row(
                d, hz=control_hz, gap_threshold=gap,
                action_scale=float(cfg.get("collection_action_scale", 0.8))))
        if meta:
            # Collection summary excludes aborted attempts and restarts, matching the
            # "completed" set of the collection metrics (only *_accumulated metrics keep them).
            # Sim nests the stats under "summary"; real-robot metas are flat.
            meta_rows.extend(
                v.get("summary", v) for k, v in meta.items()
                if isinstance(v, dict) and not v.get("aborted", False) and "restart" not in k
            )
            # The export keeps EVERY episode, aborted attempts and restart rows included, and
            # flags them: downstream analysis reports each collection metric both ways and can
            # only do that if both are on the table. It reads the flag, not the key shape.
            if registry is not None:
                prov = export.provenance(fol, fpath, meta)
                for demo, row in export.meta_rows(meta).items():
                    ep_id = ep_id_for(fol, demo, fpath)
                    record(ep_id, dict(export.collection_values(row), **prov))
                # The meta counts the mask as recorded. Under merge_mask the filled pauses are
                # part of the intervention, so the count follows the mask every other metric read.
                if MERGE_MASK_GAP:
                    for d in demos_full:
                        record(ep_id_for(fol, names.get(d["demo"], d["demo"]), fpath),
                               {"num_intervention_steps": int(d["intervention_mask"].sum())})
    collection = _summarize_collection(meta_rows) if meta_rows else None

    # ---- Training pass over PROC (or full-fallback) files ----
    if not (want_metric("ood") or want_metric("trajsim")):
        train_files = []          # only_metrics=collection: skip the kNN/DTW pass
        print("  [only_metrics] skipping the training pass (OOD + trajectory)")
    # The reference grows by one round each iteration: round N is measured against pretrain plus
    # rounds 1..N-1 of THIS method, which is what the policy that ran round N was trained on.
    ref_states = [pretrain_states]
    ref_trajs = list(pretrain_trajs)
    ref_sources = [pretrain_label]
    references: dict[str, dict] = {}
    # Many folders can share one reference — every participant's round 2 is measured against the
    # same expert data — so an index is fitted once per distinct source list.
    ref_cache: dict[tuple, RoundReference] = {}
    for iter_idx, fpath in enumerate(train_files):
        iter_label = f"iter{iter_idx + 1}"
        fol = train_folders[iter_idx]
        session, rnd = export.session_round(fol, fpath)
        print(f"\n  Iteration {iter_idx + 1}/{len(train_files)}: {os.path.basename(fpath)}")

        demos = load_collected_states(fpath, obs_keys)
        if not demos:
            # Sim writes the raw states first and renders observations into a separate _im128
            # file later; a round whose render has not been run yet has no obs to measure. Skip
            # it rather than dying, so the rounds that ARE rendered still export.
            print(f"    [warn] no episode in {os.path.basename(fpath)} carries the obs keys "
                  f"{obs_keys} — skipping this round's OOD / trajectory metrics. Its collection "
                  f"metrics are unaffected.")
            continue
        # The processed file renumbers its episodes, so its demo_N is generally NOT the
        # collection file's demo_N. Recover the real correspondence from the frames before any
        # episode id is handed out, or the kNN distance lands on another episode's row.
        full_src = folder_full.get((os.path.abspath(fol), rnd))
        name_map = match_processed_demos(fpath, full_src, obs_keys)
        # proc name -> collection-file name -> meta.json name. The middle step is the renumbering
        # data_processing does; the last is sim numbering its hdf5 and its meta differently.
        canon_src = os.path.abspath(full_src or fpath)
        if canon_src not in canon:
            # The collection pass was skipped (only_metrics), so its naming was never recorded.
            src_meta = meta_path_for(canon_src)
            keys = list(json.load(open(src_meta))) if src_meta else []
            canon[canon_src] = export.canonical_names(
                [d["demo"] for d in load_collected_states(canon_src, obs_keys)], keys)
        to_meta = canon[canon_src]
        if full_src and os.path.abspath(full_src) != os.path.abspath(fpath):
            moved = sum(1 for k, v in name_map.items() if k != v)
            if moved:
                print(f"    [demos] matched all {len(name_map)} processed episodes to the "
                      f"collection file by frame content ({moved} of them are renumbered, "
                      f"so their demo_N differs between the two files)")
            # An EMPTY map means matching failed, not that the names agree, so the check runs
            # whenever there is a collection file to match against.
            unmatched = [d["demo"] for d in demos if d["demo"] not in name_map]
            if unmatched:
                print(f"    [warn] {len(unmatched)}/{len(demos)} processed episode(s) could not "
                      f"be matched to a collection episode ({unmatched[:4]}); their rows keep "
                      f"the processed name, so their OOD / trajectory values may sit on the "
                      f"wrong row and they may carry no collection metrics")

        all_states = np.concatenate([d["states"] for d in demos], axis=0)
        all_iv     = np.concatenate([d["intervention_mask"] for d in demos], axis=0)
        n_iv, n_total = int(all_iv.sum()), len(all_states)
        print(f"    Loaded {len(demos)} demos  |  {n_total} timesteps  "
              f"({n_iv} intervention, {n_total - n_iv} rollout)")

        if reference_mode == REFERENCE_DECLARED:
            extra = list((declared_refs or {}).get(rnd, ())) or \
                list((declared_refs or {}).get(str(rnd), ()))
            sources = [pretrain_label] + [os.path.basename(os.path.normpath(p)) for p in extra]
            ref = ref_cache.get(tuple(sources))
            if ref is None and not extra:
                ref = base_ref                 # nothing declared for this round: the base policy
            elif ref is None:
                print(f"    [OOD] Fitting the declared reference "
                      f"({', '.join(sources)}) ...", end=" ", flush=True)
                loaded = [load_reference_source(p, obs_keys) for p in extra]
                ref = RoundReference(
                    np.concatenate([pretrain_states] + [s for s, _ in loaded], axis=0),
                    list(pretrain_trajs) + [t for _, ts in loaded for t in ts],
                    int(cfg.ood.k_neighbors), sources)
                print(f"done. {ref.n_states} states, p95={ref.tau[95]:.4f}")
            ref_cache[tuple(sources)] = ref
        elif reference_mode != REFERENCE_POLICY or len(ref_states) == 1:
            ref = base_ref                    # round 1's reference IS the pretrain set
        else:
            print(f"    [OOD] Fitting the round's own reference "
                  f"({len(ref_sources)} sources) ...", end=" ", flush=True)
            ref = RoundReference(np.concatenate(ref_states, axis=0), ref_trajs,
                                 int(cfg.ood.k_neighbors), ref_sources)
            print(f"done. {ref.n_states} states, p95={ref.tau[95]:.4f}")
        references[f"{method or algo_label}/{rnd if rnd is not None else iter_idx + 1}"] = \
            ref.summary()
        mean, std, nn = ref.mean, ref.std, ref.nn
        pretrain_trajs_norm = ref.trajs_norm

        # ---- OOD (calls ood_measures) ----
        states_norm = ood.normalize(all_states, mean, std)
        print(f"    [OOD] Computing kNN scores (k={cfg.ood.k_neighbors}) "
              f"for {n_total} timesteps ...", end=" ", flush=True)
        knn_scores = ood.compute_ood_stats(states_norm, all_iv, nn)
        print("done.")
        print(f"          all={knn_scores['all'].mean():.4f}±{knn_scores['all'].std():.4f}"
              f"  | intervention={knn_scores['intervention'].mean():.4f}"
              f"±{knn_scores['intervention'].std():.4f}"
              f"  | rollout={knn_scores['rollout'].mean():.4f}"
              f"±{knn_scores['rollout'].std():.4f}")

        # Joint-space obs+action: configuration plus realised joint velocity.
        knn_ja = None
        if nn_ja is not None and state.ANGLE_SPACE:
            ja = np.concatenate([joint_state_vel(d["states"], control_hz) for d in demos],
                                axis=0)
            knn_ja = ood.knn_ood_scores(ja, nn_ja)

        nn_oa, mean_oa, std_oa = oa_index
        knn_oa = None
        if nn_oa is not None and demos and demos[0].get("actions") is not None:
            acts = np.concatenate([d["actions"] for d in demos], axis=0)
            if acts.shape[0] == all_states.shape[0] and \
               all_states.shape[1] + acts.shape[1] == len(mean_oa):
                oa = ood.normalize(np.concatenate([all_states, acts], axis=1),
                                   mean_oa, std_oa)
                knn_oa = ood.knn_ood_scores(oa, nn_oa)
            else:
                print("    [OOD] obs+action width mismatch — skipping for this file.")

        iter_knn[iter_label] = knn_scores
        for cat in pool_knn:
            pool_knn[cat].append(knn_scores[cat])

        ood_stats = {cat: agg(knn_scores[cat]) for cat in ("all", "intervention", "rollout")}

        # ---- Trajectory DTW (calls traj_similarity) ----
        dtw_results = {}
        if traj.HAS_DTW and cfg.trajectory.analyze_segments is not False:
            n_ref = len(pretrain_trajs_norm)
            min_len = int(cfg.trajectory.min_segment_length)
            print(f"    [DTW] {len(demos)} trajectories × {n_ref} pretrain refs "
                  f"(+{len(demos)} segment sets, min_len={min_len}) ...")
            dtw_results = {} if not want_metric("trajsim") else traj.compute_demo_dtw(
                demos, pretrain_trajs_norm, mean, std,
                min_segment_length=min_len,
                desc=f"{algo_label} iter{iter_idx + 1}",
            )
            for cat, arr in dtw_results.items():
                if cat.endswith("_demo") or "frechet" in cat:
                    continue           # index arrays / reported separately
                valid = arr[np.isfinite(arr)]
                label = cat.replace("_segs", " segs")
                print(f"          {label}: n={len(valid)}, "
                      f"mean={valid.mean():.4f}±{valid.std():.4f}" if len(valid) else
                      f"          {label}: n=0")
            for cat in pool_dtw:
                pool_dtw[cat].append(dtw_results.get(cat, np.array([])))
        else:
            dtw_results = {cat: np.array([]) for cat in pool_dtw}

        iter_dtw[iter_label] = dtw_results

        # ---- Per-episode values for the paper figures ----
        offsets = np.concatenate([[0], np.cumsum([len(d["states"]) for d in demos])])
        knn_all = knn_scores["all"]
        seg_dtw = dtw_results.get("intervention_segs", np.array([]))
        seg_of  = dtw_results.get("intervention_segs_demo", np.array([], dtype=np.int64))
        full_dtw = dtw_results.get("full", np.array([]))
        seg_fre = dtw_results.get("frechet_intervention_segs", np.array([]))
        full_fre = dtw_results.get("frechet_full", np.array([]))
        seg_dfr = dtw_results.get("dtwfrechet_intervention_segs", np.array([]))
        full_dfr = dtw_results.get("dtwfrechet_full", np.array([]))
        seg_dmn = dtw_results.get("dtwmean_intervention_segs", np.array([]))
        full_dmn = dtw_results.get("dtwmean_full", np.array([]))
        seg_dwn = dtw_results.get("dtwwin_intervention_segs", np.array([]))
        full_dwn = dtw_results.get("dtwwin_full", np.array([]))
        seg_fwn = dtw_results.get("frechetwin_intervention_segs", np.array([]))
        full_fwn = dtw_results.get("frechetwin_full", np.array([]))
        for d_idx, d in enumerate(demos):
            _src = name_map.get(d["demo"], d["demo"])
            ep_id = ep_id_for(fol, to_meta.get(_src, _src), fpath)
            iv = d["intervention_mask"]
            ep_knn = knn_all[offsets[d_idx]:offsets[d_idx + 1]]
            # One value per intervention: per-timestep rows would oversample the same event.
            runs = _contiguous_runs(iv)
            add_points("ood_intervention", [ep_knn[r].mean() for r in runs], ep_id)
            add_points("ood_all", ep_knn, ep_id)          # whole episode
            if knn_oa is not None:
                ep_oa = knn_oa[offsets[d_idx]:offsets[d_idx + 1]]
                add_points("ood_oa_intervention", [ep_oa[r].mean() for r in runs], ep_id)
                add_points("ood_oa_all", ep_oa, ep_id)
            if knn_ja is not None:
                ep_ja = knn_ja[offsets[d_idx]:offsets[d_idx + 1]]
                add_points("ood_ja_intervention", [ep_ja[r].mean() for r in runs], ep_id)
                add_points("ood_ja_all", ep_ja, ep_id)
            if len(seg_dtw):
                add_points("dtw_intervention", seg_dtw[seg_of == d_idx], ep_id)  # per segment
            if d_idx < len(full_dtw):
                add_points("dtw_full", [full_dtw[d_idx]], ep_id)       # one per episode
            if len(seg_fre):
                add_points("frechet_intervention", seg_fre[seg_of == d_idx], ep_id)
            if d_idx < len(full_fre):
                add_points("frechet_full", [full_fre[d_idx]], ep_id)
            if len(seg_dfr):
                add_points("dtwfrechet_intervention", seg_dfr[seg_of == d_idx], ep_id)
            if d_idx < len(full_dfr):
                add_points("dtwfrechet_full", [full_dfr[d_idx]], ep_id)
            if len(seg_dmn):
                add_points("dtwmean_intervention", seg_dmn[seg_of == d_idx], ep_id)
            if d_idx < len(full_dmn):
                add_points("dtwmean_full", [full_dmn[d_idx]], ep_id)
            if len(seg_dwn):
                add_points("dtwwin_intervention", seg_dwn[seg_of == d_idx], ep_id)
            if d_idx < len(full_dwn):
                add_points("dtwwin_full", [full_dwn[d_idx]], ep_id)
            if len(seg_fwn):
                add_points("frechetwin_intervention", seg_fwn[seg_of == d_idx], ep_id)
            if d_idx < len(full_fwn):
                add_points("frechetwin_full", [full_fwn[d_idx]], ep_id)
            iv_scores = (np.concatenate([ep_knn[r] for r in runs]) if runs
                         else np.array([], dtype=np.float64))
            record(ep_id, _deviation_row(ep_knn, [ep_knn[r].mean() for r in runs], iv_scores, ref,
                                         full_dtw, full_fre, seg_dtw, seg_fre, seg_of, d_idx))

        # Grow the reference: the next round's policy is trained on this round's data too.
        if reference_mode == REFERENCE_POLICY:
            ref_states.append(all_states)
            ref_trajs = ref_trajs + [d["states"] for d in demos]
            ref_sources = ref_sources + [session]

        dtw_stats = {
            "full_trajectory":      agg(dtw_results.get("full",              np.array([]))),
            "intervention_segments": agg(dtw_results.get("intervention_segs", np.array([]))),
            "rollout_segments":      agg(dtw_results.get("rollout_segs",      np.array([]))),
        }

        # ---- Intervention timing: taken from the FULL-file collection pass ----
        timing_stats = (
            {k: agg(timing_per_iter[iter_idx][k]) for k in timing_per_iter[iter_idx]}
            if iter_idx < len(timing_per_iter) else {}
        )

        # ---- Transition jerk: taken from the FULL-file collection pass ----
        jerk_results = (jerk_per_iter[iter_idx] if iter_idx < len(jerk_per_iter)
                        else {c: np.array([]) for c in pool_jerk})
        _jlabels = {
            "ro_to_iv":             "ro→iv       (all)  ",
            "iv_to_ro":             "iv→ro       (all)  ",
            "non_transition":       "baseline    (all)  ",
            "train_ro_to_iv":       "ro→iv       (train)",
            "train_iv_to_ro":       "iv→ro       (train)",
            "train_non_transition": "baseline    (train)",
        }
        for jtype, arr in jerk_results.items():
            label = _jlabels.get(jtype, jtype)
            print(f"           {label}: n={len(arr)}, "
                  f"mean={arr.mean():.4f}±{arr.std():.4f}" if len(arr) else
                  f"           {label}: n=0")

        jerk_stats = {cat: agg(jerk_results[cat]) for cat in jerk_results}

        # ---- Joint-space jerk from sim.data.qacc (calls joint_jerk) ----
        print(f"    [JOINT JERK] Joint-acceleration change magnitude ...", end=" ", flush=True)
        jjerk_results = jjerk.compute_joint_jerk(demos)
        if jjerk_results is not None:
            print("done.")
            for jtype, arr in jjerk_results.items():
                label = _jlabels.get(jtype, jtype)
                print(f"           {label}: n={len(arr)}, "
                      f"mean={arr.mean():.4f}±{arr.std():.4f}" if len(arr) else
                      f"           {label}: n=0")
            for cat in pool_jjerk:
                pool_jjerk[cat].append(jjerk_results[cat])
        else:
            print("skipped (no joint_acc in dataset).")

        iter_jjerk[iter_label] = jjerk_results
        jjerk_stats = (
            {cat: agg(jjerk_results[cat]) for cat in jjerk_results}
            if jjerk_results is not None else None
        )

        iter_results[iter_label] = dict(
            file=os.path.basename(fpath),
            n_demos=len(demos),
            n_timesteps=int(n_total),
            n_intervention=int(n_iv),
            n_rollout=int(n_total - n_iv),
            ood_knn=ood_stats,
            dtw=dtw_stats,
            timing=timing_stats,
            jerk=jerk_stats,
            joint_jerk=jjerk_stats,
        )

    # Aggregate across all iterations
    def _cat_agg(pool):
        arr = np.concatenate(pool) if pool else np.array([])
        return agg(arr)

    _jjerk_has_data = any(len(pool_jjerk[cat]) > 0 for cat in pool_jjerk)
    algo_aggregate = dict(
        ood_knn={cat: _cat_agg(pool_knn[cat]) for cat in pool_knn},
        dtw=dict(
            full_trajectory=      _cat_agg(pool_dtw["full"]),
            intervention_segments=_cat_agg(pool_dtw["intervention_segs"]),
            rollout_segments=     _cat_agg(pool_dtw["rollout_segs"]),
        ),
        timing={k: _cat_agg(pool_timing[k]) for k in pool_timing},
        jerk={cat: _cat_agg(pool_jerk[cat]) for cat in pool_jerk},
        joint_jerk=(
            {cat: _cat_agg(pool_jjerk[cat]) for cat in pool_jjerk}
            if _jjerk_has_data else None
        ),
        collection=collection,
        # Families skipped by only_metrics report as n=0 instead of missing keys.
        response=iresp.summarise(
            collections.defaultdict(
                lambda: np.array([]),
                {k: (np.concatenate(v) if v else np.array([])) for k, v in pool_resp.items()}),
            hz=control_hz),
    )

    print(f"\n  [{algo_label}] Aggregate across {len(train_files)} training file(s), "
          f"{len(full_files)} collection file(s):")
    print(f"    OOD:    all={_fmt(algo_aggregate['ood_knn']['all'])}"
          f"  | interv={_fmt(algo_aggregate['ood_knn']['intervention'])}"
          f"  | rollout={_fmt(algo_aggregate['ood_knn']['rollout'])}")
    if traj.HAS_DTW:
        print(f"    DTW:    full={_fmt(algo_aggregate['dtw']['full_trajectory'])}"
              f"  | interv segs={_fmt(algo_aggregate['dtw']['intervention_segments'])}"
              f"  | rollout segs={_fmt(algo_aggregate['dtw']['rollout_segments'])}")
    t = algo_aggregate["timing"]
    print(f"    TIMING: seg start={_fmt(t['segment_start_fracs'])} (frac)"
          f"  seg end={_fmt(t['segment_end_fracs'])}")
    print(f"            grp start={_fmt(t['group_start_fracs'])} (frac)"
          f"  grp end={_fmt(t['group_end_fracs'])}")
    j = algo_aggregate["jerk"]
    print(f"    ACTION DIFF (all):   ro→iv={_fmt(j['ro_to_iv'])}"
          f"  | iv→ro={_fmt(j['iv_to_ro'])}"
          f"  | baseline={_fmt(j['non_transition'])}")
    print(f"    ACTION DIFF (train): ro→iv={_fmt(j['train_ro_to_iv'])}"
          f"  | iv→ro={_fmt(j['train_iv_to_ro'])}"
          f"  | baseline={_fmt(j['train_non_transition'])}")
    if algo_aggregate["joint_jerk"] is not None:
        jj = algo_aggregate["joint_jerk"]
        print(f"    JOINT JERK  (all):   ro→iv={_fmt(jj['ro_to_iv'])}"
              f"  | iv→ro={_fmt(jj['iv_to_ro'])}"
              f"  | baseline={_fmt(jj['non_transition'])}")
        print(f"    JOINT JERK  (train): ro→iv={_fmt(jj['train_ro_to_iv'])}"
              f"  | iv→ro={_fmt(jj['train_iv_to_ro'])}"
              f"  | baseline={_fmt(jj['train_non_transition'])}")
    if collection is not None:
        c = collection
        print(f"    COLLECTION: success={c['success_rate']*100:.0f}%"
              f"  task_time={_fmt(c['task_time_s'])}s"
              f"  tot_interv_mag={_fmt(c['total_interv_mag'])}"
              f"  #interv/demo={_fmt(c['num_interventions'])}")
    r = algo_aggregate["response"]
    if r["latency_cmd_s"]["n"]:
        print(f"    RESPONSE: latency cmd med={r['latency_cmd_s']['median']:.2f}s "
              f"max={r['latency_cmd_s']['max']:.2f}s"
              f"  | achieved med={r['latency_ach_s']['median']:.2f}s "
              f"max={r['latency_ach_s']['max']:.2f}s"
              f"  | never aligned={r['unresolved_ach']*100:.0f}%"
              f"  | stalled={r['stall_frac']['mean']*100:.1f}% of commanding steps"
              f"  | max interv mag={r['max_magnitude']['mean']:.2f}")

    pooled = {
        k: {"values": (np.concatenate(v["values"]) if v["values"] else np.array([])),
            "episode": (np.concatenate(v["episode"]) if v["episode"]
                        else np.array([], dtype=np.int64))}
        for k, v in pooled.items()
    }
    timing_by_round = {r: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()}
                       for r, d in timing_by_round.items()}
    return (iter_results, iter_knn, iter_dtw, iter_timing, iter_jerk, iter_jjerk,
            algo_aggregate, pooled, references, ep_rounds, timing_by_round)


# ===========================================================================
# Comparison summary table
# ===========================================================================

def print_comparison_table(
    pretrain_knn_stats: dict,
    algo_aggregate: dict[str, dict],
):
    algos  = list(algo_aggregate.keys())
    col_w  = 24
    header = f"{'Metric':<34}" + "".join(f"{a:^{col_w}}" for a in ["pretrain (ref)"] + algos)
    sep    = "─" * len(header)

    def row(label, stats_list):
        cells = [_fmt(s) for s in stats_list]
        print(f"  {label:<32}" + "".join(f"{c:^{col_w}}" for c in cells))

    print(f"\n{'='*len(header)}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(sep)
    print(header)
    print(sep)

    row("OOD kNN — all",          [pretrain_knn_stats] + [algo_aggregate[a]["ood_knn"]["all"]          for a in algos])
    row("OOD kNN — intervention", [pretrain_knn_stats] + [algo_aggregate[a]["ood_knn"]["intervention"] for a in algos])
    row("OOD kNN — rollout",      [pretrain_knn_stats] + [algo_aggregate[a]["ood_knn"]["rollout"]      for a in algos])

    if traj.HAS_DTW:
        print(sep)
        row("DTW — full trajectory",  [{"mean": None}] + [algo_aggregate[a]["dtw"]["full_trajectory"]        for a in algos])
        row("DTW — interv. segments", [{"mean": None}] + [algo_aggregate[a]["dtw"]["intervention_segments"]  for a in algos])
        row("DTW — rollout segments", [{"mean": None}] + [algo_aggregate[a]["dtw"]["rollout_segments"]       for a in algos])

    print(sep)
    row("Seg start (frac episode)", [{"mean": None}] + [algo_aggregate[a]["timing"]["segment_start_fracs"] for a in algos])
    row("Seg end   (frac episode)", [{"mean": None}] + [algo_aggregate[a]["timing"]["segment_end_fracs"]   for a in algos])
    row("Grp start (frac episode)", [{"mean": None}] + [algo_aggregate[a]["timing"]["group_start_fracs"]   for a in algos])
    row("Grp end   (frac episode)", [{"mean": None}] + [algo_aggregate[a]["timing"]["group_end_fracs"]     for a in algos])
    row("Seg length (timesteps)",   [{"mean": None}] + [algo_aggregate[a]["timing"]["segment_lengths"]     for a in algos])
    row("Grp length (timesteps)",   [{"mean": None}] + [algo_aggregate[a]["timing"]["group_lengths"]       for a in algos])

    print(sep)
    row("ActDiff — ro→iv (all)",      [{"mean": None}] + [algo_aggregate[a]["jerk"]["ro_to_iv"]              for a in algos])
    row("ActDiff — iv→ro (all)",      [{"mean": None}] + [algo_aggregate[a]["jerk"]["iv_to_ro"]              for a in algos])
    row("ActDiff — baseline (all)",   [{"mean": None}] + [algo_aggregate[a]["jerk"]["non_transition"]        for a in algos])
    row("ActDiff — ro→iv (train)",    [{"mean": None}] + [algo_aggregate[a]["jerk"]["train_ro_to_iv"]        for a in algos])
    row("ActDiff — iv→ro (train)",    [{"mean": None}] + [algo_aggregate[a]["jerk"]["train_iv_to_ro"]        for a in algos])
    row("ActDiff — baseline (train)", [{"mean": None}] + [algo_aggregate[a]["jerk"]["train_non_transition"]  for a in algos])

    if any(algo_aggregate[a].get("joint_jerk") is not None for a in algos):
        _none = {"mean": None}
        def _jj(a, key):
            jj = algo_aggregate[a].get("joint_jerk")
            return jj[key] if jj is not None else _none
        print(sep)
        row("JntJerk — ro→iv (all)",      [{"mean": None}] + [_jj(a, "ro_to_iv")              for a in algos])
        row("JntJerk — iv→ro (all)",      [{"mean": None}] + [_jj(a, "iv_to_ro")              for a in algos])
        row("JntJerk — baseline (all)",   [{"mean": None}] + [_jj(a, "non_transition")        for a in algos])
        row("JntJerk — ro→iv (train)",    [{"mean": None}] + [_jj(a, "train_ro_to_iv")        for a in algos])
        row("JntJerk — iv→ro (train)",    [{"mean": None}] + [_jj(a, "train_iv_to_ro")        for a in algos])
        row("JntJerk — baseline (train)", [{"mean": None}] + [_jj(a, "train_non_transition")  for a in algos])

    if any(algo_aggregate[a].get("collection") for a in algos):
        _none = {"mean": None}
        def _c(a, key):
            c = algo_aggregate[a].get("collection")
            return c[key] if c else _none
        def _rate(a):
            c = algo_aggregate[a].get("collection")
            return f"{c['success_rate']*100:.0f}%" if c else "n/a"
        print(sep)
        print(f"  {'Success rate (collection)':<32}" + f"{'':^{col_w}}"
              + "".join(f"{_rate(a):^{col_w}}" for a in algos))
        row("Task time (s)",             [{"mean": None}] + [_c(a, "task_time_s")      for a in algos])
        row("Total interv. magnitude",   [{"mean": None}] + [_c(a, "total_interv_mag") for a in algos])
        row("Mean interv. magnitude",    [{"mean": None}] + [_c(a, "mean_interv_mag")  for a in algos])
        row("Mean interv. mag (moving)", [{"mean": None}] + [_c(a, "mean_interv_mag_moving") for a in algos])
        row("# interventions / demo",    [{"mean": None}] + [_c(a, "num_interventions")for a in algos])
        row("# intervention steps",      [{"mean": None}] + [_c(a, "num_interv_steps") for a in algos])

    if any(algo_aggregate[a].get("response", {}).get("latency_cmd_s", {}).get("n")
           for a in algos):
        def _r(a, family, stat="median"):
            fam = algo_aggregate[a].get("response", {}).get(family)
            return f"{fam[stat]:.2f}" if isinstance(fam, dict) and np.isfinite(fam[stat]) else "n/a"
        def _pct(a, key):
            v = algo_aggregate[a].get("response", {}).get(key, np.nan)
            return f"{v*100:.0f}%" if np.isfinite(v) else "n/a"
        def line(label, cells):
            print(f"  {label:<32}" + f"{'':^{col_w}}"
                  + "".join(f"{c:^{col_w}}" for c in cells))
        print(sep)
        for stat in ("mean", "median", "max"):
            line(f"Response latency, cmd ({stat}, s)",
                 [_r(a, "latency_cmd_s", stat) for a in algos])
        for stat in ("mean", "median", "max"):
            line(f"Response latency, achieved ({stat}, s)",
                 [_r(a, "latency_ach_s", stat) for a in algos])
        line("Interventions never aligned", [_pct(a, "unresolved_ach") for a in algos])
        line("Stalled steps (frac of commanding)",
             [_r(a, "stall_frac", "mean") for a in algos])
        line("Stall events / min commanding",
             [_r(a, "stall_events_per_min", "mean") for a in algos])
        line("Stall event duration (mean, s)", [_r(a, "stall_event_s", "mean") for a in algos])
        line("Max interv. magnitude (per ep)", [_r(a, "max_magnitude", "mean") for a in algos])

    print(sep)
    print("(mean ± std; fracs: 0=episode start, 1=episode end; "
          "ActDiff = ||Δaction||₂; JntJerk = ||Δjoint_acc||₂; "
          "collection rows from *_meta.json of the full files; response rows are "
          "per-intervention latency and per-episode stall/magnitude — see "
          "intervention_response.py)")


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays for each contiguous True run — one run = one intervention."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def _save_distributions(
    output_dir: str,
    run_name: str,
    run_facts: dict,
    pretrain_knn_scores: np.ndarray,
    algo_knn_agg: dict,
    algo_dtw_agg: dict,
    algo_timing_agg: dict,
    algo_jerk_agg: dict,
    all_iter_knn: dict,
    all_iter_dtw: dict,
    runs: dict,
    algo_jjerk_agg: dict | None = None,
    all_per_demo: dict | None = None,
    pretrain_knn_per_demo: np.ndarray | None = None,
    all_ep_rounds: dict | None = None,
    all_timing_rounds: dict | None = None,
):
    """
    Save all raw distribution arrays to .npz so plots can be regenerated
    later without rerunning the expensive OOD/DTW computation.

    Writes:
        {output_dir}/{run_name}_distributions.npz  — all raw arrays
        {output_dir}/{run_name}_distributions_meta.json — algo/iter labels
    """
    algo_labels = list(runs.keys())
    arrays: dict[str, np.ndarray] = {"pretrain_knn": pretrain_knn_scores}
    if pretrain_knn_per_demo is not None:
        arrays["pretrain_knn_per_demo"] = pretrain_knn_per_demo["values"]
        arrays["pretrain_knn_per_demo_episode"] = pretrain_knn_per_demo["episode"]

    for algo in algo_labels:
        # Which round each episode came from, so plot_only can restrict a figure to one.
        rounds = (all_ep_rounds or {}).get(algo) or {}
        if rounds:
            arrays[f"{algo}__episode_round__episode"] = np.array(sorted(rounds), dtype=np.int64)
            arrays[f"{algo}__episode_round__round"] = np.array(
                [rounds[e] for e in sorted(rounds)], dtype=np.int64)
        if all_per_demo is not None:
            for k, entry in all_per_demo.get(algo, {}).items():
                arrays[f"{algo}__per_demo__{k}"] = entry["values"]
                arrays[f"{algo}__per_demo__{k}__episode"] = entry["episode"]
        for cat, arr in algo_knn_agg[algo].items():
            arrays[f"{algo}__knn__{cat}"] = arr
        for cat, arr in algo_dtw_agg[algo].items():
            arrays[f"{algo}__dtw__{cat}"] = arr
        for k, arr in algo_timing_agg[algo].items():
            arrays[f"{algo}__timing__{k}"] = arr
        for rnd, per_key in ((all_timing_rounds or {}).get(algo) or {}).items():
            for k, arr in per_key.items():
                arrays[f"{algo}__timing_r{rnd}__{k}"] = arr
        for k, arr in algo_jerk_agg[algo].items():
            arrays[f"{algo}__jerk__{k}"] = arr
        if algo_jjerk_agg is not None and algo_jjerk_agg.get(algo) is not None:
            for k, arr in algo_jjerk_agg[algo].items():
                arrays[f"{algo}__joint_jerk__{k}"] = arr
        for iter_label, knn_dict in all_iter_knn[algo].items():
            for cat, arr in knn_dict.items():
                arrays[f"{algo}__{iter_label}__knn__{cat}"] = arr
        for iter_label, dtw_dict in all_iter_dtw[algo].items():
            for cat, arr in dtw_dict.items():
                arrays[f"{algo}__{iter_label}__dtw__{cat}"] = arr

    npz_path = _cat_path(output_dir, "ood", "distributions.npz")
    # A partial recompute (only_metrics=) must not drop the metrics it skipped:
    # carry any key it did not produce over from the existing file.
    if ONLY_METRICS and os.path.exists(npz_path):
        def family(key: str) -> str:
            if "__per_demo__" in key:
                m = key.split("__per_demo__")[1]
                if m.startswith(("ood_",)):
                    return "ood"
                if m.startswith(("act_", "jacc_", "cmdlin_", "cmdang_", "timing_")):
                    return "collection"
                return "trajsim"
            if "__episode_round__" in key:
                return "collection"
            if key.startswith("pretrain_knn") or "__knn__" in key:
                return "ood"
            if "__dtw__" in key or "__traj" in key:
                return "trajsim"
            if "__jerk__" in key or "__joint_jerk__" in key or "__timing__" in key:
                return "collection"
            return "other"
        # Anything this run did not recompute is stale-but-correct in the old file;
        # anything it DID recompute for a skipped family would be empty, so drop it.
        arrays = {k: v for k, v in arrays.items()
                  if family(k) == "other" or family(k) in ONLY_METRICS}
        with np.load(npz_path) as prev:
            kept = [k for k in prev.files if k not in arrays]
            for k in kept:
                arrays[k] = prev[k]
        print(f"  [save] recomputed {len(arrays) - len(kept)}, carried over {len(kept)}")
    np.savez_compressed(npz_path, **arrays)

    meta = dict(
        run_name=run_name,
        algo_labels=algo_labels,
        iter_labels={algo: list(all_iter_knn[algo].keys()) for algo in algo_labels},
        has_joint_jerk={
            algo: (algo_jjerk_agg is not None and algo_jjerk_agg.get(algo) is not None)
            for algo in algo_labels
        },
        # The run's own facts, so a later replot need not rerun main() to recover them.
        **(run_facts or {}),
    )
    meta_path = _cat_path(output_dir, "ood", "distributions_meta.json")
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"  Distributions      → {npz_path}")
    print(f"  Distributions meta → {meta_path}")


# ===========================================================================
# Main
# ===========================================================================

def _write_alpha_sessions(cfg, runs):
    """Record per-session alpha facts for downstream plotting.

    Only for configs that opt in with an `alpha:` block -- the suite writes this
    file itself for the tasks it drives, so this covers the ones it does not.
    """
    label_map = dict(zip(cfg.plot.run_names, cfg.plot.run_labels))
    color_map = dict(zip(cfg.plot.run_names, cfg.plot.run_colors))
    alpha_dir = os.path.join(cfg.output_dir, "alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    sessions = []
    for algo, folder in runs.items():
        folders = [folder] if isinstance(folder, str) else list(folder)
        for fol in folders:
            for full in sorted(f for f in glob.glob(os.path.join(fol, "*_full.hdf5"))
                               if not os.path.basename(f).startswith("proc_")
                               and "_im128" not in f and "_trimmed" not in f):
                m = re.search(r"_round(\d+)", os.path.basename(full))
                sessions.append(dict(
                    algo=algo, round=int(m.group(1)) if m else None, full=full,
                    label=label_map.get(algo, algo), color=color_map.get(algo),
                    alpha_is_constant=bool(alpha_is_constant(full)),
                    has_blend_fields=bool(has_blend_fields(full)),
                ))
    z = cfg.alpha.zoom
    with open(os.path.join(alpha_dir, "sessions.json"), "w") as fh:
        json.dump(dict(task=cfg.get("run_name", cfg.env_name),
                       zoom=dict(demo=str(z.demo), seconds=z.seconds),
                       sessions=sessions), fh, indent=2)
    print(f"  alpha sessions → {os.path.join(alpha_dir, 'sessions.json')} "
          f"({len(sessions)} sessions)")


@hydra.main(config_path="../conf", config_name="analysis/sim/coffee", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("OOD & Trajectory Similarity Analysis")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    set_angle_keys(cfg.get("angle_keys", []))
    set_only_metrics(cfg.get("only_metrics", []))
    if bool(cfg.intervention_timing.get("merge_mask", False)):
        merged = gap_timesteps(cfg, float(cfg.get("control_hz", 30)))
        set_merge_mask_gap(merged)
        print(f"  [merge_mask] pauses of <= {merged} timesteps are filled in before any metric")
    set_angle_space(angle_space_for(cfg.get("obs_keys", []), ANGLE_KEYS))
    os.makedirs(cfg.output_dir, exist_ok=True)
    task_time.set_control_hz(cfg.get("control_hz", task_time.DEFAULT_CONTROL_HZ))

    if getattr(cfg, "plot_only", False):
        # No figures are drawn here; only the alpha manifest (which needs the hdf5) is refreshed.
        if cfg.get("alpha") is not None:
            _write_alpha_sessions(cfg, resolve_runs(cfg))
        print("plot_only: no figures are drawn here; plotting code reads the saved\n"
              f"  {os.path.join(cfg.output_dir, 'ood', 'distributions.npz')}.\n"
              "  The alpha manifest was refreshed.")
        return

    obs_keys = list(cfg.obs_keys)

    # ------------------------------------------------------------------ #
    # 1. Discover algorithm folders                                        #
    # ------------------------------------------------------------------ #
    runs = resolve_runs(cfg)
    if cfg.get("alpha") is not None:
        _write_alpha_sessions(cfg, runs)

    # ------------------------------------------------------------------ #
    # 2. Load pretrain dataset                                             #
    # ------------------------------------------------------------------ #
    pretrain_path = cfg.paths_db.lookup[cfg.model][cfg.env_name].pretrained_dataset
    print(f"\n{'='*60}")
    print(f"Pretrain dataset: {pretrain_path}")

    pretrain_all, pretrain_trajs, pretrain_acts = load_pretrain_states(pretrain_path, obs_keys)
    print(f"  {len(pretrain_trajs)} demos  |  {len(pretrain_all)} timesteps  "
          f"|  {pretrain_all.shape[1]}D state")

    if cfg.trajectory.max_pretrain_demos is not None:
        n_cap = int(cfg.trajectory.max_pretrain_demos)
        idxs  = np.random.default_rng(0).choice(len(pretrain_trajs), size=n_cap, replace=False)
        pretrain_trajs_dtw = [pretrain_trajs[i] for i in idxs]
        print(f"  DTW reference capped at {n_cap} demos (random subset, seed=0)")
    else:
        pretrain_trajs_dtw = pretrain_trajs
        print(f"  DTW reference: all {len(pretrain_trajs_dtw)} pretrain demos")

    # ------------------------------------------------------------------ #
    # 3. Fit normalizer and kNN (ood_measures)                            #
    # ------------------------------------------------------------------ #
    print(f"\n[OOD] Fitting normalizer on {len(pretrain_all)} pretrain timesteps ...", end=" ", flush=True)
    set_angle_space(angle_space_for(obs_keys, ANGLE_KEYS))
    if state.ANGLE_SPACE:
        # Pure joint-angle space: sin/cos are already commensurate and bounded, so
        # z-scoring would only re-weight joints by how much they happened to move.
        print("  [OOD] joint-angle (sin/cos) space — no z-scoring")
    print("done.")

    _declared = cfg.ood.get("reference_sets", None)
    declared = OmegaConf.to_container(_declared, resolve=True) if _declared else {}
    reference_mode = (REFERENCE_DECLARED if declared
                      else str(cfg.ood.get("reference", REFERENCE_POLICY)))
    pretrain_label = os.path.basename(os.path.dirname(str(pretrain_path))) or "pretrain"

    print(f"[OOD] Fitting kNN (k={cfg.ood.k_neighbors}) ...", end=" ", flush=True)
    base_ref = RoundReference(pretrain_all, pretrain_trajs_dtw, int(cfg.ood.k_neighbors),
                              [pretrain_label], episodes=pretrain_trajs)
    pretrain_knn_scores = base_ref.self_scores
    pretrain_knn_stats  = agg(pretrain_knn_scores)
    pretrain_traj_lengths = [len(t) for t in pretrain_trajs]
    print("done.")
    print(f"      Pretrain self-score (in-distribution baseline): {_fmt(pretrain_knn_stats)}")
    _ref_note = {REFERENCE_POLICY: "the policy training set (pretrain + rounds 1..N-1)",
                 REFERENCE_PRETRAIN: "the pretrain set, every round",
                 REFERENCE_DECLARED: "the declared ood.reference_sets for that round"}
    print(f"      Reference for round N: {_ref_note.get(reference_mode, reference_mode)}")

    # Third index: joint configuration + realised joint velocity, all in radians.
    nn_ja = None
    if state.ANGLE_SPACE:
        hz = float(cfg.get("control_hz", 30))
        ja_pre = np.concatenate([joint_state_vel(t, hz) for t in pretrain_trajs], axis=0)
        nn_ja = ood.fit_knn(ja_pre, k=int(cfg.ood.k_neighbors))
        print(f"  [OOD] joint angle+velocity index: {ja_pre.shape[1]} dims "
              f"({ja_pre.shape[1] // 3} joints)")

    # Second index over [obs | action]: kNN in the raw recorded space rather than
    # the policy's encoder space. Only built when the pretrain actions have the
    # same width as the collected ones (proc files on the real robot).
    nn_oa = mean_oa = std_oa = None
    if pretrain_acts is not None and len(pretrain_acts) == len(pretrain_all):
        oa = np.concatenate([pretrain_all, pretrain_acts], axis=1)
        mean_oa, std_oa = ood.fit_normalizer(oa)
        nn_oa = ood.fit_knn(ood.normalize(oa, mean_oa, std_oa), k=int(cfg.ood.k_neighbors))
        print(f"[OOD] Fitted obs+action kNN ({oa.shape[1]} dims: "
              f"{pretrain_all.shape[1]} obs + {pretrain_acts.shape[1]} action)")
    else:
        print("[OOD] No pretrain actions matching the state rows — skipping obs+action kNN.")

    # ------------------------------------------------------------------ #
    # 4. Process each algorithm                                            #
    # ------------------------------------------------------------------ #
    all_iter_results   = {}
    all_iter_knn       = {}
    all_iter_dtw       = {}
    all_iter_timing    = {}
    all_iter_jerk      = {}
    all_iter_jjerk     = {}
    all_algo_aggregate = {}
    all_per_demo       = {}
    all_ep_rounds      = {}
    all_timing_rounds  = {}
    all_references     = {}

    # The export leaves `participant` blank; downstream code maps sessions to participants.
    settings = export.export_settings(cfg, str(cfg.run_name), str(cfg.output_dir))
    # A partial recompute still exports: the families it skipped are carried over from the previous
    # episodes.csv, so the table is complete rather than half-filled.
    exporting = settings["enabled"]
    if settings["enabled"] and ONLY_METRICS:
        print(f"\n[export] only_metrics={sorted(ONLY_METRICS)}; the per-episode export keeps the "
              "previous values for every family this run did not recompute.")
    registry = export.EpisodeRegistry(
        settings["dataset"], settings["study"], settings["task"],
        settings["environment"]) if exporting else None

    for algo_label, folder in runs.items():
        method = export.method_of(algo_label, settings["methods"])
        (iter_results, iter_knn, iter_dtw, iter_timing, iter_jerk, iter_jjerk,
         algo_agg, per_demo, refs, ep_rounds, timing_rounds) = process_algo(
            algo_label, folder, obs_keys, base_ref, pretrain_all, pretrain_trajs_dtw, cfg,
            oa_index=(nn_oa, mean_oa, std_oa),
            nn_ja=nn_ja,
            control_hz=float(cfg.get('control_hz', 30)),
            registry=registry, method=method, pretrain_label=pretrain_label,
            reference_mode=reference_mode,
            declared_refs=declared.get(algo_label, declared.get(method)),
        )
        all_per_demo[algo_label]       = per_demo
        all_ep_rounds[algo_label]      = ep_rounds
        all_timing_rounds[algo_label]  = timing_rounds
        all_references.update(refs)

        all_iter_results[algo_label]   = iter_results
        all_iter_knn[algo_label]       = iter_knn
        all_iter_dtw[algo_label]       = iter_dtw
        all_iter_timing[algo_label]    = iter_timing
        all_iter_jerk[algo_label]      = iter_jerk
        all_iter_jjerk[algo_label]     = iter_jjerk
        all_algo_aggregate[algo_label] = algo_agg

    # Baseline on the same footing: each pretrain demo is one "episode".
    pretrain_knn_per_demo = {
        "values": pretrain_knn_scores,
        "episode": np.repeat(np.arange(len(pretrain_traj_lengths)), pretrain_traj_lengths),
    }

    # ------------------------------------------------------------------ #
    # 5. Print comparison table                                            #
    # ------------------------------------------------------------------ #
    print_comparison_table(pretrain_knn_stats, all_algo_aggregate)

    # ------------------------------------------------------------------ #
    # 6. Save JSON                                                         #
    # ------------------------------------------------------------------ #
    run_name = cfg.run_name

    # Convert numpy arrays to lists for JSON serialisation
    def _json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    results_out = _json_safe(dict(
        run_name=run_name,
        pretrain=dict(
            path=pretrain_path,
            n_demos=len(pretrain_trajs),
            n_timesteps=int(len(pretrain_all)),
            knn_self=pretrain_knn_stats,
        ),
        config=dict(
            obs_keys=obs_keys,
            k_neighbors=int(cfg.ood.k_neighbors),
            normalize=bool(cfg.ood.normalize) and not state.ANGLE_SPACE,  # effective value, matches the kNN call
        ),
        aggregate=all_algo_aggregate,
        iterations=all_iter_results,
    ))

    json_path = _cat_path(cfg.output_dir, "ood", "ood_results.json")
    with open(json_path, "w") as fp:
        json.dump(results_out, fp, indent=2)
    print(f"\nResults JSON → {json_path}")

    # ------------------------------------------------------------------ #
    # 6b. Per-episode export (format: episode_export.py)                    #
    # ------------------------------------------------------------------ #
    if registry is not None:
        control_hz = float(cfg.get("control_hz", 30))
        rollout_rows = export.build_rollouts(
            settings["rollouts"],
            {k: settings[k] for k in ("dataset", "study", "task", "environment")})
        manifest = dict(
            dataset=settings["dataset"],
            study=settings["study"],
            task=settings["task"],
            environment=settings["environment"],
            analysis_config=_config_name(cfg),
            control_rate_hz=control_hz,
            pretrain=dict(path=str(pretrain_path), n_demos=len(pretrain_trajs),
                          n_states=int(len(pretrain_all)),
                          knn_self={f"p{p}": base_ref.tau[p] for p in base_ref.tau} |
                          {"mean": float(pretrain_knn_scores.mean())}),
            ood=dict(obs_keys=obs_keys, angle_keys=sorted(ANGLE_KEYS),
                     k_neighbors=int(cfg.ood.k_neighbors),
                     normalize=bool(cfg.ood.normalize) and not state.ANGLE_SPACE,
                     normalize_note=(
                         "deliberate: in a (sin, cos) joint-angle space every dimension is already "
                         "bounded on the same scale, so z-scoring would only re-weight joints by "
                         "how much they happened to move" if state.ANGLE_SPACE else
                         "per-dimension mean/std of the round's own reference set"),
                     space="joint angle (sin, cos)" if state.ANGLE_SPACE else "z-scored feature space",
                     reference=_ref_note.get(reference_mode, reference_mode),
                     references=all_references),
            intervention_timing=dict(
                gap_seconds=cfg.intervention_timing.get("gap_seconds", None),
                gap_threshold_timesteps=gap_timesteps(cfg, control_hz),
                level="group"),
            metrics=_metric_notes(control_hz),
            methods={a: export.method_of(a, settings["methods"]) for a in runs},
        )
        export.write_export(str(cfg.output_dir), registry, rollout_rows, manifest,
                            partial=bool(ONLY_METRICS))

    # ------------------------------------------------------------------ #
    # 7. Save raw distributions                                           #
    # ------------------------------------------------------------------ #
    print("\nWriting outputs ...")

    # Aggregate arrays across all iterations per algorithm
    def _cat(parts):
        """Concatenate, tolerating a pass that was skipped by only_metrics."""
        parts = [p for p in parts if p is not None]
        return np.concatenate(parts) if parts else np.array([])

    algo_knn_agg = {
        algo: {
            cat: _cat([all_iter_knn[algo][it].get(cat) for it in all_iter_knn[algo]])
            for cat in ("all", "intervention", "rollout")
        }
        for algo in runs
    }
    algo_dtw_agg = {
        algo: {
            cat: _cat([all_iter_dtw[algo][it].get(cat) for it in all_iter_dtw[algo]])
            for cat in ("full", "intervention_segs", "rollout_segs")
        }
        for algo in runs
    }
    timing_keys = (
        "segment_starts", "segment_ends", "segment_lengths",
        "segment_start_fracs", "segment_end_fracs",
        "group_starts", "group_ends", "group_lengths",
        "group_n_segments", "group_start_fracs", "group_end_fracs",
    )
    algo_timing_agg = {
        algo: {
            k: _cat([all_iter_timing[algo][it].get(k) for it in all_iter_timing[algo]])
            for k in timing_keys
        }
        for algo in runs
    }
    jerk_keys = (
        "ro_to_iv", "iv_to_ro", "non_transition",
        "train_ro_to_iv", "train_iv_to_ro", "train_non_transition",
    )
    algo_jerk_agg = {
        algo: {
            k: _cat([all_iter_jerk[algo][it].get(k) for it in all_iter_jerk[algo]])
            for k in jerk_keys
        }
        for algo in runs
    }
    algo_jjerk_agg = {}
    for algo in runs:
        iters_with_data = [
            it for it in all_iter_jjerk[algo]
            if all_iter_jjerk[algo][it] is not None
        ]
        algo_jjerk_agg[algo] = (
            {k: _cat([all_iter_jjerk[algo][it].get(k) for it in iters_with_data])
             for k in jerk_keys}
            if iters_with_data else None
        )

    run_facts = dict(
        angle_space=bool(state.ANGLE_SPACE),
        k_neighbors=int(cfg.ood.k_neighbors),
        normalize=bool(cfg.ood.normalize) and not state.ANGLE_SPACE,  # effective value, matches the kNN call
        embedding_npz=cfg.ood.get("embedding_npz", None),   # read back by downstream plotting
    )
    _save_distributions(
        cfg.output_dir, run_name, run_facts, pretrain_knn_scores,
        algo_knn_agg, algo_dtw_agg, algo_timing_agg, algo_jerk_agg,
        all_iter_knn, all_iter_dtw, runs,
        algo_jjerk_agg=algo_jjerk_agg,
        all_per_demo=all_per_demo,
        pretrain_knn_per_demo=pretrain_knn_per_demo,
        all_ep_rounds=all_ep_rounds,
        all_timing_rounds=all_timing_rounds,
    )

    # Downstream plotting reads ood/distributions.npz (+ meta) and alpha/sessions.json.

    print("\nDone.")


if __name__ == "__main__":
    main()
