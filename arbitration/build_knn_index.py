"""
Build a k-NN index for k-NN obs-encoder uncertainty estimation.

Extracts obs encoder embeddings from training data and writes:
  <output>.pkl       — {"embeddings": np.ndarray [N, d]}
  <calib-out>.json   — {"L_MIN": [...], "L_MAX": [...]} calibration bounds
                       (action_dim-length arrays, all equal to the scalar bounds)

CombinedBlend and real_collector read L_MIN / L_MAX from the calibration JSON.

Supports two robot / framework types:
  xarm      — Hydra DiffusionUnetHybridImagePolicy loaded via _DiffusionInferenceThread.
               Use for xArm real-robot checkpoints (.ckpt format).
  robomimic — Robomimic-wrapped diffusion policy (RoboCasa simulation).
               Use for .pth checkpoints from the Robomimic training pipeline.

Usage (xArm):
  python build_knn_index.py \\
      --robot-type  xarm \\
      --ckpt-dir    path/to/checkpoint/ \\
      --dataset     path/to/full_teleop.hdf5 path/to/onpolicy_rollouts/ \\
      --output      arbitration/knn_index_cupboard.pkl \\
      --calib-out   arbitration/uncertainty_knn_cupboard.json \\
      --max-demos   100 --subsample 4

  --dataset accepts any mix of HDF5 files and directories; each directory is
  expanded to all *_full.hdf5 files found under it (recursively).  This is the
  intended way to build the index from the combined DAgger dataset (full teleop
  + on-policy rollouts + interventions).

Usage (RoboCasa simulation):
  python build_knn_index.py \\
      --robot-type  robomimic \\
      --ckpt-path   path/to/model_epoch_N.pth \\
      --dataset     path/to/robocasa_demos.hdf5 \\
      --output      arbitration/knn_index_sim.pkl \\
      --calib-out   arbitration/uncertainty_knn_sim.json \\
      --obs-horizon 2

Calibration (default: leave-one-episode-out / LOEO):
  A random subset (--n-calib) of training embeddings are evaluated to get a
  distribution of kth-nearest distances.  L_MIN is set to the --calib-low-pct
  percentile (default 5th) and L_MAX to --calib-high-pct (default 95th).  The
  resulting [0,1] range covers almost all in-distribution steps, so OOD steps
  reliably exceed 1.0 before clipping.

  By default each query's neighbours are drawn from all OTHER episodes (its own
  episode is masked out) — this matches deployment, where the live trajectory's
  frames are not in the index, and removes the same-episode bias that otherwise
  collapses the distance band.  Precedence:
    --calib-dataset  → calibrate from that held-out set (LOEO ignored)
    --no-loeo        → in-index sampling (each query's own episode stays in the index)
    otherwise        → LOEO on the training set (default)
  The deployed index (.pkl) always contains ALL episodes regardless of mode;
  LOEO only affects how the L_MIN/L_MAX band is computed.
"""

import argparse
import datetime
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

_HERE = Path(__file__).parent
_REPO  = _HERE.parent
_ROOT = _REPO.parent
for _p in [str(_ROOT), str(_REPO), str(_ROOT / "diffusion_policy"), str(_ROOT / "ACT")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(raw) -> np.ndarray:
    """Decode JPEG-encoded bytes or raw uint8 HWC → float32 HWC [0,1]."""
    if isinstance(raw, (bytes, np.bytes_)):
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = np.asarray(raw)
    if arr.dtype == np.uint8 and arr.ndim == 1:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _resolve_ckpt(path_or_dir, prefer="last") -> Path:
    p = Path(path_or_dir)
    if p.is_file():
        return p
    ckpts = sorted(p.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {p}")
    preferred = [c for c in ckpts if prefer in c.name]
    return preferred[0] if preferred else ckpts[-1]


def _resolve_datasets(paths):
    """
    Expand dataset arguments into a flat, de-duplicated list of HDF5 files.

    Each argument may be either:
      - a .hdf5 file      → used directly, or
      - a directory       → every *_full.hdf5 found under it (recursively),
                            matching the finetune pipeline's convention.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files, seen = [], set()
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            found = [pp]
        elif pp.is_dir():
            found = sorted(pp.rglob("*_full.hdf5"))
            if not found:
                raise FileNotFoundError(f"No *_full.hdf5 files found under directory: {pp}")
        else:
            raise FileNotFoundError(f"Dataset path is neither a file nor a directory: {pp}")
        for fp in found:
            s = str(fp.resolve())
            if s not in seen:
                seen.add(s)
                files.append(str(fp))

    if not files:
        raise FileNotFoundError("No datasets resolved from the given --dataset arguments")
    return files


def _sample_demos(f, max_demos):
    demos = list(f["data"].keys())
    if len(demos) > max_demos:
        np.random.shuffle(demos)
        demos = demos[:max_demos]
    return demos


def _compute_calibration(embeddings: np.ndarray, k: int,
                          n_calib: int, low_pct: float, high_pct: float,
                          action_dim: int) -> dict:
    """
    Sample n_calib queries from embeddings, compute their kth-distance against
    the full index, and return L_MIN / L_MAX as action_dim-element lists.
    """
    N   = len(embeddings)
    idx = np.random.choice(N, size=min(n_calib, N), replace=False)
    queries   = embeddings[idx]
    train_sq  = (embeddings ** 2).sum(axis=1)
    query_sq  = (queries    ** 2).sum(axis=1)

    dists = []
    for i, q in enumerate(queries):
        sq = train_sq - 2.0 * (embeddings @ q) + float(query_sq[i])
        sq = np.maximum(sq, 0.0)
        kk = min(k + 1, N - 1)
        part = np.sort(np.partition(sq, kk)[:kk + 1])
        nonzero = part[part > 1e-12]
        kth_sq = float(nonzero[k - 1]) if len(nonzero) >= k else float(part[min(k, len(part) - 1)])
        dists.append(np.sqrt(max(kth_sq, 0.0)))

    dists = np.array(dists, dtype=np.float32)
    l_min = float(np.percentile(dists, low_pct))
    l_max = float(np.percentile(dists, high_pct))
    l_max = max(l_max, l_min + 1e-4)

    print(f"[build_knn] Calibration:  k={k}  "
          f"L_MIN={l_min:.5f} ({low_pct}th pct)  L_MAX={l_max:.5f} ({high_pct}th pct)")
    print(f"  dist distribution: mean={dists.mean():.4f}  std={dists.std():.4f}  "
          f"min={dists.min():.4f}  max={dists.max():.4f}")
    return {"L_MIN": [l_min] * action_dim, "L_MAX": [l_max] * action_dim}


def _compute_calibration_loeo(embeddings: np.ndarray, episode_ids: np.ndarray,
                              k: int, n_calib: int, low_pct: float, high_pct: float,
                              action_dim: int) -> dict:
    """
    Leave-one-episode-out (LOEO) calibration.

    For each sampled query, its k-th nearest-neighbour distance is computed
    against every embedding EXCEPT those belonging to the same episode.  This
    reproduces the deployment condition — the live trajectory's own frames are
    not in the index — and removes the same-episode / self-match bias present in
    the plain in-index calibration (where temporally-adjacent frames dominate the
    neighbourhood and collapse the distance band).

    Uses the exact k-th convention of the deployed _KNNIndex.kth_distance
    (partition(sq, kk)[kk]) so the band matches inference-time distances.
    """
    N           = len(embeddings)
    episode_ids = np.asarray(episode_ids)
    idx         = np.random.choice(N, size=min(n_calib, N), replace=False)
    train_sq    = (embeddings ** 2).sum(axis=1)

    dists = []
    for i in idx:
        q  = embeddings[i]
        sq = train_sq - 2.0 * (embeddings @ q) + float(train_sq[i])
        sq = np.maximum(sq, 0.0)
        sq[episode_ids == episode_ids[i]] = np.inf      # drop the whole source episode
        n_other = int(np.count_nonzero(np.isfinite(sq)))
        if n_other < 1:
            continue
        kk = min(k, n_other - 1)
        dists.append(np.sqrt(float(np.partition(sq, kk)[kk])))

    if not dists:
        raise RuntimeError("LOEO calibration needs >=2 episodes; got a single episode. "
                           "Use --no-loeo or provide --calib-dataset.")

    dists = np.array(dists, dtype=np.float32)
    l_min = float(np.percentile(dists, low_pct))
    l_max = float(np.percentile(dists, high_pct))
    l_max = max(l_max, l_min + 1e-4)

    n_eps = len(np.unique(episode_ids))
    print(f"[build_knn] Calibration (LOEO, {len(dists)} queries over {n_eps} episodes):  k={k}  "
          f"L_MIN={l_min:.5f} ({low_pct}th pct)  L_MAX={l_max:.5f} ({high_pct}th pct)")
    print(f"  dist distribution: mean={dists.mean():.4f}  std={dists.std():.4f}  "
          f"min={dists.min():.4f}  max={dists.max():.4f}")
    return {"L_MIN": [l_min] * action_dim, "L_MAX": [l_max] * action_dim}


def _compute_calibration_against_index(index_embs: np.ndarray, query_embs: np.ndarray,
                                       k: int, low_pct: float, high_pct: float,
                                       action_dim: int) -> dict:
    """
    Compute L_MIN / L_MAX from a separate held-out calibration set whose
    embeddings are NOT in the index.  Each query's k-th nearest-neighbour
    distance is computed against the index with the exact convention used by the
    deployed _KNNIndex.kth_distance (partition(sq, k)[k]).  Because the queries
    are out-of-index, this is free of the in-index self-match / same-trajectory
    bias and the resulting band matches inference-time distances.
    """
    N        = len(index_embs)
    kk       = min(k, N - 1)
    train_sq = (index_embs ** 2).sum(axis=1)
    query_sq = (query_embs ** 2).sum(axis=1)

    dists = []
    for i, q in enumerate(query_embs):
        sq = train_sq - 2.0 * (index_embs @ q) + float(query_sq[i])
        sq = np.maximum(sq, 0.0)
        dists.append(np.sqrt(float(np.partition(sq, kk)[kk])))

    dists = np.array(dists, dtype=np.float32)
    l_min = float(np.percentile(dists, low_pct))
    l_max = float(np.percentile(dists, high_pct))
    l_max = max(l_max, l_min + 1e-4)

    print(f"[build_knn] Calibration (held-out vs index, {len(query_embs)} queries):  k={kk}  "
          f"L_MIN={l_min:.5f} ({low_pct}th pct)  L_MAX={l_max:.5f} ({high_pct}th pct)")
    print(f"  dist distribution: mean={dists.mean():.4f}  std={dists.std():.4f}  "
          f"min={dists.min():.4f}  max={dists.max():.4f}")
    return {"L_MIN": [l_min] * action_dim, "L_MAX": [l_max] * action_dim}


def _save_outputs(embeddings, calib, output_path, calib_out_path, build_meta=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f, protocol=4)
    print(f"[build_knn] Index saved  → {output_path}  "
          f"({len(embeddings)} embeddings, d={embeddings.shape[1]})")

    # Record the arguments this index was built with under a "_build_meta" key.
    # Downstream consumers (calibrate_uncertainty / CombinedBlend) only read
    # L_MIN / L_MAX, so the extra key is inert for them and purely informational.
    if build_meta is not None:
        calib = {**calib, "_build_meta": build_meta}

    Path(calib_out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(calib_out_path, "w") as f:
        json.dump(calib, f, indent=4)
    print(f"[build_knn] Calibration saved → {calib_out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# xArm  (Hydra DiffusionUnetHybridImagePolicy, .ckpt format)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_xarm_embeddings(inner, obs_meta, n_obs, dataset_paths,
                             max_demos, subsample, device):
    """
    Extract mean obs-encoder embeddings for every (subsampled) step across one or
    more datasets.  `dataset_paths` may be a single file/dir or a list of them;
    directories are expanded to their *_full.hdf5 files via _resolve_datasets.

    Every demo across all files receives a globally-unique episode id (returned
    alongside the embeddings) so LOEO calibration masks a demo's own trajectory
    regardless of which file it came from.  `max_demos` is applied per file.
    """
    from diffusion_policy.common.pytorch_util import dict_apply

    dataset_files = _resolve_datasets(dataset_paths)
    print(f"[build_knn] Resolved {len(dataset_files)} dataset file(s):")
    for fp in dataset_files:
        print(f"    - {fp}")

    all_embs = []
    all_eps  = []
    ep       = 0
    for dataset_path in dataset_files:
        f     = h5py.File(dataset_path, "r")
        demos = _sample_demos(f, max_demos)
        print(f"[build_knn] Extracting from {len(demos)} demos in {Path(dataset_path).name} "
              f"(subsample={subsample}) …")

        for di, demo_name in enumerate(demos):
            demo_grp = f[f"data/{demo_name}"]

            # Only embed steps the episode marks as training data. Aborted
            # episodes are saved to the full HDF5 with train_mask all-False
            # (preserved for analysis only) — without this filter their states
            # end up in the index and shift the calibration percentiles.
            train_mask = (demo_grp["train_mask"][:].astype(bool)
                          if "train_mask" in demo_grp else None)
            if train_mask is not None and not train_mask.any():
                print(f"  {demo_name}  ({di + 1}/{len(demos)})  — SKIPPED "
                      f"(train_mask all False, e.g. aborted episode)")
                ep += 1
                continue
            kept = int(train_mask.sum()) if train_mask is not None else None
            print(f"  {demo_name}  ({di + 1}/{len(demos)})  [episode {ep}]"
                  + (f"  train steps {kept}/{len(train_mask)}" if train_mask is not None
                     else "  (no train_mask — keeping all steps)"))
            obs_h5 = demo_grp["obs"]

            obs_full = {}
            for k_obs, meta in obs_meta.items():
                if k_obs not in obs_h5:
                    continue
                typ = meta.get("type", "low_dim")
                if typ == "rgb":
                    _, H, W = meta["shape"]
                    raw_frames = obs_h5[k_obs][:]
                    if raw_frames.ndim == 1:
                        frames = np.stack([
                            cv2.resize(_decode_image(raw_frames[t]), (W, H),
                                       interpolation=cv2.INTER_AREA)
                            for t in range(len(raw_frames))
                        ])
                    else:
                        frames = raw_frames.astype(np.float32)
                        if frames.max() > 1.5:
                            frames /= 255.0
                        frames = np.stack([
                            cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_AREA)
                            for t in range(len(frames))
                        ])
                    obs_full[k_obs] = frames.transpose(0, 3, 1, 2)   # [T, 3, H, W]
                else:
                    obs_full[k_obs] = obs_h5[k_obs][:].astype(np.float32)

            T = next(iter(obs_full.values())).shape[0]
            for i in range(0, T, subsample):
                if train_mask is not None and i < len(train_mask) and not train_mask[i]:
                    continue   # non-training step (e.g. excluded rollout segment)
                seq = []
                for t in range(max(0, i - n_obs + 1), i + 1):
                    seq.append({kk: obs_full[kk][t] for kk in obs_full})
                while len(seq) < n_obs:
                    seq.insert(0, seq[0])
                obs_dict = {
                    kk: torch.from_numpy(np.stack([seq[s][kk] for s in range(n_obs)], axis=0))
                           .unsqueeze(0).to(device)
                    for kk in obs_full
                }
                with torch.no_grad():
                    nobs = inner.normalizer.normalize(obs_dict)
                    To   = inner.n_obs_steps
                    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                    feats = inner.obs_encoder(this_nobs)       # [To, feature_dim]
                    emb   = feats.mean(dim=0).cpu().float().numpy()
                all_embs.append(emb)
                all_eps.append(ep)
            ep += 1

        f.close()

    return np.stack(all_embs, axis=0), np.asarray(all_eps, dtype=np.int32)


def build_xarm(ckpt_dir, dataset_path, output_path, calib_out_path,
               max_demos, subsample, k, n_calib, low_pct, high_pct,
               calib_dataset=None, use_loeo=True, build_meta=None):
    from xarm.policies.diffusion_policy import _DiffusionInferenceThread

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = _resolve_ckpt(ckpt_dir)
    print(f"[build_knn] Loading checkpoint: {ckpt_path}")
    if build_meta is not None:
        build_meta = {**build_meta, "resolved_ckpt": str(ckpt_path)}
    thread = _DiffusionInferenceThread(str(ckpt_path), device)
    inner  = thread._policy

    obs_meta   = dict(thread._cfg.shape_meta["obs"])
    n_obs      = thread.n_obs_steps
    action_dim = inner.action_dim

    embeddings, episode_ids = _extract_xarm_embeddings(
        inner, obs_meta, n_obs, dataset_path, max_demos, subsample, device)

    if calib_dataset is not None:
        # Build the index from the training set, but calibrate the L_MIN/L_MAX
        # scale from a held-out set queried AGAINST that index (out-of-index ⇒
        # unbiased, matches deployment-time distances).
        print(f"[build_knn] Calibrating from held-out set: {calib_dataset}")
        calib_embs, _ = _extract_xarm_embeddings(inner, obs_meta, n_obs, calib_dataset,
                                                 max_demos, subsample, device)
        calib = _compute_calibration_against_index(embeddings, calib_embs, k,
                                                   low_pct, high_pct, action_dim)
    elif use_loeo:
        # Default: leave-one-episode-out on the training set — unbiased without
        # needing a separate held-out dataset (see _compute_calibration_loeo).
        calib = _compute_calibration_loeo(embeddings, episode_ids, k, n_calib,
                                          low_pct, high_pct, action_dim)
    else:
        calib = _compute_calibration(embeddings, k, n_calib, low_pct, high_pct, action_dim)
    _save_outputs(embeddings, calib, output_path, calib_out_path, build_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Robomimic / RoboCasa simulation  (.pth checkpoint format)
# ─────────────────────────────────────────────────────────────────────────────

def build_robomimic(ckpt_path, dataset_path, output_path, calib_out_path,
                    max_demos, subsample, k, n_calib, low_pct, high_pct,
                    obs_horizon, use_loeo=True, build_meta=None):
    import json
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.tensor_utils as TensorUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.lang_utils as LangUtils

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_knn] Loading Robomimic policy from {ckpt_path}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=False
    )
    algo = policy.policy
    obs_keys = list(ckpt_dict["shape_metadata"]["all_shapes"].keys())
    lang_encoder = LangUtils.LangEncoder(device=device)

    dataset_files = _resolve_datasets(dataset_path)
    print(f"[build_knn] Resolved {len(dataset_files)} dataset file(s):")
    for fp in dataset_files:
        print(f"    - {fp}")

    all_embs   = []
    all_eps    = []
    action_dim = None
    ep         = 0
    for dataset_file in dataset_files:
        f     = h5py.File(dataset_file, "r")
        demos = _sample_demos(f, max_demos)
        print(f"[build_knn] Extracting from {len(demos)} demos in {Path(dataset_file).name} "
              f"(subsample={subsample}) …")

        for di, demo_name in enumerate(demos):
            demo   = f[f"data/{demo_name}"]
            # Skip non-training steps (train_mask all-False = aborted episode).
            train_mask = (demo["train_mask"][:].astype(bool)
                          if "train_mask" in demo else None)
            if train_mask is not None and not train_mask.any():
                print(f"  {demo_name}  ({di + 1}/{len(demos)})  — SKIPPED "
                      f"(train_mask all False, e.g. aborted episode)")
                ep += 1
                continue
            print(f"  {demo_name}  ({di + 1}/{len(demos)})  [episode {ep}]")
            obs_h5 = demo["obs"]
            if action_dim is None:
                action_dim = demo["actions"].shape[1]

            # Mirror the runtime obs pipeline: the sim env postprocesses rgb to
            # [C,H,W] float, but the dataset stores raw [H,W,C] uint8, so process it here.
            obs_full = {}
            for k_obs in obs_keys:
                if k_obs not in obs_h5:
                    continue
                arr = obs_h5[k_obs][:].astype(np.float32)
                if ObsUtils.key_is_obs_modality(k_obs, "rgb"):
                    arr = ObsUtils.process_obs(torch.from_numpy(arr), obs_key=k_obs).numpy()
                obs_full[k_obs] = arr
            T = next(iter(obs_full.values())).shape[0]

            # lang_emb for the language-conditioned encoder (from the demo's task text).
            lang_emb = None
            if "ep_meta" in demo.attrs:
                lang_str = json.loads(demo.attrs["ep_meta"]).get("lang", None)
                if lang_str:
                    le = lang_encoder.get_lang_emb(lang_str)
                    if torch.is_tensor(le):
                        le = le.cpu().numpy()
                    lang_emb = np.squeeze(le).astype(np.float32)

            for i in range(0, T, subsample):
                if train_mask is not None and i < len(train_mask) and not train_mask[i]:
                    continue
                start = max(0, i - obs_horizon + 1)
                seq = [{k_obs: obs_full[k_obs][t] for k_obs in obs_full}
                       for t in range(start, i + 1)]
                while len(seq) < obs_horizon:
                    seq.insert(0, seq[0])

                obs_batch = {}
                for k_obs in obs_full:
                    stacked = np.stack([seq[s][k_obs] for s in range(obs_horizon)], axis=0)
                    obs_batch[k_obs] = torch.from_numpy(stacked).unsqueeze(0).to(device)
                if lang_emb is not None:
                    lang_stacked = np.stack([lang_emb] * obs_horizon, axis=0)
                    obs_batch["lang_emb"] = torch.from_numpy(lang_stacked).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = TensorUtils.time_distributed(
                        {"obs": obs_batch},
                        algo.nets['policy']['obs_encoder'],
                        inputs_as_kwargs=True,
                    )   # [1, T, feature_dim]
                    emb = features.mean(dim=1).squeeze(0).cpu().float().numpy()
                all_embs.append(emb)
                all_eps.append(ep)
            ep += 1

        f.close()
    if action_dim is None:
        raise RuntimeError("No demos found — could not determine action_dim")
    embeddings  = np.stack(all_embs, axis=0)
    episode_ids = np.asarray(all_eps, dtype=np.int32)
    if use_loeo:
        calib = _compute_calibration_loeo(embeddings, episode_ids, k, n_calib,
                                          low_pct, high_pct, action_dim)
    else:
        calib = _compute_calibration(embeddings, k, n_calib, low_pct, high_pct, action_dim)
    _save_outputs(embeddings, calib, output_path, calib_out_path, build_meta)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build k-NN embedding index for OOD uncertainty estimation."
    )
    parser.add_argument("--robot-type", required=True, choices=["xarm", "robomimic"],
                        help="'xarm' for Hydra .ckpt (xArm); "
                             "'robomimic' for RoboCasa sim .pth checkpoints")
    parser.add_argument("--ckpt-dir",  default=None,
                        help="[xarm] Checkpoint file or directory (looks for *.ckpt)")
    parser.add_argument("--ckpt-path", default=None,
                        help="[robomimic] Full path to .pth checkpoint")
    parser.add_argument("--dataset",   required=True, nargs="+",
                        help="One or more paths used to build the index. Each may be an "
                             "HDF5 file OR a directory (all *_full.hdf5 under it are used, "
                             "recursively). Mix files and folders freely, e.g. the full "
                             "teleop hdf5 plus the on-policy rollout directory.")
    parser.add_argument("--calib-dataset", default=None,
                        help="[xarm] Optional separate held-out HDF5 dataset used ONLY to "
                             "calibrate L_MIN/L_MAX (queried against the training index). "
                             "Removes the in-index self-match bias. Takes precedence over "
                             "LOEO when provided.")
    parser.add_argument("--no-loeo", action="store_true",
                        help="Disable leave-one-episode-out calibration (the default) and "
                             "fall back to biased in-index sampling of the training set. "
                             "Ignored when --calib-dataset is given.")
    parser.add_argument("--output",    required=True,
                        help="Output .pkl path for the embedding index")
    parser.add_argument("--calib-out", required=True,
                        help="Output .json path for calibration bounds (L_MIN / L_MAX)")
    parser.add_argument("--max-demos", type=int, default=100,
                        help="Maximum demos to use (default: 100)")
    parser.add_argument("--subsample", type=int, default=4,
                        help="Keep every Nth timestep to reduce index size (default: 4)")
    parser.add_argument("--k",         type=int, default=10,
                        help="Nearest-neighbour rank for distance and calibration (default: 10)")
    parser.add_argument("--n-calib",   type=int, default=2000,
                        help="Number of random queries used to compute calibration percentiles "
                             "(default: 2000)")
    parser.add_argument("--calib-low-pct",  type=float, default=5.0,
                        help="Percentile of kth-distances used as L_MIN (default: 5)")
    parser.add_argument("--calib-high-pct", type=float, default=95.0,
                        help="Percentile of kth-distances used as L_MAX (default: 95)")
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="[robomimic] Observation history length (default: 2)")
    args = parser.parse_args()

    # Snapshot of the run's arguments, embedded in the calib-out JSON under
    # "_build_meta" for provenance (which checkpoint / dataset / flags produced it).
    build_meta = {
        "built_at":     datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command":      " ".join(sys.argv),
        "args":         vars(args),
    }

    if args.robot_type == "xarm":
        if not args.ckpt_dir:
            parser.error("--robot-type xarm requires --ckpt-dir")
        build_xarm(
            ckpt_dir=args.ckpt_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            calib_out_path=args.calib_out,
            max_demos=args.max_demos,
            subsample=args.subsample,
            k=args.k,
            n_calib=args.n_calib,
            low_pct=args.calib_low_pct,
            high_pct=args.calib_high_pct,
            calib_dataset=args.calib_dataset,
            use_loeo=not args.no_loeo,
            build_meta=build_meta,
        )
    elif args.robot_type == "robomimic":
        if not args.ckpt_path:
            parser.error("--robot-type robomimic requires --ckpt-path")
        build_robomimic(
            ckpt_path=args.ckpt_path,
            dataset_path=args.dataset,
            output_path=args.output,
            calib_out_path=args.calib_out,
            max_demos=args.max_demos,
            subsample=args.subsample,
            k=args.k,
            n_calib=args.n_calib,
            low_pct=args.calib_low_pct,
            high_pct=args.calib_high_pct,
            obs_horizon=args.obs_horizon,
            use_loeo=not args.no_loeo,
            build_meta=build_meta,
        )
