import abc
import numpy as np
import os
import json


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty estimator abstraction
# ─────────────────────────────────────────────────────────────────────────────

class BaseUncertaintyEstimator(abc.ABC):
    @abc.abstractmethod
    def compute(self, obs_buffer, robot_action, policy):
        """
        Returns a per-DOF uncertainty array in the 7D blending space:
          [unc_x, unc_y, unc_z, unc_roll, unc_pitch, unc_yaw, unc_gripper]
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding helper  (used by k-NN estimator)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_diffusion_obs_embedding(policy) -> np.ndarray | None:
    """
    Extract the obs encoder embedding for the current observation history from
    an XArmDiffusionPolicy.  Returns shape (feature_dim,) — mean across the
    observation-step dimension.  Returns None if the obs history is empty.
    """
    from diffusion_policy.common.pytorch_util import dict_apply
    import torch

    worker = policy._worker
    inner  = worker._policy

    obs_history = list(worker._obs_history)
    if not obs_history:
        return None
    n = worker.n_obs_steps
    while len(obs_history) < n:
        obs_history.insert(0, obs_history[0])
    obs_dict = worker._convert_obs(obs_history)

    device = inner.device
    with torch.no_grad():
        nobs = inner.normalizer.normalize(obs_dict)
        To   = inner.n_obs_steps
        this_nobs    = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = inner.obs_encoder(this_nobs)   # [To, feature_dim]
        embedding     = nobs_features.mean(dim=0)      # [feature_dim]
    return embedding.cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# k-NN uncertainty estimator
# ─────────────────────────────────────────────────────────────────────────────

class _KNNIndex:
    """
    Shared k-NN lookup backend.  Loaded once from a .pkl file produced by
    build_knn_index.py; reused by all policy-specific estimator subclasses.
    """
    def __init__(self, index_path: str, k: int = 10):
        import pickle
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self._train_embs = np.asarray(data["embeddings"], dtype=np.float32)  # [N, d]
        self._k          = min(k, len(self._train_embs) - 1)
        self._train_sq   = (self._train_embs ** 2).sum(axis=1)               # [N] precomputed
        print(f"[KNNUncertainty] index: {len(self._train_embs)} embeddings  "
              f"d={self._train_embs.shape[1]}  k={self._k}")

    def kth_distance(self, query: np.ndarray) -> float:
        # ||q - e_i||² = ||e_i||² - 2*(e_i·q) + ||q||²  (BLAS matmul for the middle term)
        sq_dists = self._train_sq - 2.0 * (self._train_embs @ query) + float(np.dot(query, query))
        sq_dists = np.maximum(sq_dists, 0.0)
        kth_sq   = float(np.partition(sq_dists, self._k)[self._k])
        return float(np.sqrt(kth_sq))


class XArmKNNUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    k-NN distance in obs encoder feature space for XArmDiffusionPolicy.

    Build the index with:
      python Shared-Control-DAgger/arbitration/build_knn_index.py \\
          --robot-type xarm --ckpt-dir <ckpt> --dataset <data.hdf5> \\
          --output knn_index.pkl --calib-out uncertainty_knn.json

    Pass --knn-index-path knn_index.pkl and --uncertainty-path uncertainty_knn.json
    to collect.py.
    """
    def __init__(self, index_path: str, k: int = 10, stale_warn_s: float = 0.5):
        """stale_warn_s: embedding age above which the confidence is flagged as
        computed from outdated data. The embedding refreshes at the inference
        worker's cadence (one per chunk inference), so ~0.5 s means "the conf
        signal has effectively frozen", not one-frame jitter."""
        self._idx = _KNNIndex(index_path, k)
        self._stale_warn_s = float(stale_warn_s)
        self._last_warn = 0.0
        self._ages = []
        self._none_returns = 0

    def reset_stats(self):
        self._ages = []
        self._none_returns = 0

    def stats(self):
        """Embedding-age statistics since reset_stats(): time between the
        observation the embedding was computed from and the frame using it."""
        a = np.asarray(self._ages)
        return {
            "n": len(a),
            "none_returns": self._none_returns,
            "median_s": float(np.median(a)) if len(a) else float("nan"),
            "p95_s": float(np.percentile(a, 95)) if len(a) else float("nan"),
            "max_s": float(a.max()) if len(a) else float("nan"),
            "stale_frac": float(np.mean(a > self._stale_warn_s)) if len(a) else 0.0,
            "stale_warn_s": self._stale_warn_s,
        }

    def compute(self, obs_buffer, robot_action, policy) -> np.ndarray | None:
        import time as _time
        now = _time.time()
        # Reuse the embedding of the chunk being executed (forward hook); re-encoding here is too slow.
        worker = getattr(policy, "_worker", None)
        cached = getattr(worker, "last_obs_embedding", None)
        if cached is not None:
            emb, emb_time = cached
            age = now - emb_time
        elif getattr(worker, "emb_hook_installed", False):
            # No inference yet this episode: report no signal (alpha_conf = 1).
            self._none_returns += 1
            return None
        else:
            emb = _extract_diffusion_obs_embedding(policy)
            if emb is None:
                self._none_returns += 1
                return None
            age = 0.0
        self._ages.append(age)
        if age > self._stale_warn_s and now - self._last_warn > 1.0:
            self._last_warn = now
            print(f"[KNNUncertainty] WARNING: confidence is {age*1000:.0f}ms stale "
                  f"(no fresh inference embedding in {age:.1f}s)")
        dist       = self._idx.kth_distance(emb)
        action_dim = policy._worker._policy.action_dim
        return np.full(action_dim, dist, dtype=np.float32)


class RobomimicKNNUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    k-NN distance in obs encoder feature space for Robomimic diffusion policies
    (RoboCasa simulation experiments).

    Build the index with:
      python Shared-Control-DAgger/arbitration/build_knn_index.py \\
          --robot-type robomimic --ckpt-path <ckpt.pth> --dataset <data.hdf5> \\
          --output knn_index.pkl --calib-out uncertainty_knn.json

    Embedding: reuses the obs-encoder embedding the policy already computes during
    inference (captured by a forward hook), so compute() is a ~1ms index lookup
    instead of re-running the encoder (and fighting the diffusion inference for GPU).
    """
    def __init__(self, index_path: str, k: int = 10):
        self._idx = _KNNIndex(index_path, k)
        self._hook_installed = False
        self._cached_emb = None   # [D] mean obs-encoder embedding from the last inference

    def attach(self, policy):
        # Install the hook before the loop so autonomous inference primes the cache.
        if not self._hook_installed:
            self._install_hook(policy)

    def _install_hook(self, policy):
        # get_action() runs obs_encoder via time_distributed -> out is [B*T, D]; mean -> [D].
        enc = policy.policy.nets['policy']['obs_encoder']
        def _hook(_module, _inp, out):
            if hasattr(out, "detach") and out.ndim == 2:
                self._cached_emb = out.detach().mean(dim=0).float().cpu().numpy()
        enc.register_forward_hook(_hook)
        self._hook_installed = True

    def compute(self, obs_buffer, robot_action, policy) -> np.ndarray | None:
        if not self._hook_installed:
            self._install_hook(policy)
        emb = self._cached_emb
        if emb is None:
            emb = self._recompute(obs_buffer, policy)   # before the hook first fires
        if emb is None:
            return None
        return np.full(robot_action.shape[0], self._idx.kth_distance(emb), dtype=np.float32)

    def _recompute(self, obs_buffer, policy):
        # Fallback that also primes the cache (its encoder call fires the hook).
        import torch
        import robomimic.utils.tensor_utils as TensorUtils
        if not obs_buffer:
            return None
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        obs_batch = {k: torch.tensor(np.stack([o[k] for o in obs_buffer], axis=0),
                                     dtype=torch.float32, device=dev).unsqueeze(0)
                     for k in obs_buffer[0].keys()}
        with torch.no_grad():
            feats = TensorUtils.time_distributed({"obs": obs_batch},
                    policy.policy.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            return feats.mean(dim=1).squeeze(0).cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration and similarity helpers
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_uncertainty(calibration_file):
        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"Could not find {calibration_file}. Build it with arbitration/build_knn_index.py first!")

        with open(calibration_file, 'r') as f:
            bounds = json.load(f)

        # Convert lists back to numpy arrays
        l_min = np.array(bounds['L_MIN'])
        l_max = np.array(bounds['L_MAX'])

        print(f"Loaded static uncertainty bounds from {calibration_file}")
        return l_min, l_max

def get_cosine_similarity(v1, v2):
    """
    Compare direction of vectors using cosine similarity.
    Maps the standard [-1, 1] output to [0, 1] where 1 = identical direction, 0 = opposite.
    If either vector is zero, it assumes no conflict (returns 1.0 similarity).
    Orientation is in angle-axis deltas, so this is valid.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0

    cos_sim = np.dot(v1, v2) / (n1 * n2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Map from [-1, 1] to [0, 1]
    return (cos_sim + 1.0) / 2.0


def get_magnitude_weighted_similarity(v1, v2):
    """
    Similarity metric that accounts for both direction and how assertively the human is acting.

    Returns cosine_similarity(v1, v2) * clamp(norm(v2), 0, 1), where v2 is the
    normalized human action (pre-normalized to [-1, 1] per DOF).
    """
    cos_sim = get_cosine_similarity(v1, v2)
    magnitude = np.linalg.norm(v2)
    # Clamp to [0, 1]; groups with >1 DOF (e.g., xy, ori) can have norm > 1
    magnitude = np.clip(magnitude, 0.0, 1.0)
    return cos_sim * magnitude
