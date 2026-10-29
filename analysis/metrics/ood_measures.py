"""
OOD (out-of-distribution) measures for interactive IL analysis.

All functions operate on pre-stacked numpy arrays.  Normalization is
fit on the pretrain distribution and applied to all collected data so
that the kNN distance is meaningful across dimensions with different scales.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fit_normalizer(states: np.ndarray, rel_floor: float = 1e-3
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean and std from pretrain states.

    Dimensions with negligible variance are given std=1 (no-op scaling), so they
    contribute ~0 to any distance instead of being blown up to unit variance.
    The threshold is RELATIVE to the widest dimension, because a near-constant
    feature (e.g. the 6D-rotation entries of a near-identity action) can still
    clear an absolute 1e-8 guard and dividing by its std amplifies numerical noise.
    """
    mean = states.mean(axis=0)
    std  = states.std(axis=0)
    floor = max(1e-8, rel_floor * float(std.max()) if std.size else 1e-8)
    std[std < floor] = 1.0
    return mean, std


def normalize(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


# ---------------------------------------------------------------------------
# kNN OOD scoring
# ---------------------------------------------------------------------------

def fit_knn(pretrain_norm: np.ndarray, k: int) -> NearestNeighbors:
    """Fit kNN index on normalized pretrain states."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(pretrain_norm)
    return nn


def knn_ood_scores(states_norm: np.ndarray, nn: NearestNeighbors) -> np.ndarray:
    """
    Mean distance to the k nearest pretrain neighbours for every row.

    Parameters
    ----------
    states_norm : (T, D) normalized state array
    nn          : fitted NearestNeighbors on pretrain data

    Returns
    -------
    (T,) array of per-timestep OOD scores (higher = more OOD)
    """
    if len(states_norm) == 0:
        return np.array([])
    distances, _ = nn.kneighbors(states_norm)
    return distances.mean(axis=1)


def knn_self_scores(states_norm: np.ndarray, groups: np.ndarray, k: int,
                    start: int = 64, cap: int = 8192) -> np.ndarray:
    """Leave-one-episode-out kNN distance for every state of a reference set.

    Scoring a reference against an index built from itself is biased low: the query is its own
    nearest neighbour and its adjacent frames are nearly the same state, which collected data
    never has in the index. So a reference state is scored only against the episodes it did not
    come from, the same question a collected state is asked.

    Exact, not approximate: candidates are fetched in a widening window and filtered to other
    episodes, and any state still short of k is re-queried with a bigger window until it has them
    or the reference is exhausted. A state whose reference holds fewer than k out-of-episode
    neighbours scores NaN, rather than a mean over fewer points than every other row.
    """
    n = len(states_norm)
    if n == 0:
        return np.array([])
    groups = np.asarray(groups)

    scores = np.full(n, np.nan)
    pending = np.arange(n)
    window = min(max(start, k + 1), n)

    while len(pending):
        nn = NearestNeighbors(n_neighbors=window, algorithm="auto", n_jobs=-1)
        nn.fit(states_norm)
        distances, indices = nn.kneighbors(states_norm[pending])

        still: list[int] = []
        for row, state in enumerate(pending):
            other = groups[indices[row]] != groups[state]
            if int(other.sum()) >= k:
                scores[state] = distances[row][other][:k].mean()
            elif window < n:
                still.append(state)      # widen and look again
            # else: the whole reference was searched and it genuinely has too few. NaN stands.
        pending = np.array(still, dtype=int)
        if window >= n or window >= cap:
            break
        window = min(window * 4, n, cap)

    return scores


def compute_ood_stats(
    states_norm: np.ndarray,
    iv_mask: np.ndarray,
    nn: NearestNeighbors,
) -> dict[str, np.ndarray]:
    """
    Compute kNN OOD scores split by intervention / rollout.

    Parameters
    ----------
    states_norm : (T, D) normalized states for an entire iteration
    iv_mask     : (T,) bool — True where human was intervening
    nn          : fitted kNN on normalized pretrain states

    Returns
    -------
    dict with keys "all", "intervention", "rollout", each a (T_k,) score array
    """
    return dict(
        all=knn_ood_scores(states_norm, nn),
        intervention=knn_ood_scores(states_norm[iv_mask],  nn),
        rollout=knn_ood_scores(states_norm[~iv_mask], nn),
    )
