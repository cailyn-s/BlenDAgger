"""
Trajectory similarity measures for interactive IL analysis.

Functions compute trajectory distances between collected trajectories and a
reference set of pretrain demonstrations: DTW variants, discrete Frechet, and the
windowed (equal-length) discrete Frechet distance used in the paper
(frechet_fixed_window_to_nearest). All inputs are expected to be in the same
normalized state space (caller is responsible for normalization).
"""

import warnings

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

try:
    from dtaidistance import dtw_ndim
    HAS_DTW = True
except ImportError:
    warnings.warn("dtaidistance not found — trajectory DTW will be skipped.")
    HAS_DTW = False


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segments(mask: np.ndarray, min_len: int) -> list[np.ndarray]:
    """
    Return index arrays for each contiguous True run with length >= min_len.

    Parameters
    ----------
    mask    : (T,) bool array
    min_len : minimum run length to include

    Returns
    -------
    list of 1-D index arrays, one per qualifying run
    """
    segments, in_run, start = [], False, 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start, in_run = i, True
        elif not val and in_run:
            if i - start >= min_len:
                segments.append(np.arange(start, i))
            in_run = False
    if in_run and len(mask) - start >= min_len:
        segments.append(np.arange(start, len(mask)))
    return segments


# ---------------------------------------------------------------------------
# DTW distance
# ---------------------------------------------------------------------------

def _resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a (T, D) trajectory to exactly n points along its index."""
    t = len(traj)
    if t == n:
        return traj
    if t < 2:
        return np.repeat(traj if t else np.zeros((1, traj.shape[1])), n, axis=0)
    src = np.linspace(0.0, t - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return traj[lo] * (1 - w) + traj[hi] * w


def discrete_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete Frechet (coupling) distance between two (T, D) curves.

    Unlike DTW, which sums along the warping path, this is the minimum over
    monotone couplings of the LARGEST pairwise distance — so it is governed by a
    curve's worst excursion rather than its accumulated deviation.
    """
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    prev = np.maximum.accumulate(d[0])
    for i in range(1, len(p)):
        cur = np.empty_like(prev)
        cur[0] = max(d[i, 0], prev[0])
        best = np.minimum(prev[1:], prev[:-1])
        for j in range(1, len(q)):
            cur[j] = max(d[i, j], min(best[j - 1], cur[j - 1]))
        prev = cur
    return float(prev[-1])


try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:                                  # pragma: no cover
    HAS_NUMBA = False

    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]


@njit(cache=True)
def _sub_frechet_dp(d):
    """Discrete Frechet DP with a free start and free end in the SECOND curve."""
    n, m = d.shape
    prev = d[0].copy()                     # free START: begin anywhere in the ref
    st_prev = np.arange(m).astype(np.int32)
    for i in range(1, n):
        cur = np.empty(m, np.float64)
        st = np.empty(m, np.int32)
        cur[0] = max(d[i, 0], prev[0])
        st[0] = st_prev[0]
        for j in range(1, m):
            a, b, c = prev[j], prev[j - 1], cur[j - 1]
            if a <= b and a <= c:
                best, s = a, st_prev[j]
            elif b <= c:
                best, s = b, st_prev[j - 1]
            else:
                best, s = c, st[j - 1]
            v = d[i, j]
            cur[j] = v if v > best else best
            st[j] = s
        prev, st_prev = cur, st
    j = int(np.argmin(prev))               # free END: stop anywhere in the ref
    return prev[j], st_prev[j], j


def subcurve_frechet(query: np.ndarray, ref: np.ndarray) -> tuple[float, int, int]:
    """(distance, start, end) for the best-matching SUBCURVE of `ref`.

    The whole query is matched against a freely-chosen contiguous piece of the
    reference — the right comparison for an intervention, which is a fragment of a
    task with no known location in any demonstration.

    Unlike subsequence DTW this needs no length bound. DTW sums along its path, so
    squashing a long query onto a couple of reference points is cheap and drives the
    score toward zero; Frechet takes the MAX over the coupling, so squashing cannot
    score better than the query's own spread about that point. Being a max, it is
    sensitive to a single outlying step.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(ref, dtype=np.float64)
    if len(q) == 0 or len(r) == 0:
        return float("nan"), 0, 0
    # cdist avoids the large (n, m, D) temporary a broadcast subtraction would build
    d = cdist(q, r)
    v, s, e = _sub_frechet_dp(d)
    return float(v), int(s), int(e)


@njit(cache=True)
def _dtw_fixed_window_min(D, L, stride):
    """Smallest DTW over every equal-length window of the reference.

    D is the (n, m) pairwise distance matrix. Each candidate is the ordinary
    pinned DTW between the whole query and D[:, s:s+L] with L = len(query), so
    the compared piece of the demonstration always has the query's length and no
    amount of warping can collapse it. Every window is a candidate, so there is
    always a nearest one — however far away it is.
    """
    n, m = D.shape
    best = np.inf
    prev = np.empty(L, np.float64)
    cur = np.empty(L, np.float64)
    s = 0
    while s + L <= m:
        acc = 0.0
        for j in range(L):                       # first row: cumulative
            acc += D[0, s + j] ** 2
            prev[j] = acc
        for i in range(1, n):
            cur[0] = D[i, s] ** 2 + prev[0]
            for j in range(1, L):
                a, b, c = prev[j], cur[j - 1], prev[j - 1]
                mn = a if (a <= b and a <= c) else (b if b <= c else c)
                cur[j] = D[i, s + j] ** 2 + mn
            for j in range(L):
                prev[j] = cur[j]
        if prev[L - 1] < best:
            best = prev[L - 1]
        s += stride
    return np.sqrt(best)


@njit(cache=True)
def _frechet_fixed_window_min(D, L, stride):
    """Smallest discrete Frechet over every equal-length window of the reference."""
    n, m = D.shape
    best = np.inf
    prev = np.empty(L, np.float64)
    cur = np.empty(L, np.float64)
    s = 0
    while s + L <= m:
        run = D[0, s]                            # first row: running max
        prev[0] = run
        for j in range(1, L):
            v = D[0, s + j]
            run = v if v > run else run
            prev[j] = run
        for i in range(1, n):
            a0 = D[i, s]
            cur[0] = a0 if a0 > prev[0] else prev[0]
            for j in range(1, L):
                a, b, c = prev[j], cur[j - 1], prev[j - 1]
                mn = a if (a <= b and a <= c) else (b if b <= c else c)
                v = D[i, s + j]
                cur[j] = v if v > mn else mn
            for j in range(L):
                prev[j] = cur[j]
        if prev[L - 1] < best:
            best = prev[L - 1]
        s += stride
    return best


def frechet_fixed_window_to_nearest(query: np.ndarray, references: list[np.ndarray],
                                    stride: int = 0) -> float:
    """Discrete Frechet to the closest EQUAL-LENGTH subtrajectory of any reference.

    The Frechet counterpart of dtw_fixed_window_to_nearest: the compared piece of
    the demonstration always has the query's length, so neither curve can be
    stretched over the other. Differs from subcurve_frechet_to_nearest, which lets
    the matched piece take any length.
    """
    if not references or len(query) == 0:
        return float("nan")
    q = np.ascontiguousarray(query, dtype=np.float64)
    L = len(q)
    step = stride if stride > 0 else 1      # exhaustive: exact minimum
    best = float("inf")
    for ref in references:
        r = np.ascontiguousarray(ref, dtype=np.float64)
        if len(r) == 0:
            continue
        if len(r) < L:                            # reference shorter than the query
            best = min(best, discrete_frechet(q, _resample(r, L)))
            continue
        best = min(best, float(_frechet_fixed_window_min(cdist(q, r), L, step)))
    return best if np.isfinite(best) else float("nan")


def dtw_fixed_window_to_nearest(query: np.ndarray, references: list[np.ndarray],
                                stride: int = 0) -> float:
    """DTW to the closest EQUAL-LENGTH subtrajectory of any reference.

    The window is pinned to the query's length, which stops the collapse that
    unconstrained subsequence DTW allows (a long intervention matching one timestep).
    Every start position is tested, so the result is the exact minimum over all
    equal-length segments. `stride` > 0 trades exactness for speed.
    """
    if not HAS_DTW or not references or len(query) == 0:
        return float("nan")
    q = np.ascontiguousarray(query, dtype=np.float64)
    L = len(q)
    step = stride if stride > 0 else 1      # exhaustive: exact minimum
    best = float("inf")
    for ref in references:
        r = np.ascontiguousarray(ref, dtype=np.float64)
        if len(r) == 0:
            continue
        if len(r) < L:                            # reference shorter than the query
            best = min(best, dtw_to_nearest(q, [r]))
            continue
        best = min(best, float(_dtw_fixed_window_min(cdist(q, r), L, step)))
    return best if np.isfinite(best) else float("nan")


def subcurve_frechet_to_nearest(query: np.ndarray,
                                references: list[np.ndarray]) -> float:
    """Minimum subcurve-Frechet distance from query to any reference."""
    if not references or len(query) == 0:
        return float("nan")
    best = float("inf")
    for ref in references:
        if len(ref):
            best = min(best, subcurve_frechet(query, ref)[0])
    return best if np.isfinite(best) else float("nan")


def frechet_to_nearest(query: np.ndarray, references: list[np.ndarray],
                       n_resample: int = 64, subsequence: bool = False) -> float:
    """Minimum discrete Frechet distance from query to any reference.

    `subsequence=True` (for intervention/rollout segments) matches the query against
    the best SUBCURVE of each reference via subcurve_frechet — a fragment of a task
    has no counterpart in a whole demonstration. Whole episodes use the plain form,
    where both curves are resampled to `n_resample` points so the discrete distance
    approximates the continuous one and the O(n*m) DP stays cheap.
    """
    if not references or len(query) == 0:
        return float("nan")
    if subsequence:
        return subcurve_frechet_to_nearest(query, references)
    q = _resample(np.asarray(query, dtype=np.float64), n_resample)
    return float(min(discrete_frechet(q, _resample(np.asarray(r, dtype=np.float64),
                                                   n_resample))
                     for r in references if len(r)))


def _psi(n_ref: int, subsequence: bool):
    """Endpoint relaxation for DTW.

    Plain DTW pins both endpoints: the query's first/last point must align with the
    reference's first/last. That is right when comparing two whole episodes, but
    wrong for an intervention segment, which is a FRAGMENT of a task — pinning
    forces a short segment to span a whole demo, so the distance mostly measures
    the length mismatch. `psi=(0, 0, n, n)` relaxes only the REFERENCE's two ends,
    i.e. open-begin/open-end subsequence DTW: the whole query must still be matched
    (junk in it is still penalised), but it may sit anywhere in the reference.
    """
    return (0, 0, n_ref, n_ref) if subsequence else None


def _best_reference(query, references, subsequence):
    """(reference, psi) minimising DTW distance to query, or (None, None)."""
    q = np.ascontiguousarray(query, dtype=np.float64)
    best, best_ref, best_psi = float("inf"), None, None
    for ref in references:
        if not len(ref):
            continue
        r = np.ascontiguousarray(ref, dtype=np.float64)
        psi = _psi(len(r), subsequence)
        kw = {"psi": psi} if psi is not None else {}
        d = dtw_ndim.distance(q, r, use_c=True, **kw)
        if d < best:
            best, best_ref, best_psi = d, r, psi
    return best_ref, best_psi


def dtw_path_deviations(query: np.ndarray,
                        references: list[np.ndarray],
                        subsequence: bool = False) -> np.ndarray | None:
    """Per-step deviations along the DTW-optimal alignment to the nearest reference.

    DTW picks the alignment (removing differences in execution speed) by minimising
    TOTAL deviation; this returns the deviation at each aligned pair along that same
    alignment, so callers can summarise it however they need — max for a Frechet-like
    worst case, mean for a length-independent average. Returns None if unavailable.
    """
    if not HAS_DTW or not references or len(query) == 0:
        return None
    q = np.ascontiguousarray(query, dtype=np.float64)
    best_ref, psi = _best_reference(q, references, subsequence)
    if best_ref is None:
        return None
    kw = {"psi": psi} if psi is not None else {}
    idx = np.asarray(dtw_ndim.warping_path(q, best_ref, **kw), dtype=np.int64)
    return np.linalg.norm(q[idx[:, 0]] - best_ref[idx[:, 1]], axis=1)


def dtw_frechet_to_nearest(query: np.ndarray, references: list[np.ndarray],
                           subsequence: bool = False) -> float:
    """Largest deviation along the DTW-optimal alignment to the nearest reference.

    An upper bound on the discrete Frechet distance, which minimises the worst
    deviation over all monotone couplings rather than over the one DTW chose.
    """
    dev = dtw_path_deviations(query, references, subsequence)
    return float("nan") if dev is None else float(dev.max())


def dtw_mean_to_nearest(query: np.ndarray, references: list[np.ndarray],
                        subsequence: bool = False) -> float:
    """MEAN deviation along the DTW-optimal alignment to the nearest reference.

    The trajectory-level counterpart of the pointwise kNN distance: same z-scored
    state space and same units, but the reference is a whole demonstration walked
    in step with the query rather than a cloud of independent states. Unlike raw
    DTW — which sums along the path and so grows like sqrt(T) even at constant
    per-step deviation — this is length-independent, so long and short episodes
    are directly comparable.
    """
    dev = dtw_path_deviations(query, references, subsequence)
    return float("nan") if dev is None else float(dev.mean())


def dtw_to_nearest(query: np.ndarray, references: list[np.ndarray],
                   subsequence: bool = False) -> float:
    """
    Minimum multivariate DTW distance from query to any reference trajectory.

    Parameters
    ----------
    query      : (T_q, D) float64 array (already normalized)
    references : list of (T_i, D) arrays (already normalized)

    Returns
    -------
    Minimum DTW distance, or nan if DTW is unavailable or inputs are empty.
    """
    if not HAS_DTW or not references or len(query) == 0:
        return float("nan")
    q = np.ascontiguousarray(query, dtype=np.float64)
    best = float("inf")
    for ref in references:
        r = np.ascontiguousarray(ref, dtype=np.float64)
        psi = _psi(len(r), subsequence)
        kw = {"psi": psi} if psi is not None else {}
        # C backend: identical result, far faster than the pure-Python default
        d = dtw_ndim.distance(q, r, use_c=True, **kw)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Per-demo DTW computation
# ---------------------------------------------------------------------------

def compute_demo_dtw(
    demos: list[dict],
    pretrain_trajs_norm: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    min_segment_length: int,
    desc: str = "DTW",
) -> dict[str, np.ndarray]:
    """
    Compute DTW distances for every demo against the pretrain reference set.

    Produces three distance lists:
      full               — one distance per demo (full trajectory)
      intervention_segs  — one distance per contiguous intervention segment
      rollout_segs       — one distance per contiguous rollout segment

    Parameters
    ----------
    demos               : list of dicts with keys 'states' (T,D) raw and
                          'intervention_mask' (T,) bool
    pretrain_trajs_norm : list of (T_i, D) normalized pretrain trajectories
    mean, std           : normalizer params from fit_normalizer() in ood_measures
    min_segment_length  : minimum segment length to include in segment analysis
    desc                : label shown in the tqdm progress bar

    Returns
    -------
    dict with keys "full", "intervention_segs", "rollout_segs",
    each a float64 np.ndarray of DTW distances
    """
    full, iv_segs, ro_segs = [], [], []
    f_full, f_iv_segs = [], []
    df_full, df_iv_segs = [], []
    dm_full, dm_iv_segs = [], []
    dw_full, dw_iv_segs = [], []
    fw_full, fw_iv_segs = [], []

    def _path_summary(traj, subsequence=False):
        """(max, mean) deviation along one DTW alignment — computed once, not twice."""
        dev = dtw_path_deviations(traj, pretrain_trajs_norm, subsequence)
        if dev is None:
            return float("nan"), float("nan")
        return float(dev.max()), float(dev.mean())

    # Demo index per segment, so callers can aggregate segments per episode.
    iv_seg_demo, ro_seg_demo = [], []

    for i, d in enumerate(tqdm(demos, desc=f"    DTW ({desc})", leave=False)):
        s_norm = (d["states"] - mean) / std
        iv     = d["intervention_mask"]

        # Full trajectory vs nearest pretrain demo
        full.append(dtw_to_nearest(s_norm, pretrain_trajs_norm))
        f_full.append(frechet_to_nearest(s_norm, pretrain_trajs_norm))
        mx, mn = _path_summary(s_norm)
        df_full.append(mx)
        dm_full.append(mn)
        dw_full.append(dtw_fixed_window_to_nearest(s_norm, pretrain_trajs_norm))
        fw_full.append(frechet_fixed_window_to_nearest(s_norm, pretrain_trajs_norm))

        # Contiguous intervention segments
        # Segments are FRAGMENTS of a task, so they match a sub-portion of a demo
        # rather than the whole thing (see _psi). Whole episodes keep pinned ends.
        for idx in extract_segments(iv, min_segment_length):
            iv_segs.append(dtw_to_nearest(s_norm[idx], pretrain_trajs_norm,
                                          subsequence=True))
            f_iv_segs.append(frechet_to_nearest(s_norm[idx], pretrain_trajs_norm,
                                                subsequence=True))
            mx, mn = _path_summary(s_norm[idx], subsequence=True)
            df_iv_segs.append(mx)
            dm_iv_segs.append(mn)
            dw_iv_segs.append(
                dtw_fixed_window_to_nearest(s_norm[idx], pretrain_trajs_norm))
            fw_iv_segs.append(
                frechet_fixed_window_to_nearest(s_norm[idx], pretrain_trajs_norm))
            iv_seg_demo.append(i)

        # Contiguous rollout segments
        for idx in extract_segments(~iv, min_segment_length):
            ro_segs.append(dtw_to_nearest(s_norm[idx], pretrain_trajs_norm,
                                          subsequence=True))
            ro_seg_demo.append(i)

    return dict(
        full=np.array(full, dtype=np.float64),
        intervention_segs=np.array(iv_segs, dtype=np.float64),
        rollout_segs=np.array(ro_segs, dtype=np.float64),
        intervention_segs_demo=np.array(iv_seg_demo, dtype=np.int64),
        rollout_segs_demo=np.array(ro_seg_demo, dtype=np.int64),
        frechet_full=np.array(f_full, dtype=np.float64),
        frechet_intervention_segs=np.array(f_iv_segs, dtype=np.float64),
        dtwfrechet_full=np.array(df_full, dtype=np.float64),
        dtwfrechet_intervention_segs=np.array(df_iv_segs, dtype=np.float64),
        dtwmean_full=np.array(dm_full, dtype=np.float64),
        dtwmean_intervention_segs=np.array(dm_iv_segs, dtype=np.float64),
        dtwwin_full=np.array(dw_full, dtype=np.float64),
        dtwwin_intervention_segs=np.array(dw_iv_segs, dtype=np.float64),
        frechetwin_full=np.array(fw_full, dtype=np.float64),
        frechetwin_intervention_segs=np.array(fw_iv_segs, dtype=np.float64),
    )
