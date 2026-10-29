"""
Intervention timing analysis: when do interventions start and end?

Two levels of granularity:
  segment-level — each contiguous run of True in intervention_mask is one segment
  group-level   — nearby segments within gap_threshold timesteps are merged into
                  one logical intervention group

The group-level representation makes methods comparable: if one intervenes at
t=1-3 and t=5-7 while another intervenes at t=2-7, both are treated as one
intervention group starting at t=1 and t=2 respectively.

Times are reported both as absolute timesteps and as fractions of episode length
(0 = episode start, 1 = episode end) so that episodes of different lengths are
comparable.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Segment / group extraction
# ---------------------------------------------------------------------------

def extract_boundaries(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Extract (start, end_excl) for each contiguous True run.
    end_excl is one past the last True index (Python-slice convention).
    """
    boundaries, in_run, start = [], False, 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start, in_run = i, True
        elif not val and in_run:
            boundaries.append((start, i))
            in_run = False
    if in_run:
        boundaries.append((start, len(mask)))
    return boundaries


def merge_groups(
    boundaries: list[tuple[int, int]],
    gap_threshold: int,
) -> list[tuple[int, int]]:
    """
    Merge consecutive segments into groups when the gap between them
    is <= gap_threshold timesteps.

    Parameters
    ----------
    boundaries    : list of (start, end_excl) from extract_boundaries
    gap_threshold : maximum gap in timesteps to still count as one group

    Returns
    -------
    list of (group_start, group_end_excl) tuples
    """
    if not boundaries:
        return []
    groups = [list(boundaries[0])]
    for start, end in boundaries[1:]:
        gap = start - groups[-1][1]
        if gap <= gap_threshold:
            groups[-1][1] = end       # extend current group
        else:
            groups.append([start, end])
    return [tuple(g) for g in groups]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_timing(demos: list[dict], gap_threshold: int) -> dict[str, np.ndarray]:
    """
    Compute intervention start/end timing across all demos.

    Parameters
    ----------
    demos         : list of dicts with 'intervention_mask' (T,) bool and optional
                    'grip_window' — (first, last) movable timestep; t=0 is its
                    start and segments are clipped to it
    gap_threshold : segments within this many timesteps are merged into one group

    Returns
    -------
    dict with float64 arrays (all values are per-segment or per-group):
        segment_starts       : absolute timestep of each segment start
        segment_ends         : absolute timestep of each segment end (inclusive)
        segment_start_fracs  : segment start as fraction of episode length
        segment_end_fracs    : segment end as fraction of episode length
        segment_lengths      : length of each raw segment in timesteps

        group_starts         : absolute timestep of each group start
        group_ends           : absolute timestep of each group end (inclusive)
        group_start_fracs    : group start as fraction of episode length
        group_end_fracs      : group end as fraction of episode length
        group_lengths        : total span of each group (end - start) in timesteps
        group_n_segments     : how many raw segments were merged into each group
    """
    seg_starts, seg_ends, seg_lengths               = [], [], []
    seg_start_fracs, seg_end_fracs                  = [], []
    grp_starts, grp_ends, grp_lengths, grp_n_segs  = [], [], [], []
    grp_start_fracs, grp_end_fracs                  = [], []

    for d in demos:
        iv = d["intervention_mask"]
        # Everything is measured over the grip-held window, matching task_time: the
        # robot cannot move outside it, so the leading steps only measure how long
        # the operator took to squeeze and the trailing ones are recording that ran
        # on after they let go. Whole episode when there is no grip mask (sim).
        t0, t1 = d.get("grip_window", (0, len(iv) - 1))
        T = t1 - t0 + 1

        # Clip to the window, dropping anything wholly outside it: the mask often
        # stays True past the last press, and that tail is not robot motion.
        boundaries = [(max(s, t0), min(e, t1 + 1))
                      for s, e in extract_boundaries(iv)]
        boundaries = [(s, e) for s, e in boundaries if e > s]

        # Raw segments
        for s, e in boundaries:
            seg_starts.append(s - t0)
            seg_ends.append(e - 1 - t0)
            seg_lengths.append(e - s)
            seg_start_fracs.append((s - t0) / T)
            seg_end_fracs.append((e - 1 - t0) / T)

        # Merged groups
        groups = merge_groups(boundaries, gap_threshold)
        for gs, ge in groups:
            # Count how many raw segments fall inside this group
            n_segs = sum(1 for s, e in boundaries if gs <= s and e <= ge)
            grp_starts.append(gs - t0)
            grp_ends.append(ge - 1 - t0)
            grp_lengths.append(ge - gs)
            grp_n_segs.append(n_segs)
            grp_start_fracs.append((gs - t0) / T)
            grp_end_fracs.append((ge - 1 - t0) / T)

    return dict(
        segment_starts      =np.array(seg_starts,      dtype=np.float64),
        segment_ends        =np.array(seg_ends,        dtype=np.float64),
        segment_lengths     =np.array(seg_lengths,     dtype=np.float64),
        segment_start_fracs =np.array(seg_start_fracs, dtype=np.float64),
        segment_end_fracs   =np.array(seg_end_fracs,   dtype=np.float64),
        group_starts        =np.array(grp_starts,      dtype=np.float64),
        group_ends          =np.array(grp_ends,        dtype=np.float64),
        group_lengths       =np.array(grp_lengths,     dtype=np.float64),
        group_n_segments    =np.array(grp_n_segs,      dtype=np.float64),
        group_start_fracs   =np.array(grp_start_fracs, dtype=np.float64),
        group_end_fracs     =np.array(grp_end_fracs,   dtype=np.float64),
    )


def fill_short_gaps(mask, gap_threshold: int):
    """`mask` with pauses of <= gap_threshold timesteps filled in.

    One correction interrupted by a momentary pause is one intervention, not two. Applied to the
    mask itself rather than per metric, so every metric reads the same interventions.
    """
    flags = np.asarray(mask, dtype=bool).copy()
    if gap_threshold <= 0 or not flags.any():
        return flags
    for start, end in merge_groups(extract_boundaries(flags), gap_threshold):
        flags[start:end] = True
    return flags
