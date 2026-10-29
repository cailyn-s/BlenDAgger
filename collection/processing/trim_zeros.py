#!/usr/bin/env python3
"""Trim stationary (no-movement) steps from a sim dataset_full.hdf5 into a SEPARATE file.

Follows the trim-zeros step of the real-robot pipeline in the companion xArm repository: keep
moving frames plus a LEAD-frame run-up and a TAIL-frame settle; drop stationary frames throughout
the episode, across ALL frames (autonomous rollout and intervention alike).

Movement signal = observed base-frame EEF motion (robot0_base_to_eef_pos/quat +
robot0_gripper_qpos, logged each step). A frame is "active" if EEF position, EEF orientation, OR
gripper moved past its threshold. The sim default of 1e-3 is higher than the real-robot 1e-4
because the sim per-step EEF-motion floor is higher (the policy is always driving).
The input dataset_full is never modified — a *_trimmed.hdf5 is written alongside it.
"""
import h5py, numpy as np, argparse, shutil, os

LEAD, TAIL = 5, 10   # frames kept before first / after last movement

def quat_angle(q):
    q = np.asarray(q, dtype=np.float64)
    d = np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), -1.0, 1.0)   # |dot| handles quat double-cover
    return np.insert(2.0 * np.arccos(d), 0, 0.0)

def active_mask(eef_pos, eef_quat, gripper_pos, pos_th, rot_th, grip_th):
    p = np.asarray(eef_pos, dtype=np.float64)
    posd = np.insert(np.linalg.norm(np.diff(p, axis=0), axis=1), 0, 0.0)
    rotd = quat_angle(eef_quat)
    g = np.asarray(gripper_pos, dtype=np.float64)
    if g.ndim == 1:
        g = g[:, None]
    gd = np.insert(np.linalg.norm(np.diff(g, axis=0), axis=1), 0, 0.0)
    return (posd > pos_th) | (rotd > rot_th) | (gd > grip_th)

def keep_mask(active):
    n = len(active); idx = np.where(active)[0]
    if len(idx) == 0:
        return np.zeros(n, dtype=bool)   # episode never moves -> keep nothing
    first, last = idx[0], idx[-1]
    keep = active.copy()
    keep[max(0, first - LEAD):first] = True
    keep[last:min(n, last + 1 + TAIL)] = True
    return keep

def trim(in_path, out_path, pos_th, rot_th, grip_th):
    shutil.copy(in_path, out_path)   # preserve structure/attrs, then subset per-step arrays in place
    ti = to = 0
    with h5py.File(out_path, "a") as f:
        grp = f["data"]
        for demo in list(grp.keys()):
            d = grp[demo]
            n = d["actions"].shape[0]
            if "eef_pos" not in d:      # no observed-eef signal (e.g. pre-existing data) -> keep all
                to += n; ti += n; continue
            keep = keep_mask(active_mask(np.asarray(d["eef_pos"]), np.asarray(d["eef_quat"]),
                                         np.asarray(d["gripper_pos"]), pos_th, rot_th, grip_th))
            ti += n; to += int(keep.sum())
            for k in list(d.keys()):
                if d[k].shape and d[k].shape[0] == n:   # per-step dataset
                    data = np.asarray(d[k])[keep]
                    del d[k]; d.create_dataset(k, data=data)
            if "num_samples" in d.attrs:
                d.attrs["num_samples"] = int(keep.sum())
    print(f"[trim] {os.path.basename(in_path)} -> {os.path.basename(out_path)}: "
          f"kept {to}/{ti} steps ({100*to/max(ti,1):.1f}%)")
    return ti, to

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default=None, help="default: <dataset>_trimmed.hdf5")
    p.add_argument("--pos-thresh", type=float, default=1e-3, help="EEF position delta (m)")
    p.add_argument("--rot-thresh", type=float, default=1e-3, help="EEF orientation delta (rad)")
    p.add_argument("--grip-thresh", type=float, default=1e-3, help="gripper qpos delta")
    a = p.parse_args()
    out = a.output or a.dataset.replace(".hdf5", "_trimmed.hdf5")
    trim(a.dataset, out, a.pos_thresh, a.rot_thresh, a.grip_thresh)
