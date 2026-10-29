"""Finish preparing a base dataset after backfill_datasets.py: add robot_actions (a copy of actions).

human_targets training falls back to robot_actions wherever the human was not commanding, and the
training config lists it as a dataset key, so every training file needs it. --value also sets
every alpha to that value; build_base_datasets.sh leaves alphas at 0.

    python scripts/data/add_dataset_keys.py <pretrain.hdf5> [--value V] [--dry-run]
"""
import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--value", type=float, default=None, help="also set every alpha to this value")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r" if args.dry_run else "r+") as f:
        demos = list(f["data"].keys())
        before, changed, added_robot = {}, 0, 0
        for d in demos:
            grp = f["data"][d]
            if "alphas" not in grp:
                raise SystemExit(f"{d} has no 'alphas' -- run backfill_datasets.py on this file first")
            vals = np.unique(grp["alphas"][:])
            key = float(vals[0]) if len(vals) == 1 else "mixed"
            before[key] = before.get(key, 0) + 1
            if "robot_actions" not in grp:
                added_robot += 1
                if not args.dry_run:
                    grp.create_dataset("robot_actions", data=grp["actions"][:])

            if args.value is not None and not (len(vals) == 1 and vals[0] == args.value):
                changed += 1
                if not args.dry_run:
                    grp["alphas"][...] = args.value

        verb = "would set" if args.dry_run else "set"
        print(f"{args.dataset}\n  {len(demos)} demos; alphas before: {before}")
        if args.value is not None:
            print(f"  {verb} alphas={args.value} on {changed} demo(s)")
        print(f"  {verb} robot_actions=actions on {added_robot} demo(s)")


if __name__ == "__main__":
    main()
