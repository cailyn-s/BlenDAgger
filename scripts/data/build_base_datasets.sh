#!/bin/bash
# Rebuild the base demonstration datasets (patched_*.hdf5) from RoboCasa's human datasets.
#
#   scripts/data/build_base_datasets.sh                    # all three tasks
#   scripts/data/build_base_datasets.sh PrepareCoffee      # or only the ones named
#
# Outputs go to paths_db.dataset_dir (default data/). Existing outputs are never overwritten.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."

declare -A OUT=(
  [PrepareCoffee]=patched_coffee_demo_im128.hdf5
  [MicrowaveThawing]=patched_thawing_demo_im128.hdf5
  [PnPCounterToCab]=patched_pnp_demo_gentex_im128_randcams.hdf5
)
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(PrepareCoffee MicrowaveThawing PnPCounterToCab)

DATA=$(python -c "import repo_paths; print(repo_paths.load_paths_db().dataset_dir)" | tail -1)
ROBOCASA=$(python -c "import os, robocasa; print(os.path.dirname(robocasa.__file__))" 2>/dev/null | tail -1)
mkdir -p "$DATA"

for task in "${TASKS[@]}"; do
  out="$DATA/${OUT[$task]:?unknown task $task (expected PrepareCoffee, MicrowaveThawing or PnPCounterToCab)}"
  if [ -e "$out" ]; then echo "exists, skipped: $out"; continue; fi
  src=$(python -c "from robocasa.utils.dataset_registry import get_ds_path; print(get_ds_path('$task', ds_type='human_im'))" 2>/dev/null | tail -1)
  [ -f "$src" ] || python "$ROBOCASA/scripts/download_datasets.py" --tasks "$task" --ds_types human_im
  tmp=$(mktemp -d)
  python scripts/data/backfill_datasets.py "$tmp" --pretrain_path "$src"
  mv "$tmp/patched_$(basename "$src")" "$out"
  rmdir "$tmp"
  python scripts/data/add_dataset_keys.py "$out"
  echo "built: $out"
done
