#!/bin/bash
# Simulation environment: conda env, pinned Python packages, and robosuite / robocasa / robomimic
# at the commits the paper's results were produced with (cloned into third_party/).
#
#   bash setup_sim.sh [env_name]        # default env name: blendagger
set -euo pipefail
cd "$(dirname "$0")"
ENV=${1:-blendagger}
DEPS=third_party

eval "$(conda shell.bash hook)"
conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
pip install -r requirements-sim.txt

# --no-deps: their setup.py pins (e.g. torch==2.0.1) are older than the versions actually used.
install_pinned() {
  local name=$1 url=$2 commit=$3
  [ -d "$DEPS/$name" ] || git clone "$url" "$DEPS/$name"
  git -C "$DEPS/$name" checkout -q "$commit"
  pip install --no-deps -e "$DEPS/$name"
}
mkdir -p "$DEPS"
install_pinned robosuite https://github.com/ARISE-Initiative/robosuite.git e94119512f737635111ae651768c17ad1be72859
install_pinned robocasa https://github.com/robocasa/robocasa.git 756598a5be52e052339bb2d957426e39015c2afb
install_pinned robomimic https://github.com/ARISE-Initiative/robomimic.git 271a76c2d55c8b0f94d3d589f26fcae0d47f64a1

# Adds robomimic's "ram" cache mode, used by the seed-rerun training configs (conf/robomimic/finetune_seed*.json).
PATCH=$PWD/training/robomimic-ram-cache.patch
if ! git -C "$DEPS/robomimic" apply --reverse --check "$PATCH" 2>/dev/null; then
  git -C "$DEPS/robomimic" apply "$PATCH"
fi

python "$DEPS/robosuite/robosuite/scripts/setup_macros.py"
python "$DEPS/robocasa/robocasa/scripts/setup_macros.py"
python "$DEPS/robocasa/robocasa/scripts/download_kitchen_assets.py"   # ~5 GB, asks before downloading

echo "Done. conda activate $ENV"
