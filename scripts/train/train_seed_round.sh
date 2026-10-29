#!/bin/bash
# Retrain one PrepareCoffee round with a different training seed on already-collected data.
#
#   scripts/train/train_seed_round.sh <blend|hg_dagger> <round> <seed_n>     # seed_n: 2 (456) or 3 (789)
#
# Round N fine-tunes the base checkpoint on base demos + collected rounds 1..N; results in results/coffee_realmatch_seed<seed_n>_*.
set -eu
cd "$(dirname "$(realpath "$0")")/../.."
ALGO=${1:?usage: $0 <blend|hg_dagger> <round> <seed_n>}
ROUND=${2:?usage: $0 <blend|hg_dagger> <round> <seed_n>}
SEED_N=${3:?usage: $0 <blend|hg_dagger> <round> <seed_n>}
case "$ALGO" in
  blend)     CFG=sim/coffee/blendagger_seed${SEED_N};     SRCNAME=coffee_realmatch_combined_blend_PrepareCoffee ;;
  hg_dagger) CFG=sim/coffee/hg_dagger_seed${SEED_N}; SRCNAME=coffee_realmatch_hg_dagger_PrepareCoffee ;;
  *) echo "algo must be blend or hg_dagger"; exit 1 ;;
esac
SEED_CFG=./conf/robomimic/finetune_seed${SEED_N}.json
DATA=$(python -c "import repo_paths; print(repo_paths.load_paths_db().sim_data_dir)")
SRC=$DATA/$SRCNAME
LINKDIR=$DATA/_seed${SEED_N}_links/${ALGO}_r1to${ROUND}
[ -d "$SRC" ] || { echo "collected data missing: $SRC"; exit 1; }
[ -f "$SEED_CFG" ] || { echo "seed train config missing: $SEED_CFG"; exit 1; }

# main.py trains on every *dataset_full.hdf5 under train_data, so link exactly rounds 1..N.
rm -rf "$LINKDIR"; mkdir -p "$LINKDIR"
for r in $(seq 1 "$ROUND"); do
  n=0
  for f in "$SRC"/*_round${r}_*; do
    [ -e "$f" ] || continue
    ln -s "$f" "$LINKDIR/$(basename "$f")" && n=$((n+1))
  done
  [ "$n" -eq 0 ] && { echo "round $r has no files in $SRC"; exit 1; }
done
have=$(ls -1 "$LINKDIR"/*_dataset_full.hdf5 2>/dev/null | wc -l)
[ "$have" -eq "$ROUND" ] || { echo "expected $ROUND dataset_full files, found $have"; exit 1; }

python -u main.py --config-name "$CFG" \
  pipeline.run_collect=false \
  training.train_data="$LINKDIR" \
  ++training.run_all_iters=false \
  training.num_epochs=150 ++training.save_every_n_epochs=150 \
  training.save_rollout_freq=1000 \
  paths.config_path="$SEED_CFG"
