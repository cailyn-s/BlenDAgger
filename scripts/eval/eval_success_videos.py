"""Re-run specific rollouts of an already-scored run and keep only the videos that succeed.

Renders supplementary success videos by retrying seeds; not used for any reported success rate.

Rollouts are recorded to a temp file and kept only if the episode succeeded, so the output is
all successes. Scores are NOT written back -- eval_composite.py owns the result JSONs.

  python scripts/eval/eval_success_videos.py <run> [epoch] [seeds]

<seeds> defaults to the rollouts that succeeded in the stored result; pass a comma list, or
"all" for every rollout. --retries N re-attempts a seed that fails (sampling is not seeded on
torch, so a rollout that succeeded once may not succeed again).
"""
import os, sys, json, glob, argparse, tempfile
import numpy as np, torch, imageio

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # so eval_composite imports from any cwd
import eval_composite as EC
from repo_paths import load_fixed_env
from evaluation.video_rollout import run_rollout
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils


def stored(run_key, round_tag, key):
    p = os.path.join(EC.OUT_DIR, f"{run_key}{round_tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("epochs", {}).get(key, {}).get("success")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("epoch", nargs="?", type=int, default=150)
    ap.add_argument("seeds", nargs="?", default="stored")
    ap.add_argument("--retries", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = EC.task_of(a.run)
    c = EC.TASKS[task]
    horizon = int(EC._HORIZON_ENV) if EC._HORIZON_ENV else c.get("horizon", 1000)

    ckpts = EC.epoch_ckpts(os.path.join(EC.RESULTS_BASE, a.run), [a.epoch])
    if not ckpts:
        raise SystemExit(f"[vid] {a.run}: no checkpoint at epoch {a.epoch}")
    ep, ckpt = ckpts[0]
    key = f"epoch{ep}"
    import re
    m = re.search(r"_round(\d+)_", ckpt)
    round_tag = f"_round{m.group(1)}" if m else ""
    label = f"{a.run}{round_tag}_{key}"

    prev = stored(a.run, round_tag, key)
    if a.seeds == "stored":
        if not prev:
            raise SystemExit(f"[vid] no stored result for {label}; pass seeds explicitly")
        seeds = [i for i, v in enumerate(prev) if v >= 0.5]
    elif a.seeds == "all":
        seeds = list(range(len(prev) if prev else EC.NUM_ROLLOUTS))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]

    outdir = EC.video_dir(label)
    os.makedirs(outdir, exist_ok=True)
    print(f"[vid] {label} | horizon {horizon} | {len(seeds)} seeds -> {outdir}", flush=True)

    _, ck0 = FileUtils.policy_from_checkpoint(ckpt_path=c["pretrained"], device=device, verbose=False)
    cfg0 = json.loads(ck0["config"]); seq_length = cfg0["train"]["frame_stack"]
    gconf = config_factory(cfg0["algo_name"]); gconf.update(cfg0)
    ObsUtils.initialize_obs_utils_with_config(gconf)
    env_meta = FileUtils.get_env_metadata_from_dataset(c["dataset"])
    env_meta["env_kwargs"]["layout_ids"] = [-1]; env_meta["env_kwargs"]["style_ids"] = [-1]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=c["dataset"], action_keys=gconf.train.action_keys, all_obs_keys=gconf.all_obs_keys)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name=env_meta["env_name"],
        render=False, render_offscreen=True, use_image_obs=shape_meta["use_images"])
    state_dict = load_fixed_env(c["fixed"])

    cdict = json.loads(FileUtils.maybe_dict_from_checkpoint(ckpt)["config"])
    conf = config_factory(cdict["algo_name"]); conf.update(cdict)
    model = algo_factory(algo_name=conf.algo_name, config=conf,
                         obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=device)
    model.deserialize(FileUtils.maybe_dict_from_checkpoint(ckpt)["model"]); model.set_eval()
    policy = RolloutPolicy(model)

    kept, missed = [], []
    for s in seeds:
        for attempt in range(a.retries + 1):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
            vw = imageio.get_writer(tmp.name, fps=20)
            with torch.no_grad():
                res = run_rollout(policy=policy, env=env, task=env_meta["env_name"],
                                  horizon=horizon, terminate_on_success=True, state_dict=state_dict,
                                  video_writer=vw, seed=s, use_sequence=True,
                                  seq_length=seq_length, device=device)
            vw.close()
            ok = res["Success_Rate"] >= 0.5
            if ok:
                dst = os.path.join(outdir, f"{label}_rollout_{s}.mp4")
                os.replace(tmp.name, dst); kept.append(s)
            else:
                os.unlink(tmp.name)
            print(f"  seed {s} attempt {attempt + 1}: {'SUCCESS -> kept' if ok else 'failed'}", flush=True)
            if ok:
                break
        else:
            missed.append(s)

    env.close()
    print(f"\n[vid] kept {len(kept)} success videos in {outdir}")
    print(f"[vid] seeds kept: {kept}")
    if missed:
        print(f"[vid] did not reproduce (no video saved): {missed}")


if __name__ == "__main__":
    main()
