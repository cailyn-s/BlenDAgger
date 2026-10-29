"""Eval one composite-task run (PrepareCoffee / MicrowaveThawing), with subtask breakdowns.

Same machinery as eval_realmatch.py, differing where the composite
tasks differ:
  * horizon 1000 for coffee, 1500 for thawing, not 500
  * finetune runs are scored at epoch 150 (the final checkpoint); the base policy is labelled
    with its training epoch (1000; 900 for Pick and Place)
  * can score the BASE (pretrained) policy, which eval_realmatch.py has no path for
  * reports per-subtask rates alongside overall success, via evaluation/patch_subtask_success.py

Resumable: rollouts already scored are read back from the per-run JSON and skipped, so a
re-run only does the remainder. One run per process so GPU/host memory is freed between runs.

    python scripts/eval/eval_composite.py <run_folder_name> [epoch] [num_rollouts]
    python scripts/eval/eval_composite.py base:PrepareCoffee   [--]  [num_rollouts]
"""
import os, sys, json, glob, gc, re
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch, imageio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)
from repo_paths import load_paths_db, load_fixed_env
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.algo.algo import RolloutPolicy
from evaluation.video_rollout import run_rollout
from evaluation.patch_subtask_success import patch_subtask_success, SUBTASKS


def _patch_diffusion_config_schema():
    """Let a fine-tuned checkpoint's embedded config (adamw + warmup) load -- robomimic's
    diffusion schema omits both keys, so config_factory().update() key-locks otherwise.
    Same schema patch as training/patch_diffusion_optim.py, which only the training path applies."""
    import robomimic.config.diffusion_policy_config as dpc
    _orig = dpc.DiffusionPolicyConfig.algo_config
    def algo_config(self):
        _orig(self)
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.num_warmup_steps = 0
    dpc.DiffusionPolicyConfig.algo_config = algo_config


_patch_diffusion_config_schema()
patch_subtask_success()

_PATHS       = load_paths_db()
_CKPT        = _PATHS.lookup.diffusion
RESULTS_BASE = os.environ.get("EVAL_RESULTS_BASE", os.path.join(_PATHS.base_save_path, "results"))
DATA_DIR     = os.environ.get("EVAL_DATA_DIR", _PATHS.dataset_dir)
OUT_DIR      = os.path.join(_PATHS.base_save_path, "eval_composite_results")
VIDEO_DIR    = os.path.join(_PATHS.base_save_path, "videos")    # videos/<run_name>/<label>/<label>_rollout_<i>.mp4
_EVAL_ROUND  = os.environ.get("EVAL_ROUND")
_RGLOB       = f"*_round{_EVAL_ROUND}_*" if _EVAL_ROUND else "*"
NUM_ROLLOUTS = 100
EVAL_EPOCHS  = [150]            # finetune runs train to 150 with the final checkpoint scored
BASE_EPOCH   = 1000             # label for the base policy's training epoch
_HORIZON_ENV = os.environ.get("EVAL_HORIZON")   # override; otherwise per-task below
VIDEO_EVERY  = 20

# `pretrained` is the base policy checkpoint from conf/paths_db.yaml.
TASKS = {
  "PrepareCoffee": dict(
    dataset=f"{DATA_DIR}/patched_coffee_demo_im128.hdf5",
    pretrained=_CKPT.PrepareCoffee.pretrained_model,
    fixed=f"{REPO}/fixed_envs/fixed_env_state_coffee.npy"),
  "MicrowaveThawing": dict(
    dataset=f"{DATA_DIR}/patched_thawing_demo_im128.hdf5",
    pretrained=_CKPT.MicrowaveThawing.pretrained_model,
    fixed=f"{REPO}/fixed_envs/fixed_env_state_thawing.npy", horizon=1500),
  # horizon 500 (its registry horizon) and no subtask patch -- SUBTASKS has no entry, so it
  # reports overall success only.
  "PnPCounterToCab": dict(
    dataset=f"{DATA_DIR}/patched_pnp_demo_gentex_im128_randcams.hdf5",
    pretrained=_CKPT.PnPCounterToCab.pretrained_model,
    fixed=f"{REPO}/fixed_envs/fixed_env_state_lemon.npy", horizon=500, base_epoch=900),
}


def task_of(run):
    for t in sorted(TASKS, key=len, reverse=True):
        if t in run:
            return t
    raise SystemExit(f"[eval] cannot infer task from run name {run!r}; expected one of {list(TASKS)}")


def video_dir(label):
    """videos/<run_name>/<algorithm>/<label>/ -- run_name is the lineage prefix, algorithm the
    slice between it and the task (base runs group by task instead)."""
    if label.startswith("base_"):
        return os.path.join(VIDEO_DIR, "base", task_of(label), label)
    group = algo = "misc"
    for a in ("combined_blend", "hg_dagger", "linear_blend"):
        i = label.find(f"_{a}_")
        if i > 0:
            group, rest = label[:i], label[i + 1:]
            j = rest.find("_" + task_of(label))
            algo = rest[:j] if j > 0 else a
            break
    return os.path.join(VIDEO_DIR, group, algo, label)


def epoch_ckpts(run_dir, epochs):
    """One checkpoint per requested epoch (latest by mtime if several match)."""
    out = []
    for ep in epochs:
        cks = glob.glob(os.path.join(run_dir, _RGLOB, "*", "models", f"model_epoch_{ep}.pth"))
        if cks:
            out.append((ep, max(cks, key=os.path.getmtime)))
    return out


def summarize(succ, subs):
    d = {"success": succ, "mean": float(np.mean(succ)),
         "sem": float(np.std(succ) / np.sqrt(len(succ)))}
    d["subtasks"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(subs.items())}
    return d


def main(run, only_epoch=None, num_rollouts=None):
    nr = num_rollouts if num_rollouts is not None else NUM_ROLLOUTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_base = run.startswith("base:")
    task = run.split(":", 1)[1] if is_base else task_of(run)
    if task not in TASKS:
        raise SystemExit(f"[eval] unknown task {task!r}")
    c = TASKS[task]
    horizon = int(_HORIZON_ENV) if _HORIZON_ENV else c.get("horizon", 1000)

    if is_base:
        ckpts = [(c.get("base_epoch", BASE_EPOCH), c["pretrained"])]
        run_key = f"base_{task}"
        round_tag = ""
    else:
        run_dir = os.path.join(RESULTS_BASE, run)
        ckpts = epoch_ckpts(run_dir, [only_epoch] if only_epoch is not None else EVAL_EPOCHS)
        run_key = run
        m = re.search(r"_round(\d+)_", ckpts[0][1]) if ckpts else None
        round_tag = f"_round{m.group(1)}" if m else ""

    if not ckpts:
        print(f"[eval] {run}: no checkpoints"
              f"{' at epoch ' + str(only_epoch) if only_epoch is not None else ''}; skipping", flush=True)
        return
    print(f"[eval] {run} ({task}): {[e for e, _ in ckpts]} x {nr} rollouts, horizon {horizon}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{run_key}{round_tag}.json")
    data = (json.load(open(out_path)) if os.path.exists(out_path)
            else {"run": run_key, "task": task, "horizon": horizon, "epochs": {}})
    data.setdefault("epochs", {})

    # Env + obs spec built once from the base checkpoint's config.
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

    for ep, ckpt in ckpts:
        key = f"epoch{ep}"
        prev = data["epochs"].get(key, {})
        succ = list(prev.get("success", []))
        subs = {k: list(v) for k, v in prev.get("_subtask_raw", {}).items()}
        if len(succ) >= nr:
            print(f"[eval] {run_key} {key} already done ({np.mean(succ):.3f}); skip", flush=True)
            continue
        cdict = json.loads(FileUtils.maybe_dict_from_checkpoint(ckpt)["config"])
        conf = config_factory(cdict["algo_name"]); conf.update(cdict)
        model = algo_factory(algo_name=conf.algo_name, config=conf,
                             obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=device)
        model.deserialize(FileUtils.maybe_dict_from_checkpoint(ckpt)["model"]); model.set_eval()
        policy = RolloutPolicy(model)

        for i in range(len(succ), nr):   # seed=i fixes env/numpy RNG (same scene per index); policy sampling (torch) is not seeded
            vw = None
            if i % VIDEO_EVERY == 0:
                label = f"{run_key}{round_tag}_{key}"
                vdir = video_dir(label)
                os.makedirs(vdir, exist_ok=True)
                vw = imageio.get_writer(os.path.join(vdir, f"{label}_rollout_{i}.mp4"), fps=20)
            with torch.no_grad():
                res = run_rollout(policy=policy, env=env, task=env_meta["env_name"],
                                  horizon=horizon, terminate_on_success=True, state_dict=state_dict,
                                  video_writer=vw, seed=i, use_sequence=True,
                                  seq_length=seq_length, device=device)
            if vw is not None:
                vw.close()
            succ.append(res["Success_Rate"])
            for st in SUBTASKS.get(task, []):
                subs.setdefault(st, []).append(res.get(f"{st}_Success_Rate", 0.0))
            entry = summarize(succ, subs); entry["ckpt"] = ckpt; entry["_subtask_raw"] = subs
            data["epochs"][key] = entry
            json.dump(data, open(out_path, "w"), indent=2)
            bits = " ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(subs.items()))
            print(f"  {run_key} {key}: {i+1}/{nr} rate={np.mean(succ):.3f} | {bits}", flush=True)
        print(f"[eval] {run_key} {key} DONE mean={np.mean(succ):.3f} -> {out_path}", flush=True)
        del model, policy; gc.collect(); torch.cuda.empty_cache()
    env.close()


if __name__ == "__main__":
    _run = sys.argv[1]
    _ep = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] not in ("--", "-") else None
    _nr = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(_run, only_epoch=_ep, num_rollouts=_nr)
