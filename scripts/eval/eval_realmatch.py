"""Eval one Pick and Place (PnPCounterToCab) run: 100 rollouts for EACH saved checkpoint
(epochs 50/100/150), so performance can be followed across training. Resumable per epoch
via the per-run JSON.
Run one run per process so GPU/host memory is freed between runs.

    python scripts/eval/eval_realmatch.py <run_folder_name> [epoch] [num_rollouts]
"""
import os, sys, json, glob, gc, re
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")       # use cached CLIP, never ping HF
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

def _patch_diffusion_config_schema():
    """Declare optimizer_type + num_warmup_steps on the diffusion config schema so a fine-tuned
    checkpoint's embedded config (adamw, 100 warmup steps) loads; robomimic's diffusion schema
    omits both, so config_factory().update(cdict) key-lock errors otherwise.
    Same schema patch as training/patch_diffusion_optim.py (which the training path applies)."""
    import robomimic.config.diffusion_policy_config as dpc
    _orig = dpc.DiffusionPolicyConfig.algo_config
    def algo_config(self):
        _orig(self)
        self.algo.optim_params.policy.optimizer_type = "adam"             # checkpoint config overrides to adamw
        self.algo.optim_params.policy.learning_rate.num_warmup_steps = 0  # checkpoint config overrides to 100
    dpc.DiffusionPolicyConfig.algo_config = algo_config

_patch_diffusion_config_schema()

_PATHS       = load_paths_db()
RESULTS_BASE = os.environ.get("EVAL_RESULTS_BASE", os.path.join(_PATHS.base_save_path, "results"))
DATA_DIR     = os.environ.get("EVAL_DATA_DIR", _PATHS.dataset_dir)
OUT_DIR      = os.path.join(_PATHS.base_save_path, "eval_realmatch_results")
_EVAL_ROUND  = os.environ.get("EVAL_ROUND")                       # if set, eval ONLY this DAgger round
_RGLOB       = f"*_round{_EVAL_ROUND}_*" if _EVAL_ROUND else "*"  # restrict run folder to that round
NUM_ROLLOUTS = 100
EVAL_EPOCHS  = [50, 100, 150]   # checkpoints saved every 50 up to 150
HORIZON      = 500
VIDEO_EVERY  = 20               # save an mp4 of every Nth rollout (i=0,20,40,... per checkpoint), to eyeball the task
VIDEO_DIR    = os.path.join(_PATHS.base_save_path, "eval_realmatch_videos")

TASKS = {
  "PnPCounterToCab": dict(
    dataset=f"{DATA_DIR}/patched_pnp_demo_gentex_im128_randcams.hdf5",
    pretrained=_PATHS.lookup.diffusion.PnPCounterToCab.pretrained_model,
    fixed=f"{REPO}/fixed_envs/fixed_env_state_lemon.npy"),
}

def epoch_ckpts(run_dir):
    # One checkpoint per target epoch (latest by mtime if several match). Single-round run.
    out = []
    for ep in EVAL_EPOCHS:
        cks = glob.glob(os.path.join(run_dir, _RGLOB, "*", "models", f"model_epoch_{ep}.pth"))
        if cks:
            out.append((ep, max(cks, key=os.path.getmtime)))
    return out

def main(run, only_epoch=None, num_rollouts=None):
    nr = num_rollouts if num_rollouts is not None else NUM_ROLLOUTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(RESULTS_BASE, run)
    task = "PnPCounterToCab"
    if task not in run:
        raise SystemExit(f"[eval] {run!r} is not a {task} run")
    c = TASKS[task]

    if only_epoch is not None:
        # Glob the requested epoch directly; epoch_ckpts() only scans EVAL_EPOCHS.
        cks = glob.glob(os.path.join(run_dir, _RGLOB, "*", "models", f"model_epoch_{only_epoch}.pth"))
        ckpts = [(only_epoch, max(cks, key=os.path.getmtime))] if cks else []
    else:
        ckpts = epoch_ckpts(run_dir)
    if not ckpts:
        print(f"[eval] {run}: no checkpoints at {('epoch '+str(only_epoch)) if only_epoch is not None else EVAL_EPOCHS}; skipping")
        return
    print(f"[eval] {run}: epochs {[e for e, _ in ckpts]} x {nr} rollouts", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    # One results JSON per DAgger round, keyed from the checkpoint's run folder (..._round{N}_...).
    _m = re.search(r"_round(\d+)_", ckpts[0][1])
    round_tag = f"_round{_m.group(1)}" if _m else ""
    out_path = os.path.join(OUT_DIR, f"{run}{round_tag}.json")
    data = (json.load(open(out_path)) if os.path.exists(out_path)
            else {"run": run, "task": task, "num_rollouts": nr, "epochs": {}})
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
        succ = list(data["epochs"].get(key, {}).get("success", []))   # resume mid-checkpoint
        if len(succ) >= nr:
            print(f"[eval] {run} {key} already done ({np.mean(succ):.3f}); skip", flush=True)
            continue
        cdict = json.loads(FileUtils.maybe_dict_from_checkpoint(ckpt)["config"])
        conf = config_factory(cdict["algo_name"]); conf.update(cdict)
        model = algo_factory(algo_name=conf.algo_name, config=conf,
                             obs_key_shapes=shape_meta["all_shapes"], ac_dim=shape_meta["ac_dim"], device=device)
        model.deserialize(FileUtils.maybe_dict_from_checkpoint(ckpt)["model"]); model.set_eval()
        policy = RolloutPolicy(model)

        for i in range(len(succ), nr):   # seed=i fixes env/numpy RNG (same scene per index); policy sampling (torch) is not seeded
            vw = None
            if i % VIDEO_EVERY == 0:               # save one rollout video every VIDEO_EVERY rollouts
                os.makedirs(VIDEO_DIR, exist_ok=True)
                vpath = os.path.join(VIDEO_DIR, f"{run}_{key}_rollout{i}.mp4")
                vw = imageio.get_writer(vpath, fps=20)
            with torch.no_grad():
                res = run_rollout(policy=policy, env=env, task=env_meta["env_name"],
                                  horizon=HORIZON, terminate_on_success=True, state_dict=state_dict,
                                  video_writer=vw, seed=i, use_sequence=True,
                                  seq_length=seq_length, device=device)
            if vw is not None:
                vw.close(); print(f"  [video] saved {vpath}", flush=True)
            succ.append(res["Success_Rate"])
            data["epochs"][key] = {"ckpt": ckpt, "success": succ,
                                   "mean": float(np.mean(succ)),
                                   "sem": float(np.std(succ) / np.sqrt(len(succ)))}
            json.dump(data, open(out_path, "w"), indent=2)   # save every rollout
            print(f"  {run} {key}: {i + 1}/{nr}  last={res['Success_Rate']:.0f}  rate={np.mean(succ):.3f}", flush=True)
        print(f"[eval] {run} {key} DONE mean={np.mean(succ):.3f} -> {out_path}", flush=True)
        del model, policy; gc.collect(); torch.cuda.empty_cache()
    env.close()

if __name__ == "__main__":
    # usage: eval_realmatch.py <run> [epoch] [num_rollouts]
    #   1 arg  -> all epochs (50/100/150), 100 rollouts each
    #   3 args -> only that epoch, that many rollouts
    _run = sys.argv[1]
    _ep = int(sys.argv[2]) if len(sys.argv) > 2 else None
    _nr = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(_run, only_epoch=_ep, num_rollouts=_nr)
