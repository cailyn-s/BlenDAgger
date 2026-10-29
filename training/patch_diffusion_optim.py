import robomimic.config.diffusion_policy_config as dpc
import robomimic.algo.diffusion_policy as dp

def patch_diffusion_optim():
    """Runtime patches so the diffusion fine-tune's optimizer/LR-scheduler config takes effect.

    (1) Declare optimizer_type + num_warmup_steps on the diffusion config schema so the sim
        fine-tuning config (adamw, 100 warmup steps) passes robomimic's key-locked update.
    (2) Step the LR scheduler once per training batch: unpatched robomimic never steps it,
        so step-based schedulers stay at warmup step 0 and the LR remains 0."""
    _orig_algo_config = dpc.DiffusionPolicyConfig.algo_config

    def algo_config(self):
        _orig_algo_config(self)
        self.algo.optim_params.policy.optimizer_type = "adam"             # JSON overrides to adamw
        self.algo.optim_params.policy.learning_rate.num_warmup_steps = 0  # JSON overrides to 100

    dpc.DiffusionPolicyConfig.algo_config = algo_config

    _orig_train = dp.DiffusionPolicyUNet.train_on_batch
    if not getattr(_orig_train, "_lr_step_patched", False):
        def train_on_batch(self, batch, epoch, validate=False):
            info = _orig_train(self, batch, epoch, validate=validate)
            if not validate:
                sched = self.lr_schedulers.get("policy", None)
                for s in (sched if isinstance(sched, (list, tuple)) else [sched]):
                    if s is not None:
                        s.step()
            return info
        train_on_batch._lr_step_patched = True
        dp.DiffusionPolicyUNet.train_on_batch = train_on_batch
