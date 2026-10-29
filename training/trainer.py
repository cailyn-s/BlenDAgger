from types import SimpleNamespace
from robomimic.scripts.train import main as robomimic_main
from training.iwr_patch import patch_dataset_factory
from training.freeze_visual import patch_freeze_visual_encoder
from training.patch_human_targets import patch_loss
from training.patch_diffusion_optim import patch_diffusion_optim

def train(config_path, run_name, freeze_encoder=False, use_iwr=False, training_mode="default", intervention_only=False):
    # declare optimizer_type/num_warmup_steps on the diffusion schema so the fine-tuning config loads
    patch_diffusion_optim()
    # apply patches based on yaml configs
    if use_iwr:
        patch_dataset_factory(intervention_only=intervention_only)
    patch_loss(training_mode=training_mode)
    if freeze_encoder:
        patch_freeze_visual_encoder()

    args = SimpleNamespace(
        config=config_path,
        algo=None,
        name=run_name,
        dataset=None,
        debug=False,
        eval_only=False,
    )

    robomimic_main(args)