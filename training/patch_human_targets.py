import torch

# Supports the robomimic diffusion policy only.
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

def patch_loss(training_mode="default"):
    if training_mode not in ("default", "human_targets"):
        raise ValueError(f"training_mode must be 'default' or 'human_targets', got {training_mode!r}")
    print(f"\n[PATCH] USING PATCHED DIFFUSION LOSS - MODE: {training_mode}")

    # Patch process_batch_for_training (target swap)
    original_process_batch = DiffusionPolicyUNet.process_batch_for_training

    def patched_process_batch_for_training(self, batch):
        # Save custom keys before Robomimic strips them
        human_actions = batch.get("human_actions")
        is_swapped = False

        if training_mode == "human_targets":
            if human_actions is not None:
                # Non-commanding frames have zero human_actions; use the policy's own action there, not the attenuated executed one.
                fallback = batch.get("robot_actions")
                if fallback is None:
                    fallback = batch["actions"]
                    if not hasattr(self, "_warned_missing_robot"):
                        print("\n[WARN] 'robot_actions' not in dataset_keys; "
                              "falling back to executed actions on non-command frames.")
                        self._warned_missing_robot = True
                active = human_actions.abs().sum(dim=-1, keepdim=True) > 0
                batch["actions"] = torch.where(active, human_actions, fallback)
                is_swapped = True
            else:
                if not hasattr(self, "_warned_missing_human"):
                    print("\n[ERROR] 'human_actions' is None! The dataloader didn't load it.")
                    self._warned_missing_human = True

        # Run normal processing
        processed_batch = original_process_batch(self, batch)
        processed_batch["_swapped_targets"] = is_swapped
        return processed_batch

    DiffusionPolicyUNet.process_batch_for_training = patched_process_batch_for_training

    # Patch train_on_batch directly on DiffusionPolicyUNet
    original_train_on_batch = DiffusionPolicyUNet.train_on_batch

    def patched_train_on_batch(self, batch, epoch, validate=False):
        if batch.get("_swapped_targets"):
            if getattr(self, "_last_printed_epoch", -1) != epoch:
                print(f"[Epoch {epoch}] Using Human Action Targets for training.")
                self._last_printed_epoch = epoch

        info = original_train_on_batch(self, batch, epoch, validate)
        if batch.get("_swapped_targets"):
            info["Using_Human_Targets"] = torch.tensor(1.0)
        return info

    DiffusionPolicyUNet.train_on_batch = patched_train_on_batch
    
    # Ensure custom loss keys are sent to Tensorboard/Console
    original_log_info = DiffusionPolicyUNet.log_info

    def patched_log_info(self, info):
        log = original_log_info(self, info)
        for key in ["Using_Human_Targets"]:
            if key in info.get("losses", {}):
                log[key] = info["losses"][key].item()
        return log
        
    DiffusionPolicyUNet.log_info = patched_log_info