import numpy as np
from torch.utils.data import Dataset

class IWRDataset(Dataset):
    """
    Intervention Weighted Regression dataset wrapper for robomimic.

    Works on a single robomimic SequenceDataset. It reads the 'intervention_mask' 
    and 'train_mask' arrays (true/false) for each demo and categorizes indices 
    at the timestep level. It only uses data where the 'train_mask' flag is True.
    It builds a balanced dataset index where 50% of the sampled sequences 
    originate from intervention timesteps and 50% from autonomous rollouts.
    """

    def __init__(self, base_dataset, intervention_only=False):
        # intervention_only: drop non-intervention frames of collected demos at
        # training time (ignoring stored train_mask). Pretrain demos (no
        # intervention_mask) stay as the rollout bucket for IWR's 50/50.
        self.intervention_only = intervention_only
        self.dataset = base_dataset
        self.reference_dataset = base_dataset

        # robomimic compatibility
        if hasattr(base_dataset, "goal_mode"):
            self.goal_mode = base_dataset.goal_mode

        self.intervention_indices = []
        self.rollout_indices = []

        # ------------------------------------------------
        # Build index lists for intervention vs rollout at timestep level
        # ------------------------------------------------
        running_index = 0
        underlying_datasets = self.dataset.datasets if hasattr(self.dataset, 'datasets') else [self.dataset]

        for ds in underlying_datasets:
            for demo in ds.demos:
                demo_len = ds._demo_id_to_demo_length[demo]
                demo_grp = ds.hdf5_file[f"data/{demo}"]

                # Read the timestep-level intervention flag
                demo_has_intervention_mask = "intervention_mask" in demo_grp
                if demo_has_intervention_mask:
                    interv_array = np.squeeze(demo_grp["intervention_mask"][:]).astype(bool)
                else:
                    print(f"[IWR Warning] 'intervention_mask' key not found in {demo}. Defaulting to rollout.")
                    interv_array = np.zeros(demo_len, dtype=bool)

                # Read the timestep-level training flag
                if "train_mask" in demo_grp:
                    training_array = np.squeeze(demo_grp["train_mask"][:]).astype(bool)
                else:
                    print(f"[IWR Warning] 'train_mask' key not found in {demo}. Defaulting to True.")
                    training_array = np.ones(demo_len, dtype=bool)

                # Iterate through the frames of this demo
                for i in range(demo_len):
                    is_intervention = interv_array[i] if i < len(interv_array) else False
                    is_training = training_array[i] if i < len(training_array) else True

                    # Interventions-only: drop collected non-intervention frames.
                    if self.intervention_only and demo_has_intervention_mask and not is_intervention:
                        is_training = False

                    # Only add lists if it's marked as valid training data
                    if is_training:
                        if is_intervention:
                            self.intervention_indices.append(running_index)
                        else:
                            self.rollout_indices.append(running_index)
                        
                    # Increment running_index to stay synced with the underlying Robomimic SequenceDataset's absolute index map
                    running_index += 1

        # ------------------------------------------------
        # Balanced 50/50 index mapping via uniform resampling from the global NumPy RNG (no local seed)
        # ------------------------------------------------
        if not self.intervention_indices and not self.rollout_indices:
            self.balanced_indices = []
        elif not self.intervention_indices:
            print("[IWR] No valid intervention timesteps found. Using 100% rollouts.")
            self.balanced_indices = self.rollout_indices
        elif not self.rollout_indices:
            print("[IWR] No valid rollout timesteps found. Using 100% interventions.")
            self.balanced_indices = self.intervention_indices
        else:
            # 50/50 balanced, no inflation: epoch size = n_interv + n_rollout.
            half = (len(self.intervention_indices) + len(self.rollout_indices)) // 2
            interv_sampled = np.random.choice(self.intervention_indices, size=half, replace=True).tolist()
            rollout_sampled = np.random.choice(self.rollout_indices, size=half, replace=True).tolist()
            self.balanced_indices = interv_sampled + rollout_sampled
            
        self._len = len(self.balanced_indices)
        
        print(f"\n[IWR] Dataset Built!")
        print(f"[IWR] Total Valid Intervention Frames: {len(self.intervention_indices)}")
        print(f"[IWR] Total Valid Rollout Frames: {len(self.rollout_indices)}")
        print(f"[IWR] Balanced Epoch Size: {self._len} frames (50/50, no inflation)\n")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Deterministic lookup allows PyTorch's DataLoader to sample safely without replacement
        real_idx = self.balanced_indices[idx]
        return self.dataset[real_idx]

    # --------------------------------------------------
    # robomimic interface compatibility
    # --------------------------------------------------

    def get_dataset_sampler(self):
        return None

    def get_action_normalization_stats(self):
        if hasattr(self.reference_dataset, "get_action_normalization_stats"):
            return self.reference_dataset.get_action_normalization_stats()
        return None

    def filter_by_attribute(self, *args, **kwargs):
        if hasattr(self.reference_dataset, "filter_by_attribute"):
            return self.reference_dataset.filter_by_attribute(*args, **kwargs)
        return self

    def close_and_delete_hdf5_handle(self):
        if hasattr(self.reference_dataset, "close_and_delete_hdf5_handle"):
            self.reference_dataset.close_and_delete_hdf5_handle()

    # --------------------------------------------------
    # automatic delegation for robomimic attributes
    # --------------------------------------------------

    def __getattr__(self, name):
        if hasattr(self.reference_dataset, name):
            return getattr(self.reference_dataset, name)

        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'"
        )