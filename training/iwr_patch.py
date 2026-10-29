import robomimic.utils.train_utils as dataset_utils
from training.iwr_dataset import IWRDataset


def patch_dataset_factory(enable_iwr=True, intervention_only=False):

    if not enable_iwr:
        return

    original_dataset_factory = dataset_utils.dataset_factory

    def dataset_factory_patched(config, obs_keys, dataset_path=None, **kwargs):

        dataset = original_dataset_factory(
            config=config,
            obs_keys=obs_keys,
            dataset_path=dataset_path,
            **kwargs
        )

        print(f"Using IWR weighting from dataset attributes (intervention_only={intervention_only})")

        return IWRDataset(dataset, intervention_only=intervention_only)

    dataset_utils.dataset_factory = dataset_factory_patched