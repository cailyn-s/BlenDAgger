from abc import ABC, abstractmethod
import numpy as np


class AbstractDataWriter(ABC):
    """
    Uniform interface for writing episode data. Concrete implementations
    handle the specific on-disk format (HDF5 for sim/Robomimic, HDF5 for
    real-robot DAgger with intervention/rollout/full split, etc.).
    """

    @abstractmethod
    def open_episode(self, episode_id: int) -> None:
        """Called at the start of each episode."""
        raise NotImplementedError

    @abstractmethod
    def write_step(self, obs: dict, action: np.ndarray, metadata: dict) -> None:
        """
        Write one timestep. metadata carries arbitration info (control_mode,
        alpha values, intervention flag, etc.).
        """
        raise NotImplementedError

    @abstractmethod
    def close_episode(self, success: bool, score: int = 0) -> None:
        """Called at the end of each episode. Flushes and finalises the data."""
        raise NotImplementedError
