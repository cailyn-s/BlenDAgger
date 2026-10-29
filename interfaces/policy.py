from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class AbstractPolicy(ABC):
    """
    Uniform interface for all policy types (diffusion, flow matching, null, etc.).

    Output convention: 7D EEF-delta vector
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Adapters convert from model-native output formats (e.g. 8D rot6d from
    diffusion policy) to this standard before returning.
    """

    @abstractmethod
    def step(self, obs: dict) -> np.ndarray | None:
        """
        Given an observation dict, return a 7D EEF-delta action, or None
        if the policy is mid-inference and has no action ready this step
        (inference gap handling).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (action history, temporal ensembling buffers, etc.)."""
        raise NotImplementedError

    def start_episode(self, obs: dict) -> None:
        """
        Called once after env.reset() with the initial observation.
        Override to perform any setup that requires the initial arm state
        (e.g. computing initial FK for EEF-frame tracking).
        Defaults to calling reset().
        """
        self.reset()


class NullPolicy(AbstractPolicy):
    """
    Returns all-zero 7D actions. Used in pure teleoperation mode so the same
    collection loop runs regardless of whether a trained policy is present.
    """

    def step(self, obs: dict) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        pass
