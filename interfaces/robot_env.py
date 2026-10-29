from abc import ABC, abstractmethod
import numpy as np


class AbstractRobotEnv(ABC):
    """
    Uniform interface that all robot environments (sim or real) must implement.

    Action convention: 7D EEF-delta vector
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    where position deltas are in world frame (metres), rotation deltas are
    Euler angles (radians), and gripper is -1.0 (open) to +1.0 (closed).
    All adapters translate to/from hardware-native formats internally.
    """

    @abstractmethod
    def reset(self) -> dict:
        """Reset the environment and return the initial observation dict."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Execute a 7D EEF-delta action.
        Returns (obs, reward, done, info).
        """
        raise NotImplementedError

    @abstractmethod
    def get_obs(self) -> dict:
        """Return the current observation dict without stepping."""
        raise NotImplementedError

    @abstractmethod
    def is_success(self) -> dict:
        """Return a dict with at least {"task": bool} indicating task completion."""
        raise NotImplementedError

    def get_state(self) -> dict:
        """
        Return a serialisable state dict that can be passed to reset_to().
        Only required for environments that support state restoration (e.g. sim).
        """
        raise NotImplementedError("This environment does not support get_state()")

    def reset_to(self, state: dict) -> dict:
        """
        Reset to a specific state returned by get_state().
        Returns the initial observation dict.
        Only required for environments that support state restoration (e.g. sim).
        """
        raise NotImplementedError("This environment does not support reset_to()")
