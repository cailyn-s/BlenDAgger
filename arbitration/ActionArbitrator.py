from abc import ABC, abstractmethod

class ActionArbitrator(ABC):
    @abstractmethod
    def reset(self):
        """Called at episode start"""
        pass

    def calibrate(self, *args, **kwargs):
        """Called at episode start"""
        pass

    @abstractmethod
    def arbitrate(
        self,
        *,
        obs_buffer,
        robot_action,
        human_action,
        human_active,
        intervening,
        policy=None,
        info=None,
    ):
        """
        Args:
            obs_buffer: deque of previous observations
            robot_action: np.ndarray
            human_action: np.ndarray or None
            human_active: bool (whether human has a nonzero action)
            intervening: bool (whether the human is intervening and has not released control back to the robot)
            info: optional dict (gripper flags, etc.)

        Returns:
            dict with at least:
                action: np.ndarray           # action to execute
                control_mode: str            # "robot", "human", "blend", etc.
                metadata: dict               # anything to be logged
        """
        pass
