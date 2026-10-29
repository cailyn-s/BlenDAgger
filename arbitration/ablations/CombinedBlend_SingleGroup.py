import numpy as np
from arbitration.CombinedBlend import CombinedBlend


class CombinedBlend_SingleGroup(CombinedBlend):
    """
    CombinedBlend with one blend group over all 6 EEF dims instead of xy / z / ori.

    Base for the single-signal variants: similarity, confidence and magnitude are each reduced
    once over the merged group (axes scaled per channel first) and give one alpha for every DoF.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Single merged blend group over all 6 EEF dims (gripper stays human).
        self.dof_groups = {"all": slice(0, 6)}
        # Broadcast the single merged group across the UI's fixed [xy, z, ori] slots.
        self._ui_term_groups = ("all", "all", "all")

        # Put translation and rotation on equal footing for the single merged
        # cosine: dividing each axis by its channel scale so neither unit
        # dominates the direction. Only synthesised when magnitude_scale is
        # available (oculus) and action_scale was not passed explicitly; for the
        # spacemouse (magnitude_scale None) the base falls back to a raw cosine,
        # matching its raw per-group handling.
        if self.magnitude_scale is not None and self.action_scale is None:
            self.action_scale = self._axis_scales()

    def _axis_scales(self):
        """Per-axis normalisers for the 6-D EEF delta, [pos, pos, pos, ori, ori, ori]."""
        if self.action_scale is not None:
            return np.asarray(self.action_scale, dtype=np.float64)
        pos = 1.0
        ori = 1.0
        if self.magnitude_scale is not None:
            pos = float(self.magnitude_scale.get("xy", 1.0))
            ori = float(self.magnitude_scale.get("ori", 1.0))
        return np.array([pos, pos, pos, ori, ori, ori], dtype=np.float64)

    def _uncertainty_group_slices(self):
        # Merge all uncertainty dims into one group: 6 euler, or 9 (pos(3) +
        # rot6d(6)) for kinova. The base then means over the slice → one confidence.
        return {"all": slice(0, 9) if self.kinova else slice(0, 6)}

    def _human_channel_magnitude(self, human_action, group_name):
        """Merged human-input magnitude in [0, 1]: the average over all 6 EEF dims.

        Per-axis channel scaling, then the plain mean of the six per-DoF
        magnitudes, so every DoF is weighted equally and no channel grouping
        survives the merge.
        """
        g = np.asarray(human_action[0:6], dtype=np.float64).copy()
        g[np.abs(g) < self._MAGNITUDE_FLOOR] = 0.0
        g_norm = np.abs(g) / np.maximum(self._axis_scales(), 1e-8)
        return float(np.clip(np.mean(g_norm), 0.0, 1.0))
