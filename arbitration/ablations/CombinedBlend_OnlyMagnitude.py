import numpy as np
from arbitration.ablations.CombinedBlend_SingleGroup import CombinedBlend_SingleGroup


class CombinedBlend_OnlyMagnitude(CombinedBlend_SingleGroup):
    """
    Single-component variant of CombinedBlend: magnitude is the ONLY blending signal.

    alpha = 1 - magnitude,  magnitude = max over the 6 EEF dims of |normalised axis|

    Authority hands over in direct proportion to how hard the human pushes on the
    single most-deflected axis: commanding x at 0.8 with every other axis idle gives
    magnitude 0.8 and alpha 0.2. An idle human leaves the robot at full authority
    (alpha = 1) and a full-deflection input on any one axis takes it completely
    (alpha = 0). Each axis is normalised by its own channel scale (translation vs
    rotation) before the max, so neither unit dominates.

    This replaces the merged magnitude CombinedBlend_SingleGroup computes, so
    authority tracks the most-deflected axis rather than the whole command.

    Everything else the base blend uses to set authority is removed:
      * no similarity term and no confidence term (any uncertainty estimator passed
        in is discarded, and no calibration file is required); both are reported as
        1.0 in the diagnostics;
      * no sigmoid, so slope_sim / offset_sim / slope_conf / offset_conf are unused;
      * no magnitude gate and no exact-zero guard — the linear ramp already gives
        the robot full authority at zero input, so the gate would be redundant;
      * one merged blend group over all 6 EEF dims (inherited from CombinedBlend_SingleGroup).

    Temporal smoothing (including magnitude_smoothing on the magnitude itself), the
    grace period and the blend modes are inherited from CombinedBlend.arbitrate().
    """

    def __init__(self, **kwargs):
        kwargs["uncertainty_estimator"] = None
        super().__init__(**kwargs)
        self.magnitude_gate = False
        self.zero_input_guard = False

    def calibrate(self, env, policy, obs_buffer, lang_emb, uncertainty_path):
        # No uncertainty calibration needed
        pass

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        alpha = float(np.clip(1.0 - magnitude, 0.0, 1.0))
        return alpha, 1.0, 1.0

    def _human_channel_magnitude(self, human_action, group_name):
        """Largest single-axis human deflection in [0, 1] over all 6 EEF dims."""
        g = np.asarray(human_action[0:6], dtype=np.float64).copy()
        g[np.abs(g) < self._MAGNITUDE_FLOOR] = 0.0
        g_norm = g / np.maximum(self._axis_scales(), 1e-8)
        return float(np.clip(np.max(np.abs(g_norm)), 0.0, 1.0))
