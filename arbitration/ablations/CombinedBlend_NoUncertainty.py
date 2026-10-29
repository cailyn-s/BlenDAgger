from arbitration.CombinedBlend import CombinedBlend

class CombinedBlend_NoUncertainty(CombinedBlend):
    """
    Ablation of CombinedBlend with no use of uncertainty/confidence.
    Blending is driven by similarity and magnitude alone; no calibration file
    or uncertainty estimator is required (any passed estimator is discarded).
    The magnitude gate and smoothing are inherited from
    CombinedBlend.arbitrate(). alpha_conf is reported as 1.0 (its neutral
    value) in the diagnostics.
    """

    def __init__(self, **kwargs):
        kwargs["uncertainty_estimator"] = None
        super().__init__(**kwargs)

    def calibrate(self, env, policy, obs_buffer, lang_emb, uncertainty_path):
        # No uncertainty calibration needed
        pass

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        alpha_sim = self.compute_sigmoid_alpha(similarity, magnitude, self.slope_sim, self.offset_sim)
        return alpha_sim, alpha_sim, 1.0
