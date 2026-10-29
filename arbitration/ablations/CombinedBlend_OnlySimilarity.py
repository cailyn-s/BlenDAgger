from arbitration.ablations.CombinedBlend_SingleGroup import CombinedBlend_SingleGroup


class CombinedBlend_OnlySimilarity(CombinedBlend_SingleGroup):
    """
    Single-component variant of CombinedBlend: similarity is the ONLY blending signal.

    alpha = sigmoid(slope_sim * (similarity - offset_sim))

    Everything else the base blend uses to set authority is removed:
      * no confidence term (any uncertainty estimator passed in is discarded, and
        no calibration file is required) — alpha_conf is reported as 1.0;
      * no magnitude term in the sigmoid offset (m = 0), no magnitude gate and no
        exact-zero guard, so how hard the human pushes never affects authority;
      * one merged blend group over all 6 EEF dims (inherited from CombinedBlend_SingleGroup), so a
        single cosine over the whole action produces one alpha for every DoF.

    Temporal smoothing, the grace period and the blend modes are inherited from
    CombinedBlend.arbitrate() unchanged.

    With the gate removed, an idle human still yields the robot full authority for
    free: get_cosine_similarity returns 1.0 against a zero human vector, so
    alpha -> sigmoid(slope_sim * (1 - offset_sim)) ~ 1.
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
        alpha_sim = self.compute_sigmoid_alpha(similarity, 0.0, self.slope_sim, self.offset_sim)
        return alpha_sim, alpha_sim, 1.0
