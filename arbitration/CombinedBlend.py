import math
import numpy as np
import torch
from arbitration.utils import get_cosine_similarity, get_magnitude_weighted_similarity, calibrate_uncertainty
from arbitration.ActionArbitrator import ActionArbitrator

class CombinedBlend(ActionArbitrator):
    def __init__(
        self,
        uncertainty_estimator=None,
        train_all_steps=True,
        slope_sim=10.0,
        offset_sim=0.5,
        slope_conf=10.0,
        offset_conf=0.5,
        alpha_smoothing=0.5,        # EMA weight on the previous alpha (lambda in the paper); 0 = no smoothing
        onset_smoothing=None,       # smoothing for the 1st step of a new intervention; None = use alpha_smoothing
        magnitude_smoothing=0.0,    # EMA on the per-group human magnitude before it feeds alpha; 0 = off
        human_grace_period=5,       # inactive steps allowed before the arbitrator hands control back to the robot
        uncertainty_path=None,
        kinova=False,               # orientation uncertainty is a 6-D rot6d block (real-robot policies)
        action_scale=None,
        magnitude_scale=None,       # dict mapping group name → max magnitude for normalising to [0,1]
        similarity_mode="cosine",   # "cosine" | "magnitude_weighted"
        norm_type="clip",           # "clip" | "deadzone"
        noise_floor=0.0,            # EEF-space deadzone (metres / radians); only used when norm_type="deadzone"
        speed_preserving=False,     # if True, blend direction but keep human's commanded speed magnitude
        additive_magnitude=False,   # if True, blend direction but sum robot+human speeds along it
        human_active_threshold=0.0, # per-group input norm (device units) at or below which the robot keeps full authority
        active_threshold=None,      # unused; kept for call-site compatibility
        gate_ramp=2.5,              # unused; kept for call-site compatibility
        gate_env_decay=0.0,         # decay of an input-norm envelope that bridges Oculus frame dropouts; 0 = off
    ):
        """
        Combined blending based on similarity, confidence, and magnitude per group of degrees of freedom.
        """
        super().__init__()
        self.uncertainty_estimator = uncertainty_estimator
        self.train_all_steps = train_all_steps

        self.slope_sim = slope_sim
        self.offset_sim = offset_sim

        self.slope_conf = slope_conf
        self.offset_conf = offset_conf

        self.alpha_smoothing = alpha_smoothing
        self.onset_smoothing = onset_smoothing
        self.magnitude_smoothing = float(magnitude_smoothing)
        self.prev_alphas = None
        self._mag_ema = None    # per-group EMA of the human magnitude (see magnitude_smoothing)
        self.kinova = kinova
        self.action_scale = np.asarray(action_scale, dtype=np.float64) if action_scale is not None else None
        self.magnitude_scale = magnitude_scale  # e.g. {"xy": 5.0, "z": 5.0, "ori": 2.6669}

        self.similarity_mode = similarity_mode
        self.norm_type = norm_type
        self.noise_floor = float(noise_floor)
        self.speed_preserving = speed_preserving
        self.additive_magnitude = additive_magnitude
        self.human_active_threshold = float(human_active_threshold)
        self.active_threshold = None if active_threshold is None else float(active_threshold)
        self.gate_ramp = float(gate_ramp)
        self.gate_env_decay = float(gate_env_decay)
        self.magnitude_gate = True
        self.zero_input_guard = True # idle groups keep full robot authority; ablations without magnitude disable it
        self._gate_env = 0.0         # envelope state (see gate_env_decay)
        self._held_targets = {}      # per-group alpha_target held across dropout frames
        self._held_human = None      # last non-zero human action[:6], executed during dropout frames

        self.human_inactive_count = 100  # Don't trigger immediately
        self.human_grace_period = human_grace_period  # timesteps of grace before switching back to full robot control

        # Each DoF group (a slice into the 6-D EEF delta) gets its own alpha; CombinedBlend_SingleGroup merges them.
        self.dof_groups = {"xy": slice(0, 2), "z": slice(2, 3), "ori": slice(3, 6)}
        # Which group feeds each of the live UI's fixed [xy, z, ori] slots.
        self._ui_term_groups = ("xy", "z", "ori")

        self.L_MIN = None
        self.L_MAX = None
        if uncertainty_path is not None:
            self.L_MIN, self.L_MAX = calibrate_uncertainty(uncertainty_path)
            print(f"[CombinedBlend] Calibration loaded from {uncertainty_path}")
        else:
            print("[CombinedBlend] WARNING: no calibration — norm_uncertainty = clip(raw, 0,1)")

    def calibrate(self, env, policy, obs_buffer, lang_emb, uncertainty_path):
        _attach = getattr(self.uncertainty_estimator, "attach", None)
        if callable(_attach):
            _attach(policy)   # install obs-encoder hook before the loop so inference primes the cache
        if self.L_MIN is not None and self.L_MAX is not None:
            return

        if uncertainty_path is not None:
            self.L_MIN, self.L_MAX = calibrate_uncertainty(uncertainty_path)

        if self.uncertainty_estimator is not None:
            supports = (hasattr(self.uncertainty_estimator, "supports")
                        and self.uncertainty_estimator.supports(policy))
            if not supports:
                print(f"[CombinedBlend] WARNING: {type(policy).__name__} does not support "
                      f"uncertainty estimation — running with similarity-only blending (no confidence weighting)")

    _MAGNITUDE_SLICES = {"xy": slice(0, 2), "z": slice(2, 3), "ori": slice(3, 6)}

    # Floor below which an axis counts as "not commanded"; shared by magnitude and the per-group gate.
    _MAGNITUDE_FLOOR = 5e-4

    def _human_channel_magnitude(self, human_action, group_name):
        """Normalised human-input magnitude in [0, 1] for `group_name`, measured
        over that group's own axes only (xy = x,y; z = z; ori = rpy).

        The group's L2 norm is divided by its magnitude_scale (if given) and
        clipped to 1, so each group's authority is independent of the others.
        """
        slc = self._MAGNITUDE_SLICES[group_name]
        g = human_action[slc].copy()
        if self.action_scale is not None:
            scale = self.action_scale[slc]
            if self.norm_type == "deadzone" and self.noise_floor > 0.0:
                sign = np.sign(g)
                eff  = np.maximum(scale - self.noise_floor, 1e-8)
                g_norm = np.clip(
                    sign * np.maximum(np.abs(g) - self.noise_floor, 0.0) / eff,
                    -1.0, 1.0,
                )
            else:
                g[np.abs(g) < self._MAGNITUDE_FLOOR] = 0.0
                g_norm = np.clip(g / scale, -1.0, 1.0)
        else:
            g[np.abs(g) < self._MAGNITUDE_FLOOR] = 0.0
            g_norm = g

        magnitude = float(np.linalg.norm(g_norm))
        if self.magnitude_scale is not None and group_name in self.magnitude_scale:
            magnitude = magnitude / self.magnitude_scale[group_name]
        return float(np.clip(magnitude, 0.0, 1.0))

    def compute_sigmoid_alpha(self, x, m, s, o):
        """
        Computes the blending value alpha based on the function:
        y = 1 / (1 + e^(-s * (x - m - o)))
        
        x = confidence or similarity
        m = magnitude of the human's action
        s = slope
        o = offset
        """
        exponent = -s * (x - m - o)
        
        # Prevent math overflow for extreme exponent values
        if exponent > 50:
            return 0.0
        elif exponent < -50:
            return 1.0
            
        return 1.0 / (1.0 + math.exp(exponent))

    def _compute_alpha_target(self, similarity, confidence, magnitude):
        """Pre-gate, pre-smoothing target alpha for one DoF group.

        Returns (alpha_target, alpha_sim, alpha_conf). This is the ablation
        hook: each CombinedBlend_No* variant overrides it to drop one term
        while inheriting everything else (normalisation, gate, smoothing,
        grace period) from arbitrate().
        """
        alpha_sim = self.compute_sigmoid_alpha(similarity, magnitude, self.slope_sim, self.offset_sim)
        if confidence is not None:
            alpha_conf = self.compute_sigmoid_alpha(confidence, magnitude, self.slope_conf, self.offset_conf)
        else:
            alpha_conf = 1.0  # no confidence signal → similarity alone drives blending
        return alpha_sim * alpha_conf, alpha_sim, alpha_conf

    def _uncertainty_group_slices(self):
        """Map each blend group to its slice into the uncertainty vector.

        Identical to self.dof_groups unless kinova=True, where orientation
        uncertainty is a 6-D rot6d block (indices 3:9) rather than 3 angles.
        """
        if self.kinova:
            return {"xy": slice(0, 2), "z": slice(2, 3), "ori": slice(3, 9)}
        return dict(self.dof_groups)

    def reset(self):
        self.prev_alphas = None
        self._mag_ema = None
        self._gate_env = 0.0
        self._held_targets = {}
        self._held_human = None
        self.human_inactive_count = 100

    def arbitrate(self, *, obs_buffer, robot_action, human_action, human_active, policy, **_):
        """
        Determine the blended action.
        """
        # Detect the first step of a new human intervention so prev_alphas can be
        # reset below and onset_smoothing can be applied
        prev_inactive_count = self.human_inactive_count

        # Grace period before switching completely to robot control when human becomes inactive to avoid jitter
        if human_active:
            self.human_inactive_count = 0
        else:
            self.human_inactive_count += 1
        human_active_with_grace = (self.human_inactive_count < self.human_grace_period)

        if not human_active_with_grace:
            return {
                "action": robot_action,
                "control_mode": "robot",
                "intervening": False,
                "training": self.train_all_steps,
                "metadata": {"algo": "CombinedBlend", "disengaged": True,
                             "human_inactive_count": int(self.human_inactive_count)},
            }

        # First step of a new intervention: reset alpha history
        is_first_intervention_step = human_active and prev_inactive_count >= self.human_grace_period
        if is_first_intervention_step:
            self.prev_alphas = {g: 1.0 for g in self.dof_groups}
            self._mag_ema = None   # start each new intervention with a fresh magnitude EMA
            self._gate_env = 0.0
            self._held_targets = {}
            self._held_human = None

        # Initialize blended action with robot action
        blended = robot_action.copy()

        dof_groups = self.dof_groups

        # Get confidence (None when no estimator or policy doesn't support it)
        robot_uncertainty = None
        if self.uncertainty_estimator is not None:
            robot_uncertainty = self.uncertainty_estimator.compute(
                obs_buffer=obs_buffer, robot_action=robot_action, policy=policy
            )

        if robot_uncertainty is not None:
            uncertainty_dof_groups = self._uncertainty_group_slices()

            # Normalise uncertainty to [0, 1] if calibration bounds are available
            if self.L_MIN is not None and self.L_MAX is not None:
                norm_uncertainty = (robot_uncertainty - self.L_MIN) / (self.L_MAX - self.L_MIN)
                norm_uncertainty = np.clip(norm_uncertainty, 0.0, 1.0)
                if np.all(norm_uncertainty == 0.0):
                    print("[CombinedBlend] WARNING: all uncertainties below L_MIN — "
                          "calibration bounds may be stale or image format mismatch")
            else:
                norm_uncertainty = np.clip(robot_uncertainty, 0.0, 1.0)

            robot_confidence = {}
            for grp in dof_groups.keys():
                indices = uncertainty_dof_groups[grp]
                grp_uncertainty = norm_uncertainty[indices].mean()
                robot_confidence[grp] = float(np.clip(1.0 - grp_uncertainty, 0.0, 1.0))
        else:
            # No estimator or policy doesn't support it: alpha_conf fixed at 1.0
            robot_confidence = {grp: None for grp in dof_groups.keys()}

        # A transport dropout zeroes every axis at once while the envelope is still high; hold the last command.
        dropout_hold = False
        if self.gate_env_decay > 0.0:
            h_raw_norm = float(np.linalg.norm(human_action[:6]))
            self._gate_env = max(h_raw_norm, self._gate_env * self.gate_env_decay)
            dropout_hold = (h_raw_norm < self._MAGNITUDE_FLOOR
                            and self._gate_env >= self._MAGNITUDE_FLOOR)
            if dropout_hold and self._held_human is not None:
                # Execute the held command too, or the blend brakes and relaunches at the dropout rate.
                human_action = human_action.copy()
                human_action[:6] = self._held_human
            elif h_raw_norm >= self._MAGNITUDE_FLOOR:
                self._held_human = np.array(human_action[:6], dtype=human_action.dtype, copy=True)

        self._arb_calls = getattr(self, "_arb_calls", 0) + 1
        _verbose = (self._arb_calls % 30 == 1)   # console diagnostics ~1 Hz

        group_alphas = {}
        diag_groups = {}   # per-group blend internals, returned for logging

        if self.prev_alphas is None:
            self.prev_alphas = {g: 1.0 for g in self.dof_groups}

        # Compute blending for each DoF group
        for group_name, slc in dof_groups.items():
            r_act_group = robot_action[slc]
            h_act_group = human_action[slc].copy()

            # Normalise to [-1, 1] for magnitude and similarity so the sigmoid
            # offsets behave as intended.
            if self.action_scale is not None:
                scale = self.action_scale[slc]
                if self.norm_type == "deadzone" and self.noise_floor > 0.0:
                    # Deadzone + linear stretch: maps [noise_floor, scale] → [0, 1].
                    effective_scale = np.maximum(scale - self.noise_floor, 1e-8)
                    h_sign = np.sign(h_act_group)
                    h_act_norm = np.clip(
                        h_sign * np.maximum(np.abs(h_act_group) - self.noise_floor, 0.0) / effective_scale,
                        -1.0, 1.0,
                    )
                    r_sign = np.sign(r_act_group)
                    r_act_norm = np.clip(
                        r_sign * np.maximum(np.abs(r_act_group) - self.noise_floor, 0.0) / effective_scale,
                        -1.0, 1.0,
                    )
                else:
                    # "clip" behaviour: divide by max scale and hard-clip.
                    h_act_group[np.abs(h_act_group) < self._MAGNITUDE_FLOOR] = 0.0
                    r_act_norm = np.clip(r_act_group / scale, -1.0, 1.0)
                    h_act_norm = np.clip(h_act_group / scale, -1.0, 1.0)
            else:
                h_act_group[np.abs(h_act_group) < self._MAGNITUDE_FLOOR] = 0.0
                r_act_norm = r_act_group
                h_act_norm = h_act_group

            # Human magnitude over this group's own axes (see _human_channel_magnitude).
            magnitude_raw = self._human_channel_magnitude(human_action, group_name)

            # Optional EMA keeps a steady magnitude through idle frames of slow Oculus motion.
            if self.magnitude_smoothing > 0.0:
                if self._mag_ema is None:
                    self._mag_ema = {}
                prev = self._mag_ema.get(group_name, magnitude_raw)
                magnitude = (self.magnitude_smoothing * prev
                             + (1.0 - self.magnitude_smoothing) * magnitude_raw)
                self._mag_ema[group_name] = magnitude
            else:
                magnitude = magnitude_raw

            # Compute similarity (direction only, or direction weighted by magnitude)
            if self.similarity_mode == "magnitude_weighted":
                similarity = get_magnitude_weighted_similarity(r_act_norm, h_act_norm)
            else:  # "cosine"
                similarity = get_cosine_similarity(r_act_norm, h_act_norm)

            # Retrieve confidence (None when no uncertainty estimator)
            confidence = robot_confidence.get(group_name)

            # Pre-gate, pre-smoothing target alpha. Overridden by the
            # CombinedBlend_No* ablations to drop individual terms.
            alpha_target, alpha_sim, alpha_conf = self._compute_alpha_target(
                similarity, confidence, magnitude
            )

            # Throttled: per-frame terminal output stalls the control loop.
            if _verbose:
                conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
                print(f"[CombinedBlend] {group_name}: mag={magnitude:.4f}  sim={similarity:.3f}  "
                      f"conf={conf_str}  α_sim={alpha_sim:.3f}  α_conf={alpha_conf:.3f}  "
                      f"α_target={alpha_target:.3f}", end="")

            group_alphas[group_name] = alpha_target

            # Raw per-group input norm, in the same units as human_active_threshold.
            if self.action_scale is not None:
                sm_input_norm = float(np.linalg.norm(human_action[slc] / self.action_scale[slc]))
            else:
                sm_input_norm = float(np.linalg.norm(human_action[slc]))

            # Hold the target across dropout frames; a true release still disengages via the grace period.
            if dropout_hold and group_name in self._held_targets:
                alpha_target = self._held_targets[group_name]
            else:
                # Human not driving this group → robot keeps full authority.
                if self.zero_input_guard and (sm_input_norm <= self.human_active_threshold
                                              or magnitude < self._MAGNITUDE_FLOOR):
                    alpha_target = 1.0
                self._held_targets[group_name] = alpha_target

            # Apply temporal smoothing to avoid abrupt changes in blending. Optionally use onset_smoothing at start of takeover
            smoothing = self.alpha_smoothing
            if is_first_intervention_step and self.onset_smoothing is not None:
                smoothing = self.onset_smoothing
            alpha_final = smoothing * self.prev_alphas[group_name] + (1.0 - smoothing) * alpha_target

            if _verbose:
                print(f"  →  α_final={alpha_final:.3f}")

            # Update history
            self.prev_alphas[group_name] = alpha_final
            group_alphas[group_name] = alpha_final

            diag_groups[group_name] = {
                "mag": float(magnitude),
                "sim": float(similarity),
                "conf": (float(confidence) if confidence is not None else float("nan")),
                "alpha_sim": float(alpha_sim),
                "alpha_conf": float(alpha_conf),
                "alpha_target": float(alpha_target),
                "alpha_final": float(alpha_final),
            }

            # Linear blending for the specific DoF group
            blended_raw = (1.0 - alpha_final) * h_act_group + alpha_final * r_act_group
            if self.additive_magnitude:
                # Direction from the convex blend; magnitude = sum of the robot and
                # human speed components that go ALONG that direction (opposing
                # components are dropped, not subtracted). So a same-direction push
                # adds to the robot's motion, while an opposing push at alpha~0 moves
                # at the human's speed in the human's direction.
                blend_norm = np.linalg.norm(blended_raw)
                if blend_norm > 1e-8:
                    d = blended_raw / blend_norm
                    r_proj = max(float(np.dot(r_act_group, d)), 0.0)
                    h_proj = max(float(np.dot(h_act_group, d)), 0.0)
                    blended[slc] = (r_proj + h_proj) * d
                else:
                    blended[slc] = blended_raw
            elif self.speed_preserving:
                # Blend direction only; rescale to human's commanded speed
                h_speed = np.linalg.norm(h_act_group)
                blend_norm = np.linalg.norm(blended_raw)
                if h_speed > 1e-8 and blend_norm > 1e-8:
                    blended[slc] = (h_speed / blend_norm) * blended_raw
                else:
                    blended[slc] = blended_raw
            else:
                blended[slc] = blended_raw

        # Always take the human's gripper action (assuming index 6)
        blended[6] = human_action[6]

        # Convert per-group alphas to the 6-D EEF-delta array by expanding each
        # group's alpha across its own slice (generic over self.dof_groups; for
        # the default groups this is [xy, xy, z, ori, ori, ori]).
        alphas = np.zeros(6, dtype=float)
        for _g, _slc in dof_groups.items():
            alphas[_slc] = group_alphas[_g]

        # Per-group blend terms in the [xy, z, ori] layout the live UI and data writers consume.
        _uig = self._ui_term_groups
        blend_terms = {
            "magnitude":    np.array([diag_groups[g]["mag"]          for g in _uig], dtype=np.float32),
            "similarity":   np.array([diag_groups[g]["sim"]          for g in _uig], dtype=np.float32),
            "confidence":   np.array([diag_groups[g]["conf"]         for g in _uig], dtype=np.float32),
            "alpha_sim":    np.array([diag_groups[g]["alpha_sim"]    for g in _uig], dtype=np.float32),
            "alpha_conf":   np.array([diag_groups[g]["alpha_conf"]   for g in _uig], dtype=np.float32),
            "alpha_target": np.array([diag_groups[g]["alpha_target"] for g in _uig], dtype=np.float32),
            "alpha":        np.array([diag_groups[g]["alpha_final"]  for g in _uig], dtype=np.float32),
        }

        return {
            "action": blended,
            "control_mode": "blend",
            "intervening": True,
            "training": True,
            "alphas": alphas,
            "blend_terms": blend_terms,
            "metadata": {"algo": "CombinedBlend",
                         "groups": diag_groups,
                         "gate_env": float(self._gate_env),
                         "target_held": bool(dropout_hold),
                         "is_first_intervention_step": bool(is_first_intervention_step),
                         "human_inactive_count": int(self.human_inactive_count)},
        }