"""Pure-Torch coupled POWER-close teacher prior and bounded hand residual.

The CEM search parameter is not an environment action: its seven arm values
describe a target offset that is tracked by a time-varying feedback command.
This module validates the authoritative CEM artifact, binds it to the exact
curriculum bytes by SHA256, and reconstructs that public 21-D command from the
actor-visible 131-D observation.

The learned policy owns only a 14-D hand residual.  During ALIGN it is
canonicalized to exact zero and the CEM wrist trajectory is replayed unchanged.
During CLOSE and HOLD it is added, with phase-specific bounds, to the CEM hand
latent.  The canonical residual -- not the resulting environment action -- is
the action that belongs in FlashSAC replay.

There are deliberately no Isaac Lab or simulator imports in this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


ARM_DIM = 7
HAND_DIM = 14
TOKEN_DIM = 9
DISTAL_DIM = 5
ACTION_DIM = ARM_DIM + HAND_DIM
OBSERVATION_DIM = 131
ALIGN_STEPS = 24

PHASE_ALIGN = 0
PHASE_CLOSE_UNLATCHED = 1
PHASE_HOLD_LATCHED = 2
PHASE_NAMES = ("align", "close_unlatched", "hold_latched")

# Stable public observation slots.  The first 115 dimensions retain the old
# layout; the coupled state is appended at [115:131].
POWER_LATCH_OBSERVATION_INDEX = 106
ARM_OFFSET_OBSERVATION_SLICE = slice(115, 122)
ALIGN_ACTIVE_OBSERVATION_INDEX = 129
ALIGN_PROGRESS_OBSERVATION_INDEX = 130

ARM_TARGET_LIMIT_RAD = 0.12
ARM_ACTION_SCALE = 0.10
ARM_TARGET_EMA = 0.30
ARM_ACTION_MULTIPLIER = 0.20
ARM_FEEDBACK_DENOMINATOR = (
    ARM_ACTION_SCALE * ARM_TARGET_EMA * ARM_ACTION_MULTIPLIER
)

ARTIFACT_CONTRACT = "strict_power_close_coupled_teacher_v1"
CURRICULUM_CONTRACT = "coupled_power_static_close_start_v1"
PHYSICAL_TASK_MODE = "coupled_power_align_close_option_v1"
POLICY_TASK_MODE = "coupled_power_teacher_residual_v1"
OBSERVATION_CONTRACT = "pick_tool_coupled_power_align_close_state131_v1"
ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
PASS_AUTHORITY = "conservative_coupled_teacher_audit_v1"

POWER_REQUIRED_OTHER_CONTACTS = 3
POWER_GRASP_QUALITY_MIN = 0.35
POWER_HOLD_QUALITY_MIN = 0.50
POWER_FORCE_LIMIT = 30.0
POWER_STABLE_FRAMES = 15
POWER_CLEARANCE_LIMIT = 0.015
POWER_XY_DRIFT_LIMIT = 0.03
POWER_ROTATION_DRIFT_LIMIT = 0.35

_VALUE_TOLERANCE = 1.0e-6


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA256 of the exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact(mapping: Mapping[str, Any], key: str, expected: object, source: str) -> None:
    value = mapping.get(key)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{source}.{key}={value!r}, expected {expected!r}")


def _require_lower_sha256(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source} must be a lowercase hexadecimal SHA256")
    return value


def _require_finite_list(
    mapping: Mapping[str, Any], key: str, length: int, source: str
) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{source}.{key} must be a list of length {length}")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{source}.{key} must contain only finite numbers")
    return tuple(float(item) for item in value)


def _require_replicate_flags(
    teacher: Mapping[str, Any], key: str, replicates: int, expected: bool
) -> None:
    values = teacher.get(key)
    if (
        not isinstance(values, list)
        or len(values) != replicates
        or any(value is not expected for value in values)
    ):
        state = "true" if expected else "false"
        raise ValueError(f"artifact.teacher_result.{key} must be all {state}")


def _validate_curriculum(path: Path) -> None:
    try:
        curriculum = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot safely load coupled curriculum {path}: {error}") from error
    if not isinstance(curriculum, dict):
        raise TypeError("curriculum root must be a dictionary")
    metadata = curriculum.get("meta")
    if not isinstance(metadata, Mapping):
        raise TypeError("curriculum metadata must be a mapping")
    for key, value in {
        "format_version": 1,
        "contract": CURRICULUM_CONTRACT,
        "task_mode": PHYSICAL_TASK_MODE,
        "observation_contract": OBSERVATION_CONTRACT,
        "action_layout": ACTION_LAYOUT,
        "top_k": 1,
    }.items():
        _require_exact(metadata, key, value, "curriculum.meta")

    boundaries = curriculum.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise TypeError("curriculum.boundaries must be a mapping")
    boundary = boundaries.get("close_start")
    if not isinstance(boundary, Mapping):
        raise TypeError("curriculum.boundaries.close_start must be a mapping")
    tensor_shapes = {
        "joint_pos": (1, 19),
        "joint_vel": (1, 19),
        "dof_targets": (1, 19),
        "object_local_pos": (1, 3),
        "object_quat": (1, 4),
        "object_velocity": (1, 6),
        "last_action": (1, ACTION_DIM),
        "contact_steps": (1,),
        "lost_contact_steps": (1,),
        "is_grasped": (1,),
    }
    for key, shape in tensor_shapes.items():
        value = boundary.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(f"curriculum close_start {key} expected {shape}, got {actual}")
    for key in (
        "joint_pos",
        "joint_vel",
        "dof_targets",
        "object_local_pos",
        "object_quat",
        "object_velocity",
        "last_action",
    ):
        if not bool(torch.isfinite(boundary[key]).all()):
            raise ValueError(f"curriculum close_start {key} contains non-finite values")
    if not torch.equal(boundary["joint_pos"], boundary["dof_targets"]):
        raise ValueError("static curriculum requires dof_targets == joint_pos")
    for key in ("joint_vel", "object_velocity", "last_action"):
        if bool((boundary[key] != 0).any()):
            raise ValueError(f"static curriculum requires exact-zero {key}")
    for key in ("contact_steps", "lost_contact_steps", "is_grasped"):
        if bool(boundary[key].any()):
            raise ValueError(f"static curriculum requires cleared {key}")


def _validate_teacher_artifact(
    path: Path, *, curriculum_sha256: str
) -> tuple[Mapping[str, Any], tuple[float, ...], tuple[float, ...]]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse coupled teacher artifact {path}: {error}") from error
    if not isinstance(artifact, dict):
        raise TypeError("teacher artifact root must be a mapping")
    for key, value in {
        "format_version": 1,
        "contract": ARTIFACT_CONTRACT,
        "controller": "coupled_native_align_close_with_runtime_shields",
        "task_mode": PHYSICAL_TASK_MODE,
        "observation_contract": OBSERVATION_CONTRACT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "action_projection": "identity_v1",
        "pass_authority": PASS_AUTHORITY,
        "native_success_authority": "pick_tool_terminal.power_close_option_success",
        "search_parameter_layout": "normalized_arm_target_offset7|hybrid14",
        "search_parameter_is_environment_action": False,
        "search_parameter_projection": "time_varying_arm_feedback_plus_hybrid14",
        "arm_feedback_formula": (
            "clip((smoothstep_target-current_target)/(action_scale*ema*0.2),-1,1)"
        ),
        "align_steps": ALIGN_STEPS,
        "arm_delta_limit_rad": ARM_TARGET_LIMIT_RAD,
        "episode_length_s": 3.0,
    }.items():
        _require_exact(artifact, key, value, "artifact")
    declared_curriculum_sha = _require_lower_sha256(
        artifact.get("curriculum_sha256"), "artifact.curriculum_sha256"
    )
    if declared_curriculum_sha != curriculum_sha256:
        raise ValueError("teacher artifact and supplied curriculum SHA256 disagree")

    phase_contract = artifact.get("coupled_phase_contract")
    if not isinstance(phase_contract, Mapping):
        raise TypeError("artifact.coupled_phase_contract must be a mapping")
    for key, value in {
        "align_steps": ALIGN_STEPS,
        "arm_action_multiplier": ARM_ACTION_MULTIPLIER,
        "arm_target_limit_rad": ARM_TARGET_LIMIT_RAD,
        "align_hand_action": "masked_hold_target",
        "close_arm_action": "masked_frozen_target",
    }.items():
        _require_exact(phase_contract, key, value, "artifact.coupled_phase_contract")

    thresholds = artifact.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("artifact.thresholds must be a mapping")
    for key, value in {
        "required_legal_other_contacts": POWER_REQUIRED_OTHER_CONTACTS,
        "power_grasp_quality": POWER_GRASP_QUALITY_MIN,
        "hold_quality": POWER_HOLD_QUALITY_MIN,
        "safe_force_n": POWER_FORCE_LIMIT,
        "confirm_steps": POWER_STABLE_FRAMES,
        "unlatched_lift_m": POWER_CLEARANCE_LIMIT,
        "horizontal_drift_m": POWER_XY_DRIFT_LIMIT,
        "rotation_drift_rad": POWER_ROTATION_DRIFT_LIMIT,
    }.items():
        _require_exact(thresholds, key, value, "artifact.thresholds")

    teacher = artifact.get("teacher_result")
    if not isinstance(teacher, dict):
        raise ValueError("teacher artifact is fail-closed: teacher_result is absent")
    if artifact.get("result") != teacher:
        raise ValueError("artifact.result must equal its authoritative teacher_result")
    results = artifact.get("results")
    if not isinstance(results, Mapping) or results.get("coupled_align_close21") != teacher:
        raise ValueError("artifact.results.coupled_align_close21 must equal teacher_result")
    if artifact.get("search_seed_result") != teacher:
        raise ValueError("artifact.search_seed_result must equal teacher_result")
    for key in (
        "strict_power_close_pass",
        "conservative_teacher_audit_pass",
        "native_option_success_all_replicates",
    ):
        _require_exact(teacher, key, True, "artifact.teacher_result")
    _require_exact(teacher, "pass_authority", PASS_AUTHORITY, "artifact.teacher_result")

    replicates = teacher.get("replicates")
    strict_replicates = teacher.get("strict_replicates")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates < 1
        or strict_replicates != replicates
    ):
        raise ValueError("teacher_result must pass every recorded physical replicate")
    for key in (
        "replicate_strict_pass",
        "native_success_per_replicate",
        "native_terminal_seen_per_replicate",
    ):
        _require_replicate_flags(teacher, key, replicates, True)
    for key in (
        "native_failure_per_replicate",
        "native_timeout_per_replicate",
        "arm_target_saturated_per_replicate",
        "native_terminal_align_active_per_replicate",
    ):
        _require_replicate_flags(teacher, key, replicates, False)

    latent = _require_finite_list(
        teacher, "latent", HAND_DIM, "artifact.teacher_result"
    )
    arm_delta = _require_finite_list(
        teacher, "arm_delta_target_rad", ARM_DIM, "artifact.teacher_result"
    )
    if max(abs(value) for value in latent) > 1.0:
        raise ValueError("teacher latent exceeds normalized [-1, 1]")
    if max(abs(value) for value in arm_delta) > ARM_TARGET_LIMIT_RAD + _VALUE_TOLERANCE:
        raise ValueError("teacher arm target offset exceeds the coupled 0.12 rad envelope")
    declared_max = teacher.get("arm_delta_abs_max_rad")
    if not isinstance(declared_max, (int, float)) or isinstance(declared_max, bool):
        raise TypeError("teacher arm_delta_abs_max_rad must be numeric")
    if not math.isfinite(float(declared_max)) or abs(
        float(declared_max) - max(abs(value) for value in arm_delta)
    ) > _VALUE_TOLERANCE:
        raise ValueError("teacher arm_delta_abs_max_rad disagrees with arm_delta_target_rad")
    return artifact, latent, arm_delta


@dataclass(frozen=True)
class HandResidualScales:
    """Raw normalized-action residual bounds for CLOSE and HOLD."""

    close_token: float = 0.04
    close_distal: float = 0.06
    hold_token: float = 0.015
    hold_distal: float = 0.025

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"residual scale {name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"residual scale {name} must be finite and in [0, 1]")

    def as_dict(self) -> dict[str, float]:
        return {
            "close_token": float(self.close_token),
            "close_distal": float(self.close_distal),
            "hold_token": float(self.hold_token),
            "hold_distal": float(self.hold_distal),
        }


@dataclass(frozen=True)
class CoupledTeacherPrior:
    """Validated state-conditioned teacher plus a phase-bounded hand residual."""

    arm_delta_target_rad: tuple[float, ...]
    hand_latent: tuple[float, ...]
    teacher_artifact_sha256: str
    curriculum_dataset_sha256: str
    teacher_artifact_path: str
    curriculum_dataset_path: str
    residual_scales: HandResidualScales = HandResidualScales()
    # These tensors are immutable implementation constants.  Keeping one copy
    # per execution device/dtype avoids allocating the teacher latent and the
    # phase-specific residual scales on every simulator step.  The cache is not
    # part of the portable prior contract or value equality.
    _constant_tensor_cache: dict[
        tuple[torch.device, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.arm_delta_target_rad) != ARM_DIM:
            raise ValueError(f"arm_delta_target_rad must contain {ARM_DIM} values")
        if len(self.hand_latent) != HAND_DIM:
            raise ValueError(f"hand_latent must contain {HAND_DIM} values")
        for name, values, limit in (
            ("arm_delta_target_rad", self.arm_delta_target_rad, ARM_TARGET_LIMIT_RAD),
            ("hand_latent", self.hand_latent, 1.0),
        ):
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} contains a non-finite value")
            if max(abs(float(value)) for value in values) > limit + _VALUE_TOLERANCE:
                raise ValueError(f"{name} exceeds its normalized safety envelope")
        _require_lower_sha256(
            self.teacher_artifact_sha256, "teacher_artifact_sha256"
        )
        _require_lower_sha256(
            self.curriculum_dataset_sha256, "curriculum_dataset_sha256"
        )

    @staticmethod
    def _validate_observation(observation: torch.Tensor) -> None:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("coupled teacher observation must be a torch.Tensor")
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
            raise ValueError(
                "coupled teacher observation must have shape "
                f"[batch, {OBSERVATION_DIM}], got {tuple(observation.shape)}"
            )
        if observation.shape[0] < 1:
            raise ValueError("coupled teacher observation batch must not be empty")
        if observation.dtype != torch.float32:
            raise TypeError(
                f"coupled teacher observation must use torch.float32, got {observation.dtype}"
            )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("coupled teacher observation contains NaN or infinity")

        latch = observation[:, POWER_LATCH_OBSERVATION_INDEX]
        align = observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX]
        progress = observation[:, ALIGN_PROGRESS_OBSERVATION_INDEX]
        for name, value in (("power latch", latch), ("ALIGN active", align)):
            if not bool(((value == 0.0) | (value == 1.0)).all()):
                raise ValueError(f"{name} observation must be an exact binary bit")
        if bool(((progress < 0.0) | (progress > 1.0)).any()):
            raise ValueError("ALIGN progress observation must lie in [0, 1]")
        if bool(((align == 1.0) & (latch == 1.0)).any()):
            raise ValueError("ALIGN-active and power-latched observations are contradictory")
        if bool(((align == 1.0) & (progress >= 1.0)).any()):
            raise ValueError("ALIGN cannot remain active at completed progress")
        if bool(((align == 0.0) & (latch == 0.0) & (progress < 1.0)).any()):
            raise ValueError(
                "unlatched CLOSE cannot begin before ALIGN progress completes"
            )

        arm_offset = observation[:, ARM_OFFSET_OBSERVATION_SLICE]
        if bool((arm_offset.abs() > 1.0 + _VALUE_TOLERANCE).any()):
            raise ValueError("normalized arm-target offset escaped [-1, 1]")

    @staticmethod
    def _phase_from_validated_observation(observation: torch.Tensor) -> torch.Tensor:
        """Classify a previously validated public observation without a sync."""

        align = observation[:, ALIGN_ACTIVE_OBSERVATION_INDEX] == 1.0
        latched = observation[:, POWER_LATCH_OBSERVATION_INDEX] == 1.0
        return torch.where(
            align,
            PHASE_ALIGN,
            torch.where(
                latched,
                PHASE_HOLD_LATCHED,
                PHASE_CLOSE_UNLATCHED,
            ),
        )

    @staticmethod
    def _validate_residual(
        observation: torch.Tensor, residual: torch.Tensor
    ) -> None:
        if not isinstance(residual, torch.Tensor):
            raise TypeError("hand residual must be a torch.Tensor")
        expected = (observation.shape[0], HAND_DIM)
        if residual.shape != expected:
            raise ValueError(
                f"hand residual must have shape {expected}, got {tuple(residual.shape)}"
            )
        if residual.device != observation.device:
            raise ValueError(
                f"hand residual is on {residual.device}, expected {observation.device}"
            )
        if residual.dtype != torch.float32:
            raise TypeError(f"hand residual must use torch.float32, got {residual.dtype}")
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("hand residual contains NaN or infinity")
        if bool((residual.abs() > 1.0 + _VALUE_TOLERANCE).any()):
            raise ValueError("hand residual exceeds normalized [-1, 1]")

    def _constant_tensors(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached arm, hand, CLOSE-scale, and HOLD-scale tensors."""

        key = (observation.device, observation.dtype)
        cached = self._constant_tensor_cache.get(key)
        if cached is not None:
            return cached
        scales = self.residual_scales
        tensors = (
            torch.tensor(
                self.arm_delta_target_rad,
                device=observation.device,
                dtype=observation.dtype,
            ),
            torch.tensor(
                self.hand_latent,
                device=observation.device,
                dtype=observation.dtype,
            ),
            torch.tensor(
                (float(scales.close_token),) * TOKEN_DIM
                + (float(scales.close_distal),) * DISTAL_DIM,
                device=observation.device,
                dtype=observation.dtype,
            ),
            torch.tensor(
                (float(scales.hold_token),) * TOKEN_DIM
                + (float(scales.hold_distal),) * DISTAL_DIM,
                device=observation.device,
                dtype=observation.dtype,
            ),
        )
        # A concurrent first call can harmlessly construct the same immutable
        # values twice; the last identical tuple stored becomes authoritative.
        self._constant_tensor_cache[key] = tensors
        return tensors

    @staticmethod
    def _canonicalize_from_phase(
        residual: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        canonical = residual.clamp(-1.0, 1.0)
        return canonical.masked_fill(
            (phase == PHASE_ALIGN).unsqueeze(-1), 0.0
        )

    def _teacher_action_from_phase(
        self, observation: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """Build the teacher action branchlessly from one phase classification."""

        arm_delta, hand_latent, _, _ = self._constant_tensors(observation)
        align = (phase == PHASE_ALIGN).unsqueeze(-1)
        progress = observation[:, ALIGN_PROGRESS_OBSERVATION_INDEX]
        next_progress = torch.clamp(
            progress + 1.0 / float(ALIGN_STEPS), 0.0, 1.0
        )
        blend = next_progress.square() * (3.0 - 2.0 * next_progress)
        desired_offset = blend.unsqueeze(-1) * arm_delta.unsqueeze(0)
        current_offset = (
            observation[:, ARM_OFFSET_OBSERVATION_SLICE] * ARM_TARGET_LIMIT_RAD
        )
        arm_action = torch.clamp(
            (desired_offset - current_offset) / ARM_FEEDBACK_DENOMINATOR,
            -1.0,
            1.0,
        )
        arm_action = torch.where(align, arm_action, 0.0)
        hand_action = torch.where(
            align,
            0.0,
            hand_latent.unsqueeze(0).expand(observation.shape[0], -1),
        )
        return torch.cat((arm_action, hand_action), dim=-1)

    def phase(self, observation: torch.Tensor) -> torch.Tensor:
        """Return 0=ALIGN, 1=CLOSE-unlatched, or 2=HOLD-latched per row."""

        self._validate_observation(observation)
        return self._phase_from_validated_observation(observation)

    def teacher_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Reconstruct the canonical pre-shield 21-D CEM teacher action."""

        self._validate_observation(observation)
        phase = self._phase_from_validated_observation(observation)
        return self._teacher_action_from_phase(observation, phase)

    def canonicalize_residual(
        self, observation: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Clamp a policy residual and erase every ALIGN row exactly.

        The returned 14-D tensor is the action that must be stored in replay.
        """

        self._validate_observation(observation)
        self._validate_residual(observation, residual)
        phase = self._phase_from_validated_observation(observation)
        return self._canonicalize_from_phase(residual, phase)

    def compose_action(
        self, observation: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose one canonical residual and its exact native action.

        Observation and residual contracts are checked once, then the public
        phase is classified once.  All phase-dependent math is expressed with
        tensor masks, so mixed CUDA batches do not incur Python ``bool(any)``
        synchronization while the canonical and native actions are composed.
        """

        self._validate_observation(observation)
        self._validate_residual(observation, residual)
        phase = self._phase_from_validated_observation(observation)
        canonical = self._canonicalize_from_phase(residual, phase)
        teacher = self._teacher_action_from_phase(observation, phase)
        _, _, close_scales, hold_scales = self._constant_tensors(observation)
        phase_column = phase.unsqueeze(-1)
        residual_scales = torch.where(
            phase_column == PHASE_CLOSE_UNLATCHED,
            close_scales.unsqueeze(0),
            hold_scales.unsqueeze(0),
        )
        environment_action = torch.cat(
            (
                teacher[:, :ARM_DIM],
                teacher[:, ARM_DIM:] + residual_scales * canonical,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        return canonical, environment_action

    def to_environment_action(
        self, observation: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Map a 14-D canonical policy residual to the native 21-D action."""

        _, environment_action = self.compose_action(observation, residual)
        return environment_action

    def contract_payload(self) -> dict[str, Any]:
        """Return portable checkpoint metadata for this immutable action map."""

        return {
            "version": 1,
            "contract": "coupled_cem_teacher_prior_plus_bounded_hand_residual_v1",
            "task_mode": POLICY_TASK_MODE,
            "observation_dim": OBSERVATION_DIM,
            "observation_contract": OBSERVATION_CONTRACT,
            "policy_action_dim": HAND_DIM,
            "policy_action_layout": (
                "crossdex_token_residual9|distal_action_residual5"
            ),
            "environment_action_dim": ACTION_DIM,
            "action_projection": (
                "coupled_cem_teacher_prior_plus_bounded_hand_residual_v1"
            ),
            "policy_action_semantics": "canonical_phase_masked_hand_residual_v1",
            "align_policy_residual": "exact_zero",
            "align_steps": ALIGN_STEPS,
            "arm_target_limit_rad": ARM_TARGET_LIMIT_RAD,
            "arm_action_scale": ARM_ACTION_SCALE,
            "arm_target_ema": ARM_TARGET_EMA,
            "arm_action_multiplier": ARM_ACTION_MULTIPLIER,
            "arm_feedback_denominator": ARM_FEEDBACK_DENOMINATOR,
            "arm_delta_target_rad": list(self.arm_delta_target_rad),
            "hand_latent": list(self.hand_latent),
            "residual_scales": self.residual_scales.as_dict(),
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "curriculum_dataset_sha256": self.curriculum_dataset_sha256,
        }


def load_coupled_teacher_prior(
    teacher_artifact: str | Path,
    curriculum_dataset: str | Path,
    *,
    residual_scales: HandResidualScales | None = None,
) -> CoupledTeacherPrior:
    """Load a strict CEM result and bind it to the supplied curriculum bytes."""

    teacher_path = Path(teacher_artifact).expanduser().resolve()
    curriculum_path = Path(curriculum_dataset).expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    if not curriculum_path.is_file():
        raise FileNotFoundError(curriculum_path)
    curriculum_sha256 = sha256_file(curriculum_path)
    _validate_curriculum(curriculum_path)
    _, latent, arm_delta = _validate_teacher_artifact(
        teacher_path, curriculum_sha256=curriculum_sha256
    )
    return CoupledTeacherPrior(
        arm_delta_target_rad=arm_delta,
        hand_latent=latent,
        teacher_artifact_sha256=sha256_file(teacher_path),
        curriculum_dataset_sha256=curriculum_sha256,
        teacher_artifact_path=str(teacher_path),
        curriculum_dataset_path=str(curriculum_path),
        residual_scales=(
            HandResidualScales() if residual_scales is None else residual_scales
        ),
    )


def coupled_teacher_prior_from_payload(
    payload: Mapping[str, Any],
    *,
    source: str = "teacher_action_prior",
) -> CoupledTeacherPrior:
    """Reconstruct and strictly validate an embedded checkpoint prior."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must be a mapping")
    expected_scalars: dict[str, object] = {
        "version": 1,
        "contract": "coupled_cem_teacher_prior_plus_bounded_hand_residual_v1",
        "task_mode": POLICY_TASK_MODE,
        "observation_dim": OBSERVATION_DIM,
        "observation_contract": OBSERVATION_CONTRACT,
        "policy_action_dim": HAND_DIM,
        "policy_action_layout": "crossdex_token_residual9|distal_action_residual5",
        "environment_action_dim": ACTION_DIM,
        "action_projection": "coupled_cem_teacher_prior_plus_bounded_hand_residual_v1",
        "policy_action_semantics": "canonical_phase_masked_hand_residual_v1",
        "align_policy_residual": "exact_zero",
        "align_steps": ALIGN_STEPS,
        "arm_target_limit_rad": ARM_TARGET_LIMIT_RAD,
        "arm_action_scale": ARM_ACTION_SCALE,
        "arm_target_ema": ARM_TARGET_EMA,
        "arm_action_multiplier": ARM_ACTION_MULTIPLIER,
        "arm_feedback_denominator": ARM_FEEDBACK_DENOMINATOR,
    }
    for key, expected in expected_scalars.items():
        _require_exact(payload, key, expected, source)
    arm_delta = _require_finite_list(payload, "arm_delta_target_rad", ARM_DIM, source)
    hand_latent = _require_finite_list(payload, "hand_latent", HAND_DIM, source)
    scales_payload = payload.get("residual_scales")
    if not isinstance(scales_payload, Mapping):
        raise TypeError(f"{source}.residual_scales must be a mapping")
    expected_scale_keys = {"close_token", "close_distal", "hold_token", "hold_distal"}
    if set(scales_payload) != expected_scale_keys:
        raise ValueError(
            f"{source}.residual_scales must contain exactly {sorted(expected_scale_keys)}"
        )
    scales = HandResidualScales(
        close_token=scales_payload["close_token"],
        close_distal=scales_payload["close_distal"],
        hold_token=scales_payload["hold_token"],
        hold_distal=scales_payload["hold_distal"],
    )
    teacher_sha = _require_lower_sha256(
        payload.get("teacher_artifact_sha256"),
        f"{source}.teacher_artifact_sha256",
    )
    curriculum_sha = _require_lower_sha256(
        payload.get("curriculum_dataset_sha256"),
        f"{source}.curriculum_dataset_sha256",
    )
    expected_keys = set(expected_scalars) | {
        "arm_delta_target_rad",
        "hand_latent",
        "residual_scales",
        "teacher_artifact_sha256",
        "curriculum_dataset_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"{source} fields differ from the closed contract: "
            f"missing={sorted(expected_keys.difference(payload))}, "
            f"extra={sorted(set(payload).difference(expected_keys))}"
        )
    return CoupledTeacherPrior(
        arm_delta_target_rad=arm_delta,
        hand_latent=hand_latent,
        teacher_artifact_sha256=teacher_sha,
        curriculum_dataset_sha256=curriculum_sha,
        teacher_artifact_path=f"<{source}:embedded-teacher-lineage>",
        curriculum_dataset_path=f"<{source}:embedded-curriculum-lineage>",
        residual_scales=scales,
    )


def load_coupled_teacher_prior_payload(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CoupledTeacherPrior:
    """Load an embedded JSON prior, optionally binding its exact file SHA256."""

    prior_path = Path(path).expanduser()
    if prior_path.is_symlink() or not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    try:
        raw_bytes = prior_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read embedded teacher prior {prior_path}: {error}") from error
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None:
        expected = _require_lower_sha256(expected_sha256, "expected prior SHA256")
        if actual_sha256 != expected:
            raise ValueError(
                f"embedded teacher prior SHA256 mismatch: {actual_sha256} != {expected}"
            )
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse embedded teacher prior {prior_path}: {error}") from error
    return coupled_teacher_prior_from_payload(
        payload, source=str(prior_path.resolve())
    )


__all__ = [
    "ACTION_DIM",
    "ALIGN_ACTIVE_OBSERVATION_INDEX",
    "ALIGN_PROGRESS_OBSERVATION_INDEX",
    "ALIGN_STEPS",
    "ARM_DIM",
    "ARM_FEEDBACK_DENOMINATOR",
    "ARM_OFFSET_OBSERVATION_SLICE",
    "CoupledTeacherPrior",
    "HAND_DIM",
    "HandResidualScales",
    "OBSERVATION_DIM",
    "PHASE_ALIGN",
    "PHASE_CLOSE_UNLATCHED",
    "PHASE_HOLD_LATCHED",
    "PHYSICAL_TASK_MODE",
    "POLICY_TASK_MODE",
    "POWER_LATCH_OBSERVATION_INDEX",
    "coupled_teacher_prior_from_payload",
    "load_coupled_teacher_prior",
    "load_coupled_teacher_prior_payload",
    "sha256_file",
]
