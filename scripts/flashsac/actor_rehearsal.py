"""GPU-native observation/action demonstrations for actor-only rehearsal.

The PickTool lift teacher datasets intentionally contain no reward, terminal
flag, or next observation.  They are valid behavior-cloning supervision, but
they are not valid SAC transitions.  This module keeps that distinction
structural: :class:`ActorRehearsalReservoir` stores only observations and
actions and is never installed as an agent replay buffer.

Rows are copied once onto the configured Torch device, then sampled using a
private device-local generator.  Optional phase labels support deterministic
largest-remainder stratification.  Dataset contents, sampler state, source
SHA256 order, and sampling configuration are checkpointed so the first batch
after an exact rehearsal-state resume is bitwise identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch


ACTOR_REHEARSAL_VERSION = 1
ACTOR_REHEARSAL_KEYS = ("observation", "action")


@dataclass(frozen=True)
class EpisodeTensorContract:
    """Required per-episode evidence carried by an actor-demo source."""

    name: str
    dtype: torch.dtype
    expected_bool: bool | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("episode tensor contract name must be non-empty")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("episode tensor contract dtype must be torch.dtype")
        if self.dtype == torch.bool:
            if (
                self.expected_bool is None
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ValueError(
                    "bool episode tensor contracts require expected_bool and no bounds"
                )
        elif self.expected_bool is not None:
            raise ValueError("numeric episode tensor contracts cannot use expected_bool")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("episode tensor minimum cannot exceed maximum")


@dataclass(frozen=True)
class ActorRehearsalSourceContract:
    """Exact metadata/phase allowlist entry for an actor-only dataset."""

    name: str
    required_metadata: Mapping[str, Any]
    required_phase: int | None = None
    required_phases: tuple[int, ...] = ()
    optional_phases: tuple[int, ...] = ()
    required_sha256_metadata: tuple[str, ...] = ()
    required_zero_action_slices: tuple[tuple[int, int, int], ...] = ()
    required_observation_values: tuple[tuple[int, float], ...] = ()
    strict_metadata_types: bool = False
    require_successful_episodes: bool = True
    required_episode_tensors: tuple[EpisodeTensorContract, ...] = ()
    required_episode_conditional_values: tuple[
        tuple[str, float | int, str, float | int], ...
    ] = ()
    required_episode_orderings: tuple[tuple[str, str], ...] = ()
    phase_observation_state_indices: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        has_single_phase = self.required_phase is not None
        has_multiple_phases = bool(self.required_phases)
        if has_single_phase == has_multiple_phases:
            raise ValueError(
                "actor rehearsal source contract must declare exactly one of "
                "required_phase or required_phases"
            )
        if has_single_phase and (
            not isinstance(self.required_phase, int)
            or isinstance(self.required_phase, bool)
            or self.required_phase < 0
        ):
            raise ValueError("required_phase must be a non-negative integer")
        if has_multiple_phases:
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in self.required_phases
            ):
                raise ValueError("required_phases must contain non-negative integers")
            if len(set(self.required_phases)) != len(self.required_phases):
                raise ValueError("required_phases must not contain duplicates")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in self.optional_phases
        ):
            raise ValueError("optional_phases must contain non-negative integers")
        if len(set(self.optional_phases)) != len(self.optional_phases):
            raise ValueError("optional_phases must not contain duplicates")
        if set(self.optional_phases).intersection(self.phase_values):
            raise ValueError("required and optional phases must be disjoint")
        if len(set(self.required_sha256_metadata)) != len(
            self.required_sha256_metadata
        ):
            raise ValueError("required SHA256 metadata keys must not contain duplicates")
        if any(
            not isinstance(key, str) or not key
            for key in self.required_sha256_metadata
        ):
            raise ValueError("required SHA256 metadata keys must be non-empty strings")
        if not isinstance(self.strict_metadata_types, bool):
            raise TypeError("strict_metadata_types must be bool")
        if not isinstance(self.require_successful_episodes, bool):
            raise TypeError("require_successful_episodes must be bool")
        for phase, start, stop in self.required_zero_action_slices:
            if phase not in self.allowed_phase_values:
                raise ValueError(
                    "zero-action slice phase must be declared by the source contract"
                )
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(stop, int)
                or isinstance(stop, bool)
                or start < 0
                or stop <= start
            ):
                raise ValueError("zero-action slices must use integer 0 <= start < stop")
        observation_indices = [
            index for index, _expected in self.required_observation_values
        ]
        if len(set(observation_indices)) != len(observation_indices):
            raise ValueError("required observation-value indices must not repeat")
        for index, expected in self.required_observation_values:
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
            ):
                raise ValueError(
                    "required observation values need non-negative integer indices"
                )
            if (
                not isinstance(expected, (int, float))
                or isinstance(expected, bool)
                or not math.isfinite(float(expected))
            ):
                raise ValueError(
                    "required observation values need finite numeric expectations"
                )
        episode_field_names = [field.name for field in self.required_episode_tensors]
        if len(set(episode_field_names)) != len(episode_field_names):
            raise ValueError("required episode tensor names must not contain duplicates")
        episode_fields = set(episode_field_names)
        if len(set(self.required_episode_conditional_values)) != len(
            self.required_episode_conditional_values
        ):
            raise ValueError("required episode conditional values must not repeat")
        for condition_name, condition_value, target_name, target_value in (
            self.required_episode_conditional_values
        ):
            if condition_name not in episode_fields or target_name not in episode_fields:
                raise ValueError(
                    "episode conditional values must reference required episode tensors"
                )
            for value in (condition_value, target_value):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        "episode conditional values must use finite numeric scalars"
                    )
        if len(set(self.required_episode_orderings)) != len(
            self.required_episode_orderings
        ):
            raise ValueError("required episode orderings must not repeat")
        for earlier_name, later_name in self.required_episode_orderings:
            if earlier_name not in episode_fields or later_name not in episode_fields:
                raise ValueError(
                    "episode orderings must reference required episode tensors"
                )
        if self.phase_observation_state_indices is not None:
            if (
                len(self.phase_observation_state_indices) != 2
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    for index in self.phase_observation_state_indices
                )
            ):
                raise ValueError(
                    "phase observation state indices must contain two non-negative integers"
                )

    @property
    def phase_values(self) -> tuple[int, ...]:
        if self.required_phase is not None:
            return (self.required_phase,)
        return self.required_phases

    @property
    def allowed_phase_values(self) -> tuple[int, ...]:
        return (*self.phase_values, *self.optional_phases)


_PICK_TOOL_ACTION_LAYOUT = "arm_delta7|crossdex_token9|distal_residual5"
_PICK_TOOL_OBSERVATION_LAYOUT = "legacy_prefix87|distal_action5|grasp_transport23"
_PICK_TOOL_PHASE_NAMES = ["approach", "close", "micro", "lift", "settle"]
_COUPLED_POWER_TASK_MODE = "coupled_power_align_close_option_v1"
_COUPLED_POWER_OBSERVATION_CONTRACT = (
    "pick_tool_coupled_power_align_close_state131_v1"
)
_COUPLED_POWER_PHASE_NAMES = ["align", "close_unlatched", "hold_latched"]

PICK_TOOL_ACTOR_DEMO_CONTRACTS = (
    ActorRehearsalSourceContract(
        name="scripted_lift_teacher",
        required_metadata={
            "action_layout": _PICK_TOOL_ACTION_LAYOUT,
            "observation_layout": _PICK_TOOL_OBSERVATION_LAYOUT,
            "phase_names": _PICK_TOOL_PHASE_NAMES,
            "collector": "online_base_close_to_scripted_lift_teacher",
            "dataset_phase": "lift",
            "close_arm_mode": "zero",
            "teacher_probability": 1.0,
            "executed_teacher_fraction": 1.0,
            "first_observation_last_action_max_error": 0.0,
        },
        required_phase=3,
    ),
    ActorRehearsalSourceContract(
        # Legacy July-2026 close datasets predate ``dataset_phase`` and used a
        # close-specific arm-action audit key. Keep this schema explicit rather
        # than silently accepting arbitrary missing metadata.
        name="frozen_base_close_teacher_legacy",
        required_metadata={
            "action_layout": _PICK_TOOL_ACTION_LAYOUT,
            "observation_layout": _PICK_TOOL_OBSERVATION_LAYOUT,
            "phase_names": _PICK_TOOL_PHASE_NAMES,
            "collector": "online_frozen_base_to_close_teacher",
            "close_arm_mode": "zero",
            "first_observation_last_action_max_error": 0.0,
            "close_teacher_arm_action_abs_max": 0.0,
        },
        required_phase=1,
    ),
    ActorRehearsalSourceContract(
        # Current base_handoff_close_dataset.py producer schema. The saved
        # action is always the close teacher label, including on DAgger rows;
        # rollout teacher-execution fractions therefore do not constrain BC.
        name="frozen_base_close_teacher",
        required_metadata={
            "action_layout": _PICK_TOOL_ACTION_LAYOUT,
            "observation_layout": _PICK_TOOL_OBSERVATION_LAYOUT,
            "phase_names": _PICK_TOOL_PHASE_NAMES,
            "collector": "online_frozen_base_to_close_teacher",
            "dataset_phase": "close",
            "close_arm_mode": "zero",
            "first_observation_last_action_max_error": 0.0,
            "option_teacher_arm_action_abs_max": 0.0,
        },
        required_phase=1,
    ),
    ActorRehearsalSourceContract(
        # Successful actions executed by the frozen V6 routed policy, with a
        # preregistered deterministic/exploratory cohort assignment.  Unlike
        # the historical CLOSE teacher datasets, these labels are policy
        # actions sampled on the exact public SEARCH handoff distribution and
        # are retained only when the native task subsequently reaches a safe
        # true-mesh 20 cm success.  Every lineage digest is mandatory so a
        # similarly shaped dataset from another SEARCH/V6/LIFT stack cannot be
        # admitted by this allowlist entry.
        name="pick_tool_v6_success_self_imitation_v1",
        required_metadata={
            "format_version": 1,
            "task_mode": "full_task",
            "observation_dim": 115,
            "observation_contract": "pick_tool_markov115_v1",
            "action_dim": 21,
            "action_layout": _PICK_TOOL_ACTION_LAYOUT,
            "action_projection": "identity_v1",
            "observation_layout": _PICK_TOOL_OBSERVATION_LAYOUT,
            "phase_names": _PICK_TOOL_PHASE_NAMES,
            "collector": "pick_tool_v6_success_self_imitation_v1",
            "dataset_phase": "close",
            "close_arm_mode": "zero",
            "action_semantics": (
                "executed_v6_action_with_latch_conditioned_exploration_v1"
            ),
            "stored_row_window": (
                "trigger_frame_through_first_latch_transition_v1"
            ),
            "trajectory_acceptance": (
                "initial_episode_triggered_true_success_safe_first_latch_persistent_v1"
            ),
            "clearance_authority": (
                "true_mesh_convex_hull_min_z_minus_table_v1"
            ),
            "terminal_observation": "not_saved_auto_reset_excluded_v1",
            "search_handoff_contract": "pick_tool_public_online_handoff_v1",
            "trigger_action_semantics": (
                "option_controls_the_trigger_frame_and_remains_sticky_until_reset"
            ),
            "policy_router": "public_latch_frozen_actor_v1",
            "latch_transition_clock": (
                "post_action_transition_observation_public_latch_v1"
            ),
            "cohort_assignment": "balanced_sha256_slot_v1",
            "cohort_names": ["deterministic", "exploratory"],
            "kit_args": "--/app/extensions/fsWatcherEnabled=false",
            "handoff_min_score": 0.30,
            "handoff_hold_steps": 4,
            "exploratory_hand_noise_scale": 0.25,
            "unlatched_arm_noise_scale": 0.0,
            "latched_arm_noise_scale": 0.0,
            "latched_hand_noise_scale": 0.0,
        },
        required_phase=1,
        required_sha256_metadata=(
            "search_checkpoint_sha256",
            "v6_actor_sha256",
            "v6_task_contract_sha256",
            "v6_bridge_state_sha256",
            "frozen_lift_actor_sha256",
            "frozen_lift_semantic_sha256",
            "frozen_lift_source_actor_sha256",
            "source_manifest_sha256",
            "runtime_asset_manifest_sha256",
        ),
        required_zero_action_slices=((1, 0, 7),),
        required_observation_values=((106, 0.0),),
        strict_metadata_types=True,
        required_episode_tensors=(
            EpisodeTensorContract(
                "episode_native_success",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_dropped",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_unsafe_force",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_unlatched_clearance_ge_5cm",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_ever_grasped",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_triggered",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_ever_latched",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_latch_released_after_first",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_cohort",
                torch.int64,
                minimum=0,
                maximum=1,
            ),
            EpisodeTensorContract(
                "episode_hand_noise_scale",
                torch.float32,
                minimum=0.0,
                maximum=0.25,
            ),
            EpisodeTensorContract(
                "episode_trigger_step",
                torch.int64,
                minimum=0,
            ),
            EpisodeTensorContract(
                "episode_trigger_score",
                torch.float32,
                minimum=0.30,
                maximum=1.0,
            ),
            EpisodeTensorContract(
                "episode_first_latch_step",
                torch.int64,
                minimum=0,
            ),
            EpisodeTensorContract(
                "episode_terminal_step",
                torch.int64,
                minimum=0,
            ),
            EpisodeTensorContract(
                "episode_trajectory_max_force",
                torch.float32,
                minimum=0.0,
                maximum=30.0,
            ),
            EpisodeTensorContract(
                "episode_max_true_clearance_m",
                torch.float32,
                minimum=0.20,
            ),
        ),
        required_episode_conditional_values=(
            ("episode_cohort", 0, "episode_hand_noise_scale", 0.0),
            ("episode_cohort", 1, "episode_hand_noise_scale", 0.25),
        ),
        required_episode_orderings=(
            ("episode_trigger_step", "episode_first_latch_step"),
            ("episode_first_latch_step", "episode_terminal_step"),
        ),
    ),
)

PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS = tuple(
    contract for contract in PICK_TOOL_ACTOR_DEMO_CONTRACTS if contract.required_phase == 3
)
PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS = tuple(
    contract for contract in PICK_TOOL_ACTOR_DEMO_CONTRACTS if contract.required_phase == 1
)

_PICK_TOOL_DAGGER_PHASE0_SOURCE_SHA256 = (
    "e4aaf0eda6a33db4a2ed04bc4d7609639da9b0471408808a5dae1b67760ce57f"
)
_PICK_TOOL_DAGGER_PHASE0_SPLIT_SALT = (
    "pick_tool_candidate44_phase0_correction_split_20260723_v1"
)
_PICK_TOOL_DAGGER_PHASE0_COMMON_METADATA = {
    "format_version": 1,
    "task_mode": "full_task",
    "observation_dim": 115,
    "observation_contract": "pick_tool_markov115_v1",
    "observation_layout": _PICK_TOOL_OBSERVATION_LAYOUT,
    "action_dim": 21,
    "action_layout": _PICK_TOOL_ACTION_LAYOUT,
    "phase_names": _PICK_TOOL_PHASE_NAMES,
    "collector": "pick_tool_dagger_oracle_phase0_correction_v1",
    "dataset_phase": "approach",
    "action_semantics": "oracle_correction_label_on_logged_state_v1",
    "episode_semantics": "correction_labels_without_outcome_claim_v1",
    "critic_replay_eligible": False,
    "source_dataset_sha256": _PICK_TOOL_DAGGER_PHASE0_SOURCE_SHA256,
    "source_dataset_rows": 797450,
    "source_dataset_episodes": 1685,
    "selection": "phase_eq_0_and_observation_106_eq_0_v1",
    "split_method": "sha256_rank_source_episode_80_20_v1",
    "split_salt": _PICK_TOOL_DAGGER_PHASE0_SPLIT_SALT,
    "row_order": "hash_rank_episode_then_source_row_v1",
    "episode_id_semantics": "source_episode_id_v1",
}

PICK_TOOL_APPROACH_CORRECTION_TRAIN_CONTRACTS = (
    ActorRehearsalSourceContract(
        name="pick_tool_dagger_phase0_correction_train_v1",
        required_metadata={
            **_PICK_TOOL_DAGGER_PHASE0_COMMON_METADATA,
            "split": "train",
            "split_episodes": 734,
            "split_rows": 220181,
        },
        required_phase=0,
        required_sha256_metadata=("source_dataset_sha256",),
        required_observation_values=((106, 0.0),),
        strict_metadata_types=True,
        require_successful_episodes=False,
        required_episode_tensors=(
            EpisodeTensorContract(
                "source_episode_id",
                torch.int64,
                minimum=0,
                maximum=1684,
            ),
        ),
    ),
)

PICK_TOOL_APPROACH_CORRECTION_VALIDATION_CONTRACTS = (
    ActorRehearsalSourceContract(
        name="pick_tool_dagger_phase0_correction_validation_v1",
        required_metadata={
            **_PICK_TOOL_DAGGER_PHASE0_COMMON_METADATA,
            "split": "validation",
            "split_episodes": 183,
            "split_rows": 54900,
        },
        required_phase=0,
        required_sha256_metadata=("source_dataset_sha256",),
        required_observation_values=((106, 0.0),),
        strict_metadata_types=True,
        require_successful_episodes=False,
        required_episode_tensors=(
            EpisodeTensorContract(
                "source_episode_id",
                torch.int64,
                minimum=0,
                maximum=1684,
            ),
        ),
    ),
)

PICK_TOOL_FULL_TASK_ACTOR_DEMO_CONTRACTS = (
    *PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS,
    *PICK_TOOL_APPROACH_CORRECTION_TRAIN_CONTRACTS,
)

PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS = (
    ActorRehearsalSourceContract(
        name="successful_coupled_power_align_close_teacher_v1",
        required_metadata={
            "format_version": 1,
            "task_mode": _COUPLED_POWER_TASK_MODE,
            "observation_dim": 131,
            "observation_contract": _COUPLED_POWER_OBSERVATION_CONTRACT,
            "observation_layout": (
                "legacy_prefix87|distal_action5|grasp_transport23|coupled_state16"
            ),
            "action_dim": 21,
            "action_layout": _PICK_TOOL_ACTION_LAYOUT,
            "action_projection": "identity_v1",
            "action_semantics": "canonical_policy_action_before_phase_shield_v1",
            "collector": "successful_coupled_power_align_close_teacher",
            "dataset_phase": "align_close_hold",
            "phase_names": _COUPLED_POWER_PHASE_NAMES,
            "teacher_probability": 1.0,
            "executed_teacher_fraction": 1.0,
            "align_hand_action_abs_max": 0.0,
            "close_arm_action_abs_max": 0.0,
            "first_observation_last_action_max_error": 0.0,
            "terminal_observation": "not_saved_auto_reset_excluded_v1",
            "trajectory_acceptance": (
                "native_success_and_conservative_teacher_audit_v1"
            ),
            "clearance_authority": (
                "true_mesh_convex_hull_min_z_minus_table_v1"
            ),
        },
        required_phases=(0, 2),
        optional_phases=(1,),
        required_sha256_metadata=(
            "teacher_artifact_sha256",
            "curriculum_dataset_sha256",
        ),
        required_zero_action_slices=(
            (0, 7, 21),
            (1, 0, 7),
            (2, 0, 7),
        ),
        strict_metadata_types=True,
        required_episode_tensors=(
            EpisodeTensorContract(
                "episode_native_success",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_native_failure",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_native_timeout",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_conservative_teacher_pass",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_terminal_stable_steps",
                torch.int64,
                minimum=15,
                maximum=15,
            ),
            EpisodeTensorContract(
                "episode_terminal_power_is_grasped",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_terminal_thumb_contact",
                torch.bool,
                expected_bool=True,
            ),
            EpisodeTensorContract(
                "episode_terminal_legal_other_contact_count",
                torch.int64,
                minimum=3,
                maximum=4,
            ),
            EpisodeTensorContract(
                "episode_terminal_power_grasp_quality",
                torch.float32,
                minimum=0.35,
                maximum=1.0,
            ),
            EpisodeTensorContract(
                "episode_terminal_hold_quality",
                torch.float32,
                minimum=0.50,
                maximum=1.0,
            ),
            EpisodeTensorContract(
                "episode_terminal_max_force",
                torch.float32,
                minimum=0.0,
                maximum=30.0,
            ),
            EpisodeTensorContract(
                "episode_trajectory_max_force",
                torch.float32,
                minimum=0.0,
                maximum=30.0,
            ),
            EpisodeTensorContract(
                "episode_trajectory_max_xy_drift",
                torch.float32,
                minimum=0.0,
                maximum=0.03,
            ),
            EpisodeTensorContract(
                "episode_trajectory_max_rotation_drift",
                torch.float32,
                minimum=0.0,
                maximum=0.35,
            ),
            EpisodeTensorContract(
                "episode_trajectory_max_true_clearance",
                torch.float32,
                maximum=0.015,
            ),
            EpisodeTensorContract(
                "episode_arm_target_saturated",
                torch.bool,
                expected_bool=False,
            ),
            EpisodeTensorContract(
                "episode_terminal_align_active",
                torch.bool,
                expected_bool=False,
            ),
        ),
        phase_observation_state_indices=(129, 106),
    ),
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _canonical_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda:0")
    return resolved


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the raw-file SHA256 used as immutable source lineage."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_fingerprints(values: Sequence[str]) -> tuple[str, ...]:
    fingerprints = tuple(str(value).lower() for value in values)
    if not fingerprints:
        raise ValueError("actor rehearsal requires at least one source fingerprint")
    for value in fingerprints:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid actor rehearsal SHA256 fingerprint {value!r}")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("actor rehearsal source fingerprints must not contain duplicates")
    return fingerprints


def _require_actor_batch(
    batch: Mapping[str, Any],
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
    validate_values: bool = True,
) -> int:
    missing = [key for key in ACTOR_REHEARSAL_KEYS if key not in batch]
    if missing:
        raise KeyError(f"actor rehearsal batch is missing required keys: {missing}")
    extras = sorted(set(batch).difference(ACTOR_REHEARSAL_KEYS))
    if extras:
        raise KeyError(f"actor rehearsal batch has unsupported keys: {extras}")
    for key in ACTOR_REHEARSAL_KEYS:
        if not isinstance(batch[key], torch.Tensor):
            raise TypeError(f"actor rehearsal {key!r} must be a torch.Tensor")

    observation = batch["observation"]
    action = batch["action"]
    if observation.ndim != 2:
        raise ValueError(
            "actor rehearsal observation must have shape [batch, observation_dim]; "
            f"got {tuple(observation.shape)}"
        )
    rows = int(observation.shape[0])
    if rows < 1:
        raise ValueError("actor rehearsal batch must not be empty")
    if action.ndim != 2 or action.shape[0] != rows:
        raise ValueError(
            "actor rehearsal action must have shape [batch, action_dim] with the same "
            f"batch size as observation; got {tuple(action.shape)}"
        )
    if observation_dim is not None and observation.shape[1] != observation_dim:
        raise ValueError(
            f"expected actor rehearsal observation_dim={observation_dim}, "
            f"got {observation.shape[1]}"
        )
    if action_dim is not None and action.shape[1] != action_dim:
        raise ValueError(
            f"expected actor rehearsal action_dim={action_dim}, got {action.shape[1]}"
        )
    for key in ACTOR_REHEARSAL_KEYS:
        value = batch[key]
        if not value.is_floating_point():
            raise TypeError(f"actor rehearsal {key!r} must have a floating dtype")
        if validate_values and not bool(torch.isfinite(value).all()):
            raise ValueError(f"actor rehearsal {key!r} contains NaN or infinity")
    if validate_values and float(action.abs().max()) > 1.0001:
        raise ValueError("actor rehearsal action exceeds the normalized [-1, 1] range")
    return rows


def _require_phase(
    phase: Any,
    *,
    rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    if phase is None:
        return None
    if not isinstance(phase, torch.Tensor):
        raise TypeError("actor rehearsal phase must be a torch.Tensor")
    if phase.ndim != 1 or phase.shape != (rows,):
        raise ValueError(f"actor rehearsal phase must have shape ({rows},)")
    if phase.dtype not in _INTEGER_DTYPES:
        raise TypeError("actor rehearsal phase must use an integer dtype")
    return phase.detach().to(device=device, dtype=torch.int64, copy=True)


def _audit_episode_partition(
    payload: Mapping[str, Any],
    *,
    rows: int,
    require_successful_episodes: bool = True,
) -> int:
    offsets = payload.get("episode_offsets")
    if not isinstance(require_successful_episodes, bool):
        raise TypeError("require_successful_episodes must be bool")
    if not isinstance(offsets, torch.Tensor):
        raise TypeError("actor rehearsal dataset requires tensor episode_offsets")
    if offsets.dtype not in _INTEGER_DTYPES:
        raise TypeError("episode_offsets must use an integer dtype")
    offsets = offsets.detach().to(device="cpu", dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("episode_offsets must be one-dimensional and contain [0, ..., N]")
    if int(offsets[0]) != 0 or int(offsets[-1]) != rows:
        raise ValueError(f"episode_offsets must start at 0 and end at {rows}")
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("episode_offsets must be strictly increasing")

    episodes = int(offsets.numel() - 1)
    if require_successful_episodes:
        successes = payload.get("episode_success")
        if not isinstance(successes, torch.Tensor):
            raise TypeError("actor rehearsal dataset requires tensor episode_success")
        if successes.dtype != torch.bool or successes.shape != (episodes,):
            raise ValueError(f"episode_success must be bool with shape ({episodes},)")
        if not bool(successes.all()):
            raise ValueError("actor rehearsal may contain only successful episodes")
    elif "episode_success" in payload:
        raise ValueError(
            "correction-label actor rehearsal must omit episode_success rather than "
            "fabricating an outcome claim"
        )

    episode_id = payload.get("episode_id")
    if episode_id is not None:
        if not isinstance(episode_id, torch.Tensor) or episode_id.dtype not in _INTEGER_DTYPES:
            raise TypeError("episode_id must be an integer tensor when present")
        if episode_id.shape != (rows,):
            raise ValueError(f"episode_id must have shape ({rows},)")
        ids = episode_id.detach().to(device="cpu", dtype=torch.int64)
        for episode, (start, stop) in enumerate(
            zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)
        ):
            segment = ids[start:stop]
            if segment.numel() == 0 or bool((segment != segment[0]).any()):
                raise ValueError(f"episode_id changes inside episode_offsets segment {episode}")
        if torch.unique(ids, sorted=False).numel() != episodes:
            raise ValueError("episode_id unique count does not match episode_offsets")
    return episodes


def _metadata_value_matches(actual: Any, expected: Any, *, strict_type: bool) -> bool:
    if strict_type and type(actual) is not type(expected):
        return False
    try:
        comparison = actual == expected
    except (TypeError, ValueError):
        return False
    return comparison if isinstance(comparison, bool) else False


def audit_actor_rehearsal_metadata(
    payload: Mapping[str, Any],
    *,
    observation_dim: int,
    action_dim: int,
    phase: torch.Tensor | None,
    expected_metadata: Mapping[str, Any] | None,
    allowed_contracts: Sequence[ActorRehearsalSourceContract] | None,
) -> tuple[Mapping[str, Any], ActorRehearsalSourceContract | None]:
    metadata = payload.get("meta")
    if not isinstance(metadata, Mapping):
        raise TypeError("actor rehearsal dataset requires mapping metadata in 'meta'")
    if metadata.get("format_version") != 1:
        raise ValueError(
            f"actor rehearsal format_version={metadata.get('format_version')!r}, expected 1"
        )
    for key in ("action_layout", "collector"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"actor rehearsal metadata requires non-empty {key!r}")
    observation_schema_keys = ("observation_layout", "observation_contract")
    if not any(
        isinstance(metadata.get(key), str) and bool(metadata[key])
        for key in observation_schema_keys
    ):
        raise ValueError(
            "actor rehearsal metadata requires a non-empty observation_layout "
            "or observation_contract"
        )
    if "dataset_phase" in metadata:
        if not isinstance(metadata["dataset_phase"], str) or not metadata["dataset_phase"]:
            raise ValueError("actor rehearsal metadata dataset_phase must be non-empty when present")
    elif allowed_contracts is None:
        # Historical close-option teachers predate this descriptive field and
        # are admitted only through their explicit allowlist contract below.
        raise ValueError("actor rehearsal metadata requires non-empty 'dataset_phase'")
    for key, expected in (
        ("observation_dim", observation_dim),
        ("action_dim", action_dim),
    ):
        if key in metadata and metadata[key] != expected:
            raise ValueError(
                f"actor rehearsal metadata {key}={metadata[key]!r}, expected {expected}"
            )
    if expected_metadata is not None and allowed_contracts is not None:
        raise ValueError("expected_metadata and allowed_contracts are mutually exclusive")
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"actor rehearsal metadata {key}={metadata.get(key)!r}, expected {expected!r}"
                )

    matched_contract: ActorRehearsalSourceContract | None = None
    if allowed_contracts is not None:
        if not allowed_contracts:
            raise ValueError("actor rehearsal allowed_contracts must not be empty")
        matches = [
            contract
            for contract in allowed_contracts
            if all(
                _metadata_value_matches(
                    metadata.get(key),
                    expected,
                    strict_type=contract.strict_metadata_types,
                )
                for key, expected in contract.required_metadata.items()
            )
        ]
        if len(matches) != 1:
            names = [contract.name for contract in allowed_contracts]
            if not matches:
                raise ValueError(
                    "actor rehearsal metadata does not match an allowed source contract; "
                    f"collector={metadata.get('collector')!r}, allowed={names}"
                )
            raise ValueError(
                "actor rehearsal metadata ambiguously matches multiple source contracts: "
                f"{[contract.name for contract in matches]}"
            )
        matched_contract = matches[0]
        if phase is None:
            raise ValueError(
                f"actor rehearsal source contract {matched_contract.name!r} requires phase labels"
            )
        values = tuple(
            sorted(int(value) for value in torch.unique(phase).detach().cpu().tolist())
        )
        actual_phases = set(values)
        required_phases = set(matched_contract.phase_values)
        allowed_phases = set(matched_contract.allowed_phase_values)
        if not required_phases.issubset(actual_phases) or not actual_phases.issubset(
            allowed_phases
        ):
            raise ValueError(
                f"actor rehearsal source contract {matched_contract.name!r} requires "
                f"phases={sorted(required_phases)}, allows={sorted(allowed_phases)}, "
                f"got {list(values)}"
            )
        for key in matched_contract.required_sha256_metadata:
            value = metadata.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"actor rehearsal metadata {key!r} must be a 64-character "
                    "lowercase hexadecimal SHA256"
                )

    phase_names = metadata.get("phase_names")
    if phase_names is not None:
        if not isinstance(phase_names, (list, tuple)) or not phase_names or not all(
            isinstance(name, str) and name for name in phase_names
        ):
            raise ValueError("metadata phase_names must be a non-empty sequence of names")
        if phase is not None and (
            int(phase.min()) < 0 or int(phase.max()) >= len(phase_names)
        ):
            raise ValueError("actor rehearsal phase is outside metadata phase_names")
    for key in ("teacher_probability", "executed_teacher_fraction"):
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"actor rehearsal metadata {key} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"actor rehearsal metadata {key} must be in [0, 1]")
    return metadata, matched_contract


def audit_actor_rehearsal_action_semantics(
    action: torch.Tensor,
    phase: torch.Tensor | None,
    source_contract: ActorRehearsalSourceContract | None,
) -> None:
    """Verify action values that metadata alone cannot prove.

    Coupled option demonstrations use canonical policy labels: inactive hand
    dimensions are exactly zero during ALIGN, and inactive arm dimensions are
    exactly zero during CLOSE/HOLD.  Checking the stored tensors prevents a
    self-reported zero maximum from allowlisting contradictory supervision.
    """

    if source_contract is None or not source_contract.required_zero_action_slices:
        return
    if phase is None:
        raise ValueError(
            f"actor rehearsal source contract {source_contract.name!r} requires phase labels"
        )
    if action.ndim != 2 or phase.shape != (action.shape[0],):
        raise ValueError("action/phase shapes are incompatible for semantics audit")
    for phase_value, start, stop in source_contract.required_zero_action_slices:
        if stop > action.shape[1]:
            raise ValueError(
                f"source contract {source_contract.name!r} action slice "
                f"[{start}:{stop}] exceeds action_dim={action.shape[1]}"
            )
        selected = action[phase == phase_value, start:stop]
        if selected.numel() < 1:
            if phase_value in source_contract.optional_phases:
                continue
            raise ValueError(
                f"source contract {source_contract.name!r} has no phase={phase_value} rows"
            )
        maximum = float(selected.abs().max())
        if maximum != 0.0:
            raise ValueError(
                f"source contract {source_contract.name!r} requires exact zero action "
                f"slice [{start}:{stop}] for phase={phase_value}, got max abs={maximum}"
            )


def audit_actor_rehearsal_observation_semantics(
    observation: torch.Tensor,
    source_contract: ActorRehearsalSourceContract | None,
) -> None:
    """Verify exact actor-state values required by a demonstration source.

    The V6 self-imitation source stores only pre-latch CLOSE observations.  A
    phase label by itself cannot prove that fact, so the public latch bit is
    audited directly on every saved row.
    """

    if source_contract is None or not source_contract.required_observation_values:
        return
    if observation.ndim != 2:
        raise ValueError("observation must be two-dimensional for semantics audit")
    for index, expected in source_contract.required_observation_values:
        if index >= observation.shape[1]:
            raise ValueError(
                f"source contract {source_contract.name!r} observation index "
                f"{index} exceeds observation_dim={observation.shape[1]}"
            )
        value = observation[:, index]
        if bool((value != expected).any()):
            mismatch = int((value != expected).nonzero(as_tuple=False)[0])
            raise ValueError(
                f"source contract {source_contract.name!r} requires exact "
                f"observation[{index}]={expected}, got {float(value[mismatch])} "
                f"at row {mismatch}"
            )


def audit_actor_rehearsal_phase_observation_semantics(
    payload: Mapping[str, Any],
    observation: torch.Tensor,
    phase: torch.Tensor | None,
    source_contract: ActorRehearsalSourceContract | None,
) -> None:
    """Cross-check coupled phase labels against Markov state and episode order."""

    if source_contract is None or source_contract.phase_observation_state_indices is None:
        return
    if phase is None or phase.shape != (observation.shape[0],):
        raise ValueError("phase labels must match observations for phase-state audit")
    align_index, latch_index = source_contract.phase_observation_state_indices
    if max(align_index, latch_index) >= observation.shape[1]:
        raise ValueError(
            f"source contract {source_contract.name!r} phase-state index exceeds "
            f"observation_dim={observation.shape[1]}"
        )
    align_value = observation[:, align_index]
    latch_value = observation[:, latch_index]
    for name, value in (("align", align_value), ("power_latch", latch_value)):
        if not bool(((value == 0.0) | (value == 1.0)).all()):
            raise ValueError(
                f"source contract {source_contract.name!r} requires exact 0/1 "
                f"{name} observation bit"
            )
    expected_phase = torch.where(
        align_value.bool(),
        torch.zeros_like(phase),
        torch.where(
            latch_value.bool(),
            torch.full_like(phase, 2),
            torch.ones_like(phase),
        ),
    )
    if not torch.equal(phase, expected_phase):
        mismatch = int((phase != expected_phase).nonzero(as_tuple=False)[0])
        raise ValueError(
            f"source contract {source_contract.name!r} phase disagrees with obs131 "
            f"at row {mismatch}: phase={int(phase[mismatch])}, "
            f"expected={int(expected_phase[mismatch])}"
        )

    offsets = payload.get("episode_offsets")
    if not isinstance(offsets, torch.Tensor):
        raise TypeError("phase-state audit requires tensor episode_offsets")
    offset_values = offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    required_phases = set(source_contract.phase_values)
    for episode, (start, stop) in enumerate(
        zip(offset_values[:-1], offset_values[1:], strict=True)
    ):
        segment = phase[start:stop]
        segment_phases = set(int(value) for value in torch.unique(segment).tolist())
        if not required_phases.issubset(segment_phases):
            raise ValueError(
                f"source contract {source_contract.name!r} episode {episode} is missing "
                f"required phases {sorted(required_phases.difference(segment_phases))}"
            )
        if segment.numel() < 1 or int(segment[-1]) != 2:
            raise ValueError(
                f"source contract {source_contract.name!r} episode {episode} must end "
                "with pre-step hold_latched phase=2"
            )
        if segment.numel() > 1 and bool((segment[1:] < segment[:-1]).any()):
            raise ValueError(
                f"source contract {source_contract.name!r} episode {episode} phase "
                "must be monotonic ALIGN->CLOSE->HOLD"
            )
        hold_rows = int((segment == 2).sum())
        if hold_rows < 14:
            raise ValueError(
                f"source contract {source_contract.name!r} episode {episode} requires "
                f"at least 14 pre-step hold_latched rows, got {hold_rows}"
            )


def audit_actor_rehearsal_episode_semantics(
    payload: Mapping[str, Any],
    *,
    episodes: int,
    source_contract: ActorRehearsalSourceContract | None,
) -> None:
    """Validate per-episode physical evidence required by a source contract."""

    if source_contract is None or not source_contract.required_episode_tensors:
        return
    for requirement in source_contract.required_episode_tensors:
        value = payload.get(requirement.name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"source contract {source_contract.name!r} requires tensor "
                f"{requirement.name!r}"
            )
        if value.dtype != requirement.dtype or value.shape != (episodes,):
            raise ValueError(
                f"source contract {source_contract.name!r} requires "
                f"{requirement.name!r} as {requirement.dtype} shape ({episodes},), "
                f"got {value.dtype}{tuple(value.shape)}"
            )
        if requirement.dtype == torch.bool:
            assert requirement.expected_bool is not None
            if bool((value != requirement.expected_bool).any()):
                raise ValueError(
                    f"source contract {source_contract.name!r} requires every "
                    f"{requirement.name}={requirement.expected_bool}"
                )
            continue
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{requirement.name} contains NaN or infinity")
        if requirement.minimum is not None and bool(
            (value < requirement.minimum).any()
        ):
            raise ValueError(
                f"source contract {source_contract.name!r} requires "
                f"{requirement.name}>={requirement.minimum}"
            )
        if requirement.maximum is not None and bool(
            (value > requirement.maximum).any()
        ):
            raise ValueError(
                f"source contract {source_contract.name!r} requires "
                f"{requirement.name}<={requirement.maximum}"
            )

    for condition_name, condition_value, target_name, target_value in (
        source_contract.required_episode_conditional_values
    ):
        condition = payload[condition_name]
        target = payload[target_name]
        selected = condition == condition_value
        if not bool(selected.any()):
            raise ValueError(
                f"source contract {source_contract.name!r} requires at least one "
                f"episode with {condition_name}={condition_value}"
            )
        if bool((target[selected] != target_value).any()):
            raise ValueError(
                f"source contract {source_contract.name!r} requires "
                f"{target_name}={target_value} whenever "
                f"{condition_name}={condition_value}"
            )

    for earlier_name, later_name in source_contract.required_episode_orderings:
        earlier = payload[earlier_name]
        later = payload[later_name]
        if bool((earlier > later).any()):
            raise ValueError(
                f"source contract {source_contract.name!r} requires every "
                f"{earlier_name}<={later_name}"
            )


def load_actor_rehearsal(
    path: str | os.PathLike[str],
    *,
    device: torch.device | str,
    observation_dim: int,
    action_dim: int,
    expected_metadata: Mapping[str, Any] | None = None,
    allowed_contracts: Sequence[ActorRehearsalSourceContract] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    """Load and audit an allowlisted observation/action supervision dataset.

    ``obs`` is the native key used by the PickTool option-teacher collectors;
    ``observation`` is accepted for projected transition demonstrations.  The
    two aliases are deliberately mutually exclusive.  No missing transition
    fields are fabricated.
    """

    if observation_dim < 1 or action_dim < 1:
        raise ValueError("observation_dim and action_dim must be positive")
    source_path = Path(path)
    resolved_device = _canonical_device(device)
    fingerprint = sha256_file(source_path)
    payload = torch.load(source_path, map_location=resolved_device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("actor rehearsal dataset root must be a mapping")
    observation_keys = [key for key in ("obs", "observation") if key in payload]
    if len(observation_keys) != 1:
        raise KeyError("actor rehearsal dataset must contain exactly one of 'obs' or 'observation'")
    if "action" not in payload:
        raise KeyError("actor rehearsal dataset is missing 'action'")
    raw_batch = {
        "observation": payload[observation_keys[0]],
        "action": payload["action"],
    }
    rows = _require_actor_batch(
        raw_batch,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    batch = {
        key: value.detach().to(device=resolved_device, dtype=torch.float32, copy=True)
        for key, value in raw_batch.items()
    }
    phase = _require_phase(payload.get("phase"), rows=rows, device=resolved_device)
    metadata, source_contract = audit_actor_rehearsal_metadata(
        payload,
        observation_dim=observation_dim,
        action_dim=action_dim,
        phase=phase,
        expected_metadata=expected_metadata,
        allowed_contracts=allowed_contracts,
    )
    require_successful_episodes = (
        True
        if source_contract is None
        else source_contract.require_successful_episodes
    )
    episodes = _audit_episode_partition(
        payload,
        rows=rows,
        require_successful_episodes=require_successful_episodes,
    )
    audit_actor_rehearsal_action_semantics(
        raw_batch["action"],
        phase,
        source_contract,
    )
    audit_actor_rehearsal_observation_semantics(
        raw_batch["observation"],
        source_contract,
    )
    audit_actor_rehearsal_phase_observation_semantics(
        payload,
        batch["observation"],
        phase,
        source_contract,
    )
    audit_actor_rehearsal_episode_semantics(
        payload,
        episodes=episodes,
        source_contract=source_contract,
    )
    phase_counts: dict[str, int] = {}
    if phase is not None:
        values, counts = torch.unique(phase, sorted=True, return_counts=True)
        phase_counts = {
            str(int(value)): int(count)
            for value, count in zip(
                values.detach().cpu().tolist(),
                counts.detach().cpu().tolist(),
                strict=True,
            )
        }
    audit = {
        "path": str(source_path.resolve()),
        "sha256": fingerprint,
        "transitions": rows,
        "episodes": episodes,
        "phase_counts": phase_counts,
        "collector": str(metadata["collector"]),
        "dataset_phase": metadata.get("dataset_phase"),
        "source_contract": source_contract.name if source_contract is not None else None,
        "episode_semantics": (
            "successful_only"
            if require_successful_episodes
            else "correction_labels_without_outcome_claim"
        ),
    }
    if source_contract is not None:
        for key in source_contract.required_sha256_metadata:
            audit[key] = str(metadata[key])
    return batch, phase, audit


def _apportion_weighted_counts(batch_size: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("actor rehearsal stratum weights need a positive finite sum")
    quotas = tuple(batch_size * weight / total for weight in weights)
    counts = [math.floor(quota) for quota in quotas]
    remainder = batch_size - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


class ActorRehearsalReservoir:
    """Seal-once GPU tensor reservoir with an independent sampler RNG."""

    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device | str,
        seed: int,
        source_fingerprints: Sequence[str],
        default_batch_size: int | None = None,
        stratum_weights: Mapping[int, float] | None = None,
    ) -> None:
        if capacity < 1 or observation_dim < 1 or action_dim < 1:
            raise ValueError("actor rehearsal capacity and dimensions must be positive")
        if default_batch_size is not None and default_batch_size < 1:
            raise ValueError("actor rehearsal default_batch_size must be positive")
        self._capacity = int(capacity)
        self._observation_dim = int(observation_dim)
        self._action_dim = int(action_dim)
        self._device = _canonical_device(device)
        self._source_fingerprints = _require_source_fingerprints(source_fingerprints)
        self._default_batch_size = (
            None if default_batch_size is None else int(default_batch_size)
        )
        self._requested_stratum_weights = (
            None
            if stratum_weights is None
            else {int(label): float(weight) for label, weight in stratum_weights.items()}
        )
        if self._requested_stratum_weights is not None:
            if not self._requested_stratum_weights:
                raise ValueError("actor rehearsal stratum_weights must not be empty")
            if any(
                not math.isfinite(weight) or weight < 0.0
                for weight in self._requested_stratum_weights.values()
            ):
                raise ValueError("actor rehearsal stratum weights must be finite and non-negative")
            if not any(weight > 0.0 for weight in self._requested_stratum_weights.values()):
                raise ValueError("at least one actor rehearsal stratum weight must be positive")

        self._storage = {
            "observation": torch.empty(
                (capacity, observation_dim), dtype=torch.float32, device=self._device
            ),
            "action": torch.empty(
                (capacity, action_dim), dtype=torch.float32, device=self._device
            ),
        }
        self._size = 0
        self._sealed = False
        self._phase: torch.Tensor | None = None
        self._stratum_values: tuple[int, ...] = ()
        self._stratum_indices: tuple[torch.Tensor, ...] = ()
        self._stratum_weights: tuple[tuple[int, float], ...] | None = None
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(int(seed))
        self._initial_seed = int(seed)
        self._sample_count = 0

    def __len__(self) -> int:
        return self._size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def source_fingerprints(self) -> tuple[str, ...]:
        return self._source_fingerprints

    @property
    def stratum_values(self) -> tuple[int, ...]:
        return self._stratum_values

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def add(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        phase: torch.Tensor | None = None,
    ) -> None:
        if self._sealed:
            raise RuntimeError("actor rehearsal reservoir is sealed and immutable")
        rows = _require_actor_batch(
            batch,
            observation_dim=self._observation_dim,
            action_dim=self._action_dim,
        )
        stop = self._size + rows
        if stop > self._capacity:
            raise OverflowError(
                f"actor rehearsal capacity exceeded: size={self._size}, add={rows}, "
                f"capacity={self._capacity}"
            )
        labels = _require_phase(phase, rows=rows, device=self._device)
        if self._size > 0 and (self._phase is None) != (labels is None):
            raise ValueError("all actor rehearsal sources must consistently include or omit phase")
        if self._size == 0 and labels is not None:
            self._phase = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
        destination = slice(self._size, stop)
        for key in ACTOR_REHEARSAL_KEYS:
            self._storage[key][destination].copy_(
                batch[key].detach().to(
                    device=self._device,
                    dtype=self._storage[key].dtype,
                )
            )
        if labels is not None:
            assert self._phase is not None
            self._phase[destination].copy_(labels)
        self._size = stop

    def seal(self) -> None:
        if self._sealed:
            return
        if self._size < 1:
            raise RuntimeError("cannot seal an empty actor rehearsal reservoir")
        self._sealed = True
        self._rebuild_strata()
        self._stratum_weights = self._canonical_stratum_weights(
            self._requested_stratum_weights
        )

    def _rebuild_strata(self) -> None:
        if self._phase is None:
            self._stratum_values = ()
            self._stratum_indices = ()
            return
        unique = torch.unique(self._phase[: self._size], sorted=True)
        self._stratum_values = tuple(int(value) for value in unique.detach().cpu().tolist())
        self._stratum_indices = tuple(
            torch.nonzero(self._phase[: self._size] == value, as_tuple=False).flatten()
            for value in self._stratum_values
        )

    def _canonical_stratum_weights(
        self,
        weights: Mapping[int, float] | None,
    ) -> tuple[tuple[int, float], ...] | None:
        if weights is None:
            return None
        if self._phase is None:
            raise ValueError("actor rehearsal stratum weights require phase labels")
        unknown = sorted(set(weights).difference(self._stratum_values))
        if unknown:
            raise ValueError(f"actor rehearsal weights name absent phases: {unknown}")
        canonical = tuple(
            (label, float(weights.get(label, 0.0))) for label in self._stratum_values
        )
        if not any(weight > 0.0 for _, weight in canonical):
            raise ValueError("actor rehearsal weights select no available phase")
        return canonical

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        if self._phase is None:
            return torch.randint(
                0,
                self._size,
                (batch_size,),
                device=self._device,
                generator=self._generator,
            )
        weights = (
            tuple(1.0 for _ in self._stratum_values)
            if self._stratum_weights is None
            else tuple(weight for _, weight in self._stratum_weights)
        )
        counts = _apportion_weighted_counts(batch_size, weights)
        slots = torch.repeat_interleave(
            torch.arange(len(self._stratum_values), device=self._device),
            torch.tensor(counts, dtype=torch.int64, device=self._device),
        )
        if batch_size > 1:
            slots = slots[
                torch.randperm(batch_size, device=self._device, generator=self._generator)
            ]
        sampled = torch.empty(batch_size, dtype=torch.int64, device=self._device)
        for slot, source_indices in enumerate(self._stratum_indices):
            positions = torch.nonzero(slots == slot, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            choices = torch.randint(
                0,
                source_indices.numel(),
                (positions.numel(),),
                device=self._device,
                generator=self._generator,
            )
            sampled[positions] = source_indices[choices]
        return sampled

    def sample(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        if not self._sealed:
            raise RuntimeError("seal actor rehearsal reservoir before sampling")
        rows = self._default_batch_size if batch_size is None else int(batch_size)
        if rows is None:
            raise ValueError("sample requires batch_size when no default was configured")
        if rows < 1:
            raise ValueError("actor rehearsal sample batch_size must be positive")
        indices = self._sample_indices(rows)
        self._sample_count += 1
        return {key: value[indices] for key, value in self._storage.items()}

    def state_dict(self) -> dict[str, Any]:
        if not self._sealed:
            raise RuntimeError("seal actor rehearsal reservoir before checkpointing")
        return {
            "version": ACTOR_REHEARSAL_VERSION,
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "device_type": str(self._device),
            "size": self._size,
            "sealed": self._sealed,
            "default_batch_size": self._default_batch_size,
            "source_fingerprints": self._source_fingerprints,
            "stratum_weights": self._stratum_weights,
            "initial_seed": self._initial_seed,
            "sample_count": self._sample_count,
            "generator_state": self._generator.get_state().clone(),
            "storage": {
                key: value[: self._size].clone() for key, value in self._storage.items()
            },
            "phase": None if self._phase is None else self._phase[: self._size].clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not self._sealed:
            raise RuntimeError(
                "build and seal actor rehearsal sources before restoring sampler state"
            )
        expected = {
            "version": ACTOR_REHEARSAL_VERSION,
            "capacity": self._capacity,
            "observation_dim": self._observation_dim,
            "action_dim": self._action_dim,
            "device_type": str(self._device),
            "size": self._size,
            "sealed": True,
            "default_batch_size": self._default_batch_size,
            "source_fingerprints": self._source_fingerprints,
            "stratum_weights": self._stratum_weights,
        }
        for key, value in expected.items():
            checkpoint_value = state.get(key)
            if checkpoint_value != value:
                raise ValueError(
                    f"actor rehearsal checkpoint {key}={checkpoint_value!r}, expected {value!r}"
                )
        storage = state.get("storage")
        if not isinstance(storage, Mapping):
            raise TypeError("actor rehearsal checkpoint storage must be a mapping")
        expected_shapes = {
            "observation": (self._size, self._observation_dim),
            "action": (self._size, self._action_dim),
        }
        for key, shape in expected_shapes.items():
            value = storage.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
                raise ValueError(
                    f"actor rehearsal checkpoint {key} shape={actual}, expected {shape}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"actor rehearsal checkpoint {key} is not finite floating data")
            self._storage[key][: self._size].copy_(
                value.to(device=self._device, dtype=self._storage[key].dtype)
            )
        checkpoint_phase = state.get("phase")
        if (checkpoint_phase is None) != (self._phase is None):
            raise ValueError("actor rehearsal checkpoint phase presence differs from sources")
        if checkpoint_phase is not None:
            labels = _require_phase(
                checkpoint_phase,
                rows=self._size,
                device=self._device,
            )
            assert labels is not None and self._phase is not None
            self._phase[: self._size].copy_(labels)
            self._rebuild_strata()
            if self._canonical_stratum_weights(self._requested_stratum_weights) != self._stratum_weights:
                raise ValueError("actor rehearsal checkpoint phase/configuration is inconsistent")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("actor rehearsal checkpoint generator_state must be a tensor")
        self._generator.set_state(generator_state.cpu())
        sample_count = state.get("sample_count")
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("actor rehearsal checkpoint sample_count must be non-negative")
        self._sample_count = sample_count
        initial_seed = state.get("initial_seed")
        if not isinstance(initial_seed, int):
            raise ValueError("actor rehearsal checkpoint initial_seed must be an integer")
        self._initial_seed = initial_seed

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, destination)

    def load(self, path: str | os.PathLike[str]) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError("actor rehearsal checkpoint root must be a mapping")
        self.load_state_dict(state)
