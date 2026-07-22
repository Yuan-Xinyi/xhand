#!/usr/bin/env python3
"""Simulation-free regression tests for actor-only rehearsal storage."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from actor_rehearsal import (  # noqa: E402
    ACTOR_REHEARSAL_KEYS,
    PICK_TOOL_ACTOR_DEMO_CONTRACTS,
    PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
    PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
    PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS,
    ActorRehearsalReservoir,
    load_actor_rehearsal,
    sha256_file,
)


OBSERVATION_DIM = 5
ACTION_DIM = 3
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _expect_error(
    error_type: type[BaseException], function: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _teacher_payload() -> dict[str, Any]:
    rows = 6
    observation = torch.arange(rows * OBSERVATION_DIM, dtype=torch.float32).reshape(
        rows, OBSERVATION_DIM
    ).mul_(0.01)
    action = torch.linspace(-0.8, 0.8, rows * ACTION_DIM).reshape(rows, ACTION_DIM)
    return {
        "obs": observation,
        "action": action,
        "phase": torch.tensor([0, 0, 1, 3, 3, 3], dtype=torch.uint8),
        "episode_id": torch.tensor([10, 10, 20, 20, 20, 20], dtype=torch.int64),
        "episode_offsets": torch.tensor([0, 2, 6], dtype=torch.int64),
        "episode_success": torch.ones(2, dtype=torch.bool),
        "meta": {
            "format_version": 1,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "observation_layout": "test-observation-layout",
            "action_layout": "test-action-layout",
            "collector": "successful_lift_teacher",
            "dataset_phase": "lift",
            "phase_names": ["approach", "close", "micro", "lift"],
            "teacher_probability": 1.0,
            "executed_teacher_fraction": 1.0,
        },
    }


def _write_payload(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = _teacher_payload() if payload is None else payload
    torch.save(selected, path)
    return selected


def test_loader_accepts_obs_action_only_and_audits_success() -> None:
    with tempfile.TemporaryDirectory(prefix="actor_rehearsal_source_") as directory:
        path = Path(directory) / "lift.pt"
        source = _write_payload(path)
        batch, phase, audit = load_actor_rehearsal(
            path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            expected_metadata={
                "collector": "successful_lift_teacher",
                "dataset_phase": "lift",
            },
        )
        assert set(batch) == set(ACTOR_REHEARSAL_KEYS)
        torch.testing.assert_close(batch["observation"], source["obs"])
        torch.testing.assert_close(batch["action"], source["action"])
        assert phase is not None and torch.equal(phase, source["phase"].long())
        assert audit["episodes"] == 2
        assert audit["transitions"] == 6
        assert audit["phase_counts"] == {"0": 2, "1": 1, "3": 3}
        assert audit["sha256"] == sha256_file(path)

        # Projected transition demos may use the canonical observation name;
        # reward/next_observation are neither required nor fabricated.
        canonical = _teacher_payload()
        canonical["observation"] = canonical.pop("obs")
        canonical_path = Path(directory) / "canonical.pt"
        _write_payload(canonical_path, canonical)
        projected, _, _ = load_actor_rehearsal(
            canonical_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
        )
        assert set(projected) == {"observation", "action"}


def _pick_tool_actor_payload(*, close: bool, production_close: bool = False) -> dict[str, Any]:
    payload = _teacher_payload()
    required_phase = 1 if close else 3
    payload["phase"] = torch.full((6,), required_phase, dtype=torch.uint8)
    metadata: dict[str, Any] = {
        "format_version": 1,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
        "observation_layout": "legacy_prefix87|distal_action5|grasp_transport23",
        "phase_names": ["approach", "close", "micro", "lift", "settle"],
        "close_arm_mode": "zero",
        "first_observation_last_action_max_error": 0.0,
    }
    if close:
        metadata.update(
            {
                "collector": "online_frozen_base_to_close_teacher",
            }
        )
        if production_close:
            metadata.update(
                {
                    "dataset_phase": "close",
                    "teacher_probability": 1.0,
                    "executed_teacher_fraction": 1.0,
                    "option_teacher_arm_action_abs_max": 0.0,
                }
            )
        else:
            metadata["close_teacher_arm_action_abs_max"] = 0.0
    else:
        metadata.update(
            {
                "collector": "online_base_close_to_scripted_lift_teacher",
                "dataset_phase": "lift",
                "teacher_probability": 1.0,
                "executed_teacher_fraction": 1.0,
            }
        )
    payload["meta"] = metadata
    return payload


def _coupled_power_actor_payload() -> dict[str, Any]:
    episodes = 2
    episode_steps = 18
    per_episode_phase = torch.tensor([0, 0, 1] + [2] * 15, dtype=torch.uint8)
    phase = per_episode_phase.repeat(episodes)
    rows = episodes * episode_steps
    observation = torch.randn(
        rows, 131, generator=torch.Generator().manual_seed(9)
    )
    observation[:, 129] = (phase == 0).float()
    observation[:, 106] = (phase == 2).float()
    action = torch.linspace(-0.7, 0.7, rows * 21).reshape(rows, 21)
    action[phase == 0, 7:] = 0.0
    action[(phase == 1) | (phase == 2), :7] = 0.0
    return {
        "obs": observation,
        "action": action,
        "phase": phase,
        "episode_id": torch.arange(episodes).repeat_interleave(episode_steps),
        "episode_offsets": torch.tensor([0, 18, 36], dtype=torch.int64),
        "episode_success": torch.ones(2, dtype=torch.bool),
        "episode_native_success": torch.ones(2, dtype=torch.bool),
        "episode_native_failure": torch.zeros(2, dtype=torch.bool),
        "episode_native_timeout": torch.zeros(2, dtype=torch.bool),
        "episode_conservative_teacher_pass": torch.ones(2, dtype=torch.bool),
        "episode_terminal_stable_steps": torch.full((2,), 15, dtype=torch.int64),
        "episode_terminal_power_is_grasped": torch.ones(2, dtype=torch.bool),
        "episode_terminal_thumb_contact": torch.ones(2, dtype=torch.bool),
        "episode_terminal_legal_other_contact_count": torch.tensor(
            [3, 4], dtype=torch.int64
        ),
        "episode_terminal_power_grasp_quality": torch.tensor(
            [0.35, 0.6], dtype=torch.float32
        ),
        "episode_terminal_hold_quality": torch.tensor(
            [0.5, 0.8], dtype=torch.float32
        ),
        "episode_terminal_max_force": torch.tensor(
            [20.0, 30.0], dtype=torch.float32
        ),
        "episode_trajectory_max_force": torch.tensor(
            [25.0, 30.0], dtype=torch.float32
        ),
        "episode_trajectory_max_xy_drift": torch.tensor(
            [0.02, 0.03], dtype=torch.float32
        ),
        "episode_trajectory_max_rotation_drift": torch.tensor(
            [0.2, 0.35], dtype=torch.float32
        ),
        "episode_trajectory_max_true_clearance": torch.tensor(
            [0.01, 0.015], dtype=torch.float32
        ),
        "episode_arm_target_saturated": torch.zeros(2, dtype=torch.bool),
        "episode_terminal_align_active": torch.zeros(2, dtype=torch.bool),
        "meta": {
            "format_version": 1,
            "task_mode": "coupled_power_align_close_option_v1",
            "observation_dim": 131,
            "observation_contract": "pick_tool_coupled_power_align_close_state131_v1",
            "observation_layout": (
                "legacy_prefix87|distal_action5|grasp_transport23|coupled_state16"
            ),
            "action_dim": 21,
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "action_projection": "identity_v1",
            "action_semantics": "canonical_policy_action_before_phase_shield_v1",
            "collector": "successful_coupled_power_align_close_teacher",
            "dataset_phase": "align_close_hold",
            "phase_names": ["align", "close_unlatched", "hold_latched"],
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
            "teacher_artifact_sha256": "a" * 64,
            "curriculum_dataset_sha256": "b" * 64,
        },
    }


def _v6_success_self_imitation_payload() -> dict[str, Any]:
    episodes = 4
    episode_steps = 3
    rows = episodes * episode_steps
    observation = torch.randn(
        rows, 115, generator=torch.Generator().manual_seed(17)
    )
    observation[:, 106] = 0.0
    action = torch.linspace(-0.8, 0.8, rows * 21).reshape(rows, 21)
    action[:, :7] = 0.0
    return {
        "obs": observation,
        "action": action,
        "phase": torch.ones(rows, dtype=torch.uint8),
        "episode_id": torch.arange(episodes).repeat_interleave(episode_steps),
        "episode_offsets": torch.arange(
            0, rows + 1, episode_steps, dtype=torch.int64
        ),
        "episode_success": torch.ones(episodes, dtype=torch.bool),
        "episode_native_success": torch.ones(episodes, dtype=torch.bool),
        "episode_dropped": torch.zeros(episodes, dtype=torch.bool),
        "episode_unsafe_force": torch.zeros(episodes, dtype=torch.bool),
        "episode_unlatched_clearance_ge_5cm": torch.zeros(
            episodes, dtype=torch.bool
        ),
        "episode_ever_grasped": torch.ones(episodes, dtype=torch.bool),
        "episode_triggered": torch.ones(episodes, dtype=torch.bool),
        "episode_ever_latched": torch.ones(episodes, dtype=torch.bool),
        "episode_latch_released_after_first": torch.zeros(
            episodes, dtype=torch.bool
        ),
        "episode_cohort": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        "episode_hand_noise_scale": torch.tensor(
            [0.0, 0.25, 0.0, 0.25], dtype=torch.float32
        ),
        "episode_trigger_step": torch.tensor(
            [400, 410, 420, 430], dtype=torch.int64
        ),
        "episode_trigger_score": torch.tensor(
            [0.30, 0.42, 0.75, 1.0], dtype=torch.float32
        ),
        "episode_first_latch_step": torch.tensor(
            [403, 414, 425, 431], dtype=torch.int64
        ),
        "episode_terminal_step": torch.tensor(
            [520, 530, 540, 550], dtype=torch.int64
        ),
        "episode_trajectory_max_force": torch.tensor(
            [22.0, 25.0, 29.0, 30.0], dtype=torch.float32
        ),
        "episode_max_true_clearance_m": torch.tensor(
            [0.20, 0.21, 0.22, 0.25], dtype=torch.float32
        ),
        "meta": {
            "format_version": 1,
            "task_mode": "full_task",
            "observation_dim": 115,
            "observation_contract": "pick_tool_markov115_v1",
            "action_dim": 21,
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "action_projection": "identity_v1",
            "observation_layout": (
                "legacy_prefix87|distal_action5|grasp_transport23"
            ),
            "phase_names": ["approach", "close", "micro", "lift", "settle"],
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
            "cohort_assignment": "balanced_sha256_slot_v1",
            "cohort_names": ["deterministic", "exploratory"],
            "kit_args": "--/app/extensions/fsWatcherEnabled=false",
            "handoff_min_score": 0.30,
            "handoff_hold_steps": 4,
            "exploratory_hand_noise_scale": 0.25,
            "unlatched_arm_noise_scale": 0.0,
            "latched_arm_noise_scale": 0.0,
            "latched_hand_noise_scale": 0.0,
            "search_checkpoint_sha256": "1" * 64,
            "v6_actor_sha256": "2" * 64,
            "v6_task_contract_sha256": "3" * 64,
            "v6_bridge_state_sha256": "8" * 64,
            "frozen_lift_actor_sha256": "4" * 64,
            "frozen_lift_semantic_sha256": "9" * 64,
            "frozen_lift_source_actor_sha256": "5" * 64,
            "source_manifest_sha256": "6" * 64,
            "runtime_asset_manifest_sha256": "7" * 64,
        },
    }


def test_pick_tool_actor_demo_allowlist_and_phase_contracts() -> None:
    with tempfile.TemporaryDirectory(prefix="actor_rehearsal_allowlist_") as directory:
        root = Path(directory)
        for close, production_close, expected_contract, expected_phase in (
            (False, False, "scripted_lift_teacher", 3),
            (True, False, "frozen_base_close_teacher_legacy", 1),
            (True, True, "frozen_base_close_teacher", 1),
        ):
            suffix = "production" if production_close else "legacy"
            path = root / f"{'close_' + suffix if close else 'lift'}.pt"
            _write_payload(
                path,
                _pick_tool_actor_payload(close=close, production_close=production_close),
            )
            _, phase, audit = load_actor_rehearsal(
                path,
                device="cpu",
                observation_dim=OBSERVATION_DIM,
                action_dim=ACTION_DIM,
                allowed_contracts=PICK_TOOL_ACTOR_DEMO_CONTRACTS,
            )
            assert phase is not None and bool((phase == expected_phase).all())
            assert audit["source_contract"] == expected_contract
            assert audit["dataset_phase"] == (
                "close" if production_close else (None if close else "lift")
            )

        assert {contract.required_phase for contract in PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS} == {1}
        assert {contract.required_phase for contract in PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS} == {3}

        lift_path = root / "lift.pt"
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            lift_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            root / "close_production.pt",
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            allowed_contracts=PICK_TOOL_LIFT_ACTOR_DEMO_CONTRACTS,
        )

        dagger_close = _pick_tool_actor_payload(close=True, production_close=True)
        dagger_close["meta"]["teacher_probability"] = 0.25
        dagger_close["meta"]["executed_teacher_fraction"] = 0.2
        dagger_path = root / "close_dagger.pt"
        _write_payload(dagger_path, dagger_close)
        _, dagger_phase, dagger_audit = load_actor_rehearsal(
            dagger_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )
        assert dagger_phase is not None and bool((dagger_phase == 1).all())
        assert dagger_audit["source_contract"] == "frozen_base_close_teacher"

        wrong_phase = _pick_tool_actor_payload(close=True)
        wrong_phase["phase"][0] = 3
        wrong_phase_path = root / "wrong_phase.pt"
        _write_payload(wrong_phase_path, wrong_phase)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            wrong_phase_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            allowed_contracts=PICK_TOOL_ACTOR_DEMO_CONTRACTS,
        )

        unknown = _pick_tool_actor_payload(close=True)
        unknown["meta"]["collector"] = "unreviewed_teacher"
        unknown_path = root / "unknown.pt"
        _write_payload(unknown_path, unknown)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            unknown_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            allowed_contracts=PICK_TOOL_ACTOR_DEMO_CONTRACTS,
        )


def test_v6_success_self_imitation_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="v6_self_imitation_") as directory:
        root = Path(directory)
        valid_path = root / "valid.pt"
        source = _write_payload(valid_path, _v6_success_self_imitation_payload())
        batch, phase, audit = load_actor_rehearsal(
            valid_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )
        assert phase is not None and bool((phase == 1).all())
        assert audit["source_contract"] == (
            "pick_tool_v6_success_self_imitation_v1"
        )
        assert audit["episodes"] == 4
        assert audit["phase_counts"] == {"1": 12}
        assert audit["search_checkpoint_sha256"] == "1" * 64
        assert audit["v6_actor_sha256"] == "2" * 64
        assert audit["v6_task_contract_sha256"] == "3" * 64
        assert audit["v6_bridge_state_sha256"] == "8" * 64
        assert audit["frozen_lift_actor_sha256"] == "4" * 64
        assert audit["frozen_lift_semantic_sha256"] == "9" * 64
        assert audit["frozen_lift_source_actor_sha256"] == "5" * 64
        assert audit["source_manifest_sha256"] == "6" * 64
        assert audit["runtime_asset_manifest_sha256"] == "7" * 64
        torch.testing.assert_close(batch["action"], source["action"])

        nonzero_arm = _v6_success_self_imitation_payload()
        nonzero_arm["action"][0, 6] = torch.finfo(torch.float32).eps
        nonzero_arm_path = root / "nonzero_arm.pt"
        _write_payload(nonzero_arm_path, nonzero_arm)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            nonzero_arm_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )

        latched_observation = _v6_success_self_imitation_payload()
        latched_observation["obs"][0, 106] = 1.0
        latched_observation_path = root / "latched_observation.pt"
        _write_payload(latched_observation_path, latched_observation)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            latched_observation_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )

        malformed_hash = _v6_success_self_imitation_payload()
        malformed_hash["meta"]["runtime_asset_manifest_sha256"] = "A" * 64
        malformed_hash_path = root / "malformed_hash.pt"
        _write_payload(malformed_hash_path, malformed_hash)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            malformed_hash_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )

        wrong_metadata_type = _v6_success_self_imitation_payload()
        wrong_metadata_type["meta"]["handoff_hold_steps"] = 4.0
        wrong_metadata_path = root / "wrong_metadata_type.pt"
        _write_payload(wrong_metadata_path, wrong_metadata_type)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            wrong_metadata_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )

        rejected_episode_evidence = {
            "episode_native_success": False,
            "episode_dropped": True,
            "episode_unsafe_force": True,
            "episode_unlatched_clearance_ge_5cm": True,
            "episode_ever_grasped": False,
            "episode_triggered": False,
            "episode_ever_latched": False,
            "episode_latch_released_after_first": True,
            "episode_cohort": 2,
            "episode_hand_noise_scale": 0.2501,
            "episode_trigger_step": -1,
            "episode_trigger_score": 0.2999,
            "episode_first_latch_step": -1,
            "episode_terminal_step": -1,
            "episode_trajectory_max_force": 30.001,
            "episode_max_true_clearance_m": 0.1999,
        }
        for index, (field, rejected_value) in enumerate(
            rejected_episode_evidence.items()
        ):
            rejected = _v6_success_self_imitation_payload()
            rejected[field][0] = rejected_value
            rejected_path = root / f"rejected_episode_{index}.pt"
            _write_payload(rejected_path, rejected)
            _expect_error(
                ValueError,
                load_actor_rehearsal,
                rejected_path,
                device="cpu",
                observation_dim=115,
                action_dim=21,
                allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
            )

        relationship_violations = (
            ("deterministic_noise", "episode_hand_noise_scale", 0, 0.25),
            ("exploratory_noise", "episode_hand_noise_scale", 1, 0.0),
            ("latch_before_trigger", "episode_first_latch_step", 0, 399),
            ("terminal_before_latch", "episode_terminal_step", 0, 402),
        )
        for name, field, episode, rejected_value in relationship_violations:
            rejected = _v6_success_self_imitation_payload()
            rejected[field][episode] = rejected_value
            rejected_path = root / f"relationship_{name}.pt"
            _write_payload(rejected_path, rejected)
            _expect_error(
                ValueError,
                load_actor_rehearsal,
                rejected_path,
                device="cpu",
                observation_dim=115,
                action_dim=21,
                allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
            )

        missing_evidence = _v6_success_self_imitation_payload()
        del missing_evidence["episode_trigger_score"]
        missing_evidence_path = root / "missing_evidence.pt"
        _write_payload(missing_evidence_path, missing_evidence)
        _expect_error(
            TypeError,
            load_actor_rehearsal,
            missing_evidence_path,
            device="cpu",
            observation_dim=115,
            action_dim=21,
            allowed_contracts=PICK_TOOL_CLOSE_ACTOR_DEMO_CONTRACTS,
        )


def test_coupled_power_multiphase_contract_and_canonical_actions() -> None:
    with tempfile.TemporaryDirectory(prefix="coupled_actor_rehearsal_") as directory:
        root = Path(directory)
        valid = root / "valid.pt"
        source = _write_payload(valid, _coupled_power_actor_payload())
        batch, phase, audit = load_actor_rehearsal(
            valid,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )
        assert batch["observation"].shape == (36, 131)
        assert phase is not None and set(phase.tolist()) == {0, 1, 2}
        assert audit["source_contract"] == (
            "successful_coupled_power_align_close_teacher_v1"
        )
        assert audit["teacher_artifact_sha256"] == "a" * 64
        assert audit["curriculum_dataset_sha256"] == "b" * 64

        early_latch = _coupled_power_actor_payload()
        early_latch["phase"][early_latch["phase"] == 1] = 2
        early_latch["obs"][:, 106] = (early_latch["phase"] == 2).float()
        early_latch_path = root / "early_latch.pt"
        _write_payload(early_latch_path, early_latch)
        _, early_phase, _ = load_actor_rehearsal(
            early_latch_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )
        assert early_phase is not None and set(early_phase.tolist()) == {0, 2}

        missing_phase = _coupled_power_actor_payload()
        missing_phase["phase"][missing_phase["phase"] == 2] = 1
        missing_phase_path = root / "missing_phase.pt"
        _write_payload(missing_phase_path, missing_phase)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            missing_phase_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        bad_hash = _coupled_power_actor_payload()
        bad_hash["meta"]["teacher_artifact_sha256"] = "not-a-sha"
        bad_hash_path = root / "bad_hash.pt"
        _write_payload(bad_hash_path, bad_hash)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            bad_hash_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        wrong_numeric_type = _coupled_power_actor_payload()
        wrong_numeric_type["meta"]["teacher_probability"] = 1
        wrong_numeric_type_path = root / "wrong_numeric_type.pt"
        _write_payload(wrong_numeric_type_path, wrong_numeric_type)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            wrong_numeric_type_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        contradictory = _coupled_power_actor_payload()
        contradictory["action"][contradictory["phase"] == 0, 7] = 0.1
        contradictory_path = root / "contradictory.pt"
        _write_payload(contradictory_path, contradictory)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            contradictory_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        contradictory_optional_close = _coupled_power_actor_payload()
        close_row = int((contradictory_optional_close["phase"] == 1).nonzero()[0])
        contradictory_optional_close["action"][close_row, 0] = 0.1
        contradictory_optional_path = root / "contradictory_optional_close.pt"
        _write_payload(contradictory_optional_path, contradictory_optional_close)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            contradictory_optional_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        wrong_phase_bit = _coupled_power_actor_payload()
        wrong_phase_bit["obs"][0, 129] = 0.0
        wrong_phase_bit_path = root / "wrong_phase_bit.pt"
        _write_payload(wrong_phase_bit_path, wrong_phase_bit)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            wrong_phase_bit_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        too_few_hold = _coupled_power_actor_payload()
        for episode_start in (0, 18):
            too_few_hold["phase"][episode_start + 3 : episode_start + 5] = 1
        too_few_hold["obs"][:, 106] = (too_few_hold["phase"] == 2).float()
        too_few_hold_path = root / "too_few_hold.pt"
        _write_payload(too_few_hold_path, too_few_hold)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            too_few_hold_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        regressed_phase = _coupled_power_actor_payload()
        regressed_phase["phase"][5] = 1
        regressed_phase["obs"][5, 106] = 0.0
        regressed_phase_path = root / "regressed_phase.pt"
        _write_payload(regressed_phase_path, regressed_phase)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            regressed_phase_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        rejected_episode_evidence = {
            "episode_native_success": False,
            "episode_native_failure": True,
            "episode_native_timeout": True,
            "episode_conservative_teacher_pass": False,
            "episode_terminal_stable_steps": 14,
            "episode_terminal_power_is_grasped": False,
            "episode_terminal_thumb_contact": False,
            "episode_terminal_legal_other_contact_count": 2,
            "episode_terminal_power_grasp_quality": 0.34,
            "episode_terminal_hold_quality": 0.49,
            "episode_terminal_max_force": 30.01,
            "episode_trajectory_max_force": 30.01,
            "episode_trajectory_max_xy_drift": 0.031,
            "episode_trajectory_max_rotation_drift": 0.351,
            "episode_trajectory_max_true_clearance": 0.016,
            "episode_arm_target_saturated": True,
            "episode_terminal_align_active": True,
        }
        for index, (field, rejected_value) in enumerate(
            rejected_episode_evidence.items()
        ):
            rejected = _coupled_power_actor_payload()
            rejected[field][0] = rejected_value
            rejected_path = root / f"rejected_episode_{index}.pt"
            _write_payload(rejected_path, rejected)
            _expect_error(
                ValueError,
                load_actor_rehearsal,
                rejected_path,
                device="cpu",
                observation_dim=131,
                action_dim=21,
                allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
            )

        post_terminal_pollution = _coupled_power_actor_payload()
        post_terminal_pollution["episode_terminal_stable_steps"][0] = 16
        post_terminal_path = root / "post_terminal_pollution.pt"
        _write_payload(post_terminal_path, post_terminal_pollution)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            post_terminal_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        missing_evidence = _coupled_power_actor_payload()
        del missing_evidence["episode_trajectory_max_force"]
        missing_evidence_path = root / "missing_episode_evidence.pt"
        _write_payload(missing_evidence_path, missing_evidence)
        _expect_error(
            TypeError,
            load_actor_rehearsal,
            missing_evidence_path,
            device="cpu",
            observation_dim=131,
            action_dim=21,
            allowed_contracts=PICK_TOOL_COUPLED_POWER_ACTOR_DEMO_CONTRACTS,
        )

        torch.testing.assert_close(batch["action"], source["action"])


def test_loader_rejects_ambiguous_failed_or_malformed_sources() -> None:
    with tempfile.TemporaryDirectory(prefix="actor_rehearsal_invalid_") as directory:
        root = Path(directory)

        failed = _teacher_payload()
        failed["episode_success"] = torch.tensor([True, False])
        failed_path = root / "failed.pt"
        _write_payload(failed_path, failed)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            failed_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
        )

        ambiguous = _teacher_payload()
        ambiguous["observation"] = ambiguous["obs"].clone()
        ambiguous_path = root / "ambiguous.pt"
        _write_payload(ambiguous_path, ambiguous)
        _expect_error(
            KeyError,
            load_actor_rehearsal,
            ambiguous_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
        )

        malformed = _teacher_payload()
        malformed["episode_offsets"] = torch.tensor([0, 3, 5])
        malformed_path = root / "offsets.pt"
        _write_payload(malformed_path, malformed)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            malformed_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
        )

        wrong_meta = _teacher_payload()
        wrong_meta_path = root / "metadata.pt"
        _write_payload(wrong_meta_path, wrong_meta)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            wrong_meta_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            expected_metadata={"collector": "different_collector"},
        )

        out_of_range = _teacher_payload()
        out_of_range["action"][0, 0] = 1.1
        range_path = root / "range.pt"
        _write_payload(range_path, out_of_range)
        _expect_error(
            ValueError,
            load_actor_rehearsal,
            range_path,
            device="cpu",
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
        )


def _reservoir(
    *,
    device: torch.device | str = "cpu",
    seed: int = 7,
    fingerprints: tuple[str, ...] = (FINGERPRINT_A, FINGERPRINT_B),
    default_batch_size: int = 6,
    weights: dict[int, float] | None = None,
) -> ActorRehearsalReservoir:
    resolved = torch.device(device)
    phase = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 3, 3], device=resolved)
    observation = torch.zeros((10, OBSERVATION_DIM), dtype=torch.float32, device=resolved)
    observation[:, 0] = phase.float()
    observation[:, 1] = torch.arange(10, dtype=torch.float32, device=resolved)
    action = torch.linspace(-0.5, 0.5, 10 * ACTION_DIM, device=resolved).reshape(
        10, ACTION_DIM
    )
    reservoir = ActorRehearsalReservoir(
        capacity=10,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        device=resolved,
        seed=seed,
        source_fingerprints=fingerprints,
        default_batch_size=default_batch_size,
        stratum_weights=weights,
    )
    # Exercise the production construction path: transition-demo projection
    # first, then the observation/action-only lift source, preserving SHA order.
    reservoir.add(
        {"observation": observation[:4], "action": action[:4]},
        phase=phase[:4],
    )
    reservoir.add(
        {"observation": observation[4:], "action": action[4:]},
        phase=phase[4:],
    )
    reservoir.seal()
    return reservoir


def _assert_batch_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert set(actual) == set(expected) == set(ACTOR_REHEARSAL_KEYS)
    for key in ACTOR_REHEARSAL_KEYS:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_gpu_style_stratification_and_private_generator() -> None:
    global_state = torch.get_rng_state().clone()
    first = _reservoir(seed=19)
    second = _reservoir(seed=19)
    batch = first.sample()
    _assert_batch_equal(batch, second.sample())
    # Equal phase mass despite the 6:2:2 source imbalance.
    sampled_phase = batch["observation"][:, 0].long()
    assert [int((sampled_phase == value).sum()) for value in (0, 1, 3)] == [2, 2, 2]
    # Sampling uses only the private generator, not Torch's process-global RNG.
    assert torch.equal(torch.get_rng_state(), global_state)

    weighted = _reservoir(seed=20, default_batch_size=8, weights={0: 0.0, 1: 1.0, 3: 3.0})
    weighted_phase = weighted.sample()["observation"][:, 0].long()
    assert [int((weighted_phase == value).sum()) for value in (0, 1, 3)] == [0, 2, 6]
    assert weighted.sample_count == 1


def test_checkpoint_restores_next_batch_and_rejects_lineage_or_config_changes() -> None:
    source = _reservoir(seed=31, weights={0: 1.0, 1: 1.0, 3: 2.0})
    _ = source.sample()  # Advance the private generator before saving.
    with tempfile.TemporaryDirectory(prefix="actor_rehearsal_checkpoint_") as directory:
        checkpoint = Path(directory) / "actor_rehearsal.pt"
        source.save(checkpoint)
        expected = source.sample()

        restored = _reservoir(seed=999, weights={0: 1.0, 1: 1.0, 3: 2.0})
        restored.load(checkpoint)
        actual = restored.sample()
        _assert_batch_equal(actual, expected)
        assert restored.sample_count == source.sample_count

        wrong_order = _reservoir(
            seed=999,
            fingerprints=(FINGERPRINT_B, FINGERPRINT_A),
            weights={0: 1.0, 1: 1.0, 3: 2.0},
        )
        _expect_error(ValueError, wrong_order.load, checkpoint)

        wrong_batch = _reservoir(
            seed=999,
            default_batch_size=5,
            weights={0: 1.0, 1: 1.0, 3: 2.0},
        )
        _expect_error(ValueError, wrong_batch.load, checkpoint)

        wrong_weights = _reservoir(
            seed=999,
            weights={0: 1.0, 1: 1.0, 3: 1.0},
        )
        _expect_error(ValueError, wrong_weights.load, checkpoint)


def test_seal_and_source_contracts_are_strict() -> None:
    _expect_error(
        ValueError,
        ActorRehearsalReservoir,
        capacity=2,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        device="cpu",
        seed=1,
        source_fingerprints=("not-a-sha",),
    )
    reservoir = _reservoir()
    batch = {
        "observation": torch.zeros((1, OBSERVATION_DIM)),
        "action": torch.zeros((1, ACTION_DIM)),
    }
    _expect_error(RuntimeError, reservoir.add, batch, phase=torch.zeros(1, dtype=torch.long))
    assert set(reservoir.sample()) == {"observation", "action"}


def test_cuda_storage_sampling_and_resume_stay_on_device() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    source = _reservoir(device=device, seed=41)
    first = source.sample()
    assert all(value.device == device for value in first.values())
    with tempfile.TemporaryDirectory(prefix="actor_rehearsal_cuda_") as directory:
        checkpoint = Path(directory) / "actor_rehearsal.pt"
        source.save(checkpoint)
        expected = source.sample()
        restored = _reservoir(device=device, seed=999)
        restored.load(checkpoint)
        actual = restored.sample()
    assert all(value.device == device for value in actual.values())
    _assert_batch_equal(actual, expected)


def main() -> None:
    test_loader_accepts_obs_action_only_and_audits_success()
    print("[PASS] obs/action loader and successful-episode audit")
    test_pick_tool_actor_demo_allowlist_and_phase_contracts()
    print("[PASS] PickTool actor-demo allowlist and phase contracts")
    test_v6_success_self_imitation_contract()
    print("[PASS] V6 successful self-imitation source contract")
    test_coupled_power_multiphase_contract_and_canonical_actions()
    print("[PASS] coupled-power multi-phase and canonical-action contract")
    test_loader_rejects_ambiguous_failed_or_malformed_sources()
    print("[PASS] malformed data and metadata are rejected")
    test_gpu_style_stratification_and_private_generator()
    print("[PASS] phase stratification and independent generator")
    test_checkpoint_restores_next_batch_and_rejects_lineage_or_config_changes()
    print("[PASS] checkpoint continuation and SHA/order/config guards")
    test_seal_and_source_contracts_are_strict()
    print("[PASS] immutable actor-only source contract")
    test_cuda_storage_sampling_and_resume_stay_on_device()
    print("[PASS] CUDA-resident sampling and exact resume")


if __name__ == "__main__":
    main()
