#!/usr/bin/env python3
"""Simulation-free tests for the coupled CEM teacher/residual action map."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from coupled_teacher_prior import (  # noqa: E402
    ACTION_DIM,
    ALIGN_ACTIVE_OBSERVATION_INDEX,
    ALIGN_PROGRESS_OBSERVATION_INDEX,
    ALIGN_STEPS,
    ARM_DIM,
    ARM_FEEDBACK_DENOMINATOR,
    ARM_OFFSET_OBSERVATION_SLICE,
    HAND_DIM,
    OBSERVATION_DIM,
    PHASE_ALIGN,
    PHASE_CLOSE_UNLATCHED,
    PHASE_HOLD_LATCHED,
    POWER_LATCH_OBSERVATION_INDEX,
    HandResidualScales,
    coupled_teacher_prior_from_payload,
    load_coupled_teacher_prior,
    load_coupled_teacher_prior_payload,
    sha256_file,
)


def _expect_error(
    error_type: type[BaseException],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        fn(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _curriculum_payload() -> dict[str, Any]:
    joint = torch.linspace(-0.5, 0.5, 19, dtype=torch.float32).unsqueeze(0)
    return {
        "boundaries": {
            "close_start": {
                "joint_pos": joint,
                "joint_vel": torch.zeros((1, 19), dtype=torch.float32),
                "dof_targets": joint.clone(),
                "object_local_pos": torch.tensor([[0.5, 0.0, 0.1]], dtype=torch.float32),
                "object_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "object_velocity": torch.zeros((1, 6), dtype=torch.float32),
                "last_action": torch.zeros((1, ACTION_DIM), dtype=torch.float32),
                "contact_steps": torch.zeros(1, dtype=torch.int64),
                "lost_contact_steps": torch.zeros(1, dtype=torch.int64),
                "is_grasped": torch.zeros(1, dtype=torch.bool),
            }
        },
        "meta": {
            "format_version": 1,
            "contract": "coupled_power_static_close_start_v1",
            "task_mode": "coupled_power_align_close_option_v1",
            "observation_contract": "pick_tool_coupled_power_align_close_state131_v1",
            "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
            "top_k": 1,
        },
    }


def _teacher_result() -> dict[str, Any]:
    arm_delta = [0.12, -0.06, 0.03, 0.0, -0.02, 0.01, 0.09]
    latent = [
        0.98,
        0.20,
        -0.30,
        0.40,
        -0.50,
        0.60,
        -0.70,
        0.80,
        -0.90,
        0.90,
        -0.80,
        0.70,
        -0.60,
        0.50,
    ]
    return {
        "replicates": 2,
        "strict_replicates": 2,
        "replicate_strict_pass": [True, True],
        "native_success_per_replicate": [True, True],
        "native_terminal_seen_per_replicate": [True, True],
        "native_failure_per_replicate": [False, False],
        "native_timeout_per_replicate": [False, False],
        "arm_target_saturated_per_replicate": [False, False],
        "native_terminal_align_active_per_replicate": [False, False],
        "strict_power_close_pass": True,
        "conservative_teacher_audit_pass": True,
        "native_option_success_all_replicates": True,
        "pass_authority": "conservative_coupled_teacher_audit_v1",
        "latent": latent,
        "arm_delta_target_rad": arm_delta,
        "arm_delta_abs_max_rad": 0.12,
    }


def _artifact_payload(curriculum_sha256: str) -> dict[str, Any]:
    teacher = _teacher_result()
    return {
        "format_version": 1,
        "contract": "strict_power_close_coupled_teacher_v1",
        "controller": "coupled_native_align_close_with_runtime_shields",
        "task_mode": "coupled_power_align_close_option_v1",
        "observation_contract": "pick_tool_coupled_power_align_close_state131_v1",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "action_layout": "arm_delta7|crossdex_token9|distal_residual5",
        "action_projection": "identity_v1",
        "pass_authority": "conservative_coupled_teacher_audit_v1",
        "native_success_authority": "pick_tool_terminal.power_close_option_success",
        "search_parameter_layout": "normalized_arm_target_offset7|hybrid14",
        "search_parameter_is_environment_action": False,
        "search_parameter_projection": "time_varying_arm_feedback_plus_hybrid14",
        "arm_feedback_formula": (
            "clip((smoothstep_target-current_target)/(action_scale*ema*0.2),-1,1)"
        ),
        "align_steps": ALIGN_STEPS,
        "arm_delta_limit_rad": 0.12,
        "episode_length_s": 3.0,
        "curriculum_sha256": curriculum_sha256,
        "coupled_phase_contract": {
            "align_steps": ALIGN_STEPS,
            "arm_action_multiplier": 0.2,
            "arm_target_limit_rad": 0.12,
            "align_hand_action": "masked_hold_target",
            "close_arm_action": "masked_frozen_target",
        },
        "thresholds": {
            "required_legal_other_contacts": 3,
            "power_grasp_quality": 0.35,
            "hold_quality": 0.5,
            "safe_force_n": 30.0,
            "confirm_steps": 15,
            "unlatched_lift_m": 0.015,
            "horizontal_drift_m": 0.03,
            "rotation_drift_rad": 0.35,
        },
        "result": teacher,
        "teacher_result": teacher,
        "search_seed_result": teacher,
        "results": {"coupled_align_close21": teacher},
    }


def _write_sources(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    curriculum = directory / "curriculum.pt"
    torch.save(_curriculum_payload(), curriculum)
    artifact_payload = _artifact_payload(sha256_file(curriculum))
    artifact = directory / "teacher.json"
    artifact.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    return artifact, curriculum, artifact_payload


def _three_phase_observation() -> torch.Tensor:
    observation = torch.zeros((3, OBSERVATION_DIM), dtype=torch.float32)
    observation[0, ALIGN_ACTIVE_OBSERVATION_INDEX] = 1.0
    observation[0, ALIGN_PROGRESS_OBSERVATION_INDEX] = 0.0
    observation[1, ALIGN_PROGRESS_OBSERVATION_INDEX] = 1.0
    observation[2, POWER_LATCH_OBSERVATION_INDEX] = 1.0
    # An early latch is legal: ALIGN stops before its timed progress reaches one.
    observation[2, ALIGN_PROGRESS_OBSERVATION_INDEX] = 0.5
    return observation


def test_three_phase_teacher_and_bounded_residual() -> None:
    with tempfile.TemporaryDirectory(prefix="coupled_teacher_prior_") as temporary:
        artifact, curriculum, _ = _write_sources(Path(temporary))
        scales = HandResidualScales(
            close_token=0.04,
            close_distal=0.06,
            hold_token=0.015,
            hold_distal=0.025,
        )
        prior = load_coupled_teacher_prior(
            artifact, curriculum, residual_scales=scales
        )
        observation = _three_phase_observation()
        phase = prior.phase(observation)
        assert phase.tolist() == [
            PHASE_ALIGN,
            PHASE_CLOSE_UNLATCHED,
            PHASE_HOLD_LATCHED,
        ]

        teacher = prior.teacher_action(observation)
        assert teacher.shape == (3, ACTION_DIM)
        assert torch.count_nonzero(teacher[0, ARM_DIM:]).item() == 0
        assert torch.count_nonzero(teacher[1:, :ARM_DIM]).item() == 0
        expected_hand = torch.tensor(prior.hand_latent, dtype=torch.float32)
        torch.testing.assert_close(teacher[1, ARM_DIM:], expected_hand)
        torch.testing.assert_close(teacher[2, ARM_DIM:], expected_hand)

        x = 1.0 / ALIGN_STEPS
        blend = x * x * (3.0 - 2.0 * x)
        expected_arm = torch.clamp(
            blend
            * torch.tensor(prior.arm_delta_target_rad, dtype=torch.float32)
            / ARM_FEEDBACK_DENOMINATOR,
            -1.0,
            1.0,
        )
        torch.testing.assert_close(teacher[0, :ARM_DIM], expected_arm)

        residual = torch.stack(
            (
                torch.ones(HAND_DIM),
                torch.ones(HAND_DIM),
                -torch.ones(HAND_DIM),
            )
        ).float()
        canonical = prior.canonicalize_residual(observation, residual)
        assert torch.count_nonzero(canonical[0]).item() == 0
        torch.testing.assert_close(canonical[1], residual[1])
        torch.testing.assert_close(canonical[2], residual[2])

        environment = prior.to_environment_action(observation, residual)
        composed_canonical, composed_environment = prior.compose_action(
            observation, residual
        )
        torch.testing.assert_close(
            composed_canonical, canonical, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            composed_environment, environment, rtol=0.0, atol=0.0
        )
        assert len(prior._constant_tensor_cache) == 1
        cached_constants = prior._constant_tensors(observation)
        assert cached_constants is prior._constant_tensors(observation)
        torch.testing.assert_close(environment[0], teacher[0])
        assert torch.count_nonzero(environment[1:, :ARM_DIM]).item() == 0
        expected_close = expected_hand.clone()
        expected_close[:9] += scales.close_token
        expected_close[9:] += scales.close_distal
        expected_hold = expected_hand.clone()
        expected_hold[:9] -= scales.hold_token
        expected_hold[9:] -= scales.hold_distal
        torch.testing.assert_close(
            environment[1, ARM_DIM:], expected_close.clamp(-1.0, 1.0)
        )
        torch.testing.assert_close(
            environment[2, ARM_DIM:], expected_hold.clamp(-1.0, 1.0)
        )
        torch.testing.assert_close(
            prior.to_environment_action(observation, torch.zeros_like(residual)),
            teacher,
        )

        contract = prior.contract_payload()
        assert contract["policy_action_dim"] == HAND_DIM
        assert contract["environment_action_dim"] == ACTION_DIM
        assert contract["teacher_artifact_sha256"] == sha256_file(artifact)
        assert contract["curriculum_dataset_sha256"] == sha256_file(curriculum)
        assert contract["residual_scales"] == scales.as_dict()
        embedded = coupled_teacher_prior_from_payload(contract)
        assert embedded.contract_payload() == contract
        embedded_path = Path(temporary) / "teacher_action_prior.json"
        embedded_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded = load_coupled_teacher_prior_payload(
            embedded_path,
            expected_sha256=sha256_file(embedded_path),
        )
        assert loaded.contract_payload() == contract
        _expect_error(
            ValueError,
            load_coupled_teacher_prior_payload,
            embedded_path,
            expected_sha256="0" * 64,
        )
        linked_path = Path(temporary) / "linked_teacher_action_prior.json"
        linked_path.symlink_to(embedded_path)
        _expect_error(
            FileNotFoundError,
            load_coupled_teacher_prior_payload,
            linked_path,
            expected_sha256=sha256_file(embedded_path),
        )


def test_public_arm_offset_feedback() -> None:
    with tempfile.TemporaryDirectory(prefix="coupled_teacher_feedback_") as temporary:
        artifact, curriculum, _ = _write_sources(Path(temporary))
        prior = load_coupled_teacher_prior(artifact, curriculum)
        observation = _three_phase_observation()[:1].clone()
        observation[:, ALIGN_PROGRESS_OBSERVATION_INDEX] = 0.5
        observation[:, ARM_OFFSET_OBSERVATION_SLICE] = 0.25
        teacher = prior.teacher_action(observation)
        x = 0.5 + 1.0 / ALIGN_STEPS
        blend = x * x * (3.0 - 2.0 * x)
        desired = blend * torch.tensor(prior.arm_delta_target_rad)
        current = torch.full((ARM_DIM,), 0.25 * 0.12)
        expected = ((desired - current) / ARM_FEEDBACK_DENOMINATOR).clamp(-1.0, 1.0)
        torch.testing.assert_close(teacher[0, :ARM_DIM], expected)


def test_fail_closed_lineage_and_tensor_contracts() -> None:
    with tempfile.TemporaryDirectory(prefix="coupled_teacher_reject_") as temporary:
        directory = Path(temporary)
        artifact, curriculum, payload = _write_sources(directory)

        missing_teacher = copy.deepcopy(payload)
        missing_teacher["teacher_result"] = None
        missing_path = directory / "missing_teacher.json"
        missing_path.write_text(json.dumps(missing_teacher), encoding="utf-8")
        _expect_error(ValueError, load_coupled_teacher_prior, missing_path, curriculum)

        rejected_replica = copy.deepcopy(payload)
        rejected_replica["teacher_result"]["replicate_strict_pass"][1] = False
        # Preserve the alias equality so rejection reaches the physical-replica audit.
        rejected_replica["result"] = rejected_replica["teacher_result"]
        rejected_replica["search_seed_result"] = rejected_replica["teacher_result"]
        rejected_replica["results"]["coupled_align_close21"] = rejected_replica[
            "teacher_result"
        ]
        rejected_path = directory / "rejected_replica.json"
        rejected_path.write_text(json.dumps(rejected_replica), encoding="utf-8")
        _expect_error(ValueError, load_coupled_teacher_prior, rejected_path, curriculum)

        wrong_curriculum = directory / "wrong_curriculum.pt"
        wrong_payload = _curriculum_payload()
        wrong_payload["boundaries"]["close_start"]["object_local_pos"][0, 0] += 0.01
        torch.save(wrong_payload, wrong_curriculum)
        _expect_error(ValueError, load_coupled_teacher_prior, artifact, wrong_curriculum)

        prior = load_coupled_teacher_prior(artifact, curriculum)
        observation = _three_phase_observation()
        contradictory = observation.clone()
        contradictory[0, POWER_LATCH_OBSERVATION_INDEX] = 1.0
        _expect_error(ValueError, prior.phase, contradictory)
        bad_progress = observation.clone()
        bad_progress[0, ALIGN_PROGRESS_OBSERVATION_INDEX] = 1.0
        _expect_error(ValueError, prior.phase, bad_progress)
        premature_close = observation.clone()
        premature_close[1, ALIGN_PROGRESS_OBSERVATION_INDEX] = 0.5
        _expect_error(ValueError, prior.phase, premature_close)
        # An early latch is a real state-machine transition and remains legal.
        early_latch = observation[2:3].clone()
        early_latch[:, ALIGN_PROGRESS_OBSERVATION_INDEX] = 0.25
        assert prior.phase(early_latch).item() == PHASE_HOLD_LATCHED
        nonfinite = observation.clone()
        nonfinite[0, 0] = float("nan")
        _expect_error(ValueError, prior.teacher_action, nonfinite)
        _expect_error(
            ValueError,
            prior.teacher_action,
            torch.zeros((3, OBSERVATION_DIM - 1), dtype=torch.float32),
        )
        out_of_range = torch.zeros((3, HAND_DIM), dtype=torch.float32)
        out_of_range[0, 0] = 1.01
        _expect_error(
            ValueError, prior.canonicalize_residual, observation, out_of_range
        )
        _expect_error(ValueError, HandResidualScales, 1.01, 0.1, 0.1, 0.1)


def main() -> None:
    test_three_phase_teacher_and_bounded_residual()
    print("[PASS] exact three-phase teacher and bounded hand residual")
    test_public_arm_offset_feedback()
    print("[PASS] public-observation arm feedback reconstruction")
    test_fail_closed_lineage_and_tensor_contracts()
    print("[PASS] fail-closed lineage, phase, shape, and finite-value guards")


if __name__ == "__main__":
    main()
