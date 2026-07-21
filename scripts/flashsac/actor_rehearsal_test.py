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
