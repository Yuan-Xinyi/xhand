#!/usr/bin/env python3
"""Simulation-free tests for Candidate44's DAgger phase-0 projection."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_candidate44_phase0_correction import (  # noqa: E402
    ACTION_DIM,
    ACTION_LAYOUT,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT,
    PHASE_NAMES,
    _atomic_json_save_or_verify,
    _atomic_torch_save_or_verify,
    audit_source_payload,
    build_manifest,
    build_projection,
    eligible_episode_rows,
    rank_source_episodes,
    validate_projection,
    validate_projection_pair,
)


TEST_SHA256 = "a" * 64
TEST_SALT = "candidate44-test-salt"


def _expect_error(
    error_type: type[BaseException], function: Any, *args: Any, **kwargs: Any
) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _source_payload() -> dict[str, Any]:
    episodes = 6
    episode_steps = 5
    rows = episodes * episode_steps
    generator = torch.Generator().manual_seed(44)
    observation = torch.randn(rows, OBSERVATION_DIM, generator=generator)
    observation[:, 106] = 0.0
    action = torch.tanh(torch.randn(rows, ACTION_DIM, generator=generator))
    per_episode_phase = torch.tensor([0, 0, 0, 1, 3], dtype=torch.uint8)
    phase = per_episode_phase.repeat(episodes)
    # One invalid phase-0/latch combination is filtered row-wise, not admitted
    # or used to discard the remainder of its source episode.
    observation[2 * episode_steps + 1, 106] = 1.0
    return {
        "obs": observation,
        "action": action,
        "phase": phase,
        "episode_id": torch.arange(episodes).repeat_interleave(episode_steps),
        "step": torch.arange(episode_steps, dtype=torch.int32).repeat(episodes),
        "episode_offsets": torch.arange(
            0, rows + 1, episode_steps, dtype=torch.int64
        ),
        "meta": {
            "format_version": 1,
            "action_layout": ACTION_LAYOUT,
            "observation_layout": OBSERVATION_LAYOUT,
            "phase_names": PHASE_NAMES,
        },
    }


def test_projection_is_deterministic_disjoint_and_reconstructable() -> None:
    source = _source_payload()
    rows, episodes = audit_source_payload(
        source,
        source_sha256=TEST_SHA256,
        expected_rows=30,
        expected_episodes=6,
    )
    assert (rows, episodes) == (30, 6)
    eligible = eligible_episode_rows(source)
    assert len(eligible) == 6
    assert sum(int(value.numel()) for value in eligible.values()) == 17
    ranked = rank_source_episodes(
        list(eligible), source_sha256=TEST_SHA256, split_salt=TEST_SALT
    )
    assert ranked == rank_source_episodes(
        list(reversed(list(eligible))),
        source_sha256=TEST_SHA256,
        split_salt=TEST_SALT,
    )
    train_ids, validation_ids = ranked[:4], ranked[4:]
    train = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=train_ids,
        split="train",
        split_salt=TEST_SALT,
    )
    validation = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=validation_ids,
        split="validation",
        split_salt=TEST_SALT,
    )
    validate_projection_pair(
        train,
        validation,
        eligible_source_episode_ids=ranked,
    )
    assert "episode_success" not in train and "episode_success" not in validation
    assert bool((train["phase"] == 0).all())
    assert bool((train["obs"][:, 106] == 0.0).all())
    assert set(train["source_episode_id"].tolist()).isdisjoint(
        validation["source_episode_id"].tolist()
    )


def test_projection_rejects_outcome_claim_overlap_latch_and_lineage_changes() -> None:
    source = _source_payload()
    audit_source_payload(
        source,
        source_sha256=TEST_SHA256,
        expected_rows=30,
        expected_episodes=6,
    )
    ranked = rank_source_episodes(
        list(eligible_episode_rows(source)),
        source_sha256=TEST_SHA256,
        split_salt=TEST_SALT,
    )
    train = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=ranked[:4],
        split="train",
        split_salt=TEST_SALT,
    )
    validation = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=ranked[4:],
        split="validation",
        split_salt=TEST_SALT,
    )

    fake_success = dict(train)
    fake_success["episode_success"] = torch.ones(4, dtype=torch.bool)
    _expect_error(
        ValueError,
        validate_projection,
        fake_success,
        source_payload=source,
        source_sha256=TEST_SHA256,
        expected_split="train",
        expected_source_episode_ids=ranked[:4],
        split_salt=TEST_SALT,
    )

    extra_field = dict(train)
    extra_field["comment"] = "not part of the closed actor-label schema"
    _expect_error(
        ValueError,
        validate_projection,
        extra_field,
        source_payload=source,
        source_sha256=TEST_SHA256,
        expected_split="train",
        expected_source_episode_ids=ranked[:4],
        split_salt=TEST_SALT,
    )

    incomplete_selection = dict(train)
    incomplete_selection["source_row_index"] = train["source_row_index"].clone()
    incomplete_selection["source_row_index"][0] = train["source_row_index"][1]
    _expect_error(
        ValueError,
        validate_projection,
        incomplete_selection,
        source_payload=source,
        source_sha256=TEST_SHA256,
        expected_split="train",
        expected_source_episode_ids=ranked[:4],
        split_salt=TEST_SALT,
    )
    _expect_error(
        ValueError,
        validate_projection_pair,
        train,
        train,
        eligible_source_episode_ids=ranked,
    )

    latched = dict(validation)
    latched["obs"] = validation["obs"].clone()
    latched["obs"][0, 106] = 1.0
    _expect_error(
        ValueError,
        validate_projection,
        latched,
        source_payload=source,
        source_sha256=TEST_SHA256,
        expected_split="validation",
        expected_source_episode_ids=ranked[4:],
        split_salt=TEST_SALT,
    )

    wrong_lineage = dict(train)
    wrong_lineage["meta"] = dict(train["meta"])
    wrong_lineage["meta"]["source_dataset_sha256"] = "b" * 64
    _expect_error(
        ValueError,
        validate_projection,
        wrong_lineage,
        source_payload=source,
        source_sha256=TEST_SHA256,
        expected_split="train",
        expected_source_episode_ids=ranked[:4],
        split_salt=TEST_SALT,
    )

    out_of_range = _source_payload()
    out_of_range["action"][0, 0] = 1.01
    _expect_error(
        ValueError,
        audit_source_payload,
        out_of_range,
        source_sha256=TEST_SHA256,
        expected_rows=30,
        expected_episodes=6,
    )


def _test_projections() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _source_payload()
    ranked = rank_source_episodes(
        list(eligible_episode_rows(source)),
        source_sha256=TEST_SHA256,
        split_salt=TEST_SALT,
    )
    train = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=ranked[:4],
        split="train",
        split_salt=TEST_SALT,
    )
    validation = build_projection(
        source,
        source_sha256=TEST_SHA256,
        source_episode_ids=ranked[4:],
        split="validation",
        split_salt=TEST_SALT,
    )
    return source, train, validation


def test_partial_publication_is_recoverable_and_manifest_is_stable() -> None:
    source, train, validation = _test_projections()
    with tempfile.TemporaryDirectory(prefix="candidate44-projection-test-") as directory:
        root = Path(directory)
        source_path = root / "source-input.pt"
        canonical_source = root / "canonical-source.pt"
        torch.save(source, source_path)
        canonical_source.write_bytes(source_path.read_bytes())
        train_path = root / "train.pt"
        validation_path = root / "validation.pt"
        manifest_path = root / "manifest.json"

        # Simulate an interrupted first run after only train was published.
        assert _atomic_torch_save_or_verify(train, train_path) == "created"
        assert not validation_path.exists()

        # The next run validates train without overwriting it and fills the gap.
        train_hash_before = train_path.read_bytes()
        assert _atomic_torch_save_or_verify(train, train_path) == "verified_existing"
        assert train_path.read_bytes() == train_hash_before
        assert _atomic_torch_save_or_verify(validation, validation_path) == "created"
        assert _atomic_torch_save_or_verify(validation, validation_path) == "verified_existing"

        manifest = build_manifest(
            source=source_path,
            canonical_source=canonical_source,
            source_rows=int(source["obs"].shape[0]),
            source_episodes=int(source["episode_offsets"].numel() - 1),
            train_path=train_path,
            train=train,
            validation_path=validation_path,
            validation=validation,
        )
        assert "copy_status" not in manifest["source"]
        assert manifest["source"]["copy_contract"] == "sha256_verified_no_clobber_v1"
        assert _atomic_json_save_or_verify(manifest, manifest_path) == "created"
        manifest_bytes = manifest_path.read_bytes()
        assert _atomic_json_save_or_verify(manifest, manifest_path) == "verified_existing"
        assert manifest_path.read_bytes() == manifest_bytes


def test_existing_projection_and_sealed_bytes_reject_tampering() -> None:
    source, train, validation = _test_projections()
    with tempfile.TemporaryDirectory(prefix="candidate44-tamper-test-") as directory:
        root = Path(directory)

        # A semantically different existing projection is never overwritten.
        different_path = root / "different-train.pt"
        tampered_train = dict(train)
        tampered_train["action"] = train["action"].clone()
        tampered_train["action"][0, 0] += 0.01
        torch.save(tampered_train, different_path)
        different_bytes = different_path.read_bytes()
        _expect_error(
            ValueError,
            _atomic_torch_save_or_verify,
            train,
            different_path,
        )
        assert different_path.read_bytes() == different_bytes

        source_path = root / "source-input.pt"
        canonical_source = root / "canonical-source.pt"
        torch.save(source, source_path)
        canonical_source.write_bytes(source_path.read_bytes())
        train_path = root / "train.pt"
        validation_path = root / "validation.pt"
        manifest_path = root / "manifest.json"
        assert _atomic_torch_save_or_verify(train, train_path) == "created"
        assert _atomic_torch_save_or_verify(validation, validation_path) == "created"
        manifest = build_manifest(
            source=source_path,
            canonical_source=canonical_source,
            source_rows=int(source["obs"].shape[0]),
            source_episodes=int(source["episode_offsets"].numel() - 1),
            train_path=train_path,
            train=train,
            validation_path=validation_path,
            validation=validation,
        )
        assert _atomic_json_save_or_verify(manifest, manifest_path) == "created"

        # Appended bytes can leave torch semantics loadable, but they violate
        # the already-sealed file hash and therefore the exact manifest bytes.
        with train_path.open("ab") as stream:
            stream.write(b"tampered-after-seal")
        rebuilt_manifest = build_manifest(
            source=source_path,
            canonical_source=canonical_source,
            source_rows=int(source["obs"].shape[0]),
            source_episodes=int(source["episode_offsets"].numel() - 1),
            train_path=train_path,
            train=train,
            validation_path=validation_path,
            validation=validation,
        )
        sealed_manifest_bytes = manifest_path.read_bytes()
        _expect_error(
            FileExistsError,
            _atomic_json_save_or_verify,
            rebuilt_manifest,
            manifest_path,
        )
        assert manifest_path.read_bytes() == sealed_manifest_bytes

        altered_manifest_path = root / "altered-manifest.json"
        altered_manifest_path.write_text("{}\n", encoding="utf-8")
        _expect_error(
            FileExistsError,
            _atomic_json_save_or_verify,
            manifest,
            altered_manifest_path,
        )


def main() -> None:
    test_projection_is_deterministic_disjoint_and_reconstructable()
    print("[PASS] deterministic episode split and exact source reconstruction")
    test_projection_rejects_outcome_claim_overlap_latch_and_lineage_changes()
    print("[PASS] outcome, overlap, latch, lineage and range failures are closed")
    test_partial_publication_is_recoverable_and_manifest_is_stable()
    print("[PASS] partial publication recovery and stable exact manifest")
    test_existing_projection_and_sealed_bytes_reject_tampering()
    print("[PASS] existing semantic and sealed-byte tampering are rejected")


if __name__ == "__main__":
    main()
