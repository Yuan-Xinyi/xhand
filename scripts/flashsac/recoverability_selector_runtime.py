#!/usr/bin/env python3
"""Fail-closed runtime for the audited recoverability handoff selector.

The selector is intentionally not a general policy.  It is evaluated only at
an existing diagnostic handoff candidate and may veto the default FlashSAC
route in favour of continuing the frozen SEARCH actor.  Its inputs are exactly
the public 115-D observation and the two deterministic 21-D candidate actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

import train_recoverability_selector as trainer


MODEL_KIND = "pick_tool_recoverability_selector_v1"
MODEL_FORMAT_VERSION = 1
DATASET_KIND = "pick_tool_recoverability_pairs_v1"
EXPECTED_HEAD_ORDER = [
    "continue_search_strict_success",
    "route_flashsac_strict_success",
]
EXPECTED_DECISION_SEMANTICS = (
    "default route to FlashSAC; veto to continue SEARCH iff p_continue >= "
    "p_continue_min and p_continue - p_route >= continue_probability_margin"
)
EXPECTED_PAIRING_SEMANTICS = "independent_gpu_rollout_diagnostic_v1"
TRAINER_SOURCE_KEY = "scripts/flashsac/train_recoverability_selector.py"
BLIND_MANIFEST_KIND = "pick_tool_recoverability_blind_runtime_manifest_v1"
DEFAULT_BLIND_MANIFEST_PATH = Path(__file__).with_name(
    "recoverability_selector_blind_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"selector report contains invalid JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"selector report contains duplicate key {key!r}")
        output[key] = value
    return output


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(
            stream,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(payload, dict):
        raise TypeError("recoverability selector report must be a JSON object")
    return payload


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _require_float(value: object, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return result


def _validate_collection_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("selector model lacks collection_canonical_provenance")
    provenance = dict(value)
    if (
        provenance.get("task_mode") != "full_task"
        or provenance.get("observation_contract") != "pick_tool_markov115_v1"
        or provenance.get("observation_dim") != trainer.OBSERVATION_DIM
        or provenance.get("action_dim") != trainer.ACTION_DIM
        or provenance.get("deterministic_policy_actions") is not True
        or provenance.get("episode_length_s") != 20.0
        or provenance.get("max_episode_steps") != 1000
    ):
        raise ValueError("selector collection provenance is incompatible with blind evaluation")
    for name in ("requested_episodes", "num_envs"):
        value = provenance.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"selector collection provenance has an invalid {name}")
    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("selector collection source hashes are missing")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise ValueError("selector collection source hashes are malformed")
    for name in (
        "approach_checkpoint_sha256",
        "flashsac_actor_sha256",
        "flashsac_task_contract_sha256",
    ):
        if not _is_sha256(provenance.get(name)):
            raise ValueError(f"selector collection provenance has an invalid {name}")
    for name in ("flashsac_fork_commit", "flashsac_upstream_commit"):
        value = provenance.get(name)
        if not isinstance(value, str) or len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"selector collection provenance has an invalid {name}")
    if not isinstance(provenance.get("use_compile"), bool):
        raise ValueError("selector collection provenance lacks use_compile")
    if provenance.get("selector_eligible_fields") != trainer.EXPECTED_INPUT_FIELDS:
        raise ValueError("selector collection provenance changes eligible input fields")
    supervisor = provenance.get("supervisor")
    if not isinstance(supervisor, Mapping):
        raise TypeError("selector collection provenance lacks its candidate supervisor")
    # This catches tensors or other non-JSON provenance values before they can
    # silently disappear from the companion metrics.
    json.dumps(provenance, sort_keys=True, allow_nan=False)
    return provenance


def _require_model_tensor(
    payload: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"selector {name} shape must be {shape}, got {actual}")
    if value.dtype != torch.float64:
        raise TypeError(f"selector {name} must be torch.float64, got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"serialized selector {name} must be a CPU tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"selector {name} contains NaN or infinity")
    return value.contiguous()


@dataclass(frozen=True)
class RecoverabilityDecision:
    """One batched selector result; ``continue_search`` is the only veto."""

    probabilities: torch.Tensor
    continue_search: torch.Tensor
    route_flashsac: torch.Tensor


@dataclass(frozen=True)
class BlindRuntimeManifest:
    path: Path
    sha256: str
    evaluator_path: Path
    evaluator_sha256: str
    runtime_path: Path
    runtime_sha256: str

    def verify_unchanged(self) -> None:
        expected = (
            (self.path, self.sha256, "blind manifest"),
            (self.evaluator_path, self.evaluator_sha256, "blind evaluator"),
            (self.runtime_path, self.runtime_sha256, "blind selector runtime"),
        )
        for path, digest, label in expected:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"{label} changed during strict blind evaluation")

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "kind": BLIND_MANIFEST_KIND,
            "format_version": 1,
            "path": str(self.path),
            "sha256": self.sha256,
            "evaluator_source_sha256": self.evaluator_sha256,
            "runtime_source_sha256": self.runtime_sha256,
            "semantics": (
                "independent manifest pins the reviewed evaluator and runtime bytes; "
                "updating either source requires an explicit manifest revision"
            ),
        }


def load_blind_runtime_manifest(
    manifest_path: Path,
    *,
    evaluator_path: Path,
    runtime_path: Path | None = None,
) -> BlindRuntimeManifest:
    """Validate the independent source pin used only for strict blind runs."""

    path = _require_regular_file(Path(manifest_path), "blind runtime manifest")
    evaluator = _require_regular_file(Path(evaluator_path), "blind evaluator source")
    runtime = _require_regular_file(
        Path(runtime_path) if runtime_path is not None else Path(__file__),
        "blind selector runtime source",
    )
    manifest_sha256 = sha256_file(path)
    payload = _load_json(path)
    if sha256_file(path) != manifest_sha256:
        raise RuntimeError("blind runtime manifest changed while it was being loaded")
    if payload.get("kind") != BLIND_MANIFEST_KIND or payload.get("format_version") != 1:
        raise ValueError("unsupported blind runtime manifest kind or version")
    source_hashes = payload.get("source_sha256")
    expected_keys = {
        "scripts/flashsac/evaluate.py",
        "scripts/flashsac/recoverability_selector_runtime.py",
    }
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != expected_keys:
        raise ValueError("blind runtime manifest source set is incompatible")
    evaluator_sha256 = source_hashes["scripts/flashsac/evaluate.py"]
    runtime_sha256 = source_hashes[
        "scripts/flashsac/recoverability_selector_runtime.py"
    ]
    if not _is_sha256(evaluator_sha256) or not _is_sha256(runtime_sha256):
        raise ValueError("blind runtime manifest contains an invalid source hash")
    if sha256_file(evaluator) != evaluator_sha256:
        raise ValueError("reviewed blind evaluator SHA-256 does not match current source")
    if sha256_file(runtime) != runtime_sha256:
        raise ValueError("reviewed blind selector runtime SHA-256 does not match current source")
    result = BlindRuntimeManifest(
        path=path,
        sha256=manifest_sha256,
        evaluator_path=evaluator,
        evaluator_sha256=evaluator_sha256,
        runtime_path=runtime,
        runtime_sha256=runtime_sha256,
    )
    result.verify_unchanged()
    return result


@dataclass(frozen=True)
class RecoverabilitySelectorRuntime:
    model_path: Path
    report_path: Path
    model_sha256: str
    report_sha256: str
    trainer_source_path: Path
    trainer_source_sha256: str
    runtime_source_path: Path
    runtime_source_sha256: str
    feature_set: str
    feature_contract: Mapping[str, Any]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    head_weight: torch.Tensor
    head_bias: torch.Tensor
    p_continue_min: float
    continue_probability_margin: float
    metadata: Mapping[str, Any]

    def decide(
        self,
        observation: torch.Tensor,
        search_action: torch.Tensor,
        flashsac_action: torch.Tensor,
    ) -> RecoverabilityDecision:
        """Return the conservative veto using public tensors only."""

        if observation.ndim != 2:
            raise ValueError("selector observation must be a batched matrix")
        rows = int(observation.shape[0])
        expected_shapes = (
            (observation, (rows, trainer.OBSERVATION_DIM), "observation"),
            (search_action, (rows, trainer.ACTION_DIM), "SEARCH action"),
            (flashsac_action, (rows, trainer.ACTION_DIM), "FlashSAC action"),
        )
        for value, shape, name in expected_shapes:
            if tuple(value.shape) != shape:
                raise ValueError(f"selector {name} shape must be {shape}")
            if value.dtype != torch.float32:
                raise TypeError(f"selector {name} must be torch.float32")
            if value.device != observation.device:
                raise ValueError("selector inputs must share one device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"selector {name} contains NaN or infinity")
        if self.feature_mean.device != observation.device:
            raise ValueError("selector model and inputs must share one device")
        with torch.no_grad():
            features = trainer.build_features(
                observation, search_action, flashsac_action, self.feature_set
            ).to(dtype=torch.float64)
            normalized = (features - self.feature_mean) / self.feature_scale
            logits = normalized @ self.head_weight.transpose(0, 1) + self.head_bias
            epsilon = torch.finfo(torch.float64).eps
            probabilities = torch.sigmoid(logits).clamp(epsilon, 1.0 - epsilon)
        if probabilities.shape != (rows, 2) or not bool(
            torch.isfinite(probabilities).all()
        ):
            raise FloatingPointError("selector produced invalid probabilities")
        p_continue = probabilities[:, 0]
        p_route = probabilities[:, 1]
        continue_search = (p_continue >= self.p_continue_min) & (
            p_continue - p_route >= self.continue_probability_margin
        )
        return RecoverabilityDecision(
            probabilities=probabilities,
            continue_search=continue_search,
            route_flashsac=~continue_search,
        )

    def verify_unchanged(self) -> None:
        """Detect artifact or executable-source replacement during a rollout."""

        expected = (
            (self.model_path, self.model_sha256, "selector model"),
            (self.report_path, self.report_sha256, "selector report"),
            (self.trainer_source_path, self.trainer_source_sha256, "selector trainer"),
            (self.runtime_source_path, self.runtime_source_sha256, "selector runtime"),
        )
        for path, digest, label in expected:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"{label} changed during strict evaluation")

    def validate_live_evaluation_contract(
        self,
        *,
        approach_checkpoint_sha256: str,
        flashsac_actor_sha256: str,
        flashsac_task_contract_sha256: str,
        flashsac_fork_commit: str,
        flashsac_upstream_commit: str,
        use_compile: bool,
        supervisor: Mapping[str, Any],
        collection_source_sha256: Mapping[str, str],
        allowed_source_drift: frozenset[str] = frozenset(),
    ) -> None:
        """Reject a blind rollout outside the model's collection contract."""

        provenance = self.metadata["collection_canonical_provenance"]
        expected_scalars = {
            "approach_checkpoint_sha256": approach_checkpoint_sha256,
            "flashsac_actor_sha256": flashsac_actor_sha256,
            "flashsac_task_contract_sha256": flashsac_task_contract_sha256,
            "flashsac_fork_commit": flashsac_fork_commit,
            "flashsac_upstream_commit": flashsac_upstream_commit,
            "use_compile": bool(use_compile),
        }
        for name, value in expected_scalars.items():
            if provenance.get(name) != value:
                raise ValueError(
                    f"blind evaluation {name} differs from selector collection"
                )
        if dict(provenance["supervisor"]) != dict(supervisor):
            raise ValueError(
                "blind evaluation candidate supervisor differs from selector collection"
            )
        expected_sources = dict(provenance["source_sha256"])
        if set(expected_sources) != set(collection_source_sha256):
            raise ValueError("blind evaluation collection source set changed")
        unknown_drift = allowed_source_drift - set(expected_sources)
        if unknown_drift:
            raise ValueError(f"unknown allowed source drift entries: {sorted(unknown_drift)}")
        mismatches = sorted(
            name
            for name, digest in expected_sources.items()
            if name not in allowed_source_drift
            and collection_source_sha256.get(name) != digest
        )
        if mismatches:
            raise ValueError(
                "blind evaluation physics/policy sources differ from collection: "
                f"{mismatches}"
            )

    def validate_blind_configuration(
        self,
        *,
        seed: int,
        episodes: int,
        num_envs: int,
        strict_blind: bool,
    ) -> dict[str, Any]:
        """Audit held-out seed and scale, rejecting violations in blind mode."""

        training_seeds = list(self.metadata["training_seeds"])
        development_seed = int(self.metadata["development_seed"])
        seen_seed = int(seed) in {*training_seeds, development_seed}
        provenance = self.metadata["collection_canonical_provenance"]
        collection_episodes = int(provenance["requested_episodes"])
        collection_num_envs = int(provenance["num_envs"])
        episodes_match = int(episodes) == collection_episodes
        num_envs_match = int(num_envs) == collection_num_envs
        scale_matches = episodes_match and num_envs_match
        audit = {
            "blind_requested": bool(strict_blind),
            "blind_claim_allowed": bool(strict_blind and not seen_seed and scale_matches),
            "evaluation_seed": int(seed),
            "training_seeds": training_seeds,
            "development_seed": development_seed,
            "seen_seed": seen_seed,
            "evaluation_episodes": int(episodes),
            "evaluation_num_envs": int(num_envs),
            "collection_requested_episodes": collection_episodes,
            "collection_num_envs": collection_num_envs,
            "episodes_match_collection": episodes_match,
            "num_envs_match_collection": num_envs_match,
            "scale_matches_collection": scale_matches,
        }
        if strict_blind and seen_seed:
            raise ValueError(
                f"strict blind evaluation rejects seen seed {seed}; training seeds are "
                f"{training_seeds} and development seed is {development_seed}"
            )
        if strict_blind and not scale_matches:
            raise ValueError(
                "strict blind evaluation must match collection scale exactly: "
                f"episodes={collection_episodes}, num_envs={collection_num_envs}"
            )
        return audit

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "kind": MODEL_KIND,
            "format_version": MODEL_FORMAT_VERSION,
            "model": str(self.model_path),
            "model_sha256": self.model_sha256,
            "report": str(self.report_path),
            "report_sha256": self.report_sha256,
            "trainer_source_sha256": self.trainer_source_sha256,
            "runtime_source_sha256": self.runtime_source_sha256,
            "dataset_kind": self.metadata["dataset_kind"],
            "dataset_sha256": self.metadata["dataset_sha256"],
            "data_semantic_sha256": self.metadata["data_semantic_sha256"],
            "chosen_feature_semantic_sha256": self.metadata[
                "chosen_feature_semantic_sha256"
            ],
            "feature_contract": dict(self.feature_contract),
            "head_order": list(EXPECTED_HEAD_ORDER),
            "decision_semantics": EXPECTED_DECISION_SEMANTICS,
            "p_continue_min": self.p_continue_min,
            "continue_probability_margin": self.continue_probability_margin,
            "default_decision": "route_flashsac",
            "veto_decision": "continue_search_for_remainder_of_episode",
            "training_seeds": list(self.metadata["training_seeds"]),
            "development_seed": int(self.metadata["development_seed"]),
            "collection_canonical_provenance": self.metadata[
                "collection_canonical_provenance"
            ],
        }


def load_recoverability_selector(
    model_path: Path,
    report_path: Path,
    *,
    device: torch.device | str,
) -> RecoverabilitySelectorRuntime:
    """Load an accepted selector and its companion report, or reject it."""

    model_path = _require_regular_file(Path(model_path), "selector model")
    report_path = _require_regular_file(Path(report_path), "selector report")
    if model_path == report_path:
        raise ValueError("selector model and report must be different files")
    model_sha256 = sha256_file(model_path)
    report_sha256 = sha256_file(report_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    report = _load_json(report_path)
    if sha256_file(model_path) != model_sha256 or sha256_file(report_path) != report_sha256:
        raise RuntimeError("selector model or report changed while it was being loaded")
    if not isinstance(payload, Mapping):
        raise TypeError("recoverability selector model must be a mapping")
    if payload.get("kind") != MODEL_KIND or payload.get("format_version") != 1:
        raise ValueError("unsupported recoverability selector kind or version")
    if (
        report.get("status") != "complete"
        or report.get("kind") != MODEL_KIND
        or report.get("format_version") != 1
    ):
        raise ValueError("selector companion report is incomplete or incompatible")
    if report.get("accepted_for_blind_simulation_evaluation") is not True:
        raise ValueError("selector report did not accept this model for blind simulation")
    acceptance_paths = (
        ("cross_validated_train_seed_report", "acceptance"),
        ("held_out_development_report", "acceptance"),
    )
    for section_name, acceptance_name in acceptance_paths:
        section = report.get(section_name)
        acceptance = (
            section.get(acceptance_name) if isinstance(section, Mapping) else None
        )
        if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
            raise ValueError(f"selector report {section_name} was not accepted")
    if report.get("model_sha256") != model_sha256:
        raise ValueError("selector model SHA-256 disagrees with its companion report")

    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise TypeError("selector model metadata must be a mapping")
    metadata = dict(metadata_value)
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    if report.get("model_metadata") != metadata:
        raise ValueError("selector report metadata disagrees with the model artifact")
    for name in ("torch_version", "max_iter", "training_configuration"):
        if report.get(name) != metadata.get(name):
            raise ValueError(f"selector report disagrees on {name}")
    if metadata.get("dataset_kind") != DATASET_KIND:
        raise ValueError("selector was trained from an unsupported dataset kind")
    for name in (
        "dataset_sha256",
        "data_semantic_sha256",
        "chosen_feature_semantic_sha256",
    ):
        if not _is_sha256(metadata.get(name)):
            raise ValueError(f"selector metadata has an invalid {name}")
        if name in report and report.get(name) != metadata[name]:
            raise ValueError(f"selector report disagrees on {name}")
    if metadata.get("pairing_semantics") != EXPECTED_PAIRING_SEMANTICS:
        raise ValueError("selector pairing semantics are unsupported")
    if metadata.get("causal_counterfactual_claim_allowed") is not False:
        raise ValueError("selector must forbid a causal counterfactual claim")
    training_seeds = metadata.get("training_seeds")
    if (
        not isinstance(training_seeds, list)
        or not training_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in training_seeds)
        or len(set(training_seeds)) != len(training_seeds)
    ):
        raise ValueError("selector metadata has invalid training seed groups")
    development_seed = metadata.get("development_seed")
    if (
        isinstance(development_seed, bool)
        or not isinstance(development_seed, int)
        or development_seed in training_seeds
    ):
        raise ValueError("selector metadata has an invalid development seed")
    if (
        report.get("training_seeds") != training_seeds
        or report.get("development_seed") != development_seed
    ):
        raise ValueError("selector report seed groups disagree with the model")
    _validate_collection_provenance(
        metadata.get("collection_canonical_provenance")
    )

    trainer_path = Path(trainer.__file__).resolve()
    trainer_sha256 = sha256_file(trainer_path)
    expected_source_hashes = {TRAINER_SOURCE_KEY: trainer_sha256}
    if metadata.get("source_sha256") != expected_source_hashes:
        raise ValueError("selector trainer source SHA-256 is stale or malformed")
    if report.get("trainer_source_sha256") != trainer_sha256:
        raise ValueError("selector report trainer SHA-256 is stale")
    if metadata.get("head_order") != EXPECTED_HEAD_ORDER:
        raise ValueError("selector head order is not [continue, route]")
    if metadata.get("decision_semantics") != EXPECTED_DECISION_SEMANTICS:
        raise ValueError("selector decision semantics do not default to route")
    if metadata.get("normalization") != (
        "training_seed_rows_only_population_std_constant_to_one"
    ) or metadata.get("dtype") != "float64":
        raise ValueError("selector normalization or dtype contract is unsupported")

    contract_value = metadata.get("feature_contract")
    if not isinstance(contract_value, Mapping):
        raise TypeError("selector feature contract must be a mapping")
    feature_set = contract_value.get("name")
    if not isinstance(feature_set, str) or feature_set not in trainer.FEATURE_DIMS:
        raise ValueError("selector feature set is unsupported")
    expected_contract = trainer.feature_contract(feature_set)
    if dict(contract_value) != expected_contract:
        raise ValueError("selector feature contract differs from current audited code")
    feature_dimension = trainer.FEATURE_DIMS[feature_set]
    selected = report.get("selected")
    if not isinstance(selected, Mapping) or (
        selected.get("feature_set") != feature_set
        or selected.get("feature_dimension") != feature_dimension
        or selected.get("l2") != metadata.get("l2")
    ):
        raise ValueError("selector report selected-model fields are inconsistent")

    p_continue_min = _require_float(
        metadata.get("p_continue_min"), "p_continue_min", lower=0.0, upper=1.0
    )
    margin = _require_float(
        metadata.get("continue_probability_margin"),
        "continue_probability_margin",
        lower=0.0,
        upper=1.0,
    )
    if (
        selected.get("p_continue_min") != p_continue_min
        or selected.get("continue_probability_margin") != margin
    ):
        raise ValueError("selector report thresholds disagree with the model")

    feature_mean = _require_model_tensor(
        payload, "feature_mean", (feature_dimension,)
    )
    feature_scale = _require_model_tensor(
        payload, "feature_scale", (feature_dimension,)
    )
    head_weight = _require_model_tensor(
        payload, "head_weight", (len(EXPECTED_HEAD_ORDER), feature_dimension)
    )
    head_bias = _require_model_tensor(
        payload, "head_bias", (len(EXPECTED_HEAD_ORDER),)
    )
    if not bool((feature_scale > 0.0).all()):
        raise ValueError("selector feature scale must be strictly positive")

    runtime_path = Path(__file__).resolve()
    runtime_sha256 = sha256_file(runtime_path)
    if sha256_file(trainer_path) != trainer_sha256:
        raise RuntimeError("selector trainer changed while the model was being validated")
    target_device = torch.device(device)
    runtime = RecoverabilitySelectorRuntime(
        model_path=model_path,
        report_path=report_path,
        model_sha256=model_sha256,
        report_sha256=report_sha256,
        trainer_source_path=trainer_path,
        trainer_source_sha256=trainer_sha256,
        runtime_source_path=runtime_path,
        runtime_source_sha256=runtime_sha256,
        feature_set=feature_set,
        feature_contract=expected_contract,
        feature_mean=feature_mean.to(device=target_device),
        feature_scale=feature_scale.to(device=target_device),
        head_weight=head_weight.to(device=target_device),
        head_bias=head_bias.to(device=target_device),
        p_continue_min=p_continue_min,
        continue_probability_margin=margin,
        metadata=metadata,
    )
    runtime.verify_unchanged()
    return runtime
