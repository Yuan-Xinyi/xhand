#!/usr/bin/env python3
"""Fit and freeze the preregistered factual public-route gate ensemble.

The fit consumes every registered train-cohort trigger row and only its
randomized factual arm labels.  Pilot, development, and blind rows are never
accepted by this module.  There is no hyperparameter, threshold, or epoch
selection: all choices are sealed in ``public_route_gate_analysis_plan.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from public_route_gate_dataset import (
    DEFAULT_ANALYSIS_PLAN,
    DEFAULT_MANIFEST,
    _load_json_regular,
    _load_trial_manifest,
    _relative_output,
    _require_nonsymlink_chain,
    _require_regular_canonical,
    load_analysis_plan,
    load_registered_factual_cohort,
    semantic_dataset_sha256,
    sha256_file,
    validate_collection_acceptance,
    validate_evidence_ledger,
    validate_factual_dataset,
)


MODEL_KIND = "pick_tool_public_route_gate_ensemble_v2"
MODEL_FORMAT_VERSION = 1
REPORT_KIND = "pick_tool_public_route_gate_training_report_v1"
FEATURE_DIM = 165
HIDDEN_DIM = 128
HEADS = (
    "continue_success",
    "continue_drop",
    "continue_unsafe_force",
    "route_success",
    "route_drop",
    "route_unsafe_force",
)
TENSOR_FIELDS = {
    "feature_mean",
    "feature_scale",
    "layer0_weight",
    "layer0_bias",
    "layer1_weight",
    "layer1_bias",
    "output_weight",
    "output_bias",
}
ANALYSIS_SOURCE_FILES = (
    "scripts/flashsac/audit_public_route_gate_development.py",
    "scripts/flashsac/public_route_gate_analysis_plan.json",
    "scripts/flashsac/public_route_gate_dataset.py",
    "scripts/flashsac/public_route_gate_runtime.py",
    "scripts/flashsac/public_route_trial_contract.py",
    "scripts/flashsac/train_public_route_gate.py",
)


class PublicRouteGateNetwork(nn.Module):
    """The exact 165-128-128-6 member architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.layer1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, len(HEADS))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.layer0(feature))
        hidden = F.silu(self.layer1(hidden))
        return self.output(hidden)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def analysis_seal_provenance(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Bind the trainer and loader bytes to the annotated analysis seal."""

    registration = plan["preregistration"]
    tag = registration["seal_tag"]
    tag_ref = f"refs/tags/{tag}"
    if _git(repository_root, "cat-file", "-t", tag_ref) != "tag":
        raise RuntimeError("analysis seal must be an annotated Git tag")
    seal_commit = _git(repository_root, "rev-parse", f"{tag_ref}^{{commit}}")
    seal_line = _git(
        repository_root, "rev-list", "--parents", "-n", "1", seal_commit
    ).split()
    implementation = registration["implementation_commit"]
    if len(seal_line) != 2 or seal_line[1] != implementation:
        raise RuntimeError("analysis seal commit must be the sole child of implementation")
    changed = _git(
        repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        seal_commit,
    ).splitlines()
    if changed != ["scripts/flashsac/public_route_gate_analysis_plan.json"]:
        raise RuntimeError("analysis seal commit may change only the analysis plan")
    implementation_line = _git(
        repository_root, "rev-list", "--parents", "-n", "1", implementation
    ).split()
    if (
        len(implementation_line) != 2
        or implementation_line[1] != registration["implementation_base_commit"]
    ):
        raise RuntimeError("analysis implementation is not the registered sole child")
    current = _git(repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", seal_commit, current),
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("current HEAD does not descend from the analysis seal")
    if _git(repository_root, "rev-parse", "--abbrev-ref", "HEAD") != registration["branch"]:
        raise RuntimeError("gate training must run on the registered branch")
    source_sha256: dict[str, str] = {}
    for relative in ANALYSIS_SOURCE_FILES:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"analysis source must be regular: {relative}")
        current_blob = _git(repository_root, "hash-object", "--", relative)
        sealed_blob = _git(repository_root, "rev-parse", f"{seal_commit}:{relative}")
        if current_blob != sealed_blob:
            raise RuntimeError(f"analysis source differs from seal: {relative}")
        source_sha256[relative] = sha256_file(path)
    dirty = _git(repository_root, "status", "--porcelain", "--", *ANALYSIS_SOURCE_FILES)
    if dirty:
        raise RuntimeError("analysis source files are dirty")
    trial_tag = registration["trial_seal_tag"]
    if _git(repository_root, "cat-file", "-t", f"refs/tags/{trial_tag}") != "tag":
        raise RuntimeError("registered collection seal is absent or lightweight")
    trial_commit = _git(
        repository_root, "rev-parse", f"refs/tags/{trial_tag}^{{commit}}"
    )
    if trial_commit != registration["implementation_base_commit"]:
        raise RuntimeError("collection seal tag moved away from the analysis base")
    for relative in (
        "scripts/flashsac/public_route_trial_manifest.json",
        "scripts/flashsac/public_route_trial_contract.py",
    ):
        current_blob = _git(repository_root, "hash-object", "--", relative)
        trial_blob = _git(repository_root, "rev-parse", f"{trial_commit}:{relative}")
        if current_blob != trial_blob:
            raise RuntimeError(f"collection-sealed analysis input drifted: {relative}")
    _manifest, manifest_sha256 = _load_trial_manifest(DEFAULT_MANIFEST)
    if manifest_sha256 != registration["trial_manifest_sha256"]:
        raise RuntimeError("collection manifest differs from the analysis plan receipt")
    return {
        "branch": registration["branch"],
        "current_commit": current,
        "analysis_seal_tag": tag,
        "analysis_seal_commit": seal_commit,
        "implementation_commit": implementation,
        "trial_seal_tag": trial_tag,
        "trial_seal_commit": trial_commit,
        "source_sha256": source_sha256,
    }


def load_tagged_evidence_ledger(
    dataset: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    cohort: str,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Require the complete cohort inventory to have its own annotated tag."""

    if cohort not in {"train", "development"}:
        raise ValueError("tagged evidence ledgers exist only for train/development")
    output_key = "train_ledger" if cohort == "train" else "development_ledger"
    tag_key = "train_ledger_tag" if cohort == "train" else "development_ledger_tag"
    root = Path(repository_root).resolve()
    path = _relative_output(root, plan["outputs"][output_key], "evidence ledger")
    _require_regular_canonical(path, root, "evidence ledger")
    ledger, ledger_sha256 = _load_json_regular(path, "evidence ledger")
    ledger = validate_evidence_ledger(ledger, dataset=dataset, plan_or_manifest=plan)
    tag = plan["preregistration"][tag_key]
    tag_ref = f"refs/tags/{tag}"
    if _git(repository_root, "cat-file", "-t", tag_ref) != "tag":
        raise RuntimeError("evidence ledger tag must be annotated")
    commit = _git(repository_root, "rev-parse", f"{tag_ref}^{{commit}}")
    predecessor_tag = plan["preregistration"][
        "seal_tag" if cohort == "train" else "frozen_model_tag"
    ]
    predecessor_ref = f"refs/tags/{predecessor_tag}"
    if _git(repository_root, "cat-file", "-t", predecessor_ref) != "tag":
        raise RuntimeError("evidence ledger predecessor tag must be annotated")
    predecessor_commit = _git(
        repository_root, "rev-parse", f"{predecessor_ref}^{{commit}}"
    )
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", predecessor_commit, commit),
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ).returncode != 0:
        raise RuntimeError("evidence ledger does not descend from its sealed predecessor")
    changed = _git(
        repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    ).splitlines()
    relative = path.relative_to(root).as_posix()
    if changed != [relative]:
        raise RuntimeError("evidence ledger tag commit may change only its ledger")
    sealed_blob = _git(repository_root, "rev-parse", f"{commit}:{relative}")
    current_blob = _git(repository_root, "hash-object", "--", relative)
    if sealed_blob != current_blob:
        raise RuntimeError("evidence ledger bytes differ from the tagged receipt")
    current = _git(repository_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, current),
        cwd=repository_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("current HEAD does not descend from the ledger tag")
    dirty = _git(repository_root, "status", "--porcelain", "--", relative)
    if dirty:
        raise RuntimeError("evidence ledger is dirty")
    return ledger, {"tag": tag, "commit": commit, "sha256": ledger_sha256}


def fit_feature_normalizer(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(feature, torch.Tensor)
        or feature.ndim != 2
        or feature.shape[0] < 1
        or feature.shape[1] != FEATURE_DIM
        or feature.device.type != "cpu"
        or not feature.dtype.is_floating_point
        or not bool(torch.isfinite(feature).all())
    ):
        raise ValueError("normalizer requires finite CPU features with shape [N,165]")
    feature64 = feature.to(torch.float64)
    mean = feature64.mean(dim=0)
    scale = feature64.std(dim=0, correction=0).clamp_min(1.0e-6)
    return mean.contiguous(), scale.contiguous()


def normalize_feature(
    feature: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if mean.shape != (FEATURE_DIM,) or scale.shape != (FEATURE_DIM,):
        raise ValueError("gate normalizer has the wrong dimension")
    normalized = (feature.to(torch.float64) - mean) / scale
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("gate normalization produced NaN or infinity")
    return normalized.to(torch.float32).contiguous()


def factual_logits_and_labels(
    logits: torch.Tensor,
    *,
    treatment_route: torch.Tensor,
    success: torch.Tensor,
    dropped: torch.Tensor,
    unsafe_force: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only the randomized arm's three heads; never invent labels."""

    rows = logits.shape[0]
    if logits.shape != (rows, len(HEADS)):
        raise ValueError("gate logits must have shape [N,6]")
    for name, value in (
        ("treatment_route", treatment_route),
        ("success", success),
        ("dropped", dropped),
        ("unsafe_force", unsafe_force),
    ):
        if value.shape != (rows,) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a bool vector matching logits")
    offsets = treatment_route.to(torch.long) * 3
    indices = offsets[:, None] + torch.arange(3, device=logits.device)[None, :]
    factual_logits = logits.gather(1, indices)
    labels = torch.stack((success, dropped, unsafe_force), dim=-1).to(logits.dtype)
    return factual_logits, labels


def _state_tensors(network: PublicRouteGateNetwork) -> dict[str, torch.Tensor]:
    return {
        "layer0_weight": network.layer0.weight.detach().cpu().contiguous(),
        "layer0_bias": network.layer0.bias.detach().cpu().contiguous(),
        "layer1_weight": network.layer1.weight.detach().cpu().contiguous(),
        "layer1_bias": network.layer1.bias.detach().cpu().contiguous(),
        "output_weight": network.output.weight.detach().cpu().contiguous(),
        "output_bias": network.output.bias.detach().cpu().contiguous(),
    }


def _load_member(
    tensors: Mapping[str, torch.Tensor], member: int
) -> PublicRouteGateNetwork:
    network = PublicRouteGateNetwork().cpu()
    with torch.no_grad():
        network.layer0.weight.copy_(tensors["layer0_weight"][member])
        network.layer0.bias.copy_(tensors["layer0_bias"][member])
        network.layer1.weight.copy_(tensors["layer1_weight"][member])
        network.layer1.bias.copy_(tensors["layer1_bias"][member])
        network.output.weight.copy_(tensors["output_weight"][member])
        network.output.bias.copy_(tensors["output_bias"][member])
    return network.eval()


def _canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous().clone()
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("semantic hash rejects non-finite tensors")
        tensor[tensor == 0.0] = 0.0
    return tensor.numpy().tobytes()


def semantic_model_sha256(model: Mapping[str, Any]) -> str:
    validated = validate_gate_model(model)
    digest = hashlib.sha256()
    metadata = json.dumps(
        validated["metadata"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest.update(metadata)
    for name in sorted(TENSOR_FIELDS):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(_canonical_tensor_bytes(validated["tensors"][name]))
    return digest.hexdigest()


def _require_cpu_tensor(
    tensors: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = tensors.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"model tensor {name} must have shape {shape}")
    if value.dtype != dtype or value.device.type != "cpu":
        raise TypeError(f"model tensor {name} must be CPU {dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"model tensor {name} contains NaN or infinity")
    return value.contiguous()


def validate_gate_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "kind",
        "format_version",
        "metadata",
        "tensors",
    }:
        raise ValueError("gate model top-level schema is not exact")
    if (
        payload.get("kind") != MODEL_KIND
        or payload.get("format_version") != MODEL_FORMAT_VERSION
    ):
        raise ValueError("unsupported public-route gate model")
    metadata_value = payload.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise TypeError("gate model metadata must be a mapping")
    metadata = dict(metadata_value)
    required_metadata = {
        "analysis_plan_sha256",
        "analysis_seal_tag",
        "collection_manifest_sha256",
        "collection_seal_tag",
        "dataset_semantic_sha256",
        "excluded_cohorts",
        "feature_contract",
        "head_order",
        "input_ledger_sha256",
        "input_ledger_tag",
        "member_seeds",
        "normalization_contract",
        "optimizer_contract",
        "decision_contract",
        "run_receipts",
        "trained_cohort",
        "training_rows",
        "trainer_source_sha256",
    }
    if set(metadata) != required_metadata:
        raise ValueError("gate model metadata schema is not exact")
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    sha_fields = (
        "analysis_plan_sha256",
        "collection_manifest_sha256",
        "dataset_semantic_sha256",
        "input_ledger_sha256",
        "trainer_source_sha256",
    )
    if any(
        not isinstance(metadata[name], str)
        or len(metadata[name]) != 64
        or any(character not in "0123456789abcdef" for character in metadata[name])
        for name in sha_fields
    ):
        raise ValueError("gate model contains an invalid SHA256 receipt")
    if (
        metadata["trained_cohort"] != "train"
        or metadata["excluded_cohorts"] != ["pilot", "development", "blind"]
        or metadata["feature_contract"]
        != "pick_tool_public_route_gate_feature165_v1"
        or metadata["head_order"] != list(HEADS)
        or not isinstance(metadata["input_ledger_tag"], str)
        or not metadata["input_ledger_tag"]
        or metadata["member_seeds"] != [0, 1, 2, 3, 4]
        or not isinstance(metadata["analysis_seal_tag"], str)
        or not metadata["analysis_seal_tag"]
        or not isinstance(metadata["collection_seal_tag"], str)
        or not metadata["collection_seal_tag"]
        or not isinstance(metadata["run_receipts"], list)
        or not isinstance(metadata["normalization_contract"], dict)
        or not isinstance(metadata["optimizer_contract"], dict)
        or not isinstance(metadata["decision_contract"], dict)
        or type(metadata["training_rows"]) is not int
        or metadata["training_rows"] < 1
    ):
        raise ValueError("gate model weakens cohort, head, or member contracts")
    tensor_value = payload.get("tensors")
    if not isinstance(tensor_value, Mapping) or set(tensor_value) != TENSOR_FIELDS:
        raise ValueError("gate model tensor schema is not exact")
    tensors = {
        "feature_mean": _require_cpu_tensor(
            tensor_value, "feature_mean", shape=(FEATURE_DIM,), dtype=torch.float64
        ),
        "feature_scale": _require_cpu_tensor(
            tensor_value, "feature_scale", shape=(FEATURE_DIM,), dtype=torch.float64
        ),
        "layer0_weight": _require_cpu_tensor(
            tensor_value,
            "layer0_weight",
            shape=(5, HIDDEN_DIM, FEATURE_DIM),
            dtype=torch.float32,
        ),
        "layer0_bias": _require_cpu_tensor(
            tensor_value, "layer0_bias", shape=(5, HIDDEN_DIM), dtype=torch.float32
        ),
        "layer1_weight": _require_cpu_tensor(
            tensor_value,
            "layer1_weight",
            shape=(5, HIDDEN_DIM, HIDDEN_DIM),
            dtype=torch.float32,
        ),
        "layer1_bias": _require_cpu_tensor(
            tensor_value, "layer1_bias", shape=(5, HIDDEN_DIM), dtype=torch.float32
        ),
        "output_weight": _require_cpu_tensor(
            tensor_value,
            "output_weight",
            shape=(5, len(HEADS), HIDDEN_DIM),
            dtype=torch.float32,
        ),
        "output_bias": _require_cpu_tensor(
            tensor_value, "output_bias", shape=(5, len(HEADS)), dtype=torch.float32
        ),
    }
    if bool((tensors["feature_scale"] < 1.0e-6).any()):
        raise ValueError("gate model violates the preregistered std floor")
    return {
        "kind": MODEL_KIND,
        "format_version": MODEL_FORMAT_VERSION,
        "metadata": metadata,
        "tensors": tensors,
    }


@torch.inference_mode()
def ensemble_probabilities(
    model: Mapping[str, Any], feature: torch.Tensor
) -> torch.Tensor:
    validated = validate_gate_model(model)
    if (
        not isinstance(feature, torch.Tensor)
        or feature.ndim != 2
        or feature.shape[1] != FEATURE_DIM
        or feature.device.type != "cpu"
        or not feature.dtype.is_floating_point
        or not bool(torch.isfinite(feature).all())
    ):
        raise ValueError("gate inference feature must be finite CPU [N,165]")
    tensors = validated["tensors"]
    x = normalize_feature(feature, tensors["feature_mean"], tensors["feature_scale"])
    probabilities = [
        torch.sigmoid(_load_member(tensors, member)(x)) for member in range(5)
    ]
    result = torch.stack(probabilities, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("gate ensemble produced NaN or infinity")
    return result


def decision_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(probabilities, torch.Tensor)
        or probabilities.ndim != 3
        or probabilities.shape[0] != 5
        or probabilities.shape[2] != len(HEADS)
        or not probabilities.dtype.is_floating_point
    ):
        raise ValueError("ensemble probabilities must have shape [5,N,6]")
    finite = torch.isfinite(probabilities).all(dim=(0, 2))
    values = probabilities.to(torch.float64)
    mean = values.mean(dim=0)
    std = values.std(dim=0, correction=0)
    lower = (mean - std).clamp(0.0, 1.0)
    upper = (mean + std).clamp(0.0, 1.0)
    decision = (
        (lower[:, 0] - upper[:, 3] >= 0.05)
        & (upper[:, 1] <= lower[:, 4])
        & (upper[:, 2] <= lower[:, 5])
    )
    return decision & finite


def _training_tensors(dataset: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    tensors = dataset["tensors"]
    required = {
        "feature",
        "treatment_route",
        "success",
        "dropped",
        "unsafe_force",
    }
    if not required.issubset(tensors):
        raise ValueError("factual dataset lacks gate training tensors")
    return tensors


def train_gate_ensemble(
    dataset: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    analysis_plan_sha256: str,
    input_ledger_sha256: str,
    input_ledger_tag: str,
    trainer_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform the single fixed train-cohort fit described by the plan."""

    if (
        plan.get("status") != "preregistered"
        or plan.get("_sha256") != analysis_plan_sha256
        or input_ledger_tag != plan.get("preregistration", {}).get("train_ledger_tag")
    ):
        raise ValueError("gate fit is not bound to the sealed analysis plan")
    for name, value in (
        ("analysis_plan_sha256", analysis_plan_sha256),
        ("input_ledger_sha256", input_ledger_sha256),
        ("trainer_source_sha256", trainer_source_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA256 receipt")
    dataset = validate_factual_dataset(dataset, expected_cohort="train")
    if dataset["metadata"]["analysis_plan_sha256"] != analysis_plan_sha256:
        raise ValueError("train dataset was built under a different analysis plan")
    acceptance = validate_collection_acceptance(dataset, plan)
    if not acceptance["accepted"]:
        failed_checks = sorted(
            name for name, passed in acceptance["checks"].items() if not passed
        )
        raise RuntimeError(
            "train cohort is underpowered; fixed-route fallback remains mandatory: "
            + ", ".join(failed_checks)
        )
    tensors = _training_tensors(dataset)
    feature = tensors["feature"]
    rows = feature.shape[0]
    mean, scale = fit_feature_normalizer(feature)
    x = normalize_feature(feature, mean, scale)
    treatment = tensors["treatment_route"]
    success = tensors["success"]
    dropped = tensors["dropped"]
    unsafe = tensors["unsafe_force"]
    optimization = plan["optimization"]
    member_seeds = plan["model"]["member_seeds"]
    if member_seeds != [0, 1, 2, 3, 4]:
        raise ValueError("analysis plan changes the five fixed member seeds")
    states: list[dict[str, torch.Tensor]] = []
    diagnostics: list[dict[str, Any]] = []
    for member_seed in member_seeds:
        torch.manual_seed(int(member_seed))
        network = PublicRouteGateNetwork().cpu().train()
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(optimization["learning_rate"]),
            betas=tuple(float(v) for v in optimization["adamw_betas"]),
            eps=float(optimization["adamw_eps"]),
            weight_decay=float(optimization["weight_decay"]),
            amsgrad=bool(optimization["adamw_amsgrad"]),
            foreach=bool(optimization["adamw_foreach"]),
            fused=bool(optimization["adamw_fused"]),
            maximize=bool(optimization["adamw_maximize"]),
            capturable=bool(optimization["adamw_capturable"]),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(member_seed))
        optimizer_steps = 0
        maximum_gradient_norm = 0.0
        batch_size = int(optimization["batch_size"])
        for _epoch in range(int(optimization["epochs"])):
            permutation = torch.randperm(rows, generator=generator)
            for start in range(0, rows, batch_size):
                indices = permutation[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                factual_logits, labels = factual_logits_and_labels(
                    network(x[indices]),
                    treatment_route=treatment[indices],
                    success=success[indices],
                    dropped=dropped[indices],
                    unsafe_force=unsafe[indices],
                )
                loss = F.binary_cross_entropy_with_logits(
                    factual_logits, labels, reduction="mean"
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("gate training loss is non-finite")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    network.parameters(),
                    max_norm=float(optimization["gradient_clip_norm"]),
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("gate gradient norm is non-finite")
                maximum_gradient_norm = max(
                    maximum_gradient_norm, float(gradient_norm.detach())
                )
                optimizer.step()
                optimizer_steps += 1
        network.eval()
        with torch.inference_mode():
            factual_logits, labels = factual_logits_and_labels(
                network(x),
                treatment_route=treatment,
                success=success,
                dropped=dropped,
                unsafe_force=unsafe,
            )
            final_loss = F.binary_cross_entropy_with_logits(
                factual_logits, labels, reduction="mean"
            )
        states.append(_state_tensors(network))
        diagnostics.append(
            {
                "member_seed": int(member_seed),
                "optimizer_steps": optimizer_steps,
                "final_factual_bce": float(final_loss),
                "maximum_preclip_gradient_norm": maximum_gradient_norm,
            }
        )
    stacked: dict[str, torch.Tensor] = {
        "feature_mean": mean,
        "feature_scale": scale,
    }
    for name in (
        "layer0_weight",
        "layer0_bias",
        "layer1_weight",
        "layer1_bias",
        "output_weight",
        "output_bias",
    ):
        stacked[name] = torch.stack([state[name] for state in states], dim=0)
    metadata = {
        "analysis_plan_sha256": analysis_plan_sha256,
        "analysis_seal_tag": plan["preregistration"]["seal_tag"],
        "collection_manifest_sha256": plan["preregistration"][
            "trial_manifest_sha256"
        ],
        "collection_seal_tag": plan["preregistration"]["trial_seal_tag"],
        "dataset_semantic_sha256": semantic_dataset_sha256(dataset),
        "excluded_cohorts": ["pilot", "development", "blind"],
        "feature_contract": "pick_tool_public_route_gate_feature165_v1",
        "head_order": list(HEADS),
        "input_ledger_sha256": input_ledger_sha256,
        "input_ledger_tag": input_ledger_tag,
        "member_seeds": list(member_seeds),
        "normalization_contract": plan["normalization"],
        "optimizer_contract": optimization,
        "decision_contract": plan["inference"],
        "run_receipts": dataset["receipts"],
        "trained_cohort": "train",
        "training_rows": rows,
        "trainer_source_sha256": trainer_source_sha256,
    }
    model = validate_gate_model(
        {
            "kind": MODEL_KIND,
            "format_version": MODEL_FORMAT_VERSION,
            "metadata": metadata,
            "tensors": stacked,
        }
    )
    probabilities = ensemble_probabilities(model, feature)
    continue_decision = decision_from_probabilities(probabilities)
    arm_counts = {
        "continue": int((~treatment).sum()),
        "route": int(treatment.sum()),
    }
    report = {
        "kind": REPORT_KIND,
        "format_version": 1,
        "status": "complete",
        "training_rows": rows,
        "dataset_semantic_sha256": metadata["dataset_semantic_sha256"],
        "analysis_plan_sha256": analysis_plan_sha256,
        "input_ledger_sha256": input_ledger_sha256,
        "input_ledger_tag": input_ledger_tag,
        "member_diagnostics": diagnostics,
        "factual_arm_rows": arm_counts,
        "factual_event_counts": {
            "success": int(success.sum()),
            "dropped": int(dropped.sum()),
            "unsafe_force": int(unsafe.sum()),
        },
        "train_continue_decisions": int(continue_decision.sum()),
        "train_continue_rate": float(continue_decision.float().mean()),
        "collection_acceptance": acceptance,
        "model_semantic_sha256": semantic_model_sha256(model),
    }
    json.dumps(report, sort_keys=True, allow_nan=False)
    return model, report


def publish_model_and_report_no_clobber(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    model_output: Path,
    report_output: Path,
    repository_root: Path,
) -> str:
    validated = validate_gate_model(model)
    root = Path(os.path.abspath(os.fspath(repository_root)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("gate output repository root must be a regular directory")
    model_output = Path(os.path.abspath(os.fspath(model_output)))
    report_output = Path(os.path.abspath(os.fspath(report_output)))
    if model_output == report_output:
        raise ValueError("gate model and report paths must differ")
    for path in (model_output, report_output):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("gate output escapes the repository") from error
        _require_nonsymlink_chain(path, root, "gate output")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"gate output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_nonsymlink_chain(path.parent, root, "gate output parent")
    model_fd, model_name = tempfile.mkstemp(
        prefix=f".{model_output.name}.tmp-", dir=model_output.parent
    )
    os.close(model_fd)
    report_fd, report_name = tempfile.mkstemp(
        prefix=f".{report_output.name}.tmp-", dir=report_output.parent
    )
    model_temp, report_temp = Path(model_name), Path(report_name)
    linked: list[Path] = []
    try:
        with model_temp.open("wb") as stream:
            torch.save(validated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(model_temp)
        final_report = {**dict(report), "model_sha256": digest}
        report_bytes = (
            json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with os.fdopen(report_fd, "wb") as stream:
            stream.write(report_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(model_temp, model_output)
        linked.append(model_output)
        os.link(report_temp, report_output)
        linked.append(report_output)
        for parent in {model_output.parent, report_output.parent}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return digest
    except BaseException:
        for path in reversed(linked):
            if path.exists() or path.is_symlink():
                path.unlink()
        raise
    finally:
        for path in (model_temp, report_temp):
            if path.exists() or path.is_symlink():
                path.unlink()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    repository_root = Path(__file__).resolve().parents[2]
    plan = load_analysis_plan(DEFAULT_ANALYSIS_PLAN, require_preregistered=True)
    plan_sha256 = plan["_sha256"]
    provenance = analysis_seal_provenance(plan, repository_root=repository_root)
    dataset = load_registered_factual_cohort("train")
    _ledger, ledger_receipt = load_tagged_evidence_ledger(
        dataset, plan, cohort="train", repository_root=repository_root
    )
    model, report = train_gate_ensemble(
        dataset,
        plan,
        analysis_plan_sha256=plan_sha256,
        input_ledger_sha256=ledger_receipt["sha256"],
        input_ledger_tag=ledger_receipt["tag"],
        trainer_source_sha256=provenance["source_sha256"][
            "scripts/flashsac/train_public_route_gate.py"
        ],
    )
    outputs = plan["outputs"]
    model_output = repository_root / outputs["frozen_model"]
    report_output = repository_root / outputs["training_report"]
    digest = publish_model_and_report_no_clobber(
        model,
        {**report, "analysis_git": provenance},
        model_output=model_output,
        report_output=report_output,
        repository_root=repository_root,
    )
    print(
        f"[public-route-gate] rows={report['training_rows']} "
        f"continue_rate={report['train_continue_rate']:.6f} sha256={digest}"
    )


if __name__ == "__main__":
    main()
