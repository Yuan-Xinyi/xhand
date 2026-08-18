from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_pose_metrics():
    path = (
        ROOT
        / "source"
        / "xhand_inhand"
        / "xhand_inhand"
        / "tasks"
        / "direct"
        / "pick_tool_token"
        / "pose_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("functional_pregrasp_pose_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_module():
    path = ROOT / "scripts" / "flashsac" / "train.py"
    spec = importlib.util.spec_from_file_location("functional_pregrasp_flashsac_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_is_ten_fixed_intent_specialists() -> None:
    path = ROOT / "configs" / "functional_pregrasp" / "tools_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    objects = manifest["objects"]
    assert len(objects) == 10
    assert [item["order"] for item in objects] == list(range(10))
    assert len({item["object_id"] for item in objects}) == 10
    assert all(item["intent"] == "use" for item in objects)
    assert manifest["phase_1"]["joint_training"] is False
    assert manifest["observation_contract"]["goal_pose_slice"] == [63, 70]
    assert all(item["pose_files"] for item in objects)


def test_symmetry_aware_heading_error() -> None:
    metrics = _load_pose_metrics()
    angle = torch.tensor([0.0, math.pi, 0.5 * math.pi, 1.5 * math.pi])
    no_symmetry = metrics.symmetry_aware_angle_error(angle, 0.0, 1)
    two_fold = metrics.symmetry_aware_angle_error(angle, 0.0, 2)
    four_fold = metrics.symmetry_aware_angle_error(angle, 0.0, 4)
    assert no_symmetry.tolist() == pytest.approx(
        [0.0, math.pi, 0.5 * math.pi, 0.5 * math.pi], abs=1.0e-6
    )
    assert two_fold.tolist() == pytest.approx(
        [0.0, 0.0, 0.5 * math.pi, 0.5 * math.pi], abs=1.0e-6
    )
    assert four_fold.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1.0e-6)


def test_continuous_roll_still_requires_horizontal_long_axis() -> None:
    metrics = _load_pose_metrics()
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 1.0]])
    support = metrics.planar_axis_alignment(axes)
    assert support.tolist() == pytest.approx([1.0, 0.0, 2**-0.5], abs=1.0e-6)


def test_goal_quaternion_is_world_yaw_times_rest() -> None:
    metrics = _load_pose_metrics()
    rest = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2**-0.5, 2**-0.5, 0.0, 0.0]])
    result = metrics.compose_yaw_with_rest_quaternion(rest, math.pi)
    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2**-0.5, 2**-0.5]], dtype=torch.float
    )
    assert torch.allclose(result, expected, atol=1.0e-6)


def test_pose_delta_speed_ignores_quaternion_sign() -> None:
    metrics = _load_pose_metrics()
    previous_position = torch.zeros((2, 3))
    position = torch.tensor([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    previous_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    half_angle = 0.05
    quaternion = torch.tensor(
        [
            [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )
    linear, angular = metrics.pose_delta_speeds(
        position, quaternion, previous_position, previous_quaternion, 0.02
    )
    assert linear.tolist() == pytest.approx([1.0, 0.0], abs=1.0e-6)
    assert angular.tolist() == pytest.approx([5.0, 0.0], abs=1.0e-4)


def test_stateful_checkpoint_restore_requires_matching_experiment(tmp_path: Path) -> None:
    train = _load_train_module()
    expected = {
        "task": "Functional-Pregrasp-Flashlight-Direct-v0",
        "experiment_manifest_sha256": "abc123",
        "object_id": "flashlight_1",
        "intent": "use",
    }
    metadata_path = tmp_path / "experiment.json"
    metadata_path.write_text(json.dumps({"schema_version": 1, **expected}), encoding="utf-8")
    assert train.validate_checkpoint_experiment(
        tmp_path, expected, weights_only=False
    ) == metadata_path

    mismatched = {**expected, "object_id": "marker_pen8"}
    with pytest.raises(ValueError, match="identity does not match"):
        train.validate_checkpoint_experiment(tmp_path, mismatched, weights_only=False)

    metadata_path.unlink()
    with pytest.raises(FileNotFoundError, match="weights_only"):
        train.validate_checkpoint_experiment(tmp_path, expected, weights_only=False)
    assert train.validate_checkpoint_experiment(tmp_path, expected, weights_only=True) is None


@pytest.mark.parametrize("bad_order", [0, -1, True, 1.5])
def test_symmetry_order_validation(bad_order) -> None:
    metrics = _load_pose_metrics()
    with pytest.raises(ValueError):
        metrics.symmetry_aware_angle_error(torch.zeros(1), 0.0, bad_order)
