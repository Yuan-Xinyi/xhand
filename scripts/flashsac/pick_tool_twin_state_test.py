#!/usr/bin/env python3
"""Simulation-free tests for PickTool twin-reset state synchronization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Any

import torch

from pick_tool_twin_state import (
    ACTION_DIM,
    CLOSE_QUALITY_INDEX,
    GRASP_LATCH_INDEX,
    OBSERVATION_DIM,
    READINESS_STRATUM_NAMES,
    TWIN_CACHE_FIELDS,
    copy_fresh_reset_to_twins,
    make_twin_pair_layout,
    paired_max_abs,
    publish_json_no_clobber,
    replicate_source_actions,
    update_public_readiness,
)


def _expect_error(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


class _FakeSensor:
    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(
            force_matrix_w=torch.zeros((num_envs, 1, 1, 3)),
            net_forces_w=torch.zeros((num_envs, 1, 3)),
        )
        self._timestamp = torch.zeros(num_envs)
        self._timestamp_last_update = torch.zeros(num_envs)

    def reset(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the exact-state helper must not write contact sensors")


class _FakeRobot:
    def __init__(self, task: "_FakeTask", num_envs: int, joints: int):
        self.task = task
        base = torch.arange(num_envs * joints, dtype=torch.float32).reshape(num_envs, joints)
        self.data = SimpleNamespace(
            joint_pos=base * 0.01,
            joint_vel=torch.zeros_like(base),
            joint_pos_target=base * 0.01,
            joint_vel_target=torch.zeros_like(base),
            joint_effort_target=torch.zeros_like(base),
        )

    def write_joint_state_to_sim(
        self, position: torch.Tensor, velocity: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("robot.write_joint_state_to_sim")
        self.data.joint_pos.index_copy_(0, env_ids, position)
        self.data.joint_vel.index_copy_(0, env_ids, velocity)

    def set_joint_position_target(
        self, target: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("robot.set_joint_position_target")
        self.data.joint_pos_target.index_copy_(0, env_ids, target)

    def set_joint_velocity_target(
        self, target: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("robot.set_joint_velocity_target")
        self.data.joint_vel_target.index_copy_(0, env_ids, target)

    def set_joint_effort_target(
        self, target: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("robot.set_joint_effort_target")
        self.data.joint_effort_target.index_copy_(0, env_ids, target)


class _FakeObject:
    def __init__(self, task: "_FakeTask", origins: torch.Tensor):
        self.task = task
        num_envs = origins.shape[0]
        local = torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3) * 0.01
        quat = torch.zeros((num_envs, 4))
        quat[:, 0] = 1.0
        self.data = SimpleNamespace(
            root_link_pos_w=origins + local,
            root_link_quat_w=quat,
            root_com_lin_vel_w=torch.zeros((num_envs, 3)),
            root_com_ang_vel_w=torch.zeros((num_envs, 3)),
        )

    def write_root_pose_to_sim(
        self, pose: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("object.write_root_pose_to_sim")
        self.data.root_link_pos_w.index_copy_(0, env_ids, pose[:, :3])
        self.data.root_link_quat_w.index_copy_(0, env_ids, pose[:, 3:])

    def write_root_velocity_to_sim(
        self, velocity: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.task.call_order.append("object.write_root_velocity_to_sim")
        self.data.root_com_lin_vel_w.index_copy_(0, env_ids, velocity[:, :3])
        self.data.root_com_ang_vel_w.index_copy_(0, env_ids, velocity[:, 3:])


class _FakeScene:
    def __init__(self, task: "_FakeTask", num_envs: int):
        self.task = task
        self.env_origins = torch.zeros((num_envs, 3))
        self.env_origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 10.0

    def write_data_to_sim(self) -> None:
        self.task.call_order.append("scene.write_data_to_sim")

    def update(self, _dt: float) -> None:
        raise AssertionError("scene.update(dt) is forbidden at the zero-time fork")


class _FakeSim:
    def __init__(self, task: "_FakeTask"):
        self.task = task

    def forward(self) -> None:
        self.task.call_order.append("sim.forward")


def _cache_shape(name: str, num_envs: int, joints: int) -> tuple[int, ...]:
    if name in ("actions", "prev_actions"):
        return num_envs, ACTION_DIM
    if name == "target_quat":
        return num_envs, 4
    if name == "dof_targets":
        return num_envs, joints
    if name in ("_last_token_hand_target", "_last_raw_hand_target"):
        return num_envs, 12
    if name == "_last_distal_delta":
        return num_envs, 5
    return (num_envs,)


class _FakeTask:
    def __init__(self, num_envs: int = 4, joints: int = 19):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            observation_space=OBSERVATION_DIM,
            action_space=ACTION_DIM,
            close_option_mode=False,
            power_close_option_mode=False,
            coupled_power_align_close_option_mode=False,
            hold_arm_until_stable_grasp=False,
        )
        self.call_order: list[str] = []
        self.scene = _FakeScene(self, num_envs)
        self.sim = _FakeSim(self)
        self.robot = _FakeRobot(self, num_envs, joints)
        self.object = _FakeObject(self, self.scene.env_origins)
        self._object_contact_sensors = {
            "index": _FakeSensor(num_envs),
            "thumb": _FakeSensor(num_envs),
        }
        bool_fields = {
            "reset_buf",
            "reset_terminated",
            "reset_time_outs",
            "_is_grasped",
            "_grasp_bonus_given",
            "_is_success",
            "_success_paid",
            "_lift_bonus_given",
            "_unlatched_lift_failure",
            "_potential_initialized",
        }
        long_fields = {
            "episode_length_buf",
            "_contact_steps",
            "_lost_contact_steps",
            "_safe_grasp_steps",
            "_grasp_age",
            "_success_steps",
            "_hard_force_steps",
            "_overforce_steps",
        }
        for name in TWIN_CACHE_FIELDS:
            shape = _cache_shape(name, num_envs, joints)
            dtype = torch.bool if name in bool_fields else torch.long if name in long_fields else torch.float32
            setattr(self, name, torch.zeros(shape, dtype=dtype))
        self.dof_targets.copy_(self.robot.data.joint_pos_target)
        self.target_quat[:, 0] = 1.0
        # Distinct source/destination reset targets prove that the copy happened.
        self.target_quat[0, :] = torch.tensor((0.5, 0.5, 0.5, 0.5))
        self.target_quat[1, :] = torch.tensor((0.5, -0.5, 0.5, -0.5))
        self.obs_buf: dict[str, torch.Tensor] = {}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.call_order.append("task._get_observations")
        obs = torch.zeros((self.num_envs, OBSERVATION_DIM))
        obs[:, :19] = self.robot.data.joint_pos
        obs[:, 19:38] = self.robot.data.joint_vel
        obs[:, 38:42] = self.target_quat
        obs[:, 42:63] = self.actions
        obs[:, 63:66] = self.object.data.root_link_pos_w - self.scene.env_origins
        return {"policy": obs, "critic": obs}


def test_pair_layout_and_action_replication() -> None:
    layout = make_twin_pair_layout(6)
    assert layout.source.tolist() == [0, 2, 4]
    assert layout.destination.tolist() == [1, 3, 5]
    _expect_error(ValueError, make_twin_pair_layout, 1)
    _expect_error(ValueError, make_twin_pair_layout, 5)
    source = torch.arange(3 * ACTION_DIM, dtype=torch.float32).reshape(3, ACTION_DIM)
    full = replicate_source_actions(source, layout)
    assert torch.equal(full.index_select(0, layout.source), source)
    assert torch.equal(full.index_select(0, layout.destination), source)
    assert paired_max_abs(full, layout) == 0.0


def test_public_readiness_uses_only_declared_public_slots_and_is_sticky() -> None:
    observation = torch.zeros((4, OBSERVATION_DIM))
    # low/low, low/high, high/low, high/high.  The smallest non-thumb is a
    # decoy; g must use the second-largest non-thumb plus the thumb.
    settings = (
        (0.20, 0.10),
        (0.20, 0.30),
        (0.40, 0.10),
        (0.40, 0.30),
    )
    for row, (score, close) in enumerate(settings):
        observation[row, 92:96] = torch.tensor((0.01, score + 0.2, score, score + 0.1))
        observation[row, 96] = score + 0.05
        observation[row, CLOSE_QUALITY_INDEX] = close
        observation[row, GRASP_LATCH_INDEX] = 0.0
    # Private/legacy-looking columns are arbitrary and must not affect readiness.
    observation[:, 86] = torch.tensor((99.0, -99.0, 12.0, -12.0))
    ready = torch.zeros(4, dtype=torch.long)
    used = torch.zeros(4, dtype=torch.bool)
    for step in range(4):
        update = update_public_readiness(observation, ready, used)
        ready, used = update.ready_count, update.fork_used
        assert bool(update.trigger.all()) is (step == 3)
    assert update.stratum.tolist() == [0, 1, 2, 3]
    assert [READINESS_STRATUM_NAMES[index] for index in update.stratum.tolist()] == list(
        READINESS_STRATUM_NAMES
    )
    assert bool(update.sticky.all())
    repeated = update_public_readiness(observation, ready, used)
    assert not bool(repeated.trigger.any())
    assert bool(repeated.sticky.all())

    latched = observation.clone()
    latched[0, GRASP_LATCH_INDEX] = 1.0
    reset = update_public_readiness(latched, ready, torch.zeros_like(used))
    assert reset.ready_count[0].item() == 0
    nonbinary = observation.clone()
    nonbinary[0, GRASP_LATCH_INDEX] = 0.5
    _expect_error(
        RuntimeError,
        update_public_readiness,
        nonbinary,
        ready,
        used,
    )
    _expect_error(
        ValueError,
        update_public_readiness,
        observation[:, :114],
        ready,
        used,
    )


def test_fresh_reset_copy_is_local_frame_exact_and_never_writes_sensors() -> None:
    task = _FakeTask()
    layout = make_twin_pair_layout(task.num_envs)
    policy, report = copy_fresh_reset_to_twins(task, layout)
    assert paired_max_abs(policy, layout) <= 1.0e-6
    assert report["contact_manifold_fabricated"] is False
    assert report["physical_write_scope"] == "all_twins_symmetrically_v1"
    assert report["scene_update_called"] is False
    assert report["sync_sequence"] == [
        "scene.write_data_to_sim",
        "sim.forward",
        "task._get_observations",
    ]
    assert task.call_order[-3:] == report["sync_sequence"]
    assert report["post_copy_pair_abs_error"]["object_root_link_local_pose"] <= 1.0e-6
    assert report["post_copy_pair_abs_error"]["joint_pos_target"] == 0.0
    assert report["post_copy_pair_abs_error"]["cache_max"] == 0.0
    for sensor in task._object_contact_sensors.values():
        assert torch.count_nonzero(sensor.data.force_matrix_w).item() == 0
        assert torch.count_nonzero(sensor.data.net_forces_w).item() == 0


def test_fresh_reset_copy_rejects_contact_timestamp_and_contract_drift() -> None:
    task = _FakeTask()
    layout = make_twin_pair_layout(task.num_envs)
    task._object_contact_sensors["thumb"].data.net_forces_w[0, 0, 0] = 2.0e-6
    _expect_error(RuntimeError, copy_fresh_reset_to_twins, task, layout)

    task = _FakeTask()
    task._object_contact_sensors["thumb"]._timestamp[2] = 1.0
    _expect_error(RuntimeError, copy_fresh_reset_to_twins, task, layout)

    task = _FakeTask()
    task.cfg.close_option_mode = True
    _expect_error(RuntimeError, copy_fresh_reset_to_twins, task, layout)

    task = _FakeTask()
    task.actions[0, 0] = 1.0e-5
    _expect_error(RuntimeError, copy_fresh_reset_to_twins, task, layout)

    task = _FakeTask()
    task.dof_targets[0, 0] += 1.0e-5
    task.robot.data.joint_pos_target[0, 0] += 1.0e-5
    _expect_error(RuntimeError, copy_fresh_reset_to_twins, task, layout)


def test_atomic_json_is_strict_and_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        output = directory / "sync.json"
        publish_json_no_clobber({"status": "passed", "pairs": 2}, output)
        original = output.read_bytes()
        _expect_error(FileExistsError, publish_json_no_clobber, {"status": "changed"}, output)
        assert output.read_bytes() == original
        assert not list(directory.glob(".sync.json.tmp-*"))

        dangling = directory / "dangling.json"
        dangling.symlink_to("missing-target.json")
        _expect_error(FileExistsError, publish_json_no_clobber, {"status": "bad"}, dangling)
        assert dangling.is_symlink()

        invalid = directory / "invalid.json"
        _expect_error(ValueError, publish_json_no_clobber, {"bad": float("nan")}, invalid)
        assert not invalid.exists()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    test_pair_layout_and_action_replication()
    test_public_readiness_uses_only_declared_public_slots_and_is_sticky()
    test_fresh_reset_copy_is_local_frame_exact_and_never_writes_sensors()
    test_fresh_reset_copy_rejects_contact_timestamp_and_contract_drift()
    test_atomic_json_is_strict_and_no_clobber()
    print("pick_tool_twin_state tests passed")


if __name__ == "__main__":
    main()
