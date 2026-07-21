#!/usr/bin/env python3
"""CPU checks for ranked pregrasp curriculum staticization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from power_pregrasp_curriculum import CURRICULUM_CONTRACT, build_static_curriculum


def _source(path: Path, offset: float) -> None:
    states = 2
    joint = torch.arange(states * 19, dtype=torch.float32).reshape(states, 19) + offset
    boundary = {
        "joint_pos": joint,
        "joint_vel": torch.full_like(joint, 3.0),
        "dof_targets": joint + 0.5,
        "object_local_pos": torch.full((states, 3), offset),
        "object_quat": torch.tensor(((1.0, 0.0, 0.0, 0.0),) * states),
        "object_velocity": torch.full((states, 6), 4.0),
        "last_action": torch.full((states, 21), 0.7),
        "contact_steps": torch.full((states,), 9, dtype=torch.long),
        "lost_contact_steps": torch.full((states,), 2, dtype=torch.long),
        "is_grasped": torch.ones(states, dtype=torch.bool),
    }
    torch.save({"boundaries": {"close_start": boundary}}, path)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.pt"
        second = root / "second.pt"
        report_path = root / "rank.json"
        _source(first, 0.0)
        _source(second, 100.0)
        report = {
            "contract": "power_pregrasp_third_finger_bottleneck_v1",
            "records": [
                {
                    "rank": 1,
                    "dataset": str(second),
                    "dataset_state_index": 1,
                    "dataset_seed": 2,
                    "score": 0.9,
                    "third_other_near": 0.3,
                },
                {
                    "rank": 2,
                    "dataset": str(first),
                    "dataset_state_index": 0,
                    "dataset_seed": 1,
                    "score": 0.8,
                    "third_other_near": 0.2,
                },
            ],
        }
        report_path.write_text(json.dumps(report))
        curriculum = build_static_curriculum(report_path, 2)

        assert curriculum["meta"]["contract"] == CURRICULUM_CONTRACT
        assert (
            curriculum["meta"]["observation_contract"]
            == "pick_tool_coupled_power_align_close_state131_v1"
        )
        assert curriculum["meta"]["top_k"] == 2
        boundary = curriculum["boundaries"]["close_start"]
        assert boundary["joint_pos"][0, 0].item() == 119.0
        assert boundary["joint_pos"][1, 0].item() == 0.0
        assert torch.equal(boundary["dof_targets"], boundary["joint_pos"])
        assert torch.count_nonzero(boundary["joint_vel"]).item() == 0
        assert torch.count_nonzero(boundary["object_velocity"]).item() == 0
        assert torch.count_nonzero(boundary["last_action"]).item() == 0
        assert torch.count_nonzero(boundary["contact_steps"]).item() == 0
        assert torch.count_nonzero(boundary["lost_contact_steps"]).item() == 0
        assert not boundary["is_grasped"].any().item()

        try:
            build_static_curriculum(report_path, 3)
        except ValueError as exc:
            assert "exceeds" in str(exc)
        else:
            raise AssertionError("top_k beyond the report was accepted")
    print("ALL POWER PREGRASP CURRICULUM TESTS PASSED")


if __name__ == "__main__":
    main()
