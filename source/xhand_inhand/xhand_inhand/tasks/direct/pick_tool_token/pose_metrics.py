"""Torch-only pose-set helpers shared by functional preparation tasks."""

from __future__ import annotations

import math

import torch


def symmetry_aware_angle_error(
    angle: torch.Tensor, target: float | torch.Tensor, symmetry_order: int
) -> torch.Tensor:
    """Smallest absolute angular error under an n-fold rotation symmetry."""

    if isinstance(symmetry_order, bool) or not isinstance(symmetry_order, int) or symmetry_order < 1:
        raise ValueError("symmetry_order must be a positive integer")
    delta = angle - torch.as_tensor(target, dtype=angle.dtype, device=angle.device)
    period = 2.0 * math.pi / symmetry_order
    return torch.remainder(delta + 0.5 * period, period).sub(0.5 * period).abs()


def planar_axis_alignment(axis_world: torch.Tensor) -> torch.Tensor:
    """Return 1 for a horizontal directed axis and 0 for a vertical one."""

    if axis_world.ndim != 2 or axis_world.shape[1] != 3:
        raise ValueError("axis_world must have shape (N, 3)")
    normalized = axis_world / axis_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return normalized[:, :2].norm(dim=-1)


def compose_yaw_with_rest_quaternion(rest_quat_wxyz: torch.Tensor, yaw: float) -> torch.Tensor:
    """Left-compose world-Z yaw with batched WXYZ rest quaternions."""

    if rest_quat_wxyz.ndim != 2 or rest_quat_wxyz.shape[1] != 4:
        raise ValueError("rest_quat_wxyz must have shape (N, 4)")
    half = torch.full(
        (rest_quat_wxyz.shape[0],), 0.5 * float(yaw), dtype=rest_quat_wxyz.dtype, device=rest_quat_wxyz.device
    )
    yaw_quat = torch.zeros_like(rest_quat_wxyz)
    yaw_quat[:, 0] = torch.cos(half)
    yaw_quat[:, 3] = torch.sin(half)
    aw, ax, ay, az = yaw_quat.unbind(dim=-1)
    bw, bx, by, bz = rest_quat_wxyz.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def pose_delta_speeds(
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    previous_position: torch.Tensor,
    previous_quaternion_wxyz: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer linear/angular speed from adjacent poses rather than noisy Fabric velocities."""

    if position.ndim != 2 or position.shape[1] != 3 or previous_position.shape != position.shape:
        raise ValueError("position tensors must both have shape (N, 3)")
    if (
        quaternion_wxyz.ndim != 2
        or quaternion_wxyz.shape[1] != 4
        or previous_quaternion_wxyz.shape != quaternion_wxyz.shape
        or quaternion_wxyz.shape[0] != position.shape[0]
    ):
        raise ValueError("quaternion tensors must both have shape (N, 4)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    linear = (position - previous_position).norm(dim=-1) / float(dt)
    denom = quaternion_wxyz.norm(dim=-1) * previous_quaternion_wxyz.norm(dim=-1)
    cos_half = (quaternion_wxyz * previous_quaternion_wxyz).sum(dim=-1).abs()
    cos_half = (cos_half / denom.clamp_min(1.0e-12)).clamp(max=1.0)
    angular = 2.0 * torch.acos(cos_half) / float(dt)
    return linear, angular
