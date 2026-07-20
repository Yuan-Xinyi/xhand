#!/usr/bin/env python3
"""CPU deployment invariants for the distal-only residual adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from distal_residual_adapter import (
    DistalResidualAdapter,
    StatefulCloseGate,
    load_adapter,
    make_payload,
)


def main() -> None:
    torch.manual_seed(4)
    adapter = DistalResidualAdapter(zero_head=False).eval()
    obs = torch.zeros((4, 115))
    obs[1:, 92:97] = 0.1
    obs[2, 106] = 1.0
    latent = torch.randn(4, 64)
    base = torch.randn(4, 21).clamp(-0.8, 0.8)
    result, gate, delta = adapter.apply(base, obs, latent)
    assert gate.tolist() == [False, True, False, True]
    assert torch.equal(result[:, :16], base[:, :16])
    assert torch.equal(result[~gate], base[~gate])
    assert float(delta.abs().max()) <= 0.100001
    deadband_obs = obs[1:2].clone()
    deadband_obs[:, 92:97] = 0.015
    _, deadband_gate, _ = adapter.apply(
        base[1:2], deadband_obs, latent[1:2], external_gate=torch.ones(1, dtype=torch.bool)
    )
    assert deadband_gate.item()
    try:
        adapter.apply(base, obs, latent, scale=1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("scale above one bypassed the hard delta limit")

    zero = DistalResidualAdapter().eval()
    identical, _, _ = zero.apply(base, obs, latent)
    assert torch.equal(identical, base)
    stateful = StatefulCloseGate(1, "cpu", timeout_steps=2)
    close_obs = torch.zeros((1, 115))
    close_obs[:, 92:97] = 0.1
    eligible = torch.ones(1, dtype=torch.bool)
    assert stateful.update(close_obs, eligible).item()
    assert stateful.update(close_obs, eligible).item()
    assert not stateful.update(close_obs, eligible).item()
    assert stateful.blocked.item()
    assert not stateful.update(close_obs, eligible).item()
    close_obs[:, 92:97] = 0.0
    assert not stateful.update(close_obs, eligible).item()
    close_obs[:, 92:97] = 0.1
    assert stateful.update(close_obs, eligible).item()
    with tempfile.TemporaryDirectory() as directory:
        base_path = Path(directory) / "base.pth"
        base_path.write_bytes(b"frozen actor")
        path = Path(directory) / "adapter.pth"
        torch.save(make_payload(adapter, base_path, {}), path)
        restored, _ = load_adapter(path, base_path, "cpu")
        restored_result, _, _ = restored.apply(base, obs, latent)
        assert torch.equal(restored_result, result)
        wrong = Path(directory) / "wrong.pth"
        wrong.write_bytes(b"wrong actor")
        try:
            load_adapter(path, wrong, "cpu")
        except RuntimeError:
            pass
        else:
            raise AssertionError("base SHA mismatch was accepted")
    print("PASS: zero initialization is exactly identical to the frozen base actor")
    print("PASS: action[:16] is bitwise unchanged and distal delta is bounded to 0.10")
    print("PASS: serialized adapters reject a different base checkpoint")
    print("PASS: timed-out close gates cannot immediately re-enter without a phase reset")
    print("PASS: stateful hysteresis is authoritative and scale cannot bypass the hard cap")


if __name__ == "__main__":
    main()
