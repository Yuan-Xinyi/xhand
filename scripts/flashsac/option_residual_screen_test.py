#!/usr/bin/env python3
"""CPU-only contract tests for Candidate 39 coherent option residuals."""

from __future__ import annotations

import torch

from option_residual_screen import (
    ACTION_DIM,
    ARM_ACTION_DIM,
    DISTAL_ACTION_DIM,
    HAND_ACTION_DIM,
    TOKEN_ACTION_DIM,
    apply_option_residual,
    build_option_design,
    episode_coherent_raw_z,
    exact_balanced_treatment_mask,
    grouped_pre_tanh_scale,
)


SEED = 391
NUM_ENVS = 64


def _expect(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_assignment_determinism_complement_and_antithetic_pairs() -> None:
    design_a = build_option_design(
        seed=SEED, num_envs=NUM_ENVS, replicate="a"
    )
    design_a_again = build_option_design(
        seed=SEED, num_envs=NUM_ENVS, replicate="a"
    )
    design_b = build_option_design(
        seed=SEED, num_envs=NUM_ENVS, replicate="b"
    )

    assert int(design_a.treatment.sum()) == NUM_ENVS // 2
    assert torch.equal(design_b.treatment, ~design_a.treatment)
    assert torch.equal(design_a.treatment, design_a_again.treatment)
    assert torch.equal(design_a.raw_z, design_a_again.raw_z)
    # Replicate swaps treatment only; it must never redraw episode latents.
    assert torch.equal(design_a.raw_z, design_b.raw_z)
    assert torch.equal(design_a.pair_slot, design_b.pair_slot)
    assert torch.equal(design_a.antithetic_sign, design_b.antithetic_sign)

    for env_slot in range(NUM_ENVS):
        mate = int(design_a.pair_slot[env_slot])
        assert mate != env_slot
        assert int(design_a.pair_slot[mate]) == env_slot
        assert int(design_a.antithetic_sign[env_slot]) == -int(
            design_a.antithetic_sign[mate]
        )
        assert torch.equal(design_a.raw_z[env_slot], -design_a.raw_z[mate])
        # The rank-half assignment makes each antithetic pair one treatment
        # and one control in each complementary replicate.
        assert bool(design_a.treatment[env_slot]) != bool(design_a.treatment[mate])

    assert torch.count_nonzero(design_a.effective_z[~design_a.treatment]) == 0
    assert torch.count_nonzero(design_b.effective_z[~design_b.treatment]) == 0
    assert torch.equal(
        design_a.effective_z[design_a.treatment],
        design_a.raw_z[design_a.treatment],
    )
    different_seed = build_option_design(
        seed=SEED + 1, num_envs=NUM_ENVS, replicate="a"
    )
    assert not torch.equal(design_a.raw_z, different_seed.raw_z)


def test_episode_coherence_and_grouped_scale_contract() -> None:
    raw_1, pair_1, sign_1 = episode_coherent_raw_z(
        seed=SEED, num_envs=NUM_ENVS, dtype=torch.float64
    )
    raw_2, pair_2, sign_2 = episode_coherent_raw_z(
        seed=SEED, num_envs=NUM_ENVS, dtype=torch.float64
    )
    assert torch.equal(raw_1, raw_2)
    assert torch.equal(pair_1, pair_2)
    assert torch.equal(sign_1, sign_2)
    assert raw_1.shape == (NUM_ENVS, HAND_ACTION_DIM)
    assert bool(torch.isfinite(raw_1).all())

    scale = grouped_pre_tanh_scale(
        token_scale=0.25, distal_scale=0.0625, dtype=torch.float64
    )
    assert scale.shape == (HAND_ACTION_DIM,)
    assert torch.equal(scale[:TOKEN_ACTION_DIM], torch.full((9,), 0.25, dtype=torch.float64))
    assert torch.equal(
        scale[TOKEN_ACTION_DIM:],
        torch.full((DISTAL_ACTION_DIM,), 0.0625, dtype=torch.float64),
    )


def _action_fixture() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    batch = 8
    baseline = torch.linspace(
        -0.8, 0.8, steps=batch * ACTION_DIM, dtype=torch.float32
    ).reshape(batch, ACTION_DIM)
    option_active = torch.tensor([True, True, True, False, True, True, False, True])
    public_latch = torch.tensor([False, True, False, False, False, True, False, False])
    treatment = torch.tensor([True, True, False, True, True, False, True, True])
    raw_z = build_option_design(
        seed=SEED, num_envs=batch, replicate="a"
    ).raw_z
    return baseline, option_active, public_latch, treatment, raw_z


def test_zero_sigma_is_bit_exact_everywhere() -> None:
    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    # Include exact tanh boundaries: the clamp must not perturb zero residuals.
    baseline[0, ARM_ACTION_DIM:] = 1.0
    baseline[4, ARM_ACTION_DIM:] = -1.0
    result = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=grouped_pre_tanh_scale(
            token_scale=0.0, distal_scale=0.0
        ),
    )
    assert torch.equal(result.action, baseline)
    assert torch.count_nonzero(result.pre_tanh_residual) == 0
    assert torch.equal(result.raw_z, raw_z)


def test_arm_latch_control_and_option_invariants() -> None:
    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    scale = grouped_pre_tanh_scale(token_scale=0.30, distal_scale=0.10)
    result = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=scale,
    )
    eligible = option_active & ~public_latch & treatment
    assert torch.equal(result.eligible, eligible)
    assert torch.equal(result.action[:, :ARM_ACTION_DIM], baseline[:, :ARM_ACTION_DIM])
    assert torch.equal(result.action[public_latch], baseline[public_latch])
    assert torch.equal(result.action[~option_active], baseline[~option_active])
    assert torch.equal(result.action[~treatment], baseline[~treatment])
    assert torch.count_nonzero(result.effective_z[~treatment]) == 0
    assert torch.equal(result.effective_z[treatment], raw_z[treatment])
    assert torch.equal(
        result.pre_tanh_residual[eligible], raw_z[eligible] * scale
    )
    assert torch.equal(
        result.unclipped_pre_tanh_residual, result.pre_tanh_residual
    )
    assert torch.equal(result.residual_z, result.effective_z)
    assert torch.count_nonzero(result.pre_tanh_residual[~eligible]) == 0
    assert bool(
        (
            result.action[eligible, ARM_ACTION_DIM:]
            != baseline[eligible, ARM_ACTION_DIM:]
        ).all()
    )


def test_saturated_actions_are_finite_bounded_and_auditable() -> None:
    batch = 4
    baseline = torch.zeros((batch, ACTION_DIM), dtype=torch.float64)
    baseline[:, ARM_ACTION_DIM:] = torch.tensor(
        [1.0, -1.0] * 7, dtype=torch.float64
    )
    raw_z = build_option_design(
        seed=SEED, num_envs=batch, replicate="a", dtype=torch.float64
    ).raw_z
    result = apply_option_residual(
        baseline_action=baseline,
        option_active=torch.ones(batch, dtype=torch.bool),
        public_latch=torch.zeros(batch, dtype=torch.bool),
        treatment=torch.ones(batch, dtype=torch.bool),
        raw_z=raw_z,
        pre_tanh_scale=grouped_pre_tanh_scale(
            token_scale=0.75, distal_scale=0.5, dtype=torch.float64
        ),
    )
    assert bool(torch.isfinite(result.action).all())
    assert bool((result.action.abs() <= 1.0).all())
    assert torch.equal(result.raw_z, raw_z)
    assert torch.equal(result.pre_tanh_residual, raw_z * torch.tensor(
        [0.75] * TOKEN_ACTION_DIM + [0.5] * DISTAL_ACTION_DIM,
        dtype=torch.float64,
    ))


def test_batched_state_dependent_scale_and_symmetric_clip_audit() -> None:
    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    batch = baseline.shape[0]
    group_scale = grouped_pre_tanh_scale(token_scale=0.4, distal_scale=0.2)
    # This emulates an explicitly supplied state-dependent multiplier.  The
    # option module does not read or infer actor std itself.
    state_multiplier = torch.linspace(0.25, 1.5, batch).unsqueeze(-1)
    batched_scale = state_multiplier * group_scale.unsqueeze(0)
    cap = torch.linspace(0.01, 0.14, HAND_ACTION_DIM).unsqueeze(0).expand(batch, -1).clone()
    result = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=batched_scale,
        raw_z_abs_cap=0.5,
        pre_tanh_abs_cap=cap,
    )
    eligible = option_active & ~public_latch & treatment
    expected_unclipped = torch.where(
        eligible.unsqueeze(-1), raw_z * batched_scale, torch.zeros_like(raw_z)
    )
    expected_z = torch.where(
        treatment.unsqueeze(-1), raw_z, torch.zeros_like(raw_z)
    ).clamp(-0.5, 0.5)
    expected_applied = torch.maximum(
        torch.minimum(expected_z * batched_scale, cap), -cap
    )
    expected_applied = torch.where(
        eligible.unsqueeze(-1), expected_applied, torch.zeros_like(raw_z)
    )
    assert torch.equal(result.unclipped_pre_tanh_residual, expected_unclipped)
    assert torch.equal(result.residual_z, expected_z)
    assert torch.equal(result.pre_tanh_residual, expected_applied)
    assert bool((result.pre_tanh_residual.abs() <= cap).all())
    assert torch.equal(result.action[:, :ARM_ACTION_DIM], baseline[:, :ARM_ACTION_DIM])
    assert torch.equal(result.action[public_latch], baseline[public_latch])

    vector_cap = torch.full((HAND_ACTION_DIM,), 0.03)
    vector_capped = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=batched_scale,
        pre_tanh_abs_cap=vector_cap,
    )
    assert bool((vector_capped.pre_tanh_residual.abs() <= vector_cap).all())


def test_global_l2_budget_is_symmetric_and_fail_closed() -> None:
    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    l2_cap = 0.075
    result = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=raw_z,
        pre_tanh_scale=grouped_pre_tanh_scale(
            token_scale=0.4, distal_scale=0.2
        ),
        raw_z_abs_cap=2.0,
        pre_tanh_abs_cap=grouped_pre_tanh_scale(
            token_scale=0.1, distal_scale=0.05
        ),
        pre_tanh_l2_cap=l2_cap,
    )
    norms = torch.linalg.vector_norm(result.pre_tanh_residual, dim=-1)
    assert bool((norms <= l2_cap + 1.0e-7).all())
    assert torch.count_nonzero(result.pre_tanh_residual[~result.eligible]) == 0

    mirrored = apply_option_residual(
        baseline_action=baseline,
        option_active=option_active,
        public_latch=public_latch,
        treatment=treatment,
        raw_z=-raw_z,
        pre_tanh_scale=grouped_pre_tanh_scale(
            token_scale=0.4, distal_scale=0.2
        ),
        raw_z_abs_cap=2.0,
        pre_tanh_abs_cap=grouped_pre_tanh_scale(
            token_scale=0.1, distal_scale=0.05
        ),
        pre_tanh_l2_cap=l2_cap,
    )
    assert torch.equal(
        mirrored.pre_tanh_residual, -result.pre_tanh_residual
    )


def test_scale_and_cap_shape_dtype_device_validation() -> None:
    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    scale = grouped_pre_tanh_scale(token_scale=0.1, distal_scale=0.1)
    kwargs = {
        "baseline_action": baseline,
        "option_active": option_active,
        "public_latch": public_latch,
        "treatment": treatment,
        "raw_z": raw_z,
        "pre_tanh_scale": scale,
    }
    _expect(
        ValueError,
        apply_option_residual,
        **{**kwargs, "pre_tanh_scale": torch.ones(HAND_ACTION_DIM - 1)},
    )
    _expect(
        ValueError,
        apply_option_residual,
        **{**kwargs, "pre_tanh_scale": scale.to(torch.float64)},
    )
    _expect(
        ValueError,
        apply_option_residual,
        **{**kwargs, "pre_tanh_scale": torch.ones(HAND_ACTION_DIM, device="meta")},
    )
    _expect(
        ValueError,
        apply_option_residual,
        **kwargs,
        pre_tanh_abs_cap=torch.ones(HAND_ACTION_DIM + 1),
    )
    _expect(
        ValueError,
        apply_option_residual,
        **kwargs,
        pre_tanh_abs_cap=torch.ones(HAND_ACTION_DIM, dtype=torch.float64),
    )
    _expect(
        ValueError,
        apply_option_residual,
        **kwargs,
        pre_tanh_abs_cap=torch.ones(HAND_ACTION_DIM, device="meta"),
    )
    _expect(ValueError, apply_option_residual, **kwargs, raw_z_abs_cap=-0.1)
    _expect(
        ValueError,
        apply_option_residual,
        **kwargs,
        raw_z_abs_cap=float("inf"),
    )
    _expect(ValueError, apply_option_residual, **kwargs, pre_tanh_l2_cap=-0.1)
    _expect(
        ValueError,
        apply_option_residual,
        **kwargs,
        pre_tanh_l2_cap=float("nan"),
    )


def test_fail_closed_validation() -> None:
    _expect(ValueError, exact_balanced_treatment_mask, seed=SEED, num_envs=63, replicate="a")
    _expect(ValueError, exact_balanced_treatment_mask, seed=SEED, num_envs=64, replicate="c")
    _expect(ValueError, exact_balanced_treatment_mask, seed=-1, num_envs=64, replicate="a")
    _expect(ValueError, grouped_pre_tanh_scale, token_scale=-0.1, distal_scale=0.1)
    _expect(ValueError, grouped_pre_tanh_scale, token_scale=float("nan"), distal_scale=0.1)

    baseline, option_active, public_latch, treatment, raw_z = _action_fixture()
    scale = grouped_pre_tanh_scale(token_scale=0.1, distal_scale=0.1)
    kwargs = {
        "baseline_action": baseline,
        "option_active": option_active,
        "public_latch": public_latch,
        "treatment": treatment,
        "raw_z": raw_z,
        "pre_tanh_scale": scale,
    }
    malformed = baseline.clone()
    malformed[0, 0] = float("nan")
    _expect(FloatingPointError, apply_option_residual, **{**kwargs, "baseline_action": malformed})
    malformed = baseline.clone()
    malformed[0, 0] = 1.01
    _expect(ValueError, apply_option_residual, **{**kwargs, "baseline_action": malformed})
    malformed_z = raw_z.clone()
    malformed_z[0, 0] = float("inf")
    _expect(FloatingPointError, apply_option_residual, **{**kwargs, "raw_z": malformed_z})
    _expect(
        ValueError,
        apply_option_residual,
        **{**kwargs, "option_active": option_active.to(torch.int64)},
    )
    _expect(
        ValueError,
        apply_option_residual,
        **{**kwargs, "pre_tanh_scale": -scale},
    )
    _expect(ValueError, apply_option_residual, **kwargs, atanh_epsilon=0.0)


if __name__ == "__main__":
    test_assignment_determinism_complement_and_antithetic_pairs()
    test_episode_coherence_and_grouped_scale_contract()
    test_zero_sigma_is_bit_exact_everywhere()
    test_arm_latch_control_and_option_invariants()
    test_saturated_actions_are_finite_bounded_and_auditable()
    test_batched_state_dependent_scale_and_symmetric_clip_audit()
    test_global_l2_budget_is_symmetric_and_fail_closed()
    test_scale_and_cap_shape_dtype_device_validation()
    test_fail_closed_validation()
    print("option_residual_screen_test: PASS")
