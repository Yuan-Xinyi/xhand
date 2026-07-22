#!/usr/bin/env python3
"""CPU-only proofs for Candidate42's unchanged collection kernel boundary."""

from __future__ import annotations

import ast
import argparse
import copy
from pathlib import Path
import tempfile

import torch

import candidate42_public_arm_ramp_collection as collector


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


class _CampaignNormalizer(ast.NodeTransformer):
    """Apply exactly the preregistered C42-to-C41 kernel normalization."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = node.id.replace("candidate42", "candidate41").replace(
            "CANDIDATE42", "CANDIDATE41"
        )
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = node.attr.replace("candidate42", "candidate41").replace(
            "CANDIDATE42", "CANDIDATE41"
        )
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            node.value = (
                node.value.replace("Candidate42", "Candidate41")
                .replace("candidate42", "candidate41")
                .replace("CANDIDATE42", "CANDIDATE41")
            )
        return node


class _BoundaryCallNormalizer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "authoritative_env_step"
        ):
            return self.generic_visit(node)
        call = node.value
        if len(call.args) != 2:
            raise AssertionError("authoritative_env_step positional arguments changed")
        progress = next(
            keyword.value for keyword in call.keywords if keyword.arg == "progress"
        )
        set_boundary = ast.Assign(
            targets=[
                ast.Attribute(
                    value=copy.deepcopy(progress),
                    attr="first_env_step_invoked",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(True),
        )
        node.value = ast.Call(
            func=ast.Attribute(
                value=copy.deepcopy(call.args[0]), attr="step", ctx=ast.Load()
            ),
            args=[copy.deepcopy(call.args[1])],
            keywords=[],
        )
        return [set_boundary, node]


def _normalized_candidate42_kernel() -> ast.FunctionDef:
    path = _root() / "scripts/flashsac/candidate42_public_arm_ramp_collection.py"
    kernel = copy.deepcopy(_function(ast.parse(path.read_text()), "run_collection"))
    index = next(
        i
        for i, argument in enumerate(kernel.args.kwonlyargs)
        if argument.arg == "durable_boundary"
    )
    kernel.args.kwonlyargs.pop(index)
    kernel.args.kw_defaults.pop(index)
    kernel = _CampaignNormalizer().visit(kernel)
    kernel = _BoundaryCallNormalizer().visit(kernel)
    ast.fix_missing_locations(kernel)
    return kernel


def test_normalized_run_collection_ast_is_bit_exact_to_candidate41() -> None:
    path = _root() / "scripts/flashsac/collect_candidate41_public_arm_ramp_ab.py"
    original = _function(ast.parse(path.read_text()), "run_collection")
    assert ast.dump(_normalized_candidate42_kernel(), include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_durable_boundary_precedes_first_step_and_runs_once() -> None:
    events: list[str] = []

    class FakeEnv:
        def step(self, action: torch.Tensor) -> tuple:
            events.append("step")
            assert marker.is_file()
            return (action, None, None, None, None)

    with tempfile.TemporaryDirectory() as name:
        marker = Path(name) / "20_first_step_armed.json"

        def durable_boundary() -> None:
            events.append("boundary")
            with marker.open("xb") as stream:
                stream.write(b"{}\n")
                stream.flush()
                import os

                os.fsync(stream.fileno())

        progress = collector.AttemptProgress()
        action = torch.zeros((2, 16), dtype=torch.float32)
        env = FakeEnv()
        collector.authoritative_env_step(
            env, action, progress=progress, durable_boundary=durable_boundary
        )
        collector.authoritative_env_step(
            env, action, progress=progress, durable_boundary=durable_boundary
        )
    assert events == ["boundary", "step", "step"]
    assert progress.first_env_step_invoked is True


def test_boundary_failure_never_enters_env_step() -> None:
    class FakeEnv:
        calls = 0

        def step(self, _action: torch.Tensor) -> tuple:
            self.calls += 1
            raise AssertionError("env.step must not run")

    def fail() -> None:
        raise OSError("synthetic fsync failure")

    env = FakeEnv()
    progress = collector.AttemptProgress()
    try:
        collector.authoritative_env_step(
            env,
            torch.zeros((1, 16), dtype=torch.float32),
            progress=progress,
            durable_boundary=fail,
        )
    except OSError:
        pass
    else:
        raise AssertionError("expected boundary failure")
    assert env.calls == 0
    assert progress.first_env_step_invoked is False


def test_exact_registered_invocation_order() -> None:
    assert collector.CANDIDATE42_INVOCATION_SEQUENCE == (
        (346, "b", 8),
        (347, "a", 64),
        (347, "b", 64),
        (348, "b", 64),
        (348, "a", 64),
    )
    assert collector.CANDIDATE42_RUN_ORDER == (
        (347, "a"),
        (347, "b"),
        (348, "b"),
        (348, "a"),
    )


def test_argument_receipt_is_an_explicit_parent_child_whitelist() -> None:
    with tempfile.TemporaryDirectory() as name:
        stem = Path(name) / "trial"
        values = {
            field: None for field in collector.USER_VISIBLE_ARGUMENT_FIELDS
        }
        values.update(
            {
                "device": "cuda:0",
                "enable_cameras": False,
                "experience": "",
                "fixed_direction": Path("fixed.pt"),
                "headless": True,
                "kit_args": "--/app/extensions/fsWatcherEnabled=false",
                "livestream": -1,
                "num_envs": 8,
                "output_stem": stem,
                "pre_tanh_l2_cap": 1.0,
                "raw_z_abs_cap": 1.0,
                "replicate": "b",
                "search_checkpoint": Path("search.pt"),
                "seed": 346,
                "token_component_cap": 1.0,
                "token_scale": 1.0,
                "distal_component_cap": 1.0,
                "distal_scale": 1.0,
                "v6_checkpoint": Path("v6"),
                "verify_steps": 15,
                "window_steps": 50,
            }
        )
        parent = argparse.Namespace(**values)
        child = argparse.Namespace(
            **values,
            xr=False,
            verbose=False,
            renderer_mode="balanced",
            _candidate42_lock_fd=99,
        )
        assert collector._argument_receipt(parent, stem) == collector._argument_receipt(
            child, stem
        )
        child.device = "cuda:1"
        assert collector._argument_receipt(parent, stem) != collector._argument_receipt(
            child, stem
        )
def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    test_normalized_run_collection_ast_is_bit_exact_to_candidate41()
    test_durable_boundary_precedes_first_step_and_runs_once()
    test_boundary_failure_never_enters_env_step()
    test_exact_registered_invocation_order()
    test_argument_receipt_is_an_explicit_parent_child_whitelist()
    print("candidate42_public_arm_ramp_collection tests passed")


if __name__ == "__main__":
    main()
