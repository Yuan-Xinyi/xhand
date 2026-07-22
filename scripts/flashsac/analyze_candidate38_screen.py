#!/usr/bin/env python3
"""Fail-closed, simulation-free fixed screen for Candidate 38."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


# Reuse the already tested equal-seed Horvitz--Thompson implementation while
# keeping its experiment globals in a private module instance. Importing this
# wrapper therefore cannot mutate the historical Candidate 35 analyzer in the
# same Python process.
_IMPLEMENTATION_NAME = "_candidate38_screen_analysis_impl"
_IMPLEMENTATION_PATH = Path(__file__).with_name(
    "analyze_candidate35_phase1_screen.py"
)
_spec = importlib.util.spec_from_file_location(
    _IMPLEMENTATION_NAME, _IMPLEMENTATION_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load screen implementation: {_IMPLEMENTATION_PATH}")
_analysis = importlib.util.module_from_spec(_spec)
sys.modules[_IMPLEMENTATION_NAME] = _analysis
_spec.loader.exec_module(_analysis)

_analysis.MANIFEST_KIND = "pick_tool_candidate38_screen_manifest_v1"
_analysis.REPORT_KIND = "pick_tool_candidate38_screen_v1"
_analysis.SEEDS = (323, 324, 325)
_analysis.RUN_ORDER = ("323a", "323b", "324b", "324a", "325a", "325b")
_analysis.PLAN_FIELD = "plan"
_analysis.PLAN_SHA_FIELD = "plan_sha256"
_analysis.PLAN_LABEL = "Candidate 38 plan"
_analysis.OUTPUT_PATTERN = (
    "logs/flashsac/pick_tool/48_close_ab_c38_s{seed}_{replicate}/trial"
)
_analysis.ANALYSIS_OUTPUT = "logs/flashsac/pick_tool/48_close_ab_c38_screen.json"
_analysis.PROTECTED_ANALYZER_PATHS = (
    "scripts/flashsac/analyze_candidate35_phase1_screen.py",
    "scripts/flashsac/analyze_candidate38_screen.py",
)
_analysis.PASS_DECISION = "accept_candidate38"
_analysis.FAIL_DECISION = "reject_candidate38"

EVENTS = _analysis.EVENTS
NUM_ENVS = _analysis.NUM_ENVS
REPLICATES = _analysis.REPLICATES
SEEDS = _analysis.SEEDS
compute_screen = _analysis.compute_screen
validate_manifest_payload = _analysis.validate_manifest_payload
analyze = _analysis.analyze


def main() -> None:
    _analysis.main()


if __name__ == "__main__":
    main()
