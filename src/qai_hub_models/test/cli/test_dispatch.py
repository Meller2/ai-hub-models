# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the heavy-side export/evaluate dispatcher."""

from __future__ import annotations

import argparse
import types
from unittest.mock import MagicMock, patch

import pytest

from qai_hub_models.cli.dispatch import run_model_script


def _fake_module(
    main_calls: list[argparse.Namespace] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> types.SimpleNamespace:
    """Build a stand-in for a model's export/evaluate module."""
    if parser is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", default="defaultdev")
        parser.add_argument("--skip-profiling", action="store_true")

    def _main(args: argparse.Namespace) -> None:
        if main_calls is not None:
            main_calls.append(args)

    return types.SimpleNamespace(build_parser=lambda cli_mode=False: parser, main=_main)


def test_dispatch_parses_with_model_parser_and_invokes_main() -> None:
    """Happy path: dispatcher parses with the model's parser and passes parsed args to main()."""
    main_calls: list[argparse.Namespace] = []
    fake_module = _fake_module(main_calls=main_calls)

    with patch(
        "qai_hub_models.cli.dispatch.importlib.import_module",
        return_value=fake_module,
    ):
        run_model_script(
            "fake_model", "export", ["--device", "S25", "--skip-profiling"]
        )

    assert len(main_calls) == 1
    assert main_calls[0].device == "S25"
    assert main_calls[0].skip_profiling is True


def test_module_not_found_exits_with_readme_hint() -> None:
    """ModuleNotFoundError on import surfaces a README pointer."""
    with (
        patch(
            "qai_hub_models.cli.dispatch.importlib.import_module",
            side_effect=ModuleNotFoundError("missing dep"),
        ),
        pytest.raises(SystemExit, match=r"README"),
    ):
        run_model_script("fake_model", "export", [])


def test_module_without_dispatch_interface_exits() -> None:
    """Imported module missing build_parser or main -> exit pointing at python -m."""
    # Module that has main() but not build_parser() (e.g. hand-written export.py).
    fake_module = MagicMock(spec=["main"])

    with (
        patch(
            "qai_hub_models.cli.dispatch.importlib.import_module",
            return_value=fake_module,
        ),
        pytest.raises(SystemExit, match=r"python -m"),
    ):
        run_model_script("fake_model", "export", [])
