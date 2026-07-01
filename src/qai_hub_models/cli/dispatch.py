# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side dispatcher for ``qai_hub_models <script> <model_id>`` subcommands.

Imports ``qai_hub_models.models.<model_id>.<script>``, builds its native
argparse parser via ``build_parser()``, parses the forwarded args, then calls
``main(parsed_args)``.

The lean CLI is responsible for resolving display name -> model id and
checking the model exists in the installed package before calling in.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from qai_hub_models.utils.asset_loaders import ASSET_CONFIG


def run_model_script(model_id: str, script: str, forwarded: list[str]) -> None:
    """Import ``qai_hub_models.models.<model_id>.<script>`` and run it.

    Parameters
    ----------
    model_id
        Model directory name (e.g. ``"mobilenet_v2"``). Must already be
        validated against ``MODEL_IDS`` by the caller.
    script
        Module name inside the model directory (``"export"`` or ``"evaluate"``).
    forwarded
        Argv tail handed to the model's parser.
    """
    module_path = f"qai_hub_models.models.{model_id}.{script}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        readme_url = (
            str(ASSET_CONFIG.get_qaihm_repo(model_id, relative=False)) + "#readme"
        )
        sys.exit(
            f"Failed to import {module_path}: {e}.\n"
            f"See the model's README for setup instructions: {readme_url}"
        )

    if not hasattr(module, "build_parser") or not hasattr(module, "main"):
        sys.exit(
            f"{module_path} doesn't expose the qai-hub-models CLI dispatch "
            f"interface (build_parser + main). "
            f"Use `python -m {module_path}` instead."
        )

    parser: argparse.ArgumentParser = module.build_parser(cli_mode=True)
    parser.prog = f"qai_hub_models {script} {model_id}"
    module.main(parser.parse_args(forwarded))
