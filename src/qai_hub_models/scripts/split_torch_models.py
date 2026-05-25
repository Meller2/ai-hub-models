# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
from typing import Literal

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen, TestRunnerSplit
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)

# Default max number of PyTorch models per split when auto-splitting
MAX_PT_MODELS_PER_SPLIT = 30

# Type for runs_on values: runner group/labels object, or null
RunsOnValue = dict[Literal["group", "labels"], str | list[str]] | None


def split_torch_models(
    models: set,
    max_pt_splits: int | None = None,
    max_pt_models_per_split: int = MAX_PT_MODELS_PER_SPLIT,
) -> list[dict[str, str | RunsOnValue]]:
    """
    Split models into chunks for parallel processing.

    Static models are all grouped into one split named "static".
    Torch models are split into multiple chunks.

    Parameters
    ----------
    models
        Set of model IDs or special settings (from EnabledModelsEnvvar.get())
    max_pt_splits
        Maximum number of default splits to create for torch models (does not
        include custom splits). If None, automatically calculate based on
        max_pt_models_per_split.
    max_pt_models_per_split
        Maximum number of models per default split when auto-calculating
        num_default_splits (does not include custom splits).

    Returns
    -------
    list[dict[str, str | RunsOnValue]]
        List of dicts with 'split_name', 'models', and 'runs_on' keys for each split.
    """
    torch_models, static_models = validate_and_split_enabled_models(models)

    splits: list[dict[str, str | RunsOnValue]] = []

    # Add all static models as one split
    if static_models:
        splits.append(
            {
                "split_name": "static",
                "models": ",".join(sorted(static_models)),
            }
        )

    # Split torch models into chunks
    all_torch_models = sorted(torch_models)
    if all_torch_models:
        # Group models by test_split
        custom_splits: dict[TestRunnerSplit, list[str]] = {}
        # Divide the AOT and JIT models separately to ensure they are distributed as evenly as possible across splits,
        # since AOT models take much longer to compile and we want to avoid having one split with mostly AOT models and another with mostly JIT models
        all_models_jit = []
        all_models_aot = []
        for model in all_torch_models:
            code_gen = QAIHMModelCodeGen.from_model(model)
            if code_gen.test_split != TestRunnerSplit.DEFAULT:
                custom_splits.setdefault(code_gen.test_split, []).append(model)
            elif code_gen.requires_aot_prepare:
                all_models_aot.append(model)
            else:
                all_models_jit.append(model)

        # Add custom splits
        for split_enum, split_models in sorted(
            custom_splits.items(), key=lambda x: x[0].value
        ):
            splits.append(
                {
                    "split_name": split_enum.name,
                    "models": ",".join(split_models),
                    "runs_on": split_enum.runs_on,
                }
            )

        # Split remaining torch models (JIT + AOT) into chunks
        num_default_splits = math.ceil(
            len(all_models_jit + all_models_aot) / max_pt_models_per_split
        )
        if max_pt_splits is not None:
            num_default_splits = min(num_default_splits, max_pt_splits)

        # Create splits by taking chunks of the JIT and AOT models separately, then combining them
        if num_default_splits > 0:
            jit_split_size = math.ceil(len(all_models_jit) / num_default_splits)
            aot_split_size = math.ceil(len(all_models_aot) / num_default_splits)
            for i in range(num_default_splits):
                jit_start_idx = i * jit_split_size
                jit_end_idx = min((i + 1) * jit_split_size, len(all_models_jit))

                aot_start_idx = i * aot_split_size
                aot_end_idx = min((i + 1) * aot_split_size, len(all_models_aot))

                jit_models_in_split = all_models_jit[jit_start_idx:jit_end_idx]
                aot_models_in_split = all_models_aot[aot_start_idx:aot_end_idx]
                models_in_split = (
                    aot_models_in_split + jit_models_in_split
                )  # AOT models first because they take longer to compile

                if models_in_split:
                    splits.append(
                        {
                            "split_name": f"torch_{i + 1}_of_{num_default_splits}",
                            "models": ",".join(models_in_split),
                        }
                    )

    # If there's only one split and it's an auto-generated default split, simplify the name to "torch"
    _custom_names = {s.name for s in TestRunnerSplit if s != TestRunnerSplit.DEFAULT}
    if len(splits) == 1 and splits[0]["split_name"] not in ("static", *_custom_names):
        splits[0]["split_name"] = "torch"

    for split in splits:
        if "runs_on" not in split:
            split["runs_on"] = None

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split models into chunks for parallel scorecard runs"
    )
    EnabledModelsEnvvar.add_arg(parser)
    parser.add_argument(
        "--max-num-pt-splits",
        type=int,
        default=None,
        help="Maximum number of default splits to create (does not include custom splits).",
    )
    parser.add_argument(
        "--max-models-per-pt-split",
        type=int,
        default=MAX_PT_MODELS_PER_SPLIT,
        help=f"Maximum models per default split (does not include custom splits, default: {MAX_PT_MODELS_PER_SPLIT})",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="json",
        help="Output format: 'json' for pretty JSON, 'github' for GitHub Actions matrix format",
    )

    args = parser.parse_args()
    splits = split_torch_models(
        args.models, args.max_num_pt_splits, args.max_models_per_pt_split
    )
    if args.output_format == "github":
        # Output as a single line JSON for GitHub Actions
        print(json.dumps(splits))
    else:
        # Pretty print JSON
        print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()
