# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from qai_hub_models.scripts import file_scorecard_regression_issue as mod
from qai_hub_models.scripts.file_scorecard_regression_issue import (
    MAX_ISSUE_BODY_LEN,
    build_issue_body,
)

# --- Fixtures ---
# Keys match the canonical PrettyTable column names emitted by
# performance_diff.py and numerics_diff.py (no env suffix). The issue builder
# re-tags them with the deployment of each side at render time.

PERF_REGRESSIONS = [
    {
        "Model ID": "resnet50",
        "Precision": "float",
        "Component": "resnet50",
        "Device": "Snapdragon 8 Gen 3",
        "Runtime": "TFLITE",
        "Prev Inference time": "10.0",
        "New Inference time": "20.0",
        "Kx slower": "2.0",
        "Job ID": "jnew789",
        "Compile Job ID": "jcnew789",
        "Previous Job ID": "jprev789",
        "Previous Compile Job ID": "jcprev789",
    },
    {
        "Model ID": "mobilenet",
        "Precision": "w8a8",
        "Component": "mobilenet",
        "Device": "Snapdragon 8 Elite",
        "Runtime": "ONNX",
        "Prev Inference time": "5.0",
        "New Inference time": "15.0",
        "Kx slower": "3.0",
        "Job ID": "jnew012",
        "Compile Job ID": "jcnew012",
        "Previous Job ID": "jprev012",
        "Previous Compile Job ID": "jcprev012",
    },
]

NUMERICS_REGRESSIONS = [
    {
        "Model ID": "yolov8_det",
        "Dataset Name": "coco-2017",
        "Metric Name": "mAP",
        "Device": "Snapdragon 8 Gen 3",
        "Precision": "float",
        "Runtime": "TFLITE",
        "FP Accuracy": "45.2 mAP",
        "Device Accuracy": "38.1 mAP",
        "Previous FP Accuracy": "45.2 mAP",
        "Previous Device Accuracy": "42.5 mAP",
        "Inference Job ID": "jinew_yolo",
        "Previous Inference Job ID": "jiprev_yolo",
    },
]


# --- Tests ---


def test_build_issue_body() -> None:
    """Full template rendering with both perf and numerics regressions."""
    body = build_issue_body(
        PERF_REGRESSIONS,
        NUMERICS_REGRESSIONS,
        "https://run",
        "https://perf",
        "https://num",
    )
    # Both sections present
    assert "## Performance Regressions" in body
    assert "## Numerics Regressions" in body
    # Perf data rendered
    assert "resnet50" in body
    assert "mobilenet" in body
    # Job IDs rendered as markdown links
    assert "[jnew789]" in body
    # Numerics data rendered
    assert "yolov8_det" in body
    # Links section
    assert "[Scorecard Run](https://run)" in body


def test_build_issue_body_truncates_to_github_limit() -> None:
    """A huge regression list must be trimmed to fit GitHub's 65536-char limit."""
    # ~5000 rows is well over the issue body limit when rendered as a markdown table.
    huge_perf = [{**PERF_REGRESSIONS[0], "Model ID": f"model_{i}"} for i in range(5000)]

    body = build_issue_body(
        huge_perf,
        [],
        "https://run",
        "https://perf",
        "https://num",
    )

    assert len(body) <= MAX_ISSUE_BODY_LEN
    # Some rows are kept and the truncation footer is present.
    assert "model_0" in body
    assert "more performance regression(s) omitted" in body
    assert "[Performance Diff](https://perf)" in body


def test_build_issue_body_truncates_numerics_table() -> None:
    """Truncation drops from the numerics table when it is the larger one."""
    huge_numerics = [
        {**NUMERICS_REGRESSIONS[0], "Model ID": f"model_{i}"} for i in range(5000)
    ]
    body = build_issue_body(
        PERF_REGRESSIONS,  # small
        huge_numerics,
        "https://run",
        "https://perf",
        "https://num",
    )
    assert len(body) <= MAX_ISSUE_BODY_LEN
    assert "more numerics regression(s) omitted" in body
    assert "[Numerics Diff](https://num)" in body


def test_dev_run_default_previous_deployment_links_to_dev() -> None:
    """A dev scorecard run with no --previous-deployment override links the
    previous columns to the dev subdomain.

    Default workflow plumbing: scorecard.yml passes github.ref_name as
    previous_results_branch, so on a dev branch the previous baseline is also
    a dev run — the previous-job links should point to dev.aihub.qualcomm.com,
    and the column header should read "(dev)" not "(prod)".
    """
    body = build_issue_body(
        PERF_REGRESSIONS[:1],
        [],
        "https://run",
        "https://perf",
        "https://num",
        deployment="dev",
    )
    # Both columns get a (dev) suffix in the header.
    assert "Job ID (dev)" in body
    assert "Previous Job ID (dev)" in body
    assert "Previous Compile Job ID (dev)" in body
    # No (prod) anywhere — that would mislead an engineer reading a dev issue.
    assert "(prod)" not in body
    # Both new and previous links route to dev.
    assert "[jnew789](https://dev.aihub.qualcomm.com/jobs/jnew789/)" in body
    assert "[jprev789](https://dev.aihub.qualcomm.com/jobs/jprev789/)" in body


def test_dev_run_with_prod_baseline_routes_previous_to_workbench() -> None:
    """A dev run that explicitly compares against a prod baseline (e.g. a
    workflow_dispatch with previous_results_branch=main) routes the previous
    columns to workbench — and labels them (prod).
    """
    body = build_issue_body(
        PERF_REGRESSIONS[:1],
        [],
        "https://run",
        "https://perf",
        "https://num",
        deployment="dev",
        previous_deployment="workbench",
    )
    # New columns labelled (dev), previous columns labelled (prod).
    assert "Job ID (dev)" in body
    assert "Previous Job ID (prod)" in body
    # Previous links go to workbench; new links go to dev.
    assert "[jnew789](https://dev.aihub.qualcomm.com/jobs/jnew789/)" in body
    assert "[jprev789](https://workbench.aihub.qualcomm.com/jobs/jprev789/)" in body
    assert "https://dev.aihub.qualcomm.com/jobs/jprev789/" not in body


def test_historical_suffixed_columns_pass_through() -> None:
    """JSON keys that already carry an env suffix (e.g. from older S3 dumps)
    are passed through verbatim — the linkifier still routes them by suffix.
    """
    row = {
        **PERF_REGRESSIONS[0],
        "Job ID": "jignored",
    }
    # Replace canonical with a suffixed historical-shape key.
    row.pop("Job ID")
    row["Job ID (prod)"] = "jhistoric"

    body = build_issue_body(
        [row], [], "https://run", "https://perf", "https://num", deployment="dev"
    )
    # Suffixed key wins -> link routes to workbench despite run being on dev.
    assert "[jhistoric](https://workbench.aihub.qualcomm.com/jobs/jhistoric/)" in body


def test_build_issue_body_rejects_malformed_job_ids() -> None:
    """Job IDs that don't match the expected format are not linkified."""
    bad_row = {
        **PERF_REGRESSIONS[0],
        "Job ID": "abc](javascript:alert(1)",
    }
    body = build_issue_body(
        [bad_row],
        [],
        "https://run",
        "https://perf",
        "https://num",
    )
    assert "javascript:" not in body
    assert "alert(1)" not in body


def test_main_writes_output(tmp_path: Path) -> None:
    """End-to-end: main() loads JSON, renders template, writes output file."""
    perf_file = tmp_path / "perf-regressions-2x-2026-01-01.json"
    perf_file.write_text(json.dumps(PERF_REGRESSIONS))
    numerics_file = tmp_path / "numerics-regressions-2026-01-01.json"
    numerics_file.write_text(json.dumps(NUMERICS_REGRESSIONS))
    output_file = tmp_path / "regression-issue.json"

    with mock.patch(
        "sys.argv",
        [
            "file_scorecard_regression_issue.py",
            "--perf-regressions-json",
            str(perf_file),
            "--numerics-regressions-json",
            str(numerics_file),
            "--run-url",
            "https://run",
            "--perf-diff-url",
            "https://perf",
            "--numerics-diff-url",
            "https://num",
            "--output",
            str(output_file),
        ],
    ):
        mod.main()

    assert output_file.exists()
    issue = json.loads(output_file.read_text())
    assert "[Scorecard - Prod]" in issue["title"]
    assert "resnet50" in issue["body"]
    assert "yolov8_det" in issue["body"]
    assert issue["labels"] == ["p1", "scorecard"]
