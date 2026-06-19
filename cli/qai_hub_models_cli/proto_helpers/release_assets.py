# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.common import model_repo_url
from qai_hub_models_cli.proto import info_pb2
from qai_hub_models_cli.proto.platform_pb2 import ChipsetInfo, PlatformInfo
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
from qai_hub_models_cli.proto_helpers._common import fetch_model_proto
from qai_hub_models_cli.proto_helpers.info import get_model_info
from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry
from qai_hub_models_cli.proto_helpers.platform import (
    get_runtime_info,
    resolve_chipset,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    precision_str_to_proto,
    runtime_proto_to_str,
    runtime_str_to_proto,
)
from qai_hub_models_cli.utils import build_table
from qai_hub_models_cli.versions import CURRENT_VERSION


class AssetNotFoundError(FileNotFoundError):
    def __init__(self, *args: object, model_sharing_restricted: bool = False) -> None:
        self.model_sharing_restricted = model_sharing_restricted
        super().__init__(*args)


# Tool version proto fields paired with their display labels, in display order.
_TOOL_VERSION_LABELS: list[tuple[str, str]] = [
    ("qairt", "QAIRT"),
    ("onnx", "ONNX"),
    ("onnx_runtime", "ONNX Runtime"),
    ("tflite", "TFLite"),
    ("litert", "LiteRT"),
    ("ai_hub_models", "AI Hub Models"),
]


def format_tool_versions(tool_versions: ToolVersions) -> str:
    """Format the set tool versions as a comma-separated ``Label X.Y`` string."""
    parts = [
        f"{label} {value}"
        for field, label in _TOOL_VERSION_LABELS
        if (value := getattr(tool_versions, field))
    ]
    return ", ".join(parts) if parts else "—"


@functools.lru_cache(maxsize=1)
def get_model_release_assets(
    model: str,
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ModelReleaseAssets:
    """
    Fetch and cache the model release assets protobuf for a given model.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name
        (e.g. ``"MobileNet-v2"``).
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local release assets protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ModelReleaseAssets
        Parsed release assets protobuf containing download URLs for
        each available runtime, precision, and chipset combination.

    Raises
    ------
    KeyError
        If *model* is not found in the manifest for *version*.
    """
    proto = fetch_model_proto(
        model,
        version,
        ModelReleaseAssets,
        cache_filename="release_assets.pb",
        manifest_url_field="release_assets",
        source_getter="get_release_assets_proto",
        local_path=local_path,
    )

    if not proto.assets:
        entry = get_manifest_entry(model, version)
        info_proto = get_model_info(model, version)

        if info_proto.HasField("llm_details"):
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_COMING_SOON
            ):
                raise AssetNotFoundError(
                    f"No pre-compiled model files are available for {entry.display_name}, but assets are coming soon. Stay tuned!"
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_US
            ):
                raise AssetNotFoundError(
                    f"If you have interest in downloading {entry.display_name}, reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_DOWNLOAD
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for download. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )
            if (
                info_proto.llm_details.call_to_action
                == info_pb2.ModelInfo.LLMDetails.CALL_TO_ACTION_CONTACT_FOR_PURCHASE
            ):
                raise AssetNotFoundError(
                    f"Pre-compiled model files for {entry.display_name} are available for purchase. Reach out to us at qai-hub-support@qti.qualcomm.com."
                )

        if info_proto.restrict_model_sharing:
            raise AssetNotFoundError(
                f"No pre-compiled model files for {entry.display_name} are available due to licensing"
                " restrictions. You can use the AI Hub Models package to manually"
                " export the model. See"
                f" {model_repo_url(entry.id, version)} for export instructions.",
                model_sharing_restricted=True,
            )

        raise AssetNotFoundError(
            f"No pre-compiled model files are available for {entry.display_name}. Reach out to us at"
            " qai-hub-support@qti.qualcomm.com."
        )

    return proto


def format_release_assets_table(
    release_assets: ModelReleaseAssets,
    chipsets: Iterable[ChipsetInfo],
    model: str | None = None,
    title: str | None = None,
) -> str:
    """Format a table of download options for a model.

    *chipsets* is used to display each asset's chipset by its marketing name.
    When *model* is provided, the table is followed by ``fetch`` instructions
    for that model; otherwise only the table is returned.
    """
    chipset_names = {c.name: c.marketing_name for c in chipsets}

    grouped: dict[tuple[str, str, str], list[str | None]] = {}
    for asset in release_assets.assets:
        prec = precision_proto_to_str(asset.precision)
        rt = runtime_proto_to_str(asset.runtime)
        sdk = (
            format_tool_versions(asset.tool_versions)
            if asset.HasField("tool_versions")
            else "—"
        )
        key = (prec, rt, sdk)
        chipset = (
            chipset_names.get(asset.chipset, asset.chipset)
            if asset.HasField("chipset")
            else None
        )
        grouped.setdefault(key, []).append(chipset)

    rows = []
    for (prec, rt, sdk), grouped_chipsets in grouped.items():
        if all(c is None for c in grouped_chipsets):
            chipset_str = "Universal"
        else:
            chipset_str = ", ".join(sorted(c for c in grouped_chipsets if c))
        rows.append([prec, rt, chipset_str, sdk])
    table = build_table(
        ["Precision", "Runtime", "Chipsets", "SDK Versions"],
        rows,
        wrap_column="Chipsets",
        title=title,
        wrap_on_commas=True,
    )

    if model is None:
        return table

    chipset_or_device_flag = (
        " [ -c <chipset> || -d <device> ]"
        if any(asset.HasField("chipset") for asset in release_assets.assets)
        else ""
    )
    lines = [
        table,
        (
            "Use `qai_hub_models devices` to list the devices compatible with each chipset.\n"
            if chipset_or_device_flag
            else ""
        )
        + f"Run `qai_hub_models fetch {model} -r <runtime> -p <precision>"
        f"{chipset_or_device_flag}` to download the model.\n",
    ]
    return "\n".join(lines)


def get_model_asset_details(
    release_assets: ModelReleaseAssets,
    platform: PlatformInfo,
    runtime: Runtime.ValueType | str,
    precision: Precision.ValueType | str,
    chipset: str | None = None,
    device: str | None = None,
) -> ModelReleaseAssets.AssetDetails:
    """
    Look up a specific asset from a model's release assets.

    If the runtime is AOT-compiled, *chipset* (or *device*)
    is required and a chipset-specific asset is returned. Otherwise
    *chipset* is ignored and a universal asset is returned.

    Parameters
    ----------
    release_assets
        The model's release assets to search.
    platform
        Platform registry used to resolve *chipset*/*device* and the runtime.
    runtime
        Runtime enum value (e.g. ``RUNTIME_TFLITE``) or string
        (e.g. ``"tflite"``).
    precision
        Precision enum value (e.g. ``PRECISION_FLOAT``) or string
        (e.g. ``"float"``).
    chipset
        Chipset reference: canonical ID, name, or alias. Resolved to
        the canonical chipset ID. Required for AOT-compiled runtimes, ignored
        otherwise.
    device
        Device name to select the asset by; resolved to its chipset. Mutually
        exclusive with *chipset*.

    Returns
    -------
    ModelReleaseAssets.AssetDetails
        Matching asset entry with download URL, tool versions, etc.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    KeyError
        If no matching asset is found, if *chipset* is missing for an
        AOT-compiled runtime, or if *chipset*/*device* is not known.
    """
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")
    if device is not None or chipset is not None:
        chipset = resolve_chipset(platform, device=device, chipset=chipset).name

    runtime_val = runtime_str_to_proto(runtime)
    precision_val = precision_str_to_proto(precision)
    rt_info = get_runtime_info(platform, runtime_val)
    model = release_assets.model_id

    errmsg: str | None = None
    if rt_info.is_aot_compiled:
        if chipset is None:
            errmsg = f"Chipset is required for AOT-compiled runtime {runtime!r}.\n\n"
        else:
            for asset in release_assets.assets:
                if (
                    asset.runtime == runtime_val
                    and asset.precision == precision_val
                    and asset.chipset == chipset
                ):
                    return asset
    else:
        for asset in release_assets.assets:
            if asset.runtime == runtime_val and asset.precision == precision_val:
                return asset

    errmsg = errmsg or (
        f"No asset found for model={model!r} with runtime={runtime!r}, "
        f"precision={precision!r}, chipset={chipset!r}.\n"
        f"The model was found, but not with the requested runtime, precision, or chipset.\n\n"
    )
    errmsg += f"The following are valid fetch options for {model}:\n"
    errmsg += format_release_assets_table(release_assets, platform.chipsets)
    raise AssetNotFoundError(errmsg)
