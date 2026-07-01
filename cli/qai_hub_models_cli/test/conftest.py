# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import socket
import sys
from collections.abc import Generator
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any

import pytest

from qai_hub_models_cli.versions import get_published_versions, get_supported_versions

_HEAVY_PACKAGE = "qai_hub_models"


class _BlockHeavyImport(MetaPathFinder):
    """
    Refuse to import the heavy ``qai_hub_models`` package.
    No tests should require `qai_hub_models` are installed.
    """

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> ModuleSpec | None:
        if fullname == _HEAVY_PACKAGE or fullname.startswith(_HEAVY_PACKAGE + "."):
            raise ImportError(
                f"Importing '{fullname}' is not allowed. You should mock/monkeypatch any {_HEAVY_PACKAGE} module used by the CLI tests."
            )
        return None


@pytest.fixture(autouse=True)
def _block_heavy_import() -> Generator[None, None, None]:
    """Fail any test that triggers a real import of the heavy ``qai_hub_models``."""
    finder = _BlockHeavyImport()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


@pytest.fixture(autouse=True)
def _clear_version_cache() -> Generator[None, None, None]:
    """Clear lru_caches on version functions between tests."""
    get_published_versions.cache_clear()
    get_supported_versions.cache_clear()
    yield
    get_published_versions.cache_clear()
    get_supported_versions.cache_clear()


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fail tests if they make a real network connection.
    All network paths are expected to be mocked.
    """

    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Network access is disabled in CLI tests. Mock the download / request "
            "instead of hitting the network."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
