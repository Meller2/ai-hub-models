# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.datasets.bsd import BSD300Dataset
from qai_hub_models.datasets.common import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

NUM_TEST_IMAGES = 100


class BSD100Dataset(BSD300Dataset):
    """BSD100 super-resolution benchmark dataset.

    The 100-image test split of the BSDS300 dataset, commonly used to
    benchmark 4x super-resolution models.

    Published here: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        scaling_factor: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        split
            Dataset split to use. BSD100 only has a test split; any split is mapped to TEST.
        input_spec
            Model input spec; determines the spatial dimensions used when resizing images.
        scaling_factor
            Super-resolution upscaling factor (e.g. 4 for 4x SR).
        """
        super().__init__(
            split=DatasetSplit.TEST,
            input_spec=input_spec,
            scaling_factor=scaling_factor,
        )
        self.image_files: list[str] = sorted(self.image_files)[:NUM_TEST_IMAGES]

    def __len__(self) -> int:
        """BSD100 is always the 100-image test split."""
        return NUM_TEST_IMAGES

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
            split_description="test split (100 images)",
        )
