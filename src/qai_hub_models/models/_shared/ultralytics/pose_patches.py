# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from typing import cast

import torch
from ultralytics.nn.modules.head import Pose
from ultralytics.nn.tasks import PoseModel

from qai_hub_models.models._shared.ultralytics.detect_patches import (
    patched_ultryaltics_det_head_forward,
    patched_ultryaltics_det_head_inference,
)


def patch_ultralytics_pose_head(model: PoseModel) -> None:
    """
    Patch the pose model head for export / quantization compatibility.

    After patching the model returns:
        boxes:
            Shape [batch, 4, num_anchors]  (x, y, w, h in pixel space)
        scores:
            Shape [batch, 1, num_anchors]  (objectness confidence)
        keypoints:
            Shape [batch, num_keypoints * 3, num_anchors]
            Flattened as [x0, y0, v0, x1, y1, v1, ...] per anchor.
    """
    head = cast(Pose, model.model[-1])

    # Makes the model traceable and disables built-in post-processing.
    head.export = True
    head.end2end = False

    # Patch inference head to skip concat of boxes & scores (required for int8).
    head._inference = functools.partial(patched_ultryaltics_det_head_inference, head)  # type: ignore[assignment]
    head.forward = functools.partial(patched_ultralytics_pose_head_forward, head)  # type: ignore[assignment]


def patched_ultralytics_pose_head_forward(
    self: Pose, x: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adjusted Pose::forward that returns boxes, scores, and keypoints as
    separate tensors (no concatenation), enabling int8 quantization.

    Parameters
    ----------
    self
        Pose head module instance.
    x
        List of feature maps from different detection layers.

    Returns
    -------
    boxes : torch.Tensor
        Decoded bounding boxes. Shape [batch, 4, num_anchors].
    scores : torch.Tensor
        Class scores after sigmoid. Shape [batch, num_classes, num_anchors].
    keypoints : torch.Tensor
        Decoded keypoint predictions. Shape [batch, num_keypoints * 3, num_anchors].
        x,y are in pixel space; visibility is sigmoid-activated.
    """
    bs = x[0].shape[0]

    # Collect keypoint predictions from each detection level.
    kpt = torch.cat(
        [self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], dim=2
    )

    # Reuse the patched detection head for boxes + scores.
    boxes, scores = cast(
        tuple[torch.Tensor, torch.Tensor],
        patched_ultryaltics_det_head_forward(self, x),
    )

    # Decode keypoints to pixel space (x,y) with sigmoid-activated visibility.
    # kpts_decode expects shape [batch, nk, num_anchors] and returns the same shape.
    kpt_decoded = self.kpts_decode(kpt)

    return boxes, scores, kpt_decoded
