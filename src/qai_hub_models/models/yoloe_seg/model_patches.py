# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
import torch.nn.functional as F
from ultralytics.nn.modules.head import BNContrastiveHead


class BNContrastiveHeadInf(BNContrastiveHead):
    def __init__(self, source: BNContrastiveHead) -> None:
        torch.nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Patched forward - replaces einsum with equivalent ops for QAIRT compatibility.

        Parameters
        ----------
        x
            Image features (b, c, h, w).
        w
            Text features (b, k, c).

        Returns
        -------
        torch.Tensor
            Similarity scores (b, k, h, w).

        Notes
        -----
        Source : https://github.com/ultralytics/ultralytics/blob/f4d0fda2cb5aa9925f3b08c56fc883664cf3872d/ultralytics/nn/modules/block.py#L811
        """
        x = self.norm(x)  # BN normalization (differs from ContrastiveHead)
        w = F.normalize(w, dim=-1, p=2)

        # Begin Qualcomm modification
        # Replaced torch.einsum("bchw,bkc->bkhw", x, w) using permute and matmul
        # x: (b, c, h, w) -> (b, h, w, c)
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        # w: (b, k, c) -> (b, c, k)
        w = w.permute(0, 2, 1)  # (b, c, k)
        # matmul: (b, h, w, c) x (b, c, k) -> (b, h, w, k)
        x = torch.matmul(x, w)  # (b, h, w, k)
        # Restore to (b, k, h, w)
        x = x.permute(0, 3, 1, 2)  # (b, k, h, w)
        # End Qualcomm modification

        return x * self.logit_scale.exp() + self.bias
