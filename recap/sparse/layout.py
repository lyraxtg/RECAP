"""Reorder a native token sequence into [Hilbert-ordered video | text] and back."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Layout:
    permutation: torch.Tensor  # reordered video position -> native video position
    inverse: torch.Tensor
    video_tokens: int
    text_tokens: int
    text_first: bool
    _gather_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_permutation(
        cls,
        permutation: np.ndarray | torch.Tensor,
        *,
        text_tokens: int,
        text_first: bool,
    ) -> "Layout":
        p = torch.as_tensor(permutation, dtype=torch.long, device="cpu")
        if p.ndim != 1 or torch.unique(p).numel() != p.numel():
            raise ValueError("invalid layout permutation")
        return cls(
            permutation=p,
            inverse=torch.argsort(p),
            video_tokens=p.numel(),
            text_tokens=int(text_tokens),
            text_first=bool(text_first),
        )

    @property
    def video_offset(self) -> int:
        """Position of the first video token in the native sequence."""
        return self.text_tokens if self.text_first else 0

    def _gather_indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather indices for reorder and restore, cached per device."""
        key = str(device)
        cached = self._gather_cache.get(key)
        if cached is not None:
            return cached
        nv, tt = self.video_tokens, self.text_tokens
        text_pos = torch.arange(tt, dtype=torch.long)
        if self.text_first:
            # input [text | video_native] -> output [video_reordered | text]
            fwd = torch.cat((self.permutation + tt, text_pos))
            # input [video_reordered | text] -> output [text | video_native]
            inv = torch.cat((text_pos + nv, self.inverse))
        else:
            # input [video_native | text] -> output [video_reordered | text]
            fwd = torch.cat((self.permutation, text_pos + nv))
            # input [video_reordered | text] -> output [video_native | text]
            inv = torch.cat((self.inverse, text_pos + nv))
        pair = (fwd.to(device), inv.to(device))
        self._gather_cache[key] = pair
        return pair

    def video_permutation(self, device: torch.device) -> torch.Tensor:
        """The video permutation on ``device``, cached."""
        key = f"video:{device}"
        cached = self._gather_cache.get(key)
        if cached is None:
            cached = self.permutation.to(device)
            self._gather_cache[key] = cached
        return cached

    def _check(self, x: torch.Tensor) -> None:
        if x.shape[-2] != self.video_tokens + self.text_tokens:
            raise ValueError(
                f"sequence {x.shape[-2]} != video+text "
                f"{self.video_tokens + self.text_tokens}"
            )

    def reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Native model order -> [reordered video | text]."""
        self._check(x)
        fwd, _ = self._gather_indices(x.device)
        return x.index_select(-2, fwd)

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        """[reordered video | text] -> native model order."""
        self._check(x)
        _, inv = self._gather_indices(x.device)
        return x.index_select(-2, inv)
