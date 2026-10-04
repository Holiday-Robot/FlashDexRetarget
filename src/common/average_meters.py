"""Scalar-metric accumulators for training-loop logging."""

from __future__ import annotations

from typing import Any


class AverageMeter:
    """Accumulate scalar metrics and report their mean since the last reset."""

    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._cnt: dict[str, int] = {}

    def update(self, metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            self._sum[k] = self._sum.get(k, 0.0) + float(v)
            self._cnt[k] = self._cnt.get(k, 0) + 1

    def mean(self) -> dict[str, float]:
        return {k: self._sum[k] / self._cnt[k] for k in self._sum}

    def reset(self) -> None:
        self._sum.clear()
        self._cnt.clear()
