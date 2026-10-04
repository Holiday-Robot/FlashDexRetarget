from __future__ import annotations

import itertools

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


def _inside(hulls: list[ConvexHull], pts: np.ndarray, tol: float) -> np.ndarray:
    ins = np.zeros(len(pts), bool)
    for h in hulls:
        ins |= (pts @ h.equations[:, :3].T + h.equations[:, 3] <= tol).all(1)
    return ins


def _surface(h: ConvexHull, n: int, seed: int) -> np.ndarray:
    return trimesh.sample.sample_surface(trimesh.Trimesh(h.points, h.simplices), n, seed=seed)[0]


class _Parts:
    """Merged hulls, each with the unmerged parts it covers: stick-out is measured against those, so a chain of
    merges cannot pile its allowances up."""

    def __init__(self, parts: list[np.ndarray]):
        self.raw = [ConvexHull(np.asarray(p, dtype=np.float64)) for p in parts]
        self.raw_pts = [_surface(h, 1500, i) for i, h in enumerate(self.raw)]
        self.hulls, self.trees, self.boxes, self.members = [], [], [], []
        for i, h in enumerate(self.raw):
            self.add(h, [i])

    def add(self, h: ConvexHull, members: list[int]) -> int:
        v = h.points[h.vertices]
        self.hulls.append(ConvexHull(v))
        self.trees.append(cKDTree(np.concatenate([self.raw_pts[m] for m in members])))
        self.boxes.append((v.min(0), v.max(0)))
        self.members.append(members)
        return len(self.hulls) - 1

    def joint(self, i: int, j: int) -> ConvexHull:
        return ConvexHull(np.concatenate([self.hulls[i].points, self.hulls[j].points]))

    def stick_out(self, i: int, j: int, limit: float) -> tuple[float, ConvexHull | None]:
        """Max distance from the joint hull's surface to the unmerged parts of the two (inf when farther apart
        than 2 * limit)."""
        (lo_i, hi_i), (lo_j, hi_j) = self.boxes[i], self.boxes[j]
        if np.maximum(0, np.maximum(lo_i - hi_j, lo_j - hi_i)).max() > 2 * limit:
            return np.inf, None
        h = self.joint(i, j)
        x = _surface(h, 1000, i * 7919 + j)
        x = x[~_inside([self.raw[m] for m in self.members[i] + self.members[j]], x, 1e-6)]
        if not len(x):
            return 0.0, h
        return float(np.minimum(self.trees[i].query(x)[0], self.trees[j].query(x)[0]).max()), h


def _greedy(parts: _Parts, alive: set[int], cost, stop) -> set[int]:
    costs = {(i, j): cost(i, j) for i, j in itertools.combinations(sorted(alive), 2)}
    while costs and len(alive) > 1:
        (i, j), (c, h) = min(costs.items(), key=lambda kv: kv[1][0])
        if stop(c, len(alive)):
            break
        k = parts.add(h, parts.members[i] + parts.members[j])
        alive -= {i, j}
        costs = {p: v for p, v in costs.items() if i not in p and j not in p}
        costs.update({(o, k): cost(o, k) for o in alive})
        alive.add(k)
    return alive


def merge_parts(parts: list[np.ndarray], threshold: float, max_parts: int) -> list[np.ndarray]:
    """Convex parts (vertex arrays) after CoACD's merge: cheapest first, pairs whose joint hull sticks out of their
    unmerged parts by at most threshold * half the overall extent; then, above max_parts, the pairs sticking out
    least."""
    if len(parts) < 2:
        return parts
    parts = [np.asarray(p, dtype=np.float64) for p in parts]
    parts = sorted((p[np.lexsort(p.T[::-1])] for p in parts), key=lambda p: (*np.round(p.mean(0), 6), len(p)))
    allp = np.concatenate(parts)
    half = float((allp.max(0) - allp.min(0)).max()) / 2
    limit = threshold * half
    p = _Parts(parts)
    alive = _greedy(p, set(range(len(parts))), lambda i, j: p.stick_out(i, j, limit), lambda c, n: c > limit)
    if len(alive) > max_parts:
        alive = _greedy(p, alive, lambda i, j: p.stick_out(i, j, half), lambda c, n: n <= max_parts)
    return [p.hulls[i].points for i in sorted(alive)]
