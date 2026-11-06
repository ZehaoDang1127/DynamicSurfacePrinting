"""Native XY patterns only; arbitrary slicer macros are deliberately unsupported."""

import json
from pathlib import Path

import numpy as np


def subdivide(vertices, spacing_mm):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError("path requires at least two finite XY vertices")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("path spacing must be positive")
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        length = float(np.linalg.norm(b - a))
        if length == 0:
            raise ValueError("duplicate adjacent path vertices")
        n = int(np.ceil(length / spacing_mm))
        if n > 100000 or len(result) + n > 100000:
            raise ValueError("path exceeds 100000-point limit")
        result.extend(a + (b - a) * i / n for i in range(1, n + 1))
    return np.asarray(result)


def load_path(path, spacing_mm):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    kind = raw.get("kind")
    if kind == "line" and set(raw) == {"kind", "start_xy_mm", "end_xy_mm"}:
        vertices = [raw["start_xy_mm"], raw["end_xy_mm"]]
    elif kind == "rectangle" and set(raw) == {"kind", "min_xy_mm", "max_xy_mm"}:
        x0, y0 = raw["min_xy_mm"]
        x1, y1 = raw["max_xy_mm"]
        if x1 <= x0 or y1 <= y0:
            raise ValueError("rectangle bounds must increase")
        vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    elif kind == "polyline" and set(raw) == {"kind", "vertices_xy_mm"}:
        vertices = raw["vertices_xy_mm"]
    else:
        raise ValueError("unsupported path schema (line, rectangle, or polyline)")
    return subdivide(vertices, spacing_mm)
