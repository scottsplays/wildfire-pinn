"""Turn a level-set field into a Cartesian GeoJSON perimeter.

Coordinates are easting and northing in meters. They are not longitude and latitude.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import orient

from wildfire_pinn.geometry import signed_distance, zero_contours


def rasterize_polygon(geojson: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    geom = shape(geojson)
    if geom.geom_type != "Polygon":
        raise ValueError("only a Polygon perimeter is accepted")
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or geom.geom_type != "Polygon":
        raise ValueError("perimeter is not a valid polygon")
    return signed_distance(geom, x, y)


def burned_area_m2(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    cell = float(x[1] - x[0]) * float(y[1] - y[0])
    return float(np.sum(np.asarray(phi) < 0) * cell)


def perimeter_geometry(phi: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Longest valid zero contour, as a Polygon. Extra contours become a MultiPolygon."""
    polygons = []
    for line in zero_contours(phi, x, y):
        if len(line) < 4:
            continue
        coords = [(float(px), float(py)) for px, py in line]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        if polygon.geom_type == "MultiPolygon":
            polygons.extend(polygon.geoms)
        elif polygon.geom_type == "Polygon":
            polygons.append(polygon)
    polygons = [orient(polygon, sign=1.0) for polygon in polygons if polygon.area > 0 and polygon.is_valid]
    if not polygons:
        raise RuntimeError("the forecast has no closed zero contour")
    polygons.sort(key=lambda item: item.area, reverse=True)
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": [_ring(polygons[0])]}
    multi = MultiPolygon(polygons)
    return {
        "type": "MultiPolygon",
        "coordinates": [[_ring(polygon)] for polygon in multi.geoms],
    }


def _ring(polygon: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]


def geometry_is_valid(geojson: dict) -> bool:
    geom = shape(geojson)
    return bool(geom.is_valid and not geom.is_empty and geom.area > 0)
