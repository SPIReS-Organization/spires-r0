"""Validation and explicit metadata-only repair of georeferenced R0 grids."""

import numpy as np
from pyproj import CRS
import xarray as xr

from spires_contract import validate_persisted_grid


def validate_r0_grid(dataset: xr.Dataset) -> None:
    """Check center coordinates against the persisted grid when georeferenced.

    Unreferenced, in-memory pixel grids remain supported by the compositing API.
    A declared spatial reference must be complete and internally consistent.
    """
    if "spatial_ref" in dataset or any(
        key in dataset.attrs for key in ("crs_wkt", "geospatial_transform")
    ):
        validate_persisted_grid(dataset)


def repair_r0_grid_metadata(
    dataset: xr.Dataset, reference_grid: xr.Dataset
) -> xr.Dataset:
    """Relabel a proven legacy edge grid without resampling R0 data.

    The caller must supply an independently verified source grid. Recorded tile
    bounds, dimensions and CRS must agree with it, and existing coordinates must
    describe either its pixel centers or its legacy upper-left pixel edges.
    Arbitrary shifts, different tiles and CRS changes are rejected.
    """
    validate_persisted_grid(reference_grid)
    for dim in ("x", "y"):
        if dataset.sizes.get(dim) != reference_grid.sizes.get(dim):
            raise ValueError(f"R0 {dim} dimension differs from the reference grid")
    if "spatial_ref" not in dataset:
        raise ValueError("R0 repair requires its original spatial reference")

    def crs(value):
        attrs = value.spatial_ref.attrs
        return CRS.from_user_input(attrs.get("crs_wkt") or attrs.get("spatial_ref"))

    if crs(dataset) != crs(reference_grid):
        raise ValueError("R0 CRS differs from the reference grid")
    parts = np.fromstring(reference_grid.spatial_ref.attrs["GeoTransform"], sep=" ")
    tolerance = max(1e-6, max(abs(parts[1]), abs(parts[5])) * 1e-8)
    expected_bounds = {
        "geospatial_upper_left_meters": [parts[0], parts[3]],
        "geospatial_lower_right_meters": [
            parts[0] + dataset.sizes["x"] * parts[1],
            parts[3] + dataset.sizes["y"] * parts[5],
        ],
    }
    for key, expected in expected_bounds.items():
        recorded = np.asarray(dataset.attrs.get(key, []), dtype=np.float64)
        if recorded.shape != (2,) or not np.allclose(
            recorded, expected, rtol=0.0, atol=tolerance
        ):
            raise ValueError(f"R0 recorded {key} differs from the reference grid")

    conventions = []
    for dim, origin, step in (("x", parts[0], parts[1]), ("y", parts[3], parts[5])):
        values = dataset[dim].values
        centers = reference_grid[dim].values
        edges = origin + np.arange(values.size, dtype=np.float64) * step
        if np.allclose(values, centers, rtol=0.0, atol=tolerance):
            conventions.append("centers")
        elif np.allclose(values, edges, rtol=0.0, atol=tolerance):
            conventions.append("upper_left_edges")
        else:
            raise ValueError(f"R0 {dim} is neither the reference centers nor legacy edges")
    if conventions[0] != conventions[1]:
        raise ValueError("R0 axes use inconsistent coordinate conventions")

    result = dataset.assign_coords(
        x=reference_grid.x.copy(), y=reference_grid.y.copy(),
        spatial_ref=reference_grid.spatial_ref.copy(),
    )
    result.attrs = dict(dataset.attrs)
    result.attrs.update(expected_bounds)
    result.attrs["geospatial_transform"] = reference_grid.spatial_ref.attrs["GeoTransform"]
    result.attrs["spires_r0_grid_repair"] = (
        f"metadata only: {conventions[0]} to canonical pixel centers; "
        "full-precision transform from independently verified source grid; no resampling"
    )
    validate_r0_grid(result)
    return result
