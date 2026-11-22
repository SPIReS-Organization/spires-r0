"""Permanent-water mask from 30-degree ``landWater2020_{lon:04d}_{lat:03d}.tif`` tiles.

The fold's water-blue fallback (min raw blue over water pixels) only runs where a scene
carries ``mask_water``. MODIS/VIIRS scenes bring that flag from their QA; raster scenes
(Landsat / Sentinel-2 / HLS) do not, so without this layer a lake that is dark and
NDSI-positive in every snow-free scene gets no background at all and the inversion
skips it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr

LAND_WATER_TILE_DEG = 30
LAND_WATER_EDGE_BUFFER_DEG = 0.01
LAND_WATER_WATER_VALUE = 0


def _tile_origins(low: float, high: float) -> list[int]:
    first = math.floor(low / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    last = math.floor(high / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    return list(range(first, last + 1, LAND_WATER_TILE_DEG))


def land_water_tile_path(bounds_4326: tuple[float, float, float, float], directory: str | Path) -> Path:
    """The single tile covering ``bounds_4326``; tiles are EPSG:4326 squares named by their south-west corner."""
    minx, miny, maxx, maxy = bounds_4326
    lons, lats = _tile_origins(minx, maxx), _tile_origins(miny, maxy)
    if len(lons) != 1 or len(lats) != 1:
        raise ValueError(f"bounds {bounds_4326} span multiple land/water tiles (lon {lons}, lat {lats})")
    path = Path(directory) / f"landWater2020_{lons[0]:04d}_{lats[0]:03d}.tif"
    if not path.is_file():
        raise FileNotFoundError(f"land/water tile is not staged: {path}")
    return path


def water_mask_from_tiles(template: xr.Dataset, directory: str | Path) -> xr.DataArray:
    """True where the tile says WATER, nearest-resampled onto ``template``'s (y, x) grid.

    ``directory`` has no default on purpose: a silently absent mask reproduces the very
    failure this layer exists to prevent.
    """
    import rioxarray  # noqa: F401

    if template.rio.crs is None:
        raise ValueError("template must carry a CRS")
    target = xr.DataArray(np.zeros((template.sizes["y"], template.sizes["x"]), np.uint8), dims=("y", "x"),
                          coords={"y": template["y"], "x": template["x"]}).rio.write_crs(template.rio.crs)
    bounds = tuple(float(b) for b in target.rio.transform_bounds("EPSG:4326"))
    path = land_water_tile_path(bounds, directory)
    minx, miny, maxx, maxy = bounds
    with rioxarray.open_rasterio(path) as tile:
        window = tile.squeeze(drop=True).rio.clip_box(
            minx - LAND_WATER_EDGE_BUFFER_DEG, miny - LAND_WATER_EDGE_BUFFER_DEG,
            maxx + LAND_WATER_EDGE_BUFFER_DEG, maxy + LAND_WATER_EDGE_BUFFER_DEG).load()
        crs = tile.rio.crs
    # Nearest (rioxarray default) is the only correct choice for a categorical mask. Pixels the
    # tile does not cover fill with 255, so they count as not-water rather than water.
    window = window.rio.write_crs(crs).rio.write_nodata(255)
    matched = window.rio.reproject_match(target)
    mask = xr.DataArray(np.asarray(matched.values == LAND_WATER_WATER_VALUE, dtype=bool), dims=("y", "x"),
                        coords={"y": template["y"], "x": template["x"]}, name="mask_water")
    mask.attrs = {"long_name": "Permanent water (True = water)", "source": "landWater2020",
                  "tile": path.name, "resampling": "nearest"}
    return mask
