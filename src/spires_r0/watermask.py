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
_NODATA = 255


def _tile_origins(low: float, high: float) -> list[int]:
    first = math.floor(low / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    last = math.floor(high / LAND_WATER_TILE_DEG) * LAND_WATER_TILE_DEG
    return list(range(first, last + 1, LAND_WATER_TILE_DEG))


def land_water_tile_paths(bounds_4326: tuple[float, float, float, float], directory: str | Path) -> list[Path]:
    """Every tile touching ``bounds_4326``; tiles are EPSG:4326 squares named by their south-west corner.

    A Landsat scene is ~2.7 degrees wide, so straddling a tile edge is routine, not an error.
    """
    minx, miny, maxx, maxy = bounds_4326
    paths = [Path(directory) / f"landWater2020_{lon:04d}_{lat:03d}.tif"
             for lat in _tile_origins(miny, maxy) for lon in _tile_origins(minx, maxx)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"land/water tile(s) not staged: {', '.join(str(p) for p in missing)}")
    return paths


def water_mask_from_tiles(template: xr.Dataset, directory: str | Path) -> xr.DataArray:
    """True where the tiles say WATER, nearest-resampled onto ``template``'s (y, x) grid.

    ``directory`` has no default on purpose: a silently absent mask reproduces the very
    failure this layer exists to prevent. Pixels no tile covers count as not-water.
    """
    import rioxarray  # noqa: F401

    if template.rio.crs is None:
        raise ValueError("template must carry a CRS")
    target = xr.DataArray(np.zeros((template.sizes["y"], template.sizes["x"]), np.uint8), dims=("y", "x"),
                          coords={"y": template["y"], "x": template["x"]}).rio.write_crs(template.rio.crs)
    bounds = tuple(float(b) for b in target.rio.transform_bounds("EPSG:4326"))
    minx, miny, maxx, maxy = bounds
    paths = land_water_tile_paths(bounds, directory)
    water = np.zeros(target.shape, bool)
    for path in paths:
        with rioxarray.open_rasterio(path) as tile:
            tile_bounds = tile.rio.bounds()
            # Clip to the overlap of the target footprint and this tile, buffered where the tile allows.
            box = (max(minx - LAND_WATER_EDGE_BUFFER_DEG, tile_bounds[0]), max(miny - LAND_WATER_EDGE_BUFFER_DEG, tile_bounds[1]),
                   min(maxx + LAND_WATER_EDGE_BUFFER_DEG, tile_bounds[2]), min(maxy + LAND_WATER_EDGE_BUFFER_DEG, tile_bounds[3]))
            window = tile.squeeze(drop=True).rio.clip_box(*box).load()
            crs = tile.rio.crs
        window = window.rio.write_crs(crs).rio.write_nodata(_NODATA)
        # Nearest (rioxarray default) is the only correct choice for a categorical mask.
        matched = window.rio.reproject_match(target)
        water |= np.asarray(matched.values == LAND_WATER_WATER_VALUE)
    mask = xr.DataArray(water, dims=("y", "x"), coords={"y": template["y"], "x": template["x"]}, name="mask_water")
    mask.attrs = {"long_name": "Permanent water (True = water)", "source": "landWater2020",
                  "tile": ",".join(p.name for p in paths), "resampling": "nearest"}
    return mask
