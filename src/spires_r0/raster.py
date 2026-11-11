"""Landsat / Sentinel-2 / HLS R0 via spires_io raster adapters and the shared streaming engine."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import xarray as xr

from spires_r0.core import (
    R0_BUILD_SIGNATURE_ATTR,
    R0_SOURCE_INVENTORY_DIGEST_ATTR,
    _finalize_composite,
    _r0_product_attrs,
    build_request_digest,
    load_existing_r0_if_valid,
    sanitize_r0_dataset_attrs,
    source_inventory_digest,
    write_r0_dataset_atomically,
)
from spires_r0.fold import RASTER_POLICY, R0Composite, R0Policy, _scene_time
from spires_r0.io import sanitize_netcdf_dataset
from spires_r0.prefetch import one_ahead

log = logging.getLogger(__name__)
OpenScene = Callable[[Path], xr.Dataset]


def prepare_raster_scene_for_r0(scene: xr.Dataset) -> xr.Dataset:
    """spires_io raster scene -> R0 staging shape (valid_r0_mask, acquisition_date). No geometry is fabricated."""
    if "valid" not in scene or "reflectance" not in scene:
        raise ValueError("expected a spires_io raster scene with 'reflectance' and 'valid'")
    if scene.rio.crs is None:
        raise ValueError("raster scene has no CRS")
    out = xr.Dataset(
        {
            "reflectance": scene["reflectance"].transpose("y", "x", "band"),
            "valid_r0_mask": scene["valid"].astype(bool),
        },
        coords={"y": scene["y"], "x": scene["x"], "band": scene["band"]},
        attrs=dict(scene.attrs),
    )
    if "solar_zenith" in scene:
        out["solar_zenith"] = scene["solar_zenith"].astype(np.float32)
    if "acquisition_date" not in out.attrs:
        if "acquisition_time" not in scene.attrs:
            raise ValueError("raster scene has neither 'acquisition_time' nor 'acquisition_date' attrs")
        out.attrs["acquisition_date"] = str(scene.attrs["acquisition_time"])
    return out.rio.write_crs(scene.rio.crs)


def _default_opener(sensor: str, platform: str | None, bands: list[str] | None) -> OpenScene:
    from spires_io.registry import get_sensor_adapter

    adapter = get_sensor_adapter(sensor, platform)
    kwargs = {"bands": bands} if bands else {}
    return lambda d: adapter.open_surface_reflectance(d, **kwargs)


def _align(scene: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Reindex a scene onto the first scene's grid. Pixels with no counterpart become invalid."""
    if scene.rio.crs != template.rio.crs:
        raise ValueError(f"scene CRS {scene.rio.crs} != template CRS {template.rio.crs}")
    if scene["y"].equals(template["y"]) and scene["x"].equals(template["x"]):
        return scene
    res = float(abs(template["x"].values[1] - template["x"].values[0])) if template["x"].size > 1 else 1.0
    aligned = scene.drop_vars("spatial_ref", errors="ignore").reindex(
        y=template["y"].values, x=template["x"].values, method="nearest", tolerance=0.1 * res)
    aligned["valid_r0_mask"] = aligned["valid_r0_mask"].fillna(False).astype(bool)
    return aligned.rio.write_crs(template.rio.crs)


def _inventory_sources(dirs: list[Path]) -> list[Path]:
    """Scene directories plus the regular files inside each. source_inventory_digest only stats
    regular files, so including them makes in-place rewrites of scene data change the digest."""
    out: list[Path] = []
    for d in dirs:
        out.append(d)
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.is_file()))
    return out


def build_raster_r0(
    scene_dirs: Iterable[str | Path],
    *,
    sensor: str,
    r0_path: str | Path,
    platform: str | None = None,
    bands: list[str] | None = None,
    policy: R0Policy = RASTER_POLICY,
    cache_root: str | Path | None = None,
    overwrite: bool = False,
    open_scene: OpenScene | None = None,
    prefetch: bool = True,
    logger: logging.Logger | None = None,
) -> xr.Dataset:
    """Stream spires_io scene directories through the R0 engine and write `r0_path` (SPIReS_R0 netCDF).

    Directories are processed in sorted-name order; the first defines the grid. Because the
    fold's tie-breaks are feed-order independent, the order only fixes the grid and the
    r0_source_index <-> r0_source_ids mapping. The build signature is computed before any scene
    is loaded so an existing product can be reused without touching the data.
    """
    logger = logger or log
    r0_path = Path(r0_path)
    dirs = sorted((Path(d) for d in scene_dirs), key=lambda p: p.name)
    if not dirs:
        raise ValueError("no scene directories given")
    scene_ids = [d.name for d in dirs]
    inventory = source_inventory_digest(_inventory_sources(dirs))
    signature = build_request_digest(
        source_inventory_digest_value=inventory,
        lut_file=None,
        adapter=f"raster:{sensor}:{platform or ''}",
        algorithm_parameters={"policy": asdict(policy)},
        prepare_kwargs={"bands": bands},
    )

    if not overwrite and r0_path.exists():
        existing = load_existing_r0_if_valid(
            r0_path, expected_band_count=None, max_scene_index=None,
            expected_source_inventory_digest=inventory, expected_build_signature=signature)
        if existing is not None:
            logger.info("R0 exists and matches the request, reusing %s", r0_path)
            return existing

    cache = None
    if cache_root is not None:
        from spires_r0.cache import R0Cache

        cache = R0Cache(cache_root)
        if cache.get(signature, r0_path):
            logger.info("R0 cache hit %s -> %s", signature[:12], r0_path)
            with xr.open_dataset(r0_path) as hit:
                return hit.load()

    opener = open_scene or _default_opener(sensor, platform, bands)

    def load(d: Path) -> xr.Dataset:
        return prepare_raster_scene_for_r0(opener(d))

    stream = one_ahead(dirs, load) if prefetch else ((d, load(d)) for d in dirs)
    composite: R0Composite | None = None
    template: xr.Dataset | None = None
    times: list[np.int64] = []
    for i, (d, scene) in enumerate(stream, start=1):
        if composite is None:
            template = scene
            composite = R0Composite(scene, policy)
        times.append(_scene_time(scene))
        frac = composite.update(_align(scene, template))
        logger.info("[%d/%d] %s: %.1f%% of pixels are snow-free candidates", i, len(dirs), d.name, 100 * frac)

    result = _finalize_composite(composite)
    result.attrs = {**template.attrs, **result.attrs}
    result = result.rio.write_crs(template.rio.crs)
    time_values = np.array(times, dtype="datetime64[ns]")
    params = {
        "max_sensor_zenith": policy.max_sensor_zenith,
        "ndvi_tie_epsilon": policy.ndvi_tie_epsilon,
        "ndvi_red_band": policy.bands["red"],
        "ndvi_nir_band": policy.bands["nir"],
        "ndsi_visible_band": policy.bands["visible"],
        "ndsi_swir_band": policy.bands["swir"],
        "blue_band": policy.bands["blue"],
        "min_blue_reflectance": policy.min_blue_reflectance,
    }
    result.attrs.update(_r0_product_attrs(time_values, n_scenes=composite.n_scenes, params=params))
    result.attrs.update({
        "r0_sensor": sensor,
        "r0_platform": platform or "",
        "r0_source_ids": json.dumps(scene_ids),
        R0_SOURCE_INVENTORY_DIGEST_ATTR: inventory,
        R0_BUILD_SIGNATURE_ATTR: signature,
    })
    result = sanitize_r0_dataset_attrs(result)
    covered = float((result["r0_source_index"].values >= 0).mean())
    logger.info("R0 complete: %.1f%% of pixels have a background observation", 100 * covered)

    r0_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = sanitize_netcdf_dataset(result)
    # xarray writes a non-index spatial_ref coord as a plain data variable, which rioxarray can't read
    # back as a CRS. Tagging each gridded variable with coordinates="spatial_ref" makes it a coord again.
    for name in to_write.data_vars:
        if {"y", "x"} <= set(to_write[name].dims):
            to_write[name].attrs["coordinates"] = "spatial_ref"
    write_r0_dataset_atomically(
        to_write, r0_path,
        expected_band_count=int(result.sizes["band"]),
        max_scene_index=composite.n_scenes - 1,
        expected_source_inventory_digest=inventory,
        expected_build_signature=signature,
    )
    if cache is not None:
        cache.put(signature, r0_path, meta={"sensor": sensor, "platform": platform, "scene_ids": scene_ids})
    return result
