"""Single streaming R0 engine, parameterised by a per-sensor policy.

Semantics are those of the original cube reducer (core.build_r0 at 08f598f):
eligible = valid_r0_mask ∧ (sensor_zenith ≤ max_sensor_zenith if geometry is used);
primary pick = among eligible with NDSI < 0, the candidate with the smallest tie-break
metric among those whose NDVI is within `ndvi_tie_epsilon` of the per-pixel max NDVI;
fallback 1 = min blue among eligible with blue ≥ min_blue_reflectance;
fallback 2 = min raw blue among eligible water pixels.
Nothing is ever stacked along time; peak memory is ~(tiebreak_depth + 2) scenes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

CASCADE_VERSION = "streaming-fold-1"


@dataclass(frozen=True)
class R0Policy:
    bands: dict[str, str]                       # roles: red, nir, visible, swir, blue → band coord labels
    use_sensor_geometry: bool                   # zenith cut + view-geometry tie-break need sensor_zenith/azimuth
    max_sensor_zenith: float = 30.0
    ndvi_tie_epsilon: float = 0.02
    min_blue_reflectance: float = 0.10          # fallback-rule threshold (Ross)
    min_candidate_blue: float | None = None     # raster-only: drop candidates with blue ≤ this (path radiance)
    require_blue_below_green: bool = False      # raster-only
    tiebreak_depth: int = 4                     # top-K NDVI candidates kept per pixel
    name: str = "custom"

    def required_bands(self) -> tuple[str, ...]:
        return tuple(self.bands[r] for r in ("red", "nir", "visible", "swir", "blue"))


MODIS_POLICY = R0Policy(
    bands={"red": "1", "nir": "2", "visible": "4", "swir": "6", "blue": "3"},
    use_sensor_geometry=True, name="modis",
)
VIIRS_POLICY = R0Policy(
    bands={"red": "I1", "nir": "I2", "visible": "M4", "swir": "I3", "blue": "M2"},
    use_sensor_geometry=True, name="viirs",
)
RASTER_POLICY = R0Policy(
    bands={"red": "red", "nir": "nir", "visible": "green", "swir": "swir1", "blue": "blue"},
    use_sensor_geometry=False, ndvi_tie_epsilon=0.0, min_candidate_blue=0.01,
    require_blue_below_green=True, tiebreak_depth=1, name="raster",
)


def policy_from_legacy_kwargs(base: R0Policy, **kw) -> R0Policy:
    """Map build_r0's keyword arguments onto a policy (keeps Ross's API intact)."""
    bands = dict(base.bands)
    for role, key in (("red", "ndvi_red_band"), ("nir", "ndvi_nir_band"), ("visible", "ndsi_visible_band"),
                      ("swir", "ndsi_swir_band"), ("blue", "blue_band")):
        if kw.get(key) is not None:
            bands[role] = kw[key]
    fields = {k: kw[k] for k in ("max_sensor_zenith", "ndvi_tie_epsilon", "min_blue_reflectance", "tiebreak_depth")
              if kw.get(k) is not None}
    return replace(base, bands=bands, **fields)


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / np.where(total != 0, total, np.nan)
    return out.astype(np.float32)


def _view_metric(zen: np.ndarray, azi: np.ndarray) -> np.ndarray:
    az = np.deg2rad(azi); sz = np.sin(np.deg2rad(zen))
    return (zen.astype(np.float32)
            + np.abs(sz * np.cos(az)).astype(np.float32) * np.float32(1e-3)
            + np.abs(sz * np.sin(az)).astype(np.float32) * np.float32(1e-6))


def compute_scene_metrics(scene: xr.Dataset, policy: R0Policy) -> dict[str, np.ndarray]:
    """Per-pixel metrics for one prepared scene. All (y, x) numpy arrays; NaN = not a candidate."""
    present = set(scene["band"].values.tolist())
    missing = [b for b in policy.required_bands() if b not in present]
    if missing:
        raise ValueError(f"scene lacks band(s) {missing} required by policy {policy.name}")
    refl = scene["reflectance"].transpose("y", "x", "band")
    sel = lambda role: refl.sel(band=policy.bands[role]).values
    red, nir, vis, swir, blue = (sel(r) for r in ("red", "nir", "visible", "swir", "blue"))
    valid = scene["valid_r0_mask"].values.astype(bool)

    if policy.use_sensor_geometry:
        zen = scene["sensor_zenith"].values; azi = scene["sensor_azimuth"].values
        eligible = valid & (zen <= policy.max_sensor_zenith)
        view = _view_metric(zen, azi)
    else:
        eligible = valid
        view = np.zeros(valid.shape, np.float32)

    nan = np.float32(np.nan)
    ndvi = np.where(eligible, _nd(nir, red), nan)
    ndsi = np.where(eligible, _nd(vis, swir), nan)
    raw_blue = np.where(eligible, blue, nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        blue_metric = np.where(raw_blue >= policy.min_blue_reflectance, raw_blue, nan)
        cand = eligible & (ndsi < 0)
        if policy.min_candidate_blue is not None:
            cand &= blue > policy.min_candidate_blue
        if policy.require_blue_below_green:
            cand &= blue < vis
    # Ross's reducer uses xarray .notnull() (keeps +/-inf), so test for NaN, not finiteness.
    candidate_ndvi = np.where(cand & ~np.isnan(ndvi), ndvi, nan)
    water = scene["mask_water"].values.astype(bool) if "mask_water" in scene else np.zeros_like(valid)
    return {
        "eligible": eligible,
        "valid": valid,
        "ndsi": ndsi,
        "candidate_ndvi": candidate_ndvi.astype(np.float32),
        "candidate_blue": np.where(eligible & ~np.isnan(blue_metric), blue_metric, nan).astype(np.float32),
        "candidate_water_blue": np.where(eligible & water & ~np.isnan(raw_blue), raw_blue, nan).astype(np.float32),
        "view": view,
    }
