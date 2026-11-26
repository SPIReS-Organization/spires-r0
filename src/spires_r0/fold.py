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


_NAT = np.iinfo(np.int64).min
_TMAX = np.iinfo(np.int64).max
_IMAX32 = np.iinfo(np.int32).max


def _scene_time_value(scene: xr.Dataset) -> np.datetime64:
    if "time" in scene.coords and scene["time"].size == 1:
        return np.asarray(scene["time"].values).reshape(-1)[0]
    for key in ("acquisition_date", "acquisition_time"):
        if key in scene.attrs:
            return np.datetime64(scene.attrs[key])
    raise ValueError("scene has neither a 'time' coord nor attrs['acquisition_date'/'acquisition_time']")


def _scene_time(scene: xr.Dataset) -> np.int64:
    """Acquisition time as int64 ns (NaT = int64 min). Output unit is set by _time_dtype."""
    return _scene_time_value(scene).astype("datetime64[ns]").astype(np.int64)


def _time_dtype(scene: xr.Dataset) -> np.dtype:
    """datetime64 dtype the cube reducer carries for this scene's time: a time coord keeps its own
    dtype; attrs-derived times get the dtype xarray gives a 1-element time coord built from them."""
    if "time" in scene.coords and scene["time"].size == 1:
        return np.asarray(scene["time"].values).dtype
    return xr.DataArray(np.array([_scene_time_value(scene)])).dtype


def _opt(scene: xr.Dataset, name: str, shape) -> np.ndarray:
    if name in scene:
        return scene[name].values.astype(np.float32)
    return np.full(shape, np.nan, np.float32)


class _MinSlot:
    """Single-slot running minimum per pixel. Strict min; ties go to earliest (time, feed index)."""

    def __init__(self, ny: int, nx: int, nb: int):
        self.metric = np.full((ny, nx), np.inf, np.float32)
        self.time = np.full((ny, nx), _NAT, np.int64)
        self.idx = np.full((ny, nx), -1, np.int32)
        self.refl = np.full((ny, nx, nb), np.nan, np.float32)
        self.geom = np.full((4, ny, nx), np.nan, np.float32)

    def offer(self, metric, when, idx, refl, geom):
        with np.errstate(invalid="ignore"):
            better = (metric < self.metric) | ((metric == self.metric)
                                               & ((when < self.time) | ((when == self.time) & (idx < self.idx))))
        take = ~np.isnan(metric) & ((self.idx < 0) | better)
        self.metric = np.where(take, metric, self.metric).astype(np.float32)
        self.time = np.where(take, when, self.time)
        self.idx = np.where(take, idx, self.idx).astype(np.int32)
        self.refl = np.where(take[..., None], refl, self.refl).astype(np.float32)
        self.geom = np.where(take[None], geom, self.geom).astype(np.float32)


class R0Composite:
    """Running per-pixel R0 state. Feed scenes in any order; call finalize() once.

    Keeps the top-K NDVI candidates per pixel (K = policy.tiebreak_depth) and single-slot
    fallback minima. Top-K retention ranks by (NDVI desc, time asc, feed index asc), so which
    candidates survive is independent of feed order.

    r0_tiebreak_window_overflow is conservative: it is set wherever the result MAY differ from
    the full-stack reducer, and it is never unset where the result does differ.
    """

    def __init__(self, template: xr.Dataset, policy: R0Policy):
        self.policy = policy
        self.y = template["y"].values
        self.x = template["x"].values
        self.band = template["band"].values
        ny, nx, nb = self.y.size, self.x.size, self.band.size
        self.shape = (ny, nx)
        K = max(1, int(policy.tiebreak_depth))
        self.K = K
        self.c_ndvi = np.full((K, ny, nx), -np.inf, np.float32)
        self.c_view = np.full((K, ny, nx), np.nan, np.float32)
        self.c_time = np.full((K, ny, nx), _NAT, np.int64)
        self.c_idx = np.full((K, ny, nx), -1, np.int32)
        self.c_refl = np.full((K, ny, nx, nb), np.nan, np.float32)
        self.c_geom = np.full((K, 4, ny, nx), np.nan, np.float32)
        self.evicted_ndvi = np.full((ny, nx), -np.inf, np.float32)
        self.blue = _MinSlot(ny, nx, nb)
        self.water = _MinSlot(ny, nx, nb)
        self.ndsi_min = np.full((ny, nx), np.inf, np.float32)
        self.ndsi_seen = np.zeros((ny, nx), bool)
        self.count = np.zeros((ny, nx), np.int32)
        self.n_scenes = 0
        self.time_dtype: np.dtype | None = None

    def _push(self, ndvi, view, when, idx, refl, geom):
        K = self.K
        ok = ~np.isnan(ndvi)
        ndvi_all = np.concatenate([self.c_ndvi, np.where(ok, ndvi, -np.inf).astype(np.float32)[None]], 0)
        time_all = np.concatenate([self.c_time, np.where(ok, when, _NAT).astype(np.int64)[None]], 0)
        idx_all = np.concatenate([self.c_idx, np.where(ok, idx, -1).astype(np.int32)[None]], 0)
        # Retention order: highest NDVI, then earliest time, then earliest feed index (lexsort's last key is primary).
        # Feed order therefore cannot decide which candidate survives an NDVI tie.
        order = np.lexsort((idx_all, time_all, -ndvi_all), axis=0)
        ranked = np.take_along_axis(ndvi_all, order, axis=0)
        self.evicted_ndvi = np.maximum(self.evicted_ndvi, ranked[K])  # (K+1)-th ranked is dropped
        keep = order[:K]
        self.c_ndvi = ranked[:K]
        self.c_view = np.take_along_axis(
            np.concatenate([self.c_view, np.where(ok, view, np.nan).astype(np.float32)[None]], 0), keep, 0)
        self.c_time = np.take_along_axis(time_all, keep, 0)
        self.c_idx = np.take_along_axis(idx_all, keep, 0)
        full_refl = np.concatenate([self.c_refl, refl[None]], 0)
        self.c_refl = np.take_along_axis(full_refl, np.broadcast_to(keep[..., None], (K,) + refl.shape), 0)
        full_geom = np.concatenate([self.c_geom, geom[None]], 0)
        self.c_geom = np.take_along_axis(full_geom, np.broadcast_to(keep[:, None], (K,) + geom.shape), 0)

    def update(self, scene: xr.Dataset) -> float:
        """Fold one prepared scene. Returns the fraction of pixels that are primary candidates."""
        if list(scene["band"].values) != list(self.band):
            raise ValueError(f"band mismatch: {list(scene['band'].values)} vs {list(self.band)}")
        if "time" in scene.dims:
            scene = scene.squeeze("time", drop=False)
        dt = _time_dtype(scene)
        if self.time_dtype is None:
            self.time_dtype = dt
        elif dt != self.time_dtype:
            raise ValueError(f"mixed time resolutions in one composite: {dt} vs {self.time_dtype}")
        when = _scene_time(scene)
        m = compute_scene_metrics(scene, self.policy)
        refl = scene["reflectance"].transpose("y", "x", "band").values.astype(np.float32)
        geom = np.stack([_opt(scene, "sensor_zenith", self.shape), _opt(scene, "sensor_azimuth", self.shape),
                         _opt(scene, "solar_zenith", self.shape), _opt(scene, "solar_azimuth", self.shape)])
        idx = self.n_scenes
        self._push(m["candidate_ndvi"], m["view"], when, idx, refl, geom)
        self.blue.offer(m["candidate_blue"], when, idx, refl, geom)
        self.water.offer(m["candidate_water_blue"], when, idx, refl, geom)
        ndsi = m["ndsi"]
        ok = ~np.isnan(ndsi)
        self.ndsi_min = np.where(ok, np.minimum(self.ndsi_min, ndsi), self.ndsi_min).astype(np.float32)
        self.ndsi_seen |= ok
        self.count += m["valid"].astype(np.int32)
        self.n_scenes += 1
        return float(np.mean(~np.isnan(m["candidate_ndvi"])))

    def finalize(self) -> xr.Dataset:
        p = self.policy
        ny, nx = self.shape
        nan = np.float32(np.nan)
        has = (self.c_idx >= 0).any(axis=0)
        top = self.c_ndvi[0]
        thr = (top - np.float32(p.ndvi_tie_epsilon)).astype(np.float32)
        window = (self.c_idx >= 0) & (self.c_ndvi >= thr[None])
        pick = window & ~np.isnan(self.c_view)
        any_pick = pick.any(axis=0)
        # lexicographic (view, time, feed index) minimum over the pickable window
        vkey = np.where(pick, self.c_view, np.inf)
        in_v = pick & (self.c_view == vkey.min(axis=0)[None])
        tkey = np.where(in_v, self.c_time, _TMAX)
        in_t = in_v & (self.c_time == tkey.min(axis=0)[None])
        ikey = np.where(in_t, self.c_idx, _IMAX32)
        slot = ikey.argmin(axis=0)

        prim_valid = has & any_pick
        prim_idx = np.take_along_axis(self.c_idx, slot[None], 0)[0]
        prim_time = np.take_along_axis(self.c_time, slot[None], 0)[0]
        prim_refl = np.take_along_axis(self.c_refl, np.broadcast_to(slot[None, ..., None], (1, ny, nx, self.band.size)), 0)[0]
        prim_geom = np.take_along_axis(self.c_geom, np.broadcast_to(slot[None, None], (1, 4, ny, nx)), 0)[0]

        use_blue = ~prim_valid & (self.blue.idx >= 0)
        use_water = ~prim_valid & ~use_blue & (self.water.idx >= 0)
        src_idx = np.where(prim_valid, prim_idx,
                           np.where(use_blue, self.blue.idx, np.where(use_water, self.water.idx, -1))).astype(np.int32)
        src_time = np.where(prim_valid, prim_time,
                            np.where(use_blue, self.blue.time, np.where(use_water, self.water.time, _NAT)))
        refl = np.where(prim_valid[..., None], prim_refl,
                        np.where(use_blue[..., None], self.blue.refl,
                                 np.where(use_water[..., None], self.water.refl, nan))).astype(np.float32)
        geom = np.where(prim_valid[None], prim_geom,
                        np.where(use_blue[None], self.blue.geom,
                                 np.where(use_water[None], self.water.geom, nan))).astype(np.float32)
        overflow = has & (self.evicted_ndvi >= thr)

        yx = ("y", "x")
        ds = xr.Dataset(
            data_vars={
                "r0_reflectance": (("y", "x", "band"), refl),
                "r0_source_index": (yx, src_idx),
                "r0_source_time": (yx, src_time.view("datetime64[ns]").astype(self.time_dtype or np.dtype("datetime64[ns]"))),
                "r0_used_min_blue_rule": (yx, use_blue | use_water),
                "r0_used_water_blue_rule": (yx, use_water),
                "r0_count": (yx, self.count.astype(np.int32)),
                "r0_sensor_zenith": (yx, geom[0]),
                "r0_sensor_azimuth": (yx, geom[1]),
                "r0_solar_zenith": (yx, geom[2]),
                "r0_solar_azimuth": (yx, geom[3]),
                "max_ndvi": (yx, np.where(has, top, nan).astype(np.float32)),
                "min_ndsi": (yx, np.where(self.ndsi_seen, self.ndsi_min, nan).astype(np.float32)),
                "min_blue_metric": (yx, np.where(self.blue.idx >= 0, self.blue.metric, nan).astype(np.float32)),
                "min_water_blue_metric": (yx, np.where(self.water.idx >= 0, self.water.metric, nan).astype(np.float32)),
                "has_negative_ndsi": (yx, has),
                "r0_tiebreak_window_overflow": (yx, overflow),
            },
            coords={"y": self.y, "x": self.x, "band": self.band},
            attrs={
                "r0_tiebreak_window_overflow_pixels": int(overflow.sum()),
                "r0_tiebreak_depth": int(self.K),
                "r0_policy": p.name,
                "r0_cascade_version": CASCADE_VERSION,
            },
        )
        return ds
