"""Verbatim snapshot of spires_r0.core.build_r0 at 08f598f (Ross Palomaki, 2026-07-29),
kept as the equivalence oracle for the streaming fold. Do not edit; if the algorithm
changes on purpose, change fold.py and re-snapshot deliberately."""
from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from spires_r0.core import CopySpatialMetadataFn


def safe_normalized_difference(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    total = numerator + denominator
    safe_total = total.where(total != 0)
    return (numerator - denominator) / safe_total


def compute_r0_indices(
    prepared_ds: xr.Dataset,
    *,
    max_sensor_zenith: float,
    ndvi_red_band: str,
    ndvi_nir_band: str,
    ndsi_visible_band: str,
    ndsi_swir_band: str,
    blue_band: str,
    min_blue_reflectance: float,
) -> xr.Dataset:
    reflectance = prepared_ds["reflectance"]
    sensor_zenith = prepared_ds["sensor_zenith"]
    valid_r0_mask = prepared_ds["valid_r0_mask"]

    low_zenith = sensor_zenith <= max_sensor_zenith

    red = reflectance.sel(band=ndvi_red_band)
    nir = reflectance.sel(band=ndvi_nir_band)
    visible = reflectance.sel(band=ndsi_visible_band)
    swir = reflectance.sel(band=ndsi_swir_band)
    blue = reflectance.sel(band=blue_band)

    valid_low_zenith = valid_r0_mask & low_zenith

    ndvi = safe_normalized_difference(nir.where(valid_low_zenith), red.where(valid_low_zenith)).rename("ndvi")
    ndsi = safe_normalized_difference(visible.where(valid_low_zenith), swir.where(valid_low_zenith)).rename("ndsi")
    raw_blue_metric = blue.where(valid_low_zenith).rename("raw_blue_metric")
    blue_metric = raw_blue_metric.where(raw_blue_metric >= min_blue_reflectance).rename("blue_metric")

    return xr.Dataset(
        data_vars={
            "ndvi": ndvi,
            "ndsi": ndsi,
            "raw_blue_metric": raw_blue_metric,
            "blue_metric": blue_metric,
            "r0_low_sensor_zenith_mask": low_zenith.astype(bool),
        }
    )


def build_r0_candidate_metrics(
    prepared_timeseries: xr.Dataset,
    *,
    max_sensor_zenith: float,
    ndvi_red_band: str,
    ndvi_nir_band: str,
    ndsi_visible_band: str,
    ndsi_swir_band: str,
    blue_band: str,
    min_blue_reflectance: float,
) -> xr.Dataset:
    indices_ds = compute_r0_indices(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )

    valid_r0_mask = prepared_timeseries["valid_r0_mask"]
    ndsi = indices_ds["ndsi"]
    ndvi = indices_ds["ndvi"]
    raw_blue_metric = indices_ds["raw_blue_metric"]
    blue_metric = indices_ds["blue_metric"]
    mask_water = prepared_timeseries.get("mask_water")
    if mask_water is None:
        mask_water = xr.zeros_like(valid_r0_mask, dtype=bool)

    candidate_negative_ndsi_mask = valid_r0_mask & (ndsi < 0)
    candidate_blue_mask = valid_r0_mask & blue_metric.notnull()
    candidate_water_blue_mask = valid_r0_mask & mask_water.astype(bool) & raw_blue_metric.notnull()

    candidate_ndvi = ndvi.where(candidate_negative_ndsi_mask)
    candidate_blue_metric = blue_metric.where(candidate_blue_mask)
    candidate_water_blue_metric = raw_blue_metric.where(candidate_water_blue_mask)
    has_negative_ndsi = candidate_ndvi.notnull().any(dim="time")

    return xr.Dataset(
        data_vars={
            **indices_ds.data_vars,
            "candidate_ndvi": candidate_ndvi,
            "candidate_blue_metric": candidate_blue_metric,
            "candidate_water_blue_metric": candidate_water_blue_metric,
            "candidate_negative_ndsi_mask": candidate_negative_ndsi_mask.astype(bool),
            "candidate_blue_mask": candidate_blue_mask.astype(bool),
            "candidate_water_blue_mask": candidate_water_blue_mask.astype(bool),
            "has_negative_ndsi": has_negative_ndsi.astype(bool),
        }
    )


def select_time_indices(metric: xr.DataArray, *, mode: str) -> tuple[xr.DataArray, xr.DataArray]:
    metric = xr.DataArray(
        metric.data,
        dims=metric.dims,
        coords={dim: metric.coords[dim] for dim in metric.dims},
        name=metric.name,
    )
    invalid = metric.isnull().all(dim="time")

    if mode == "max":
        filled = metric.fillna(-np.inf)
        index = filled.argmax(dim="time")
    elif mode == "min":
        filled = metric.fillna(np.inf)
        index = filled.argmin(dim="time")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    index = index.where(~invalid, other=-1).astype(np.int32)
    return index, invalid


def build_view_geometry_tiebreak_metric(
    sensor_zenith: xr.DataArray,
    sensor_azimuth: xr.DataArray,
) -> xr.DataArray:
    """Build a stable near-nadir tie-break metric with directional components as secondary keys."""
    azimuth_radians = np.deg2rad(sensor_azimuth)
    view_x = np.sin(np.deg2rad(sensor_zenith)) * np.cos(azimuth_radians)
    view_y = np.sin(np.deg2rad(sensor_zenith)) * np.sin(azimuth_radians)
    return (
        sensor_zenith.astype(np.float32)
        + np.abs(view_x).astype(np.float32) * np.float32(1.0e-3)
        + np.abs(view_y).astype(np.float32) * np.float32(1.0e-6)
    ).rename("view_geometry_tiebreak_metric")


def gather_values_by_index(
    values: xr.DataArray,
    source_index: xr.DataArray,
    invalid_mask: xr.DataArray,
) -> xr.DataArray:
    source_index = xr.DataArray(
        source_index.data,
        dims=source_index.dims,
        coords={dim: source_index.coords[dim] for dim in source_index.dims},
    )
    invalid_mask = xr.DataArray(
        invalid_mask.data,
        dims=invalid_mask.dims,
        coords={dim: invalid_mask.coords[dim] for dim in invalid_mask.dims},
    )

    if hasattr(source_index.data, "compute"):
        source_index = source_index.compute()
    if hasattr(invalid_mask.data, "compute"):
        invalid_mask = invalid_mask.compute()

    safe_index = source_index.where(~invalid_mask, other=0)
    return values.isel(time=safe_index).where(~invalid_mask)


def reference_build_r0(
    prepared_timeseries: xr.Dataset,
    *,
    logger: logging.Logger | None,
    event_name: str,
    max_sensor_zenith: float,
    ndvi_tie_epsilon: float,
    ndvi_red_band: str,
    ndvi_nir_band: str,
    ndsi_visible_band: str,
    ndsi_swir_band: str,
    blue_band: str,
    min_blue_reflectance: float,
    copy_spatial_metadata_fn: CopySpatialMetadataFn,
) -> xr.Dataset:
    if "time" not in prepared_timeseries.dims:
        raise ValueError("prepared_timeseries must have a 'time' dimension")

    candidate_ds = build_r0_candidate_metrics(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )

    ndsi = candidate_ds["ndsi"]
    candidate_ndvi = candidate_ds["candidate_ndvi"]
    candidate_blue_metric = candidate_ds["candidate_blue_metric"]
    candidate_water_blue_metric = candidate_ds["candidate_water_blue_metric"]
    has_negative_ndsi = candidate_ds["has_negative_ndsi"]
    sensor_zenith = prepared_timeseries["sensor_zenith"]
    sensor_azimuth = prepared_timeseries["sensor_azimuth"]
    solar_zenith = prepared_timeseries["solar_zenith"]
    solar_azimuth = prepared_timeseries["solar_azimuth"]
    ndvi_near_max_threshold = candidate_ndvi.max(dim="time", skipna=True) - np.float32(ndvi_tie_epsilon)
    near_max_ndvi_mask = candidate_ndvi >= ndvi_near_max_threshold
    view_geometry_tiebreak_metric = build_view_geometry_tiebreak_metric(sensor_zenith, sensor_azimuth)
    candidate_ndvi_tiebreak_metric = view_geometry_tiebreak_metric.where(near_max_ndvi_mask)

    idx_max_ndvi, invalid_ndvi = select_time_indices(candidate_ndvi_tiebreak_metric, mode="min")
    idx_min_blue, invalid_blue = select_time_indices(candidate_blue_metric, mode="min")
    idx_min_water_blue, invalid_water_blue = select_time_indices(candidate_water_blue_metric, mode="min")

    min_ndsi = ndsi.min(dim="time", skipna=True)
    max_ndvi = candidate_ndvi.max(dim="time", skipna=True)
    min_blue = candidate_blue_metric.min(dim="time", skipna=True)
    min_water_blue = candidate_water_blue_metric.min(dim="time", skipna=True)

    use_max_ndvi = has_negative_ndsi.fillna(False).astype(bool) & (~invalid_ndvi)
    use_min_blue = (~use_max_ndvi) & (~invalid_blue)
    use_min_water_blue = (~use_max_ndvi) & invalid_blue & (~invalid_water_blue)
    invalid_final = ~(use_max_ndvi | use_min_blue | use_min_water_blue)

    source_index = xr.where(
        use_max_ndvi,
        idx_max_ndvi,
        xr.where(use_min_blue, idx_min_blue, idx_min_water_blue),
    ).where(~invalid_final, other=-1).astype(np.int32)

    reflectance = prepared_timeseries["reflectance"]
    r0_values = gather_values_by_index(reflectance, source_index, invalid_final).astype(np.float32)
    r0_sensor_zenith = gather_values_by_index(sensor_zenith, source_index, invalid_final).astype(np.float32)
    r0_sensor_azimuth = gather_values_by_index(sensor_azimuth, source_index, invalid_final).astype(np.float32)
    r0_solar_zenith = gather_values_by_index(solar_zenith, source_index, invalid_final).astype(np.float32)
    r0_solar_azimuth = gather_values_by_index(solar_azimuth, source_index, invalid_final).astype(np.float32)

    safe_source_index = xr.DataArray(
        source_index.data,
        dims=source_index.dims,
        coords={dim: source_index.coords[dim] for dim in source_index.dims},
    )
    if hasattr(safe_source_index.data, "compute"):
        safe_source_index = safe_source_index.compute()
    safe_source_index = safe_source_index.where(safe_source_index >= 0, other=0)
    source_time = prepared_timeseries["time"].isel(time=safe_source_index).where(source_index >= 0)

    valid_count = prepared_timeseries["valid_r0_mask"].sum(dim="time").astype(np.int32)
    used_any_min_blue = use_min_blue | use_min_water_blue

    result = xr.Dataset(
        data_vars={
            "r0_reflectance": xr.DataArray(
                r0_values.data,
                dims=("y", "x", "band"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values, "band": prepared_timeseries["band"].values},
            ),
            "r0_source_index": xr.DataArray(
                source_index.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_source_time": xr.DataArray(
                source_time.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_used_min_blue_rule": xr.DataArray(
                used_any_min_blue.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_used_water_blue_rule": xr.DataArray(
                use_min_water_blue.data,
                dims=("y", "x"),
                coords={"y": prepared_timeseries["y"].values, "x": prepared_timeseries["x"].values},
            ),
            "r0_count": valid_count.astype(np.int32),
            "r0_sensor_zenith": r0_sensor_zenith,
            "r0_sensor_azimuth": r0_sensor_azimuth,
            "r0_solar_zenith": r0_solar_zenith,
            "r0_solar_azimuth": r0_solar_azimuth,
            "max_ndvi": max_ndvi,
            "min_ndsi": min_ndsi,
            "min_blue_metric": min_blue,
            "min_water_blue_metric": min_water_blue,
            "has_negative_ndsi": has_negative_ndsi,
        },
        attrs=prepared_timeseries.attrs.copy(),
    )
    result = copy_spatial_metadata_fn(prepared_timeseries, result)
    return result
