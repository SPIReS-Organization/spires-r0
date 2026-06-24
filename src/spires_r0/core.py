"""Shared R0 compositing helpers for multisensor SPIReS adapters."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Callable
import uuid

import numpy as np
import xarray as xr

from spires_io.logging_utils import log_event

from spires_r0.io import sanitize_netcdf_dataset


LOGGER = logging.getLogger(__name__)
ZARR_SERIALIZATION_ATTRS = {"_FillValue", "missing_value", "add_offset", "scale_factor"}
ZARR_WRITE_KWARGS = {"zarr_format": 2}
R0_REQUIRED_VARS = (
    "r0_reflectance",
    "r0_source_index",
    "r0_source_time",
    "r0_used_min_blue_rule",
    "r0_count",
    "r0_sensor_zenith",
    "r0_sensor_azimuth",
    "r0_solar_zenith",
    "r0_solar_azimuth",
)
R0_EXCLUDED_DATASET_ATTRS = {
    "acquisition_date",
    "processing_timestamp",
    "lut_file",
}

ParseFilenameFn = Callable[[str | Path], object]
PrepareSceneFn = Callable[..., xr.Dataset]
CopySpatialMetadataFn = Callable[[xr.Dataset, xr.Dataset], xr.Dataset]


def infer_source_date_bounds(
    sources: list[str | Path | xr.Dataset],
    *,
    parse_filename_fn: ParseFilenameFn,
) -> tuple[str | None, str | None]:
    dates: list[np.datetime64] = []
    for source in sources:
        acquisition_date: str | None = None
        if isinstance(source, xr.Dataset):
            if "time" in source.coords and source["time"].size:
                time_values = source["time"].values
                dates.append(np.datetime64(time_values.min()))
                dates.append(np.datetime64(time_values.max()))
                continue
            acquisition_date = source.attrs.get("acquisition_date")
            if acquisition_date is None:
                source_path = source.attrs.get("source_path")
                if source_path:
                    try:
                        acquisition_date = parse_filename_fn(source_path).acquisition_date
                    except ValueError:
                        acquisition_date = None
        else:
            try:
                acquisition_date = parse_filename_fn(source).acquisition_date
            except ValueError:
                acquisition_date = None

        if acquisition_date is not None:
            dates.append(np.datetime64(acquisition_date))

    if not dates:
        return None, None
    return str(min(dates))[:10], str(max(dates))[:10]


def safe_normalized_difference(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    total = numerator + denominator
    safe_total = total.where(total != 0)
    return (numerator - denominator) / safe_total


def scalar_dataarray_to_float(value: xr.DataArray) -> float:
    if hasattr(value.data, "compute"):
        value = value.compute()
    return float(value.item())


def sanitize_dataset_for_zarr_write(dataset: xr.Dataset) -> xr.Dataset:
    """Drop serialization attrs that conflict with xarray zarr encoding."""
    sanitized = dataset.copy()
    sanitized.attrs = sanitized.attrs.copy()
    for variable_name in sanitized.variables:
        sanitized[variable_name].attrs = {
            key: value
            for key, value in sanitized[variable_name].attrs.items()
            if key not in ZARR_SERIALIZATION_ATTRS
        }
    return sanitized


def sanitize_r0_dataset_attrs(dataset: xr.Dataset) -> xr.Dataset:
    sanitized = dataset.copy()
    sanitized.attrs = {
        key: value
        for key, value in sanitized.attrs.items()
        if key not in R0_EXCLUDED_DATASET_ATTRS
    }
    return sanitized


def validate_r0_dataset(
    dataset: xr.Dataset,
    *,
    expected_band_count: int | None = None,
    max_scene_index: int | None = None,
) -> xr.Dataset:
    missing = [name for name in R0_REQUIRED_VARS if name not in dataset]
    if missing:
        raise ValueError(f"R0 dataset is missing required variables: {missing}")
    if "band" not in dataset.coords:
        raise ValueError("R0 dataset is missing required 'band' coordinate")

    band_count = int(dataset.sizes.get("band", 0))
    if band_count <= 0:
        raise ValueError("R0 dataset must contain at least one band")
    if expected_band_count is not None and band_count != expected_band_count:
        raise ValueError(f"R0 dataset band count mismatch: {band_count} != {expected_band_count}")

    if dataset["r0_reflectance"].dims != ("y", "x", "band"):
        raise ValueError(f"Unexpected r0_reflectance dims: {dataset['r0_reflectance'].dims}")
    for name in (
        "r0_source_index",
        "r0_source_time",
        "r0_used_min_blue_rule",
        "r0_count",
        "r0_sensor_zenith",
        "r0_sensor_azimuth",
        "r0_solar_zenith",
        "r0_solar_azimuth",
    ):
        if dataset[name].dims != ("y", "x"):
            raise ValueError(f"Unexpected {name} dims: {dataset[name].dims}")

    min_r0_count = int(dataset["r0_count"].min().item())
    max_r0_count = int(dataset["r0_count"].max().item())
    if min_r0_count < 0:
        raise ValueError(f"R0 count contains negative values: min={min_r0_count}")
    if max_scene_index is not None and max_r0_count > max_scene_index + 1:
        raise ValueError(f"R0 count exceeds available scene count: max={max_r0_count}, scenes={max_scene_index + 1}")

    min_source_index = int(dataset["r0_source_index"].min().item())
    max_source_index = int(dataset["r0_source_index"].max().item())
    if min_source_index < -1:
        raise ValueError(f"R0 source index contains invalid values: min={min_source_index}")
    if max_scene_index is not None and max_source_index > max_scene_index:
        raise ValueError(f"R0 source index exceeds available scene count: max={max_source_index}, scenes={max_scene_index + 1}")

    has_finite_r0 = bool(np.isfinite(dataset["r0_reflectance"]).any().item())
    if not has_finite_r0:
        raise ValueError("R0 reflectance contains no finite values")

    build_status = dataset.attrs.get("build_status")
    if build_status is not None and build_status != "complete":
        raise ValueError(f"R0 dataset has non-complete build_status={build_status!r}")

    return dataset


def load_existing_r0_if_valid(
    path: Path,
    *,
    expected_band_count: int | None,
    max_scene_index: int | None,
) -> xr.Dataset | None:
    try:
        dataset = xr.open_dataset(path)
    except Exception as exc:
        LOGGER.warning("Failed to open existing R0 file %s; rebuilding. Error: %s", path, exc)
        return None

    try:
        return validate_r0_dataset(
            dataset,
            expected_band_count=expected_band_count,
            max_scene_index=max_scene_index,
        )
    except Exception as exc:
        dataset.close()
        LOGGER.warning("Existing R0 file %s failed validation; rebuilding. Error: %s", path, exc)
        return None


def write_r0_dataset_atomically(
    dataset: xr.Dataset,
    path: Path,
    *,
    expected_band_count: int | None,
    max_scene_index: int | None,
) -> None:
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        dataset.to_netcdf(temp_path)
        with xr.open_dataset(temp_path) as written:
            validate_r0_dataset(
                written,
                expected_band_count=expected_band_count,
                max_scene_index=max_scene_index,
            )
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def reduce_prepared_scene_for_r0(
    prepared_ds: xr.Dataset,
    *,
    staging_variables: tuple[str, ...],
    sensor_display_name: str,
    copy_spatial_metadata_fn: CopySpatialMetadataFn,
) -> xr.Dataset:
    missing = [name for name in staging_variables if name not in prepared_ds]
    if missing:
        raise ValueError(f"Prepared {sensor_display_name} scene is missing required R0 staging variables: {missing}")

    reduced = prepared_ds[list(staging_variables)].copy()
    reduced.attrs = prepared_ds.attrs.copy()
    return copy_spatial_metadata_fn(prepared_ds, reduced)


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


def build_r0(
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
    logger = logger or LOGGER

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
    time_values = prepared_timeseries["time"].values
    used_any_min_blue = use_min_blue | use_min_water_blue
    used_min_blue_fraction = scalar_dataarray_to_float(used_any_min_blue.mean()) if used_any_min_blue.size else float("nan")
    used_water_blue_fraction = scalar_dataarray_to_float(use_min_water_blue.mean()) if use_min_water_blue.size else float("nan")
    valid_source = source_index >= 0
    valid_source_fraction = scalar_dataarray_to_float(valid_source.mean()) if valid_source.size else float("nan")

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

    log_event(
        logger,
        event_name,
        stage="r0",
        event_type="summary",
        time_count=int(prepared_timeseries.sizes["time"]),
        time_coverage_start=str(time_values.min())[:10],
        time_coverage_end=str(time_values.max())[:10],
        selected_bands=prepared_timeseries["band"].values.tolist(),
        output_shape=list(result["r0_reflectance"].shape),
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        used_min_blue_fraction=round(used_min_blue_fraction, 6),
        used_water_blue_fraction=round(used_water_blue_fraction, 6),
        valid_source_fraction=round(valid_source_fraction, 6),
        mean_r0_count=round(scalar_dataarray_to_float(valid_count.mean()), 6),
    )
    return result


def build_timeseries(
    sources: list[str | Path | xr.Dataset],
    *,
    lut_file: str | Path | None,
    logger: logging.Logger | None,
    show_progress: bool,
    progress_desc: str,
    keep_variables: tuple[str, ...] | list[str] | str | None,
    zarr_path: str | Path | None,
    zarr_mode: str,
    chunks: dict[str, int] | None,
    parse_filename_fn: ParseFilenameFn,
    prepare_scene_fn: PrepareSceneFn,
    reduce_scene_for_r0_fn: Callable[[xr.Dataset], xr.Dataset],
    start_event_name: str,
    scene_event_name: str,
    summary_event_name: str,
    **prepare_kwargs,
) -> xr.Dataset:
    prepared_scenes = []
    total_sources = len(sources)
    prepared_count = 0
    logger = logger or LOGGER
    iterator = sources
    requested_start_date, requested_end_date = infer_source_date_bounds(sources, parse_filename_fn=parse_filename_fn)
    resolved_zarr_path = Path(zarr_path).expanduser().resolve() if zarr_path is not None else None
    wrote_any_scene = False

    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("show_progress=True requires tqdm to be installed") from exc
        iterator = tqdm(sources, desc=progress_desc)

    log_event(
        logger,
        start_event_name,
        stage="timeseries",
        event_type="start",
        status="started",
        scenes_requested=total_sources,
        requested_time_coverage_start=requested_start_date,
        requested_time_coverage_end=requested_end_date,
    )

    for idx, source in enumerate(iterator, start=1):
        if isinstance(source, xr.Dataset) and "time" in source.dims:
            prepared_scenes.append(source)
            prepared_count += int(source.sizes["time"])
            log_event(
                logger,
                scene_event_name,
                stage="timeseries",
                event_type="detail",
                status="reused_dataset",
                source_type="dataset_with_time",
                processed_scenes=f"{idx}/{total_sources}",
            )
            continue

        if isinstance(source, xr.Dataset) and "reflectance" in source.data_vars:
            prepared = source
            scene_name = prepared.attrs.get("source_path") or f"dataset_{idx}"
            log_event(
                logger,
                scene_event_name,
                stage="timeseries",
                event_type="detail",
                status="reused_dataset",
                source_type="prepared_dataset",
                scene_name=Path(scene_name).name,
                processed_scenes=f"{idx}/{total_sources}",
            )
        else:
            prepared = prepare_scene_fn(source, lut_file=lut_file, logger=logger, **prepare_kwargs)
            scene_name = (
                Path(source).name
                if isinstance(source, (str, Path))
                else prepared.attrs.get("source_path") or f"dataset_{idx}"
            )
            log_event(
                logger,
                scene_event_name,
                stage="timeseries",
                event_type="detail",
                status="prepared",
                source_type="path" if isinstance(source, (str, Path)) else "dataset",
                scene_name=Path(scene_name).name,
                input_path=str(source) if isinstance(source, (str, Path)) else None,
                processed_scenes=f"{idx}/{total_sources}",
            )

        acquisition_date = prepared.attrs.get("acquisition_date")
        if acquisition_date is None:
            raise ValueError("Prepared scene is missing 'acquisition_date' in attrs")

        if keep_variables == "r0":
            prepared = reduce_scene_for_r0_fn(prepared)
        elif keep_variables is not None:
            variable_names = list(keep_variables)
            missing = [name for name in variable_names if name not in prepared]
            if missing:
                raise ValueError(f"Prepared scene is missing requested variables: {missing}")
            prepared = prepared[variable_names].copy()
            prepared.attrs = prepared.attrs.copy()

        prepared = prepared.expand_dims(time=[np.datetime64(acquisition_date)])
        prepared_count += 1

        if chunks is not None:
            normalized_chunks = {
                dim: prepared.sizes[dim] if size == -1 else size
                for dim, size in chunks.items()
                if dim in prepared.dims
            }
            if normalized_chunks:
                prepared = prepared.chunk(normalized_chunks)

        if resolved_zarr_path is not None:
            prepared_for_write = sanitize_dataset_for_zarr_write(prepared)
            if not wrote_any_scene:
                resolved_zarr_path.parent.mkdir(parents=True, exist_ok=True)
                prepared_for_write.to_zarr(resolved_zarr_path, mode=zarr_mode, **ZARR_WRITE_KWARGS)
                wrote_any_scene = True
            else:
                prepared_for_write.to_zarr(
                    resolved_zarr_path,
                    mode="a",
                    append_dim="time",
                    **ZARR_WRITE_KWARGS,
                )
        else:
            prepared_scenes.append(prepared)

    if not prepared_scenes and resolved_zarr_path is None:
        raise ValueError("At least one scene is required to build a timeseries")

    if resolved_zarr_path is not None:
        if not wrote_any_scene:
            raise ValueError("At least one scene is required to build a timeseries")
        timeseries = xr.open_zarr(resolved_zarr_path)
    else:
        timeseries = xr.concat(prepared_scenes, dim="time")
        # The concatenated stack owns the data needed from the per-scene list.
        # Drop the individual scene references before downstream metric building.
        prepared_scenes.clear()
        gc.collect()

    if "time" in timeseries.coords:
        timeseries = timeseries.sortby("time")
    timeseries.attrs["time_coverage_start"] = str(timeseries["time"].min().values)
    timeseries.attrs["time_coverage_end"] = str(timeseries["time"].max().values)
    log_event(
        logger,
        summary_event_name,
        stage="timeseries",
        event_type="summary",
        status="completed",
        time_count=int(timeseries.sizes["time"]),
        time_coverage_start=timeseries.attrs["time_coverage_start"][:10],
        time_coverage_end=timeseries.attrs["time_coverage_end"][:10],
        selected_bands=timeseries["band"].values.tolist() if "band" in timeseries.coords else None,
        output_shape=list(timeseries["reflectance"].shape) if "reflectance" in timeseries else None,
        scenes_requested=total_sources,
        scenes_prepared=prepared_count,
        zarr_path=str(resolved_zarr_path) if resolved_zarr_path is not None else None,
    )
    return timeseries


def build_r0_from_sources(
    sources: list[str | Path | xr.Dataset],
    *,
    r0_path: str | Path | None,
    overwrite: bool,
    lut_file: str | Path | None,
    logger: logging.Logger | None,
    show_progress: bool,
    progress_desc: str,
    max_sensor_zenith: float,
    ndvi_tie_epsilon: float,
    zarr_path: str | Path | None,
    zarr_mode: str,
    chunks: dict[str, int] | None,
    ndvi_red_band: str,
    ndvi_nir_band: str,
    ndsi_visible_band: str,
    ndsi_swir_band: str,
    blue_band: str,
    min_blue_reflectance: float,
    expected_band_count: int | None,
    parse_filename_fn: ParseFilenameFn,
    prepare_scene_fn: PrepareSceneFn,
    reduce_scene_for_r0_fn: Callable[[xr.Dataset], xr.Dataset],
    copy_spatial_metadata_fn: CopySpatialMetadataFn,
    r0_event_name: str,
    r0_build_event_name: str,
    timeseries_start_event_name: str,
    timeseries_scene_event_name: str,
    timeseries_summary_event_name: str,
    **prepare_kwargs,
) -> xr.Dataset:
    logger = logger or LOGGER
    resolved_r0_path = Path(r0_path).expanduser().resolve() if r0_path is not None else None
    max_scene_index = len(sources) - 1 if sources else None

    if resolved_r0_path is not None and resolved_r0_path.exists() and not overwrite:
        result = load_existing_r0_if_valid(
            resolved_r0_path,
            expected_band_count=expected_band_count,
            max_scene_index=max_scene_index,
        )
        if result is not None:
            log_event(
                logger,
                r0_event_name,
                stage="r0",
                event_type="summary",
                status="loaded_existing",
                r0_path=str(resolved_r0_path),
                time_coverage_start=result.attrs.get("time_coverage_start"),
                time_coverage_end=result.attrs.get("time_coverage_end"),
                selected_bands=result["band"].values.tolist() if "band" in result.coords else None,
                output_shape=list(result["r0_reflectance"].shape) if "r0_reflectance" in result else None,
            )
            return result
        log_event(
            logger,
            r0_event_name,
            stage="r0",
            event_type="detail",
            status="invalid_existing_rebuild",
            r0_path=str(resolved_r0_path),
        )

    requested_start_date, requested_end_date = infer_source_date_bounds(sources, parse_filename_fn=parse_filename_fn)
    log_event(
        logger,
        r0_event_name,
        stage="r0",
        event_type="start",
        status="started",
        scenes_requested=len(sources),
        requested_time_coverage_start=requested_start_date,
        requested_time_coverage_end=requested_end_date,
        r0_path=str(resolved_r0_path) if resolved_r0_path is not None else None,
    )

    timeseries = build_timeseries(
        sources,
        lut_file=lut_file,
        logger=logger,
        show_progress=show_progress,
        progress_desc=progress_desc,
        keep_variables="r0",
        zarr_path=zarr_path,
        zarr_mode=zarr_mode,
        chunks=chunks,
        parse_filename_fn=parse_filename_fn,
        prepare_scene_fn=prepare_scene_fn,
        reduce_scene_for_r0_fn=reduce_scene_for_r0_fn,
        start_event_name=timeseries_start_event_name,
        scene_event_name=timeseries_scene_event_name,
        summary_event_name=timeseries_summary_event_name,
        **prepare_kwargs,
    )
    result = build_r0(
        timeseries,
        logger=logger,
        event_name=r0_build_event_name,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
        copy_spatial_metadata_fn=copy_spatial_metadata_fn,
    )
    result.attrs["time_coverage_start"] = str(timeseries["time"].min().values)[:10]
    result.attrs["time_coverage_end"] = str(timeseries["time"].max().values)[:10]
    result.attrs["build_status"] = "complete"
    # The output product no longer depends on the full source stack after the
    # summary attrs are copied, so drop it before final serialization.
    del timeseries
    gc.collect()
    result = sanitize_r0_dataset_attrs(result)

    if resolved_r0_path is not None:
        resolved_r0_path.parent.mkdir(parents=True, exist_ok=True)
        result = sanitize_netcdf_dataset(result)
        write_r0_dataset_atomically(
            result,
            resolved_r0_path,
            expected_band_count=expected_band_count,
            max_scene_index=max_scene_index,
        )

    return result
