"""Background reflectance (R0) helpers for MODIS prepared scenes."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from spires_io.modis.geospatial import copy_spatial_metadata
from spires_io.modis.load_surface_reflectance import (
    parse_modis_surface_reflectance_filename,
    prepare_modis_scene_for_inversion,
)
from spires_r0.core import (
    build_r0 as build_shared_r0,
    build_r0_candidate_metrics as build_shared_r0_candidate_metrics,
    build_r0_from_sources as build_shared_r0_from_sources,
    build_timeseries as build_shared_timeseries,
    compute_r0_indices as compute_shared_r0_indices,
    reduce_prepared_scene_for_r0 as reduce_shared_prepared_scene_for_r0,
)


MODIS_R0_NDVI_RED_BAND = "1"
MODIS_R0_NDVI_NIR_BAND = "2"
MODIS_R0_NDSI_VISIBLE_BAND = "4"
MODIS_R0_NDSI_SWIR_BAND = "6"
MODIS_R0_BLUE_BAND = "3"
MODIS_R0_STAGING_VARIABLES = (
    "reflectance",
    "sensor_zenith",
    "sensor_azimuth",
    "solar_zenith",
    "solar_azimuth",
    "valid_r0_mask",
    "mask_water",
)
LOGGER = logging.getLogger(__name__)


def reduce_modis_prepared_scene_for_r0(prepared_ds: xr.Dataset) -> xr.Dataset:
    """Keep only the variables required for MODIS R0 screening and compositing."""
    return reduce_shared_prepared_scene_for_r0(
        prepared_ds,
        staging_variables=MODIS_R0_STAGING_VARIABLES,
        sensor_display_name="MODIS",
        copy_spatial_metadata_fn=copy_spatial_metadata,
    )


def compute_modis_r0_indices(
    prepared_ds: xr.Dataset,
    *,
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = MODIS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = MODIS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = MODIS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = MODIS_R0_NDSI_SWIR_BAND,
    blue_band: str = MODIS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Compute the MODIS screening indices used by the shared R0 compositing logic."""
    return compute_shared_r0_indices(
        prepared_ds,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )


def build_modis_r0_candidate_metrics(
    prepared_timeseries: xr.Dataset,
    *,
    max_sensor_zenith: float = 30.0,
    ndvi_red_band: str = MODIS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = MODIS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = MODIS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = MODIS_R0_NDSI_SWIR_BAND,
    blue_band: str = MODIS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Build screened MODIS candidate metrics for downstream R0 selection."""
    return build_shared_r0_candidate_metrics(
        prepared_timeseries,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
    )


def build_modis_r0(
    prepared_timeseries: xr.Dataset,
    *,
    logger: logging.Logger | None = None,
    max_sensor_zenith: float = 30.0,
    ndvi_tie_epsilon: float = 0.02,
    ndvi_red_band: str = MODIS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = MODIS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = MODIS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = MODIS_R0_NDSI_SWIR_BAND,
    blue_band: str = MODIS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
) -> xr.Dataset:
    """Build a MODIS R0 composite from a prepared time series."""
    return build_shared_r0(
        prepared_timeseries,
        logger=logger or LOGGER,
        event_name="build_modis_r0",
        max_sensor_zenith=max_sensor_zenith,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
        copy_spatial_metadata_fn=copy_spatial_metadata,
    )


def build_modis_timeseries(
    sources: list[str | Path | xr.Dataset],
    *,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    show_progress: bool = False,
    progress_desc: str = "Preparing MODIS scenes",
    keep_variables: tuple[str, ...] | list[str] | str | None = None,
    zarr_path: str | Path | None = None,
    zarr_mode: str = "w",
    chunks: dict[str, int] | None = None,
    **prepare_kwargs,
) -> xr.Dataset:
    """Prepare and concatenate MODIS scenes into a time stack for R0 workflows."""
    prepare_kwargs.setdefault("keep_r0_masks", True)
    return build_shared_timeseries(
        sources,
        lut_file=lut_file,
        logger=logger or LOGGER,
        show_progress=show_progress,
        progress_desc=progress_desc,
        keep_variables=keep_variables,
        zarr_path=zarr_path,
        zarr_mode=zarr_mode,
        chunks=chunks,
        parse_filename_fn=parse_modis_surface_reflectance_filename,
        prepare_scene_fn=prepare_modis_scene_for_inversion,
        reduce_scene_for_r0_fn=reduce_modis_prepared_scene_for_r0,
        start_event_name="build_modis_timeseries",
        scene_event_name="build_modis_timeseries_scene",
        summary_event_name="build_modis_timeseries",
        **prepare_kwargs,
    )


def build_modis_r0_from_sources(
    sources: list[str | Path | xr.Dataset],
    *,
    r0_path: str | Path | None = None,
    overwrite: bool = False,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    show_progress: bool = False,
    progress_desc: str = "Building MODIS R0",
    max_sensor_zenith: float = 30.0,
    ndvi_tie_epsilon: float = 0.02,
    zarr_path: str | Path | None = None,
    zarr_mode: str = "w",
    chunks: dict[str, int] | None = None,
    ndvi_red_band: str = MODIS_R0_NDVI_RED_BAND,
    ndvi_nir_band: str = MODIS_R0_NDVI_NIR_BAND,
    ndsi_visible_band: str = MODIS_R0_NDSI_VISIBLE_BAND,
    ndsi_swir_band: str = MODIS_R0_NDSI_SWIR_BAND,
    blue_band: str = MODIS_R0_BLUE_BAND,
    min_blue_reflectance: float = 0.10,
    **prepare_kwargs,
) -> xr.Dataset:
    """Build a MODIS R0 composite from raw or prepared input scenes."""
    prepare_kwargs.setdefault("keep_r0_masks", True)
    return build_shared_r0_from_sources(
        sources,
        r0_path=r0_path,
        overwrite=overwrite,
        lut_file=lut_file,
        logger=logger or LOGGER,
        show_progress=show_progress,
        progress_desc=progress_desc,
        max_sensor_zenith=max_sensor_zenith,
        ndvi_tie_epsilon=ndvi_tie_epsilon,
        zarr_path=zarr_path,
        zarr_mode=zarr_mode,
        chunks=chunks,
        ndvi_red_band=ndvi_red_band,
        ndvi_nir_band=ndvi_nir_band,
        ndsi_visible_band=ndsi_visible_band,
        ndsi_swir_band=ndsi_swir_band,
        blue_band=blue_band,
        min_blue_reflectance=min_blue_reflectance,
        expected_band_count=None,
        parse_filename_fn=parse_modis_surface_reflectance_filename,
        prepare_scene_fn=prepare_modis_scene_for_inversion,
        reduce_scene_for_r0_fn=reduce_modis_prepared_scene_for_r0,
        copy_spatial_metadata_fn=copy_spatial_metadata,
        r0_event_name="build_modis_r0_from_sources",
        r0_build_event_name="build_modis_r0",
        timeseries_start_event_name="build_modis_timeseries",
        timeseries_scene_event_name="build_modis_timeseries_scene",
        timeseries_summary_event_name="build_modis_timeseries",
        **prepare_kwargs,
    )
