"""Background (R_0) reflectance production for the SPIReS package family."""

__version__ = "0.1.0"

from spires_r0.core import (
    build_r0,
    build_r0_candidate_metrics,
    build_r0_from_sources,
    build_timeseries,
    compute_r0_indices,
    reduce_prepared_scene_for_r0,
    validate_r0_dataset,
)
from spires_r0.modis import (
    build_modis_r0,
    build_modis_r0_candidate_metrics,
    build_modis_r0_from_sources,
    build_modis_timeseries,
    compute_modis_r0_indices,
    reduce_modis_prepared_scene_for_r0,
)
from spires_r0.viirs import (
    build_r0_from_sources as build_viirs_r0_from_sources,
    build_viirs_r0,
    build_viirs_r0_candidate_metrics,
    build_viirs_timeseries,
    compute_viirs_r0_indices,
    reduce_viirs_prepared_scene_for_r0,
)

__all__ = [
    "__version__",
    "build_modis_r0",
    "build_modis_r0_candidate_metrics",
    "build_modis_r0_from_sources",
    "build_modis_timeseries",
    "build_r0",
    "build_r0_candidate_metrics",
    "build_r0_from_sources",
    "build_timeseries",
    "build_viirs_r0",
    "build_viirs_r0_candidate_metrics",
    "build_viirs_r0_from_sources",
    "build_viirs_timeseries",
    "compute_modis_r0_indices",
    "compute_r0_indices",
    "compute_viirs_r0_indices",
    "reduce_modis_prepared_scene_for_r0",
    "reduce_prepared_scene_for_r0",
    "reduce_viirs_prepared_scene_for_r0",
    "validate_r0_dataset",
]
