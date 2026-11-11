"""Background (R_0) reflectance production for the SPIReS package family."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spires-r0")
except PackageNotFoundError:
    __version__ = "unknown"

from spires_r0.core import validate_r0_dataset
from spires_r0.modis import (
    build_modis_r0,
    build_modis_r0_candidate_metrics,
    build_modis_r0_from_sources,
    build_modis_timeseries,
    compute_modis_r0_indices,
    reduce_modis_prepared_scene_for_r0,
)
from spires_r0.viirs import (
    build_viirs_r0,
    build_viirs_r0_candidate_metrics,
    build_viirs_r0_from_sources,
    build_viirs_timeseries,
    compute_viirs_r0_indices,
    reduce_viirs_prepared_scene_for_r0,
)
from spires_r0.fold import CASCADE_VERSION, MODIS_POLICY, RASTER_POLICY, VIIRS_POLICY, R0Composite, R0Policy
from spires_r0.raster import build_raster_r0, prepare_raster_scene_for_r0
from spires_r0.cache import R0Cache
from spires_r0.prefetch import one_ahead

__all__ = [
    "CASCADE_VERSION",
    "MODIS_POLICY",
    "R0Cache",
    "R0Composite",
    "R0Policy",
    "RASTER_POLICY",
    "VIIRS_POLICY",
    "__version__",
    "build_modis_r0",
    "build_modis_r0_candidate_metrics",
    "build_modis_r0_from_sources",
    "build_modis_timeseries",
    "build_raster_r0",
    "build_viirs_r0",
    "build_viirs_r0_candidate_metrics",
    "build_viirs_r0_from_sources",
    "build_viirs_timeseries",
    "compute_modis_r0_indices",
    "compute_viirs_r0_indices",
    "one_ahead",
    "prepare_raster_scene_for_r0",
    "reduce_modis_prepared_scene_for_r0",
    "reduce_viirs_prepared_scene_for_r0",
    "validate_r0_dataset",
]
