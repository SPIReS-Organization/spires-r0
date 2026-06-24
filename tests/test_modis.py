import numpy as np
import xarray as xr

from spires_r0.modis import (
    build_modis_r0,
    build_modis_r0_candidate_metrics,
    build_modis_timeseries,
)


MODIS_BANDS = ["1", "2", "3", "4", "5", "6", "7"]


def build_mock_modis_prepared_scene(acquisition_date: str, values: list[float]) -> xr.Dataset:
    reflectance = xr.DataArray(
        np.tile(np.asarray(values, dtype=np.float32), (2, 2, 1)),
        dims=("y", "x", "band"),
        coords={"y": [0, 1], "x": [0, 1], "band": MODIS_BANDS},
    )
    scalar = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    return xr.Dataset(
        data_vars={
            "reflectance": reflectance,
            "sensor_zenith": scalar.copy(),
            "sensor_azimuth": scalar.copy(),
            "solar_zenith": scalar.copy(),
            "solar_azimuth": scalar.copy(),
            "valid_r0_mask": xr.ones_like(scalar, dtype=bool),
            "mask_water": xr.zeros_like(scalar, dtype=bool),
        },
        coords={"y": [0, 1], "x": [0, 1], "band": MODIS_BANDS},
        attrs={"acquisition_date": acquisition_date},
    )


def test_build_modis_r0_candidate_metrics_marks_negative_ndsi_dates():
    scene = build_mock_modis_prepared_scene("2026-06-15", [0.2, 0.7, 0.15, 0.1, 0.3, 0.4, 0.2])
    timeseries = build_modis_timeseries([scene], keep_variables="r0")

    candidates = build_modis_r0_candidate_metrics(timeseries)

    assert bool(candidates["has_negative_ndsi"].all())
    assert bool(candidates["candidate_negative_ndsi_mask"].all())


def test_build_modis_r0_uses_max_ndvi_when_any_negative_ndsi_exists():
    scene_1 = build_mock_modis_prepared_scene("2026-06-15", [0.2, 0.6, 0.15, 0.1, 0.3, 0.4, 0.2])
    scene_2 = build_mock_modis_prepared_scene("2026-06-16", [0.2, 0.8, 0.25, 0.1, 0.3, 0.4, 0.2])
    timeseries = build_modis_timeseries([scene_1, scene_2], keep_variables="r0")

    r0 = build_modis_r0(timeseries)

    assert bool((r0["r0_source_index"] == 1).all())
    assert bool((r0["r0_used_min_blue_rule"] == 0).all())


def test_build_modis_r0_uses_min_blue_when_ndsi_stays_positive():
    scene_1 = build_mock_modis_prepared_scene("2026-06-15", [0.2, 0.6, 0.3, 0.5, 0.3, 0.2, 0.2])
    scene_2 = build_mock_modis_prepared_scene("2026-06-16", [0.2, 0.8, 0.2, 0.5, 0.3, 0.2, 0.2])
    timeseries = build_modis_timeseries([scene_1, scene_2], keep_variables="r0")

    r0 = build_modis_r0(timeseries)

    assert bool((r0["r0_source_index"] == 1).all())
    assert bool(r0["r0_used_min_blue_rule"].all())
