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


def _assert_r0_products_equal(a, b):
    assert set(a.data_vars) == set(b.data_vars)
    for v in ("r0_reflectance", "r0_source_index", "r0_source_time", "r0_count", "r0_used_min_blue_rule"):
        x, y = a[v].values, b[v].values
        if x.dtype.kind == "f":
            np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=v)
            np.testing.assert_array_equal(x[~np.isnan(x)], y[~np.isnan(y)], err_msg=v)
        else:
            np.testing.assert_array_equal(x, y, err_msg=v)


def test_build_modis_r0_from_sources_streaming_equals_zarr_cube(tmp_path, rng):
    from tests.conftest import random_stack
    from spires_r0.modis import build_modis_r0_from_sources

    scenes = random_stack(rng, n_time=4)
    streamed = build_modis_r0_from_sources(scenes, r0_path=tmp_path / "s.nc", overwrite=True)
    cubed = build_modis_r0_from_sources(
        scenes, r0_path=tmp_path / "c.nc", overwrite=True, zarr_path=tmp_path / "ts.zarr",
    )
    _assert_r0_products_equal(streamed, cubed)
    assert streamed.attrs["r0_source_scene_count"] == 4


def test_build_modis_r0_from_sources_streaming_equals_zarr_cube_shuffled(tmp_path, rng):
    # The cube path sorts by time; the streaming path must remap its feed-order indices to the same order.
    from tests.conftest import random_stack
    from spires_r0.modis import build_modis_r0_from_sources

    scenes = random_stack(rng, n_time=5)
    shuffled = [scenes[i] for i in (3, 0, 4, 1, 2)]
    streamed = build_modis_r0_from_sources(shuffled, r0_path=tmp_path / "s.nc", overwrite=True)
    cubed = build_modis_r0_from_sources(
        shuffled, r0_path=tmp_path / "c.nc", overwrite=True, zarr_path=tmp_path / "ts.zarr",
    )
    _assert_r0_products_equal(streamed, cubed)
