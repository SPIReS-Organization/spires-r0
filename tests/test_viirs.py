import numpy as np
import pytest
import xarray as xr

from spires_r0.viirs import (
    build_viirs_r0,
    build_viirs_r0_candidate_metrics,
    build_viirs_r0_from_sources,
    build_viirs_timeseries,
    compute_viirs_r0_indices,
    reduce_viirs_prepared_scene_for_r0,
)


VIIRS_BANDS = ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]


def build_mock_prepared_scene(
    acquisition_date: str,
    spectra: np.ndarray,
    *,
    sensor_zenith: np.ndarray | None = None,
    sensor_azimuth: np.ndarray | None = None,
    solar_zenith: np.ndarray | None = None,
    solar_azimuth: np.ndarray | None = None,
    mask_water: np.ndarray | None = None,
) -> xr.Dataset:
    reflectance = xr.DataArray(
        spectra.reshape(1, 2, len(VIIRS_BANDS)),
        dims=("y", "x", "band"),
        coords={"y": [0], "x": [0, 1], "band": VIIRS_BANDS},
    )
    zenith_values = np.array([[10.0, 10.0]], dtype=np.float32) if sensor_zenith is None else sensor_zenith.astype(np.float32)
    azimuth_values = np.array([[0.0, 90.0]], dtype=np.float32) if sensor_azimuth is None else sensor_azimuth.astype(np.float32)
    solar_zenith_values = np.array([[45.0, 45.0]], dtype=np.float32) if solar_zenith is None else solar_zenith.astype(np.float32)
    solar_azimuth_values = np.array([[180.0, 180.0]], dtype=np.float32) if solar_azimuth is None else solar_azimuth.astype(np.float32)
    water_values = np.zeros((1, 2), dtype=bool) if mask_water is None else mask_water.astype(bool)

    return xr.Dataset(
        data_vars={
            "reflectance": reflectance,
            "sensor_zenith": xr.DataArray(zenith_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
            "sensor_azimuth": xr.DataArray(azimuth_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
            "solar_zenith": xr.DataArray(solar_zenith_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
            "solar_azimuth": xr.DataArray(solar_azimuth_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
            "valid_r0_mask": xr.DataArray(np.array([[True, True]]), dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
            "mask_water": xr.DataArray(water_values, dims=("y", "x"), coords={"y": [0], "x": [0, 1]}),
        },
        coords={"y": [0], "x": [0, 1], "band": VIIRS_BANDS},
        attrs={"acquisition_date": acquisition_date},
    )


def test_compute_viirs_r0_indices_uses_expected_bands():
    scene = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.6, 0.1, 0.3, 0.5, 0.4, 0.4],
                [0.3, 0.4, 0.2, 0.25, 0.45, 0.35, 0.35],
            ],
            dtype=np.float32,
        ),
    )

    indices = compute_viirs_r0_indices(scene)

    assert np.isclose(indices["ndvi"].isel(y=0, x=0).item(), (0.6 - 0.2) / (0.6 + 0.2))
    assert np.isclose(indices["ndsi"].isel(y=0, x=0).item(), (0.5 - 0.1) / (0.5 + 0.1))
    assert np.isclose(indices["blue_metric"].isel(y=0, x=0).item(), 0.3)


def test_build_viirs_r0_matches_selection_rules():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    r0 = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]))

    assert r0["r0_source_index"].isel(y=0, x=0).item() == 1
    assert r0["r0_source_index"].isel(y=0, x=1).item() == 0
    assert not bool(r0["has_negative_ndsi"].isel(y=0, x=0))
    assert bool(r0["has_negative_ndsi"].isel(y=0, x=1))
    np.testing.assert_allclose(r0["r0_reflectance"].isel(y=0, x=0).values, scene_2["reflectance"].isel(y=0, x=0).values)
    np.testing.assert_allclose(r0["r0_reflectance"].isel(y=0, x=1).values, scene_1["reflectance"].isel(y=0, x=1).values)


def test_build_viirs_r0_prefers_near_nadir_candidate_when_ndvi_is_within_epsilon():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array(
            [
                [0.20, 0.596, 0.4, 0.30, 0.2, 0.4, 0.4],
                [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[28.0, 10.0]], dtype=np.float32),
        sensor_azimuth=np.array([[70.0, 10.0]], dtype=np.float32),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array(
            [
                [0.20, 0.58, 0.4, 0.20, 0.2, 0.4, 0.4],
                [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        sensor_zenith=np.array([[10.0, 10.0]], dtype=np.float32),
        sensor_azimuth=np.array([[5.0, 20.0]], dtype=np.float32),
    )

    r0 = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]), ndvi_tie_epsilon=0.02)

    assert r0["r0_source_index"].isel(y=0, x=0).item() == 1
    assert np.isclose(r0["r0_sensor_zenith"].isel(y=0, x=0).item(), 10.0)
    assert np.isclose(r0["r0_sensor_azimuth"].isel(y=0, x=0).item(), 5.0)


def test_build_viirs_r0_uses_dark_water_fallback_when_blue_is_below_land_threshold():
    scene_1 = build_mock_prepared_scene(
        "2026-06-01",
        np.array([[0.02, 0.03, 0.02, 0.06, 0.05, 0.03, 0.04], [0.02, 0.03, 0.02, 0.06, 0.05, 0.03, 0.04]], dtype=np.float32),
        mask_water=np.array([[True, False]]),
    )
    scene_2 = build_mock_prepared_scene(
        "2026-07-01",
        np.array([[0.02, 0.03, 0.02, 0.05, 0.05, 0.03, 0.04], [0.02, 0.03, 0.02, 0.05, 0.05, 0.03, 0.04]], dtype=np.float32),
        mask_water=np.array([[True, False]]),
    )

    r0 = build_viirs_r0(build_viirs_timeseries([scene_1, scene_2]))

    assert r0["r0_source_index"].isel(y=0, x=0).item() == 1
    assert bool(r0["r0_used_water_blue_rule"].isel(y=0, x=0))
    assert r0["r0_source_index"].isel(y=0, x=1).item() == -1
    assert np.isnan(r0["r0_reflectance"].isel(y=0, x=1).values).all()


def test_reduce_viirs_prepared_scene_for_r0_keeps_required_variables():
    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.1, dtype=np.float32))
    scene["extra"] = xr.DataArray(np.ones((1, 2), dtype=np.float32), dims=("y", "x"), coords={"y": [0], "x": [0, 1]})

    reduced = reduce_viirs_prepared_scene_for_r0(scene)

    assert "extra" not in reduced
    assert {"reflectance", "sensor_zenith", "sensor_azimuth", "solar_zenith", "solar_azimuth", "valid_r0_mask"}.issubset(
        reduced.data_vars
    )


def test_build_viirs_r0_from_sources_writes_and_reuses_existing_file(tmp_path):
    r0_path = tmp_path / "snpp_r0.nc"
    scene = build_mock_prepared_scene("2026-06-01", np.full((2, 7), 0.2, dtype=np.float32))
    scene.attrs["processing_timestamp"] = "2026160123456"
    scene.attrs["lut_file"] = "/tmp/lut_viirs.mat"

    original = build_viirs_r0_from_sources([scene], r0_path=r0_path)
    loaded = build_viirs_r0_from_sources([], r0_path=r0_path)

    assert r0_path.exists()
    assert "acquisition_date" not in original.attrs
    assert "processing_timestamp" not in original.attrs
    assert "lut_file" not in original.attrs
    np.testing.assert_allclose(original["r0_reflectance"].values, loaded["r0_reflectance"].values)


def test_build_viirs_r0_accepts_chunked_timeseries():
    da = pytest.importorskip("dask.array")

    scene_1 = build_mock_prepared_scene("2026-06-01", np.array([[0.2, 0.3, 0.1, 0.30, 0.5, 0.4, 0.4], [0.2, 0.6, 0.4, 0.40, 0.2, 0.3, 0.3]], dtype=np.float32))
    scene_2 = build_mock_prepared_scene("2026-07-01", np.array([[0.2, 0.4, 0.1, 0.20, 0.3, 0.4, 0.4], [0.3, 0.5, 0.1, 0.20, 0.2, 0.3, 0.3]], dtype=np.float32))
    timeseries = build_viirs_timeseries([scene_1, scene_2])
    chunked = timeseries.copy()
    chunked["reflectance"] = xr.DataArray(
        da.from_array(timeseries["reflectance"].values, chunks=(2, 1, 1, 7)),
        dims=timeseries["reflectance"].dims,
        coords=timeseries["reflectance"].coords,
    )

    r0 = build_viirs_r0(chunked)

    assert r0["r0_source_index"].isel(y=0, x=0).compute().item() == 1
    assert r0["r0_source_index"].isel(y=0, x=1).compute().item() == 0
