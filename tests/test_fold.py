import numpy as np
import pytest

from spires_r0.fold import MODIS_POLICY, RASTER_POLICY, R0Policy, compute_scene_metrics


def test_policy_presets():
    assert MODIS_POLICY.bands == {"red": "1", "nir": "2", "visible": "4", "swir": "6", "blue": "3"}
    assert MODIS_POLICY.use_sensor_geometry is True and MODIS_POLICY.ndvi_tie_epsilon == 0.02
    assert MODIS_POLICY.max_sensor_zenith == 30.0 and MODIS_POLICY.min_blue_reflectance == 0.10
    assert MODIS_POLICY.min_candidate_blue is None and MODIS_POLICY.require_blue_below_green is False
    assert RASTER_POLICY.bands == {"red": "red", "nir": "nir", "visible": "green", "swir": "swir1", "blue": "blue"}
    assert RASTER_POLICY.use_sensor_geometry is False and RASTER_POLICY.ndvi_tie_epsilon == 0.0
    assert RASTER_POLICY.min_candidate_blue == 0.01 and RASTER_POLICY.require_blue_below_green is True


def test_metrics_match_ross_definitions(rng):
    from tests.conftest import random_stack
    from tests.reference_cube import compute_r0_indices, build_r0_candidate_metrics
    scene = random_stack(rng, n_time=1)[0]
    m = compute_scene_metrics(scene, MODIS_POLICY)
    cube = scene.expand_dims(time=[np.datetime64(scene.attrs["acquisition_date"])])
    ref = build_r0_candidate_metrics(
        cube, max_sensor_zenith=30.0, ndvi_red_band="1", ndvi_nir_band="2", ndsi_visible_band="4",
        ndsi_swir_band="6", blue_band="3", min_blue_reflectance=0.10,
    ).isel(time=0)
    np.testing.assert_array_equal(m["eligible"], (ref["r0_low_sensor_zenith_mask"] & scene["valid_r0_mask"]).values)
    np.testing.assert_allclose(m["ndsi"], ref["ndsi"].values, equal_nan=True)
    np.testing.assert_allclose(m["candidate_ndvi"], ref["candidate_ndvi"].values, equal_nan=True)
    np.testing.assert_allclose(m["candidate_blue"], ref["candidate_blue_metric"].values, equal_nan=True)
    np.testing.assert_allclose(m["candidate_water_blue"], ref["candidate_water_blue_metric"].values, equal_nan=True)


@pytest.mark.xfail(strict=True, reason="raster.py lands in Task 5")
def test_raster_policy_candidate_filters():
    from tests.conftest import make_raster_scene
    from spires_r0.raster import prepare_raster_scene_for_r0
    base = {"blue": 0.05, "green": 0.08, "red": 0.10, "nir": 0.40, "swir1": 0.25, "swir2": 0.15}
    ok = compute_scene_metrics(prepare_raster_scene_for_r0(make_raster_scene(base)), RASTER_POLICY)
    assert np.isfinite(ok["candidate_ndvi"]).all()
    dark = compute_scene_metrics(prepare_raster_scene_for_r0(make_raster_scene({**base, "blue": 0.005})), RASTER_POLICY)
    assert np.isnan(dark["candidate_ndvi"]).all()
    bluish = compute_scene_metrics(prepare_raster_scene_for_r0(make_raster_scene({**base, "blue": 0.09})), RASTER_POLICY)
    assert np.isnan(bluish["candidate_ndvi"]).all()


@pytest.mark.xfail(strict=True, reason="raster.py lands in Task 5")
def test_missing_role_band_raises():
    from tests.conftest import make_raster_scene
    from spires_r0.raster import prepare_raster_scene_for_r0
    s = make_raster_scene({"blue": .05, "green": .08, "red": .1, "nir": .4}, bands=["blue", "green", "red", "nir"])
    with pytest.raises(ValueError, match="swir1"):
        compute_scene_metrics(prepare_raster_scene_for_r0(s), RASTER_POLICY)
