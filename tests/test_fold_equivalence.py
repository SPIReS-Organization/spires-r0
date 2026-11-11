import numpy as np
import pytest
import xarray as xr

from spires_r0.fold import MODIS_POLICY, R0Composite, policy_from_legacy_kwargs
from tests.conftest import random_stack
from tests.reference_cube import reference_build_r0

COMPARE = ["r0_reflectance", "r0_source_index", "r0_source_time", "r0_used_min_blue_rule",
           "r0_used_water_blue_rule", "r0_count", "r0_sensor_zenith", "r0_sensor_azimuth",
           "r0_solar_zenith", "r0_solar_azimuth", "max_ndvi", "min_ndsi", "min_blue_metric",
           "min_water_blue_metric", "has_negative_ndsi"]


def _identity(src, dst):
    return dst


def _fold(scenes, policy):
    comp = R0Composite(scenes[0], policy)
    for s in scenes:
        comp.update(s)
    return comp.finalize()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("n_time", [1, 3, 8])
def test_fold_equals_cube_reducer(seed, n_time):
    rng = np.random.default_rng(seed)
    scenes = random_stack(rng, n_time=n_time)
    cube = xr.concat([s.expand_dims(time=[np.datetime64(s.attrs["acquisition_date"])]) for s in scenes], dim="time")
    ref = reference_build_r0(
        cube, logger=None, event_name="ref", max_sensor_zenith=30.0, ndvi_tie_epsilon=0.02, ndvi_red_band="1", ndvi_nir_band="2",
        ndsi_visible_band="4", ndsi_swir_band="6", blue_band="3", min_blue_reflectance=0.10,
        copy_spatial_metadata_fn=_identity,
    )
    # depth = n_time makes the fold exact by construction; the default depth is tested separately
    got = _fold(scenes, policy_from_legacy_kwargs(MODIS_POLICY, tiebreak_depth=n_time))
    for v in COMPARE:
        a, b = got[v].values, ref[v].values
        assert a.dtype == b.dtype, f"{v}: {a.dtype} vs {b.dtype}"
        if a.dtype.kind == "f":
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=v)
            np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)], err_msg=v)
        elif a.dtype.kind == "M":
            np.testing.assert_array_equal(np.isnat(a), np.isnat(b), err_msg=v)
            np.testing.assert_array_equal(a[~np.isnat(a)].astype("datetime64[ns]"), b[~np.isnat(b)].astype("datetime64[ns]"), err_msg=v)
        else:
            np.testing.assert_array_equal(a, b, err_msg=v)


def test_default_depth_reports_overflow_but_matches_where_not_overflowed(rng):
    scenes = random_stack(rng, n_time=12)
    exact = _fold(scenes, policy_from_legacy_kwargs(MODIS_POLICY, tiebreak_depth=12))
    approx = _fold(scenes, MODIS_POLICY)  # depth 4
    overflow = approx["r0_tiebreak_window_overflow"].values.astype(bool)
    assert approx.attrs["r0_tiebreak_window_overflow_pixels"] == int(overflow.sum())
    same = ~overflow
    np.testing.assert_array_equal(approx["r0_source_index"].values[same], exact["r0_source_index"].values[same])


def test_fold_is_order_independent(rng):
    scenes = random_stack(rng, n_time=6)
    a = _fold(scenes, policy_from_legacy_kwargs(MODIS_POLICY, tiebreak_depth=6))
    b = _fold(scenes[::-1], policy_from_legacy_kwargs(MODIS_POLICY, tiebreak_depth=6))
    for v in ("r0_source_time", "r0_count", "r0_used_min_blue_rule"):
        np.testing.assert_array_equal(a[v].values, b[v].values, err_msg=v)
    np.testing.assert_array_equal(a["r0_reflectance"].values, b["r0_reflectance"].values)
