import json
import numpy as np
import pytest
import xarray as xr
from spires_contract.spectra import validate_background_spectra

from spires_r0.core import validate_r0_dataset
from spires_r0.raster import build_raster_r0, prepare_raster_scene_for_r0
from tests.conftest import make_raster_scene

SNOWFREE = {"blue": 0.05, "green": 0.08, "red": 0.10, "nir": 0.40, "swir1": 0.25, "swir2": 0.15}


def test_prepare_adds_r0_staging_variables():
    p = prepare_raster_scene_for_r0(make_raster_scene(SNOWFREE))
    assert set(("reflectance", "valid_r0_mask", "solar_zenith")) <= set(p.data_vars)
    assert "sensor_zenith" not in p and "mask_water" not in p
    assert p.attrs["acquisition_date"] == "2024-08-01T18:30:00"
    assert p.rio.crs is not None


def _dirs(tmp_path, names):
    out = []
    for n in names:
        d = tmp_path / "landsat" / n; d.mkdir(parents=True); out.append(d)
    return out


def test_build_picks_max_ndvi_ties_to_earliest_and_validates(tmp_path):
    dirs = _dirs(tmp_path, ["B_late", "A_early", "C_hi"])
    scenes = {
        "A_early": make_raster_scene(SNOWFREE, acquisition_time="2024-07-01T18:00:00"),
        "B_late": make_raster_scene({**SNOWFREE, "swir2": 0.9}, acquisition_time="2024-09-01T18:00:00"),  # same NDVI
        "C_hi": make_raster_scene({**SNOWFREE, "nir": 0.6}, acquisition_time="2024-08-01T18:00:00"),
    }
    out = tmp_path / "r0.nc"
    ds = build_raster_r0(dirs, sensor="landsat", r0_path=out, open_scene=lambda d: scenes[d.name], prefetch=False)
    assert np.allclose(ds["r0_reflectance"].sel(band="nir").values, 0.6)
    assert (ds["r0_count"].values == 3).all()
    assert np.isnan(ds["r0_sensor_zenith"].values).all()
    validate_background_spectra(ds["r0_reflectance"])
    with xr.open_dataset(out) as disk:
        validate_r0_dataset(disk, expected_band_count=6, max_scene_index=2,
                            expected_source_inventory_digest=None, expected_build_signature=None)
        assert disk.rio.crs is not None
        assert json.loads(disk.attrs["r0_source_ids"]) == ["A_early", "B_late", "C_hi"]
    # tie case: drop C → A (earlier) must win over B even though B is fed... (dirs are sorted by name so A is fed first anyway;
    # feed B first explicitly to prove time, not feed order, decides)
    ds2 = build_raster_r0([dirs[0], dirs[1]], sensor="landsat", r0_path=tmp_path / "r0b.nc",
                          open_scene=lambda d: scenes[d.name], prefetch=False)
    assert np.allclose(ds2["r0_reflectance"].sel(band="swir2").values, 0.15)
    assert (ds2["r0_source_time"].values.astype("datetime64[D]") == np.datetime64("2024-07-01")).all()


def test_all_snow_pixels_fall_back_to_min_blue_rule(tmp_path):
    dirs = _dirs(tmp_path, ["A"])
    snow = make_raster_scene({**SNOWFREE, "blue": 0.6, "green": 0.9, "swir1": 0.1})  # NDSI>0, blue≥0.10
    ds = build_raster_r0(dirs, sensor="landsat", r0_path=tmp_path / "r0.nc", open_scene=lambda d: snow, prefetch=False)
    assert ds["r0_used_min_blue_rule"].values.all()
    assert np.allclose(ds["r0_reflectance"].sel(band="blue").values, 0.6)


def test_shifted_scene_is_reindexed_onto_first_grid(tmp_path):
    dirs = _dirs(tmp_path, ["A", "B"])
    a = make_raster_scene(SNOWFREE, nx=3)
    b = make_raster_scene({**SNOWFREE, "nir": 0.6}, nx=3, x0=300000.0 + 30.0)
    ds = build_raster_r0(dirs, sensor="landsat", r0_path=tmp_path / "r0.nc",
                         open_scene=lambda d: {"A": a, "B": b}[d.name], prefetch=False)
    nir = ds["r0_reflectance"].sel(band="nir").values
    assert np.allclose(nir[:, 0], 0.40) and np.allclose(nir[:, 1:], 0.60)


def test_crs_mismatch_raises(tmp_path):
    dirs = _dirs(tmp_path, ["A", "B"])
    a = make_raster_scene(SNOWFREE); b = make_raster_scene(SNOWFREE).rio.write_crs("EPSG:32610")
    with pytest.raises(ValueError, match="CRS"):
        build_raster_r0(dirs, sensor="landsat", r0_path=tmp_path / "r0.nc",
                        open_scene=lambda d: {"A": a, "B": b}[d.name], prefetch=False)


def test_existing_output_reused_unless_overwrite(tmp_path):
    dirs = _dirs(tmp_path, ["A"]); n = {"c": 0}
    def opener(d): n["c"] += 1; return make_raster_scene(SNOWFREE)
    p = tmp_path / "r0.nc"
    build_raster_r0(dirs, sensor="landsat", r0_path=p, open_scene=opener, prefetch=False)
    build_raster_r0(dirs, sensor="landsat", r0_path=p, open_scene=opener, prefetch=False)
    assert n["c"] == 1
    build_raster_r0(dirs, sensor="landsat", r0_path=p, open_scene=opener, prefetch=False, overwrite=True)
    assert n["c"] == 2


def test_no_scenes_raises(tmp_path):
    with pytest.raises(ValueError, match="no scene"):
        build_raster_r0([], sensor="landsat", r0_path=tmp_path / "r0.nc", prefetch=False)
