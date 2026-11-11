import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

MODIS_BANDS = ["1", "2", "3", "4", "5", "6", "7"]
RASTER_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
CRS = "EPSG:32611"


def _grid(ny, nx, x0=300000.0, y0=4170000.0, res=30.0):
    return y0 - res * (np.arange(ny) + 0.5), x0 + res * (np.arange(nx) + 0.5)


def make_prepared_scene(
    reflectance: np.ndarray,            # (ny, nx, 7) float32
    *,
    acquisition_date: str,
    valid: np.ndarray | None = None,
    sensor_zenith: np.ndarray | float = 10.0,
    sensor_azimuth: np.ndarray | float = 90.0,
    solar_zenith: float = 40.0,
    solar_azimuth: float = 150.0,
    mask_water: np.ndarray | None = None,
    bands=MODIS_BANDS,
) -> xr.Dataset:
    """MODIS-shaped prepared scene (what spires_io.modis hands to spires_r0)."""
    refl = np.asarray(reflectance, dtype=np.float32)
    ny, nx, _ = refl.shape
    y, x = _grid(ny, nx)
    f = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float32), (ny, nx)).copy()
    return xr.Dataset(
        {
            "reflectance": (("y", "x", "band"), refl),
            "sensor_zenith": (("y", "x"), f(sensor_zenith)),
            "sensor_azimuth": (("y", "x"), f(sensor_azimuth)),
            "solar_zenith": (("y", "x"), f(solar_zenith)),
            "solar_azimuth": (("y", "x"), f(solar_azimuth)),
            "valid_r0_mask": (("y", "x"), np.ones((ny, nx), bool) if valid is None else valid),
            "mask_water": (("y", "x"), np.zeros((ny, nx), bool) if mask_water is None else mask_water),
        },
        coords={"y": y, "x": x, "band": list(bands)},
        attrs={"acquisition_date": acquisition_date},
    )


def make_raster_scene(values, *, acquisition_time="2024-08-01T18:30:00", valid=None, ny=2, nx=3,
                      x0=300000.0, bands=RASTER_BANDS) -> xr.Dataset:
    """spires_io.raster_scene-shaped scene; `values` = {band: scalar} or (ny,nx,nb) array."""
    if isinstance(values, dict):
        refl = np.empty((ny, nx, len(bands)), np.float32)
        for k, b in enumerate(bands):
            refl[..., k] = values[b]
    else:
        refl = np.asarray(values, np.float32)
        ny, nx = refl.shape[:2]
    y, x = _grid(ny, nx, x0=x0)
    ds = xr.Dataset(
        {
            "reflectance": (("y", "x", "band"), refl),
            "valid": (("y", "x"), np.ones((ny, nx), bool) if valid is None else valid),
            "solar_zenith": (("y", "x"), np.full((ny, nx), 30.0, np.float32)),
        },
        coords={"y": y, "x": x, "band": list(bands)},
        attrs={"acquisition_time": acquisition_time, "platform": "test"},
    )
    return ds.rio.write_crs(CRS)


def random_stack(rng: np.random.Generator, *, n_time: int, ny: int = 6, nx: int = 5) -> list[xr.Dataset]:
    """Random MODIS-shaped scenes engineered to exercise every branch: NaNs, snow, water,
    invalid pixels, high zenith, and near-tie NDVI values (quantised to 0.01 so ε-window ties occur)."""
    scenes = []
    dates = np.datetime64("2024-07-01") + np.arange(n_time)
    for t in range(n_time):
        refl = rng.uniform(0.02, 0.6, (ny, nx, 7)).astype(np.float32)
        # quantise red/nir so NDVI collides within ε often
        refl[..., 0] = np.round(refl[..., 0], 2); refl[..., 1] = np.round(refl[..., 1], 2)
        snow = rng.random((ny, nx)) < 0.3
        refl[snow, 3] = 0.8; refl[snow, 5] = 0.1          # band 4 bright, band 6 dark → NDSI > 0
        nanpix = rng.random((ny, nx)) < 0.1
        refl[nanpix, :] = np.nan
        scenes.append(make_prepared_scene(
            refl,
            acquisition_date=str(dates[t]),
            valid=rng.random((ny, nx)) < 0.85,
            sensor_zenith=rng.uniform(0, 45, (ny, nx)).astype(np.float32),
            sensor_azimuth=rng.uniform(0, 360, (ny, nx)).astype(np.float32),
            mask_water=rng.random((ny, nx)) < 0.15,
        ))
    return scenes


@pytest.fixture
def rng():
    return np.random.default_rng(20261111)
