# spires-r0

Background (R_0) reflectance production for the
[SPIReS](https://github.com/SPIReS-Organization) package family: builds the
snow-free background reflectance the inversion uses as a reference.

Consumes prepared scenes from
[`spires-io`](https://github.com/SPIReS-Organization/spires-io) and produces
background spectra for the inversion boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).

## Current API

- `build_viirs_timeseries(...)` and `build_modis_timeseries(...)` prepare
  explicit source inventories into time stacks.
- `build_viirs_r0(...)` and `build_modis_r0(...)` composite already prepared
  time stacks.
- `build_viirs_r0_from_sources(...)` and
  `build_modis_r0_from_sources(...)` provide the complete source-to-artifact
  workflow, including validated atomic NetCDF writing and reuse.
- `validate_r0_dataset(...)` validates a complete R0 artifact.

The source builders accept prepared `xarray.Dataset` scenes or raw paths that
`spires-io` can prepare, then select an annual snow-free background using the
same NDVI/NDSI/blue-reflectance rules migrated from the original SpiPy workflow.

## One streaming engine for every sensor

All R0 products — MODIS, VIIRS, Landsat 8/9, Sentinel-2, HLS L30/S30 — are built by the same
per-scene fold (`spires_r0.fold.R0Composite`). Sensor differences live in an `R0Policy`:
band roles, whether sensor geometry drives the 30° zenith cut and the near-nadir tie-break
(MODIS/VIIRS: yes, ε = 0.02 NDVI window; Landsat/S2/HLS: no — ties go to the earliest
acquisition, and `r0_sensor_zenith/azimuth` are NaN), and raster-only candidate filters
(blue > 0.01, blue < green). Memory is ~(tiebreak_depth + 2) scenes regardless of stack length.

- `build_modis_r0(...)`, `build_viirs_r0(...)`, `*_from_sources(...)`: unchanged signatures and
  output schema; `*_from_sources` streams scene by scene and never materialises a time stack.
- `build_raster_r0(scene_dirs, sensor="hls_l30", r0_path=..., water_mask_dir=..., cache_root=...)`:
  Landsat/S2/HLS from `spires_io`-materialised scene directories. Grid = first scene in sorted-name
  order. Raster scenes carry no water flag, so pass `water_mask_dir` (a directory of
  `landWater2020_{lon}_{lat}.tif` tiles) or permanent water that is dark in every scene gets no
  background and the inversion skips it.
- `R0Cache`: keyed on the product's `spires_r0_build_signature_sha256`; hits are re-validated and copied.

## Scientific and file boundary

The canonical inversion input is `r0_reflectance`, a float32
`(y, x, band)` array satisfying the `spires-contract` background-spectra
contract. The surrounding R0 Dataset also records the selected source index and
time, observation count, selection-rule flags, and source geometry.

Written products use R0 schema version 1 and carry explicit product type,
completion status, time coverage, algorithm settings, and source-scene count.
An existing artifact is reused only after the complete Dataset passes
`validate_r0_dataset` and its source-inventory and build-configuration
signatures match the current request.

`spires-io` reads the product directly into the shared scientific object:

```python
import spires_io
from spires_contract import SpiresData, validate_for_inversion

scene = spires_io.prepare_scene_for_inversion(
    "VNP09GA.A2026182.h09v05.002.h5",
    sensor="viirs",
    platform="snpp",
)
background = spires_io.load_background_reflectance(
    "r0_20260601_20260930.nc",
    target_scene=scene,
)
data = SpiresData(scene=scene, background=background)
validate_for_inversion(data)
```

## In-memory execution

`build_modis_r0(timeseries)` / `build_viirs_r0(timeseries)` take a caller-supplied
summer stack for one tile; a lazy dataset is loaded once before selection, so
allocate RAM for the full stack. `*_from_sources(...)` and `build_raster_r0(...)`
stream instead and hold only ~(tiebreak_depth + 2) scenes at a time.

Zarr staging and chunked R0 execution have been removed. The retired
`zarr_path`, `zarr_mode` and `chunks` arguments raise `TypeError`, including when
set to `None`. Remove these arguments from old callers; use `r0_path` for the
atomic NetCDF product. Do not reinstate a disk-backed R0 execution mode.

Progress bars remain optional:

```bash
pip install "spires-r0[progress]"
```

## Georeferenced R0 grids

Georeferenced inputs and products must use pixel-center `x`/`y` coordinates that
agree with their scalar `spatial_ref` GeoTransform. R0 preparation, compositing,
validation, and reuse now enforce that contract. The current MODIS/VIIRS reader
derives centers from full-precision HDF-EOS tile bounds; do not round transform
strings or reuse source edge coordinates as centers.

For a legacy product whose scientific arrays are already on the correct tile,
`spires_r0.repair_r0_grid_metadata(dataset, reference_grid)` provides an explicit
metadata-only repair. Supply a reference grid independently parsed from its
source product. The function requires matching dimensions, CRS, and recorded
tile bounds, and accepts only canonical centers or the recognized legacy edge
coordinate convention. It never resamples spectra or repairs arbitrary shifts.
Back up the original file, verify unchanged scientific arrays, and replace it
atomically after validation. Bare, unreferenced in-memory grids remain supported
by the compositing API; they are not georeferenced operational products.
