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
same NDVI/NDSI/blue-reflectance rules carried over from an earlier
implementation.

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

All VIIRS and MODIS R0 builds hold the complete summer stack for one tile in
memory before selecting the background. Caller-supplied lazy datasets are loaded
once before selection. Allocate RAM for the scenes, concatenated stack and
selection intermediates; full-tile jobs can require several hundred GiB.

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
