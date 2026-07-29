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

## Optional execution features

Zarr-backed staging, Dask chunking, and progress bars are optional:

```bash
pip install "spires-r0[all]"
```
