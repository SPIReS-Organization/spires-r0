# spires-r0

Background (R_0) reflectance production for the
[SPIReS](https://github.com/SPIReS-Organization) package family: builds the
snow-free background reflectance the inversion uses as a reference.

Consumes prepared scenes from
[`spires-io`](https://github.com/SPIReS-Organization/spires-io) and produces
background spectra for the inversion boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).

## Current API

- `spires_r0.core`: shared multisensor R0 compositing helpers
- `spires_r0.viirs`: VIIRS defaults and source/time-series builders
- `spires_r0.modis`: MODIS defaults and source/time-series builders

The source builders accept prepared `xarray.Dataset` scenes or raw paths that
`spires-io` can prepare, then select an annual snow-free background using the
same NDVI/NDSI/blue-reflectance rules migrated from the original SpiPy workflow.
