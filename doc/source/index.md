# spires-r0

Background (R₀) reflectance production for the SPIReS package family: the
snow-free reference reflectance the inversion uses to unmix each scene.

This package is part of the [SPIReS family](https://spires.readthedocs.io/).

## Implemented scope

- Prepare explicit VIIRS and MODIS scene inventories through `spires-io`.
- Build optional in-memory or Zarr-backed time stacks.
- Select snow-free background spectra using the NDVI, NDSI,
  blue-reflectance, and view-geometry rules migrated from SpiPy.
- Record source time/index, observation count, rule choice, and geometry.
- Validate and atomically write reusable NetCDF R0 artifacts.
- Produce canonical float32 `(y, x, band)` background spectra that
  `spires-io` loads directly into `SpiresData.background`.

## API reference

```{eval-rst}
.. automodule:: spires_r0
   :members:
```
