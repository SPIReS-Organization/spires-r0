# spires-r0

Background (R₀) reflectance production for the SPIReS package family: the
snow-free reference reflectance the inversion uses to unmix each scene.

This package is part of the [SPIReS family](https://spires.readthedocs.io/).

```{note}
**Status: scaffolding.** `spires-r0` is an early scaffold — the public API is
not implemented yet, so this site is a placeholder. It will grow an API
reference (autodoc) as the background-reflectance code lands. Track progress in
the [repository](https://github.com/SPIReS-Organization/spires-r0).
```

## Planned scope

- Produce background (R₀, snow-free) reflectance against the spectra / r0
  contract boundary.
- Consume `spires-io`-loaded scenes; feed `spires-inversion`.
