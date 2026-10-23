# Changelog

## 0.10.0 (unreleased)

### New features

- Add the `burst_id` argument to `Sentinel1SarProduct` to process a single SLC IW burst.
- Select a single SLC IW burst by position with a `measurement_group` like `IW1/VV/3`.
- Add the abstract `OrbitInterpolator` class; `SarProduct.orbit_interpolator()`
  returns the best one for the product.

### Breaking changes

- `OrbitPolyfitInterpolator.position()`, `velocity()` and `acceleration()` need
  an explicit time; `azimuth_time_range()` is removed.
- Remove the misspelled `OrbitPolyfitIterpolator` alias.
- Add `zarr` and `numcodecs` to the required dependencies.
- Remove the `lab` extra: `pip install sarsen[lab]` no longer works.

### Bug fixes

- `sarsen stc` no longer writes a stray `GTC.tif`.
- Fix the `grouping_area_factor` default in the `terrain_correction` docstring.
