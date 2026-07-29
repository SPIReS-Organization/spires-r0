"""Output serialization helpers for R0 products."""

from __future__ import annotations

import json
from numbers import Number

import numpy as np
import xarray as xr


NETCDF_ATTR_TYPES = (str, Number, np.number, bytes)
NETCDF_RESERVED_ATTRS = {"_FillValue", "missing_value", "add_offset", "scale_factor", "DIMENSION_LIST"}


def _netcdf_safe_attr_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value.tolist()
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        array_value = np.asarray(value, dtype=object)
        if array_value.ndim <= 1:
            return list(value)
        return json.dumps(value)
    if isinstance(value, NETCDF_ATTR_TYPES):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _is_fill_value_compatible(variable: xr.DataArray, value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.upper() in {"N/A", "NA"}:
        return False
    try:
        np.asarray(value, dtype=variable.dtype)
    except (TypeError, ValueError):
        return False
    return True


def _is_valid_range_compatible(variable: xr.DataArray, value) -> bool:
    try:
        values = np.asarray(value, dtype=variable.dtype)
    except (TypeError, ValueError):
        return False
    return values.ndim <= 1 and values.size == 2


def _sanitize_variable_attrs(variable: xr.DataArray) -> dict[str, object]:
    sanitized_attrs = {}
    for key, value in variable.attrs.items():
        if key in NETCDF_RESERVED_ATTRS:
            continue
        if key in {"_FillValue", "missing_value"} and not _is_fill_value_compatible(variable, value):
            continue
        if key == "valid_range" and not _is_valid_range_compatible(variable, value):
            continue
        sanitized_attrs[key] = _netcdf_safe_attr_value(value)
    return sanitized_attrs


def sanitize_netcdf_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return a copy of a dataset with NetCDF-safe attributes."""
    sanitized = dataset.copy()
    sanitized.attrs = {key: _netcdf_safe_attr_value(value) for key, value in dataset.attrs.items()}
    for variable_name in sanitized.variables:
        sanitized[variable_name].attrs = _sanitize_variable_attrs(sanitized[variable_name])
    return sanitized
