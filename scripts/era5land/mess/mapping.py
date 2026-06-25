from typing import Any, Dict

import xarray as xr


def identify_data_var(ds: xr.Dataset, rule: Dict[str, Any]) -> str:
    """Identify the data variable to convert from a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset.
    rule : dict[str, Any]
        Conversion rule.

    Returns
    -------
    str
        Name of the selected data variable.
    """
    explicit = rule.get("source_name_in_dataset")
    if explicit:
        if explicit not in ds.data_vars:
            raise KeyError(
                f"Configured data variable {explicit!r} not present in dataset."
            )
        return str(explicit)

    for candidate in (rule["source"], rule["target"]):
        if candidate in ds.data_vars:
            return str(candidate)

    if len(ds.data_vars) == 1:
        return str(next(iter(ds.data_vars)))

    raise ValueError(
        "Could not infer which data variable to use for "
        f"{rule['source']!r}; dataset variables are {list(ds.data_vars)!r}."
    )


def apply_variable_mapping(
    da: xr.DataArray,
    *,
    rule: Dict[str, Any],
    source_cfg: Dict[str, Any],
) -> xr.DataArray:
    """Apply scale, offset, variable name, unit, and metadata updates.

    Parameters
    ----------
    da : xarray.DataArray
        Source data array.
    rule : dict[str, Any]
        Conversion rule.
    source_cfg : dict[str, Any]
        Source configuration.

    Returns
    -------
    xarray.DataArray
        Converted data array with updated attributes and target name.
    """
    del source_cfg
    result = da
    factor = float(rule.get("factor", rule.get("scale", 1.0)))
    offset = float(rule.get("offset", 0.0))
    if factor != 1.0 or offset != 0.0:
        result = result * factor + offset

    attrs = dict(da.attrs)
    source_units = rule.get("source_units")
    if source_units and "units" not in attrs:
        attrs["units"] = source_units
    attrs["original_name"] = str(rule["source"])
    if "units" in attrs:
        attrs["original_units"] = str(attrs["units"])
    target_units = rule.get("target_units")
    if target_units:
        attrs["units"] = str(target_units)
    extra_attrs = rule.get("attrs", {})
    if not isinstance(extra_attrs, dict):
        raise TypeError(f"'attrs' for {rule['source']!r} must be a JSON object.")
    attrs.update(extra_attrs)
    result.attrs = attrs
    result.name = str(rule["target"])
    return result


def apply_rule_coordinates(ds: xr.Dataset, *, rule: Dict[str, Any]) -> xr.Dataset:
    """Attach rule-defined coordinates to a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to augment.
    rule : dict[str, Any]
        Conversion rule.

    Returns
    -------
    xarray.Dataset
        Dataset with any configured coordinates attached.
    """
    coordinates = rule.get("coordinates", {})
    if not coordinates:
        return ds
    if not isinstance(coordinates, dict):
        raise TypeError(f"'coordinates' for {rule['source']!r} must be a JSON object.")

    result = ds
    for coord_name, spec in coordinates.items():
        if isinstance(spec, dict):
            if "value" not in spec:
                raise KeyError(
                    f"Coordinate {coord_name!r} for {rule['source']!r} "
                    "must define a 'value'."
                )
            coord_attrs = spec.get("attrs", {})
            if not isinstance(coord_attrs, dict):
                raise TypeError(
                    f"'attrs' for coordinate {coord_name!r} in {rule['source']!r} "
                    "must be a JSON object."
                )
            coord_value = spec["value"]
        else:
            coord_value = spec
            coord_attrs = {}

        result = result.assign_coords(
            {
                str(coord_name): xr.DataArray(
                    coord_value,
                    attrs=dict(coord_attrs),
                )
            }
        )

    return result


def transform_rule_dataset(
    ds: xr.Dataset,
    *,
    rule: Dict[str, Any],
    source_cfg: Dict[str, Any],
) -> xr.Dataset:
    """Convert a source dataset for a single rule.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset.
    rule : dict[str, Any]
        Conversion rule.
    source_cfg : dict[str, Any]
        Source configuration.

    Returns
    -------
    xarray.Dataset
        Converted dataset containing the rule's target variable.
    """
    data_var = identify_data_var(ds, rule)
    converted = apply_variable_mapping(ds[data_var], rule=rule, source_cfg=source_cfg)
    result = converted.to_dataset(name=str(rule["target"]))
    return apply_rule_coordinates(result, rule=rule)
