# -*- coding: utf-8 -*-
"""Model-independent user specification for fixed or sampled n-point observables."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Any, Dict, List, Mapping


class ObservableSpecificationError(ValueError):
    """Raised when a user observable or its sampling request is malformed."""


DEFAULT_COORDINATE_CONVENTION = {
    "mode": "lattice_indices",
    "spacing_parameter": "lattice_spacing_a",
    "temporal_spacing": "same_as_spatial",
    "units": "natural_units_c_1",
}


def _coordinate_convention(value: Any) -> Dict[str, Any]:
    if value is None or value == "lattice_indices":
        return dict(DEFAULT_COORDINATE_CONVENTION)
    if value == "physical_coordinates":
        return {"mode": "physical_coordinates", "units": "model_native"}
    _require(isinstance(value, dict),
             "observable.coordinate_convention must be an object or known mode")
    mode = value.get("mode")
    _require(mode in ("lattice_indices", "physical_coordinates"),
             "coordinate convention mode must be lattice_indices or physical_coordinates")
    if mode == "lattice_indices":
        normalized = dict(DEFAULT_COORDINATE_CONVENTION)
        normalized.update(value)
        _require(_nonempty(normalized.get("spacing_parameter")),
                 "lattice_indices requires spacing_parameter")
        _require(normalized.get("temporal_spacing") == "same_as_spatial",
                 "only same_as_spatial temporal spacing is currently supported")
        return normalized
    normalized = {"mode": "physical_coordinates", "units": "model_native"}
    normalized.update(value)
    _require(_nonempty(normalized.get("units")),
             "physical_coordinates requires units")
    return normalized


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservableSpecificationError(message)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def legacy_observable_specification(expression: str) -> Dict[str, Any]:
    """Wrap the former free-form observable string without guessing its structure."""
    _require(_nonempty(expression), "observable expression must be nonempty")
    return {
        "observable_schema_version": "1.0",
        "name": "legacy_free_form_observable",
        "expression": expression,
        "ordering": "as_written",
        "insertions": [
            {
                "id": "legacy_expression",
                "operator": expression,
                "position": "as_written_in_expression",
            }
        ],
        "requested_components": ["real", "imag"],
        "sampling": {
            "mode": "explicit_points",
            "points": [{"sample_id": "default", "values": {}}],
        },
    }


def concise_observable_specification(specification: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand the user-facing operator/grid shorthand into the strict schema."""
    expression = specification.get("expression")
    operators = specification.get("operators")
    _require(_nonempty(expression), "observable.expression must be nonempty")
    _require(isinstance(operators, list) and operators,
             "observable.operators must be a nonempty list")
    insertions = []
    identifiers = set()
    for index, raw in enumerate(operators, start=1):
        _require(isinstance(raw, dict),
                 "every observable operator must be an object")
        operator = raw.get("operator")
        position = raw.get("position")
        _require(_nonempty(operator), "every observable operator needs operator")
        _require("position" in raw, "every observable operator needs position")
        identifier = raw.get("id", "operator_{}".format(index))
        _require(_nonempty(identifier) and identifier not in identifiers,
                 "observable operator IDs must be unique nonempty strings")
        identifiers.add(identifier)
        insertion = dict(raw)
        insertion.update({"id": identifier, "operator": operator, "position": position})
        insertions.append(insertion)

    position_grid = specification.get("position_grid", {})
    if isinstance(position_grid, dict):
        if position_grid:
            sampling = {
                "mode": "cartesian_product",
                "variables": position_grid,
                "maximum_points": specification.get("maximum_points", 10000),
            }
        else:
            sampling = {
                "mode": "explicit_points",
                "points": [{"sample_id": "default", "values": {}}],
            }
    elif isinstance(position_grid, list) and position_grid:
        sampling = {
            "mode": "explicit_points",
            "points": [
                {
                    "sample_id": "sample_{:04d}".format(index),
                    "values": values,
                }
                for index, values in enumerate(position_grid)
            ],
        }
    else:
        raise ObservableSpecificationError(
            "observable.position_grid must be an object or nonempty list"
        )

    expanded = {
        "observable_schema_version": "1.0",
        "name": specification.get("name", "user_observable"),
        "expression": expression,
        "ordering": specification.get("ordering", "as_written"),
        "insertions": insertions,
        "requested_components": specification.get(
            "components", specification.get("requested_components", ["real", "imag"])
        ),
        "sampling": sampling,
        "coordinate_convention": _coordinate_convention(
            specification.get("coordinate_convention")
        ),
    }
    transporters = specification.get(
        "transporters", specification.get("gauge_transporters")
    )
    if transporters is not None:
        expanded["gauge_transporters"] = transporters
    if "constraints" in specification:
        expanded["constraints"] = specification["constraints"]
    if "matrix_elements" in specification:
        expanded["matrix_elements"] = specification["matrix_elements"]
    return expanded


def validate_observable_specification(specification: Any) -> Dict[str, Any]:
    """Validate and return a JSON-safe copy of a generic n-point specification."""
    if isinstance(specification, str):
        specification = legacy_observable_specification(specification)
    _require(isinstance(specification, dict), "agent1.observable must be an object")
    if "observable_schema_version" not in specification:
        specification = concise_observable_specification(specification)
    spec = json.loads(json.dumps(specification, ensure_ascii=False))
    _require(
        spec.get("observable_schema_version") == "1.0",
        "observable_schema_version must be '1.0'",
    )
    for key in ("name", "expression", "ordering"):
        _require(_nonempty(spec.get(key)), "observable.{} must be nonempty".format(key))
    spec["coordinate_convention"] = _coordinate_convention(
        spec.get("coordinate_convention")
    )

    insertions = spec.get("insertions")
    _require(isinstance(insertions, list) and insertions,
             "observable.insertions must be a nonempty list")
    identifiers = set()
    for index, insertion in enumerate(insertions):
        _require(isinstance(insertion, dict),
                 "observable.insertions[{}] must be an object".format(index))
        identifier = insertion.get("id")
        _require(_nonempty(identifier) and identifier not in identifiers,
                 "observable insertion IDs must be unique nonempty strings")
        identifiers.add(identifier)
        _require(_nonempty(insertion.get("operator")),
                 "observable insertion operators must be nonempty strings")
        _require("position" in insertion,
                 "every observable insertion must declare position")

    components = spec.get("requested_components")
    _require(
        isinstance(components, list)
        and components
        and len(components) == len(set(components))
        and set(components).issubset({"real", "imag"}),
        "requested_components must contain unique real and/or imag values",
    )
    transporters = spec.get("gauge_transporters", [])
    _require(isinstance(transporters, list), "gauge_transporters must be a list")
    constraints = spec.get("constraints", [])
    _require(
        isinstance(constraints, list) and all(_nonempty(item) for item in constraints),
        "observable constraints must be strings",
    )

    matrix_elements = spec.get("matrix_elements")
    if matrix_elements is not None:
        _require(
            isinstance(matrix_elements, list) and matrix_elements,
            "observable.matrix_elements must be a nonempty list when supplied",
        )
        term_ids = set()
        normalized_terms = []
        for index, term in enumerate(matrix_elements, start=1):
            _require(isinstance(term, dict), "every matrix element must be an object")
            term_id = term.get("id", "matrix_element_{}".format(index))
            coefficient = term.get("coefficient")
            bra = term.get("bra")
            ket = term.get("ket")
            _require(
                _nonempty(term_id) and term_id not in term_ids,
                "matrix-element IDs must be unique nonempty strings",
            )
            _require(
                isinstance(coefficient, (int, float))
                and not isinstance(coefficient, bool)
                and math.isfinite(float(coefficient)),
                "matrix-element coefficients must be finite real numbers",
            )
            _require(_nonempty(bra) and _nonempty(ket),
                     "every matrix element needs bra and ket state labels")
            term_ids.add(term_id)
            normalized_terms.append({
                "id": term_id,
                "coefficient": float(coefficient),
                "bra": bra,
                "ket": ket,
            })
        spec["matrix_elements"] = normalized_terms

    # Expansion also validates the complete sampling request and unique IDs.
    expand_observable_samples(spec)
    return spec


def expand_observable_samples(specification: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand explicit assignments or a model/dimension-independent Cartesian grid."""
    sampling = specification.get("sampling")
    _require(isinstance(sampling, dict), "observable.sampling must be an object")
    mode = sampling.get("mode")
    maximum = sampling.get("maximum_points", 10000)
    _require(
        isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0,
        "sampling.maximum_points must be a positive integer",
    )
    expanded: List[Dict[str, Any]] = []
    if mode == "explicit_points":
        points = sampling.get("points")
        _require(isinstance(points, list) and points,
                 "explicit_points sampling requires a nonempty points list")
        for point in points:
            _require(isinstance(point, dict), "every sampling point must be an object")
            sample_id = point.get("sample_id")
            values = point.get("values")
            _require(_nonempty(sample_id), "every sampling point needs sample_id")
            _require(isinstance(values, dict), "sampling point values must be an object")
            expanded.append({"sample_id": sample_id, "values": values})
    elif mode == "cartesian_product":
        variables = sampling.get("variables")
        _require(isinstance(variables, dict) and variables,
                 "cartesian_product sampling requires nonempty variables")
        names = sorted(variables)
        axes = []
        for name in names:
            _require(_nonempty(name), "sampling variable names must be nonempty")
            values = variables[name]
            _require(isinstance(values, list) and values,
                     "every sampling variable must have a nonempty values list")
            axes.append(values)
        for index, combination in enumerate(itertools.product(*axes)):
            expanded.append({
                "sample_id": "sample_{:04d}".format(index),
                "values": dict(zip(names, combination)),
            })
            _require(len(expanded) <= maximum,
                     "observable sampling exceeds maximum_points={}".format(maximum))
    else:
        raise ObservableSpecificationError(
            "sampling.mode must be explicit_points or cartesian_product"
        )
    _require(len(expanded) <= maximum,
             "observable sampling exceeds maximum_points={}".format(maximum))
    identifiers = [point["sample_id"] for point in expanded]
    _require(len(identifiers) == len(set(identifiers)),
             "observable sample IDs must be unique")
    return expanded


_INTEGER_EXPRESSION = re.compile(
    r"^\s*([+-]?\d+|[A-Za-z_]\w*)(?:\s*([+-])\s*(\d+))?\s*$"
)


def _integer_symbols(parameters: Mapping[str, Any]) -> Dict[str, int]:
    """Collect unambiguous integer parameter aliases from user input or Agent2."""
    entries = parameters.get("parameters")
    if isinstance(entries, list):
        raw_entries = entries
    else:
        raw_entries = [
            {"name": name, "symbol": name, "selected_value": value}
            for name, value in parameters.items()
        ]

    candidates: Dict[str, List[int]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("selected_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if int(value) != value:
            continue
        aliases = {entry.get("name"), entry.get("symbol")}
        name = entry.get("name")
        if isinstance(name, str) and ":" in name:
            aliases.add(name.rsplit(":", 1)[1].strip())
        for alias in aliases:
            if isinstance(alias, str) and re.fullmatch(r"[A-Za-z_]\w*", alias):
                candidates.setdefault(alias, []).append(int(value))
    return {
        alias: values[0]
        for alias, values in candidates.items()
        if len(set(values)) == 1
    }


def _resolve_integer_expression(value: Any, symbols: Mapping[str, int]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    _require(isinstance(value, str), "grid range bounds must be integer expressions")
    match = _INTEGER_EXPRESSION.fullmatch(value)
    _require(match is not None, "unsupported grid expression: {}".format(value))
    base, sign, offset = match.groups()
    if re.fullmatch(r"[+-]?\d+", base):
        result = int(base)
    else:
        _require(base in symbols, "unknown integer grid parameter: {}".format(base))
        result = symbols[base]
    if offset:
        result += int(offset) if sign == "+" else -int(offset)
    return result


def _inclusive_range(start: int, stop: int, step: int, maximum: int) -> List[int]:
    _require(step != 0, "grid range step cannot be zero")
    _require((stop - start) * step >= 0, "grid range step points away from its endpoint")
    count = abs(stop - start) // abs(step) + 1
    _require(count <= maximum,
             "observable sampling exceeds maximum_points={}".format(maximum))
    return list(range(start, stop + (1 if step > 0 else -1), step))


def _resolve_grid_axis(
    values: List[Any], symbols: Mapping[str, int], maximum: int
) -> List[Any]:
    # Preferred compact form: ["0:L-1"] or ["0:L-1:2"]. Bounds are inclusive.
    if len(values) == 1 and isinstance(values[0], str) and ":" in values[0]:
        pieces = [piece.strip() for piece in values[0].split(":")]
        _require(len(pieces) in (2, 3), "grid range must be start:stop[:step]")
        start = _resolve_integer_expression(pieces[0], symbols)
        stop = _resolve_integer_expression(pieces[1], symbols)
        step = (
            _resolve_integer_expression(pieces[2], symbols)
            if len(pieces) == 3 else (1 if stop >= start else -1)
        )
        return _inclusive_range(start, stop, step, maximum)

    # Backward compatibility for the former UI example: 0,1,...,L-1.
    ellipses = [index for index, value in enumerate(values) if value in ("...", "…")]
    if ellipses:
        _require(len(ellipses) == 1, "a grid axis may contain only one ellipsis")
        index = ellipses[0]
        _require(index > 0 and index == len(values) - 2,
                 "ellipsis form must be prefix,...,endpoint")
        prefix = [_resolve_integer_expression(value, symbols) for value in values[:index]]
        stop = _resolve_integer_expression(values[-1], symbols)
        step = prefix[-1] - prefix[-2] if len(prefix) >= 2 else (1 if stop >= prefix[0] else -1)
        expanded = _inclusive_range(prefix[0], stop, step, maximum)
        _require(expanded[:len(prefix)] == prefix,
                 "values before the ellipsis do not match the inferred step")
        return expanded

    resolved: List[Any] = []
    for value in values:
        if isinstance(value, str) and _INTEGER_EXPRESSION.fullmatch(value):
            resolved.append(_resolve_integer_expression(value, symbols))
        else:
            resolved.append(value)
    return resolved


def resolve_observable_sampling(
    specification: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Dict[str, Any]:
    """Resolve compact grid ranges using already-authoritative model parameters."""
    spec = json.loads(json.dumps(specification, ensure_ascii=False))
    sampling = spec.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("mode") != "cartesian_product":
        return spec
    variables = sampling.get("variables")
    _require(isinstance(variables, dict), "sampling.variables must be an object")
    symbols = _integer_symbols(parameters)
    maximum = sampling.get("maximum_points", 10000)
    sampling["variables"] = {
        name: _resolve_grid_axis(values, symbols, maximum)
        if isinstance(values, list) else values
        for name, values in variables.items()
    }
    # Reuse the normal size and shape checks after resolution.
    expand_observable_samples(spec)
    return spec


def observable_specification_digest(specification: Any) -> str:
    validated = validate_observable_specification(specification)
    canonical = json.dumps(
        validated, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_observable_specification(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return validate_observable_specification(json.load(handle))
