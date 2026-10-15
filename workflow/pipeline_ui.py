# -*- coding: utf-8 -*-
"""Small local Streamlit front end for the concise Jianmu-Q pipeline input."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from pipeline_runtime import (
    PipelineError,
    find_approved_reuse_source_for_config,
    set_program_reuse_approval,
    validate_config,
)
from pipeline_support import atomic_json_write
from llm_support import (
    DEFAULT_UI_MODEL, MODEL_PRESETS_BY_PROVIDER,
    api_key_for_model, api_key_variable, provider_for_model,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIRECTORY = os.path.join(PROJECT_ROOT, "user_configs")
LOG_DIRECTORY = os.path.join(PROJECT_ROOT, "ui_logs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIRECTORY, "default_input.json")

MODEL_PRESETS = {
    "Schwinger model": [
        {"name": "mass_to_coupling_ratio_m_over_g", "value": "1.0"},
        {"name": "gauge_coupling_g", "value": "0.5"},
        {"name": "lattice_spacing_a", "value": "1.0"},
        {"name": "spatial_site_count", "value": "4"},
    ],
    "Z2 gauge theory": [
        {"name": "spatial_lattice_shape", "value": "[4, 4]"},
    ],
    "SU(2) gauge theory": [
        {"name": "spatial_lattice_shape", "value": "[4, 4]"},
    ],
}

DEFAULT_OPERATORS = [
    {
        "id": "bar_psi_zn",
        "operator": r"\bar{\psi}",
        "position": "t_index=z_index, x_index=z_index",
    },
    {
        "id": "psi_0",
        "operator": r"\psi",
        "position": "t_index=0, x_index=0",
    },
]

DEFAULT_TRANSPORTERS = [
    {
        "id": "W_zn_0",
        "from": "psi_0",
        "to": "bar_psi_zn",
        "path": "straight lattice path along n^mu=(1,1)",
    }
]

LIGHTCONE_CONVENTIONS = {
    'No light-cone coordinates': {},
    'Normalized: x±=(x0±x1)/√2': {
        "lightcone": {
            "coordinate_definition": "x^plus/minus=(x^0 plus/minus x^1)/sqrt(2)",
            "gamma_definition": "gamma^plus/minus=(gamma^0 plus/minus gamma^1)/sqrt(2)",
            "normalization": 0.7071067811865476,
        }
    },
    'Unnormalized: x±=x0±x1': {
        "lightcone": {
            "coordinate_definition": "x^plus/minus=x^0 plus/minus x^1",
            "gamma_definition": "gamma^plus/minus=gamma^0 plus/minus gamma^1",
            "normalization": 1.0,
        }
    },
}


class UIInputError(ValueError):
    """Raised when a UI table cannot be converted into concise pipeline JSON."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or (isinstance(value, str) and not value.strip())
    )


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def parse_value(value: Any) -> Any:
    """Parse JSON-like table values while keeping ordinary text user-friendly."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        raise UIInputError("value cannot be empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _canonical_lattice_parameter(name: str, value: Any) -> tuple[str, Any]:
    """Convert the former ``L*L = Nx*Ny`` UI spelling into one clear value."""
    legacy_name = re.fullmatch(
        r"number\s+of\s+lattice\s+sites?\s*:\s*l\s*\*\s*l",
        name.strip(),
        flags=re.IGNORECASE,
    )
    if not legacy_name or not isinstance(value, str):
        return name, value
    match = re.fullmatch(
        r"\s*l\s*\*\s*l\s*=\s*(\d+)\s*\*\s*(\d+)"
        r"(?:\s*=\s*(\d+))?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return name, value
    first, second = int(match.group(1)), int(match.group(2))
    stated_total = int(match.group(3)) if match.group(3) else first * second
    if first * second != stated_total:
        raise UIInputError("lattice shape and stated total site count disagree")
    return "spatial_lattice_shape", [first, second]


def parse_position(value: Any) -> Any:
    """Accept JSON or a compact ``coordinate=value`` position expression."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        raise UIInputError("operator position cannot be empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "=" not in text:
        return text
    position: Dict[str, Any] = {}
    for assignment in text.split(","):
        if "=" not in assignment:
            raise UIInputError("positions must use coordinate=value pairs")
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        if not name or name in position:
            raise UIInputError("position coordinate names must be unique and nonempty")
        position[name] = parse_value(raw_value)
    return position


def parse_grid_values(value: Any) -> List[Any]:
    """Accept a JSON list or a comma-separated sequence of grid values."""
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise UIInputError("position-grid values cannot be empty")
        if text.startswith("["):
            try:
                values = json.loads(text)
            except json.JSONDecodeError as error:
                raise UIInputError("invalid JSON list in position grid") from error
        else:
            values = [parse_value(item) for item in text.split(",")]
    else:
        raise UIInputError("position-grid values must be a list or comma-separated text")
    if not isinstance(values, list) or not values:
        raise UIInputError("position-grid values must be a nonempty list")
    return values


def rows_to_parameters(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    for row in rows:
        name = _text(row.get("name", ""))
        raw_value = row.get("value", "")
        if not name and _blank(raw_value):
            continue
        if not name:
            raise UIInputError("every model parameter needs a name")
        name, raw_value = _canonical_lattice_parameter(name, raw_value)
        if name in parameters:
            raise UIInputError("model parameter names must be unique")
        parameters[name] = parse_value(raw_value)
    if not parameters:
        raise UIInputError("at least one model parameter is required")
    return parameters


def rows_to_operators(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    operators: List[Dict[str, Any]] = []
    identifiers = set()
    for row in rows:
        operator = _text(row.get("operator", ""))
        position = row.get("position", "")
        identifier = _text(row.get("id", ""))
        if not operator and _blank(position) and not identifier:
            continue
        if not operator:
            raise UIInputError("every operator needs an operator expression")
        if identifier and identifier in identifiers:
            raise UIInputError("operator IDs must be unique")
        item = {"operator": operator, "position": parse_position(position)}
        if identifier:
            item["id"] = identifier
            identifiers.add(identifier)
        operators.append(item)
    if not operators:
        raise UIInputError("at least one observable operator is required")
    return operators


def rows_to_grid(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for row in rows:
        name = _text(row.get("variable", ""))
        values = row.get("values", "")
        if not name and _blank(values):
            continue
        if not name:
            raise UIInputError("every position-grid row needs a variable name")
        if name in grid:
            raise UIInputError("position-grid variable names must be unique")
        grid[name] = parse_grid_values(values)
    return grid


def rows_to_transporters(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    transporters = []
    for row in rows:
        item = {
            key: _text(row.get(key, ""))
            for key in ("id", "from", "to", "path")
        }
        if not any(item.values()):
            continue
        if not all(item.values()):
            raise UIInputError("each transporter needs id, from, to, and path")
        transporters.append(item)
    return transporters


def rows_to_states(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Convert the compact named-state table into pipeline state targets."""
    states: Dict[str, Any] = {}
    for row in rows:
        label = _text(row.get("label", ""))
        kind = _text(row.get("kind", ""))
        if not label and not kind:
            continue
        if not label or label in states:
            raise UIInputError("state labels must be unique and nonempty")
        if kind in {"Ground state / Vacuum", "Vacuum", "Ground state"}:
            # Vacuum and unrestricted ground state use the same runtime target.
            # The legacy spellings remain readable for old saved configurations.
            target: Any = "Ground state / vacuum"
        elif kind == "Sector eigenstate":
            raw_level = row.get("level_index", 0)
            try:
                level = int(raw_level)
            except (TypeError, ValueError) as error:
                raise UIInputError("sector level_index must be a nonnegative integer") from error
            if isinstance(raw_level, float) and raw_level != level:
                raise UIInputError("sector level_index must be a nonnegative integer")
            target = build_sector_state(
                "", level,
                _text(row.get("quantum_numbers", "")),
            )
        else:
            raise UIInputError(
                "state kind must be Ground state / Vacuum or Sector eigenstate"
            )
        states[label] = target
    if not states:
        raise UIInputError("at least one named state is required")
    return states


def rows_to_matrix_elements(
    rows: Iterable[Mapping[str, Any]], state_labels: Iterable[str],
) -> List[Dict[str, Any]]:
    """Convert c * <bra|O|ket> rows and validate their state references."""
    labels = set(state_labels)
    terms = []
    identifiers = set()
    for index, row in enumerate(rows, start=1):
        bra = _text(row.get("bra", ""))
        ket = _text(row.get("ket", ""))
        raw_coefficient = row.get("coefficient", "")
        term_id = _text(row.get("id", "")) or "matrix_element_{}".format(index)
        if not bra and not ket and _blank(raw_coefficient):
            continue
        coefficient = parse_value(raw_coefficient)
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(float(coefficient))
        ):
            raise UIInputError("matrix-element coefficients must be finite real numbers")
        if term_id in identifiers:
            raise UIInputError("matrix-element IDs must be unique")
        if bra not in labels or ket not in labels:
            raise UIInputError("every matrix-element bra/ket must name a defined state")
        identifiers.add(term_id)
        terms.append({
            "id": term_id,
            "coefficient": float(coefficient),
            "bra": bra,
            "ket": ket,
        })
    if not terms:
        raise UIInputError("at least one matrix element is required")
    return terms


def build_user_config(
    *,
    model: str,
    parameter_rows: Sequence[Mapping[str, Any]],
    expression: str,
    operator_rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
    transporter_rows: Sequence[Mapping[str, Any]],
    ordering: str,
    components: Sequence[str],
    constraints: str,
    maximum_points: int,
    bra: Any,
    ket: Any,
    llm: str,
    through: str = "crosscheck",
    coordinate_mode: str = "lattice_indices",
    hardware_validation: bool = False,
    hardware_mode: str = "representative",
    continuum_conventions: Mapping[str, Any] | None = None,
    states: Mapping[str, Any] | None = None,
    matrix_elements: Sequence[Mapping[str, Any]] | None = None,
    clarifications: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build exactly the concise, five-category user configuration."""
    for label, value in (("model", model), ("expression", expression),
                         ("llm", llm)):
        if not _nonempty(value):
            raise UIInputError("{} cannot be empty".format(label))
    if states is None:
        for label, value in (("bra", bra), ("ket", ket)):
            if not (_nonempty(value) or isinstance(value, Mapping)):
                raise UIInputError("{} cannot be empty".format(label))
    else:
        if not states:
            raise UIInputError("states cannot be empty")
        if not matrix_elements:
            raise UIInputError("named states require matrix_elements")
    observable: Dict[str, Any] = {
        "expression": expression.strip(),
        "ordering": ordering.strip() or "as_written",
        "operators": rows_to_operators(operator_rows),
        "position_grid": rows_to_grid(grid_rows),
        "components": list(components),
        "maximum_points": int(maximum_points),
        "coordinate_convention": coordinate_mode,
    }
    transporters = rows_to_transporters(transporter_rows)
    if transporters:
        observable["transporters"] = transporters
    constraint_items = [line.strip() for line in constraints.splitlines() if line.strip()]
    if constraint_items:
        observable["constraints"] = constraint_items
    if matrix_elements is not None:
        observable["matrix_elements"] = [dict(item) for item in matrix_elements]
    if hardware_validation:
        through = "final_report"
    if hardware_mode not in ("representative", "full_d1_nt1"):
        raise UIInputError("hardware_mode must be representative or full_d1_nt1")
    config = {
        "model": model.strip(),
        "parameters": rows_to_parameters(parameter_rows),
        "observable": observable,
        "llm": llm.strip(),
        "through": through,
        "hardware_validation": hardware_validation,
        "hardware_mode": hardware_mode,
    }
    if states is None:
        config["bra"] = bra.strip() if isinstance(bra, str) else dict(bra)
        config["ket"] = ket.strip() if isinstance(ket, str) else dict(ket)
    else:
        config["states"] = {
            label: value.strip() if isinstance(value, str) else dict(value)
            for label, value in states.items()
        }
    if continuum_conventions:
        config["continuum_conventions"] = dict(continuum_conventions)
    if clarifications:
        config["clarifications"] = [dict(item) for item in clarifications]
    return config


def _safe_stem(model: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", model.strip()).strip("_")
    return stem.lower() or "pipeline"


def save_user_config(config: Mapping[str, Any]) -> str:
    os.makedirs(CONFIG_DIRECTORY, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(
        CONFIG_DIRECTORY,
        "{}_{}.json".format(_safe_stem(str(config["model"])), stamp),
    )
    atomic_json_write(path, config)
    return path


def save_default_config(
    config: Mapping[str, Any], path: str = DEFAULT_CONFIG_PATH,
) -> str:
    """Persist reusable inputs without carrying run-specific Q&A forward."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    reusable = json.loads(json.dumps(config, ensure_ascii=False))
    reusable.pop("clarifications", None)
    atomic_json_write(path, reusable)
    return path


def load_default_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _editable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _state_editor_row(label: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping) and value.get("kind") == "sector_eigenstate":
        return {
            "label": label,
            "kind": "Sector eigenstate",
            "level_index": value.get("level_index", 0),
            "quantum_numbers": json.dumps(
                value.get("quantum_numbers", {}), ensure_ascii=False
            ),
        }
    return {
        "label": label, "kind": "Ground state / Vacuum",
        "level_index": 0, "quantum_numbers": "",
    }


def _single_state_defaults(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping) and value.get("kind") == "sector_eigenstate":
        return {
            "choice": 'Sector eigenstate',
            "custom": "",
            "level": int(value.get("level_index", 0)),
            "quantum_numbers": json.dumps(
                value.get("quantum_numbers", {}), ensure_ascii=False
            ),
        }
    text = str(value or "")
    lowered = text.lower()
    if "vacuum" in lowered or "ground" in lowered:
        choice = 'Global ground state / vacuum'
    else:
        choice = 'Custom'
    return {
        "choice": choice, "custom": text if choice == 'Custom' else "",
        "level": 0, "quantum_numbers": '{"charge": 0}',
    }


def config_to_ui_defaults(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Convert a saved concise config back to editable widget values."""
    config = dict(config or {})
    has_saved_config = bool(config)
    model = str(config.get("model", "Schwinger model"))
    observable = dict(config.get("observable", {}))
    model_choice = model if model in MODEL_PRESETS else 'Custom'
    parameters = config.get("parameters")
    parameter_rows = (
        [{"name": canonical_name, "value": _editable_text(canonical_value)}
         for name, value in parameters.items()
         for canonical_name, canonical_value in [_canonical_lattice_parameter(name, value)]]
        if isinstance(parameters, Mapping) and parameters
        else MODEL_PRESETS.get(model_choice, [{"name": "spatial_site_count", "value": "4"}])
    )
    operators = observable.get("operators", DEFAULT_OPERATORS)
    operator_rows = [
        {
            "id": item.get("id", ""),
            "operator": item.get("operator", ""),
            "position": _editable_text(item.get("position", "")),
        }
        for item in operators
    ]
    grid = observable.get("position_grid", {"z_index": [1]})
    grid_rows = [
        {"variable": name, "values": _editable_text(values)}
        for name, values in grid.items()
    ]
    transporters = observable.get(
        "transporters", [] if has_saved_config else DEFAULT_TRANSPORTERS
    )
    transporter_rows = [dict(item) for item in transporters]
    conventions = config.get("continuum_conventions", {})
    lightcone_choice = next(
        (label for label, value in LIGHTCONE_CONVENTIONS.items() if value == conventions),
        'Agent1 defaults / custom conventions',
    )
    coordinate = observable.get("coordinate_convention", "lattice_indices")
    if isinstance(coordinate, Mapping):
        coordinate = coordinate.get("mode", "lattice_indices")
    llm = str(config.get("llm", DEFAULT_UI_MODEL))
    llm_provider = provider_for_model(llm)
    llm_options = MODEL_PRESETS_BY_PROVIDER[llm_provider]
    through = str(config.get("through", "crosscheck"))
    run_scope = {
        "agent5_observable": 'Generate Agent1–5 interfaces only',
        "crosscheck": 'Run through classical cross-check',
        "final_report": 'Run through final academic report',
    }.get(through, 'Run through classical cross-check')
    states = config.get("states")
    if isinstance(states, Mapping) and states:
        state_mode = 'Linear combination of matrix elements'
        state_rows = [_state_editor_row(label, value) for label, value in states.items()]
        matrix_rows = [dict(item) for item in observable.get("matrix_elements", [])]
        bra_defaults = _single_state_defaults(None)
        ket_defaults = _single_state_defaults(None)
    else:
        state_mode = 'Single matrix element'
        state_rows = []
        matrix_rows = []
        bra_defaults = _single_state_defaults(config.get("bra", "Vacuum state <Omega|"))
        ket_defaults = _single_state_defaults(config.get("ket", "Vacuum state |Omega>"))
    constraints = observable.get("constraints", [])
    return {
        "model_choice": model_choice,
        "custom_model": model if model_choice == 'Custom' else "",
        "parameter_rows": parameter_rows,
        "expression": str(observable.get(
            "expression", r"\bar{\psi}(z n) W(z n,0) \psi(0), n^mu=(1,1)"
        )),
        "operator_rows": operator_rows,
        "grid_rows": grid_rows,
        "transporter_rows": transporter_rows,
        "include_transporters": bool(transporters),
        "lightcone_choice": lightcone_choice,
        "convention_notes": str(
            conventions.get("user_notes", "") if isinstance(conventions, Mapping) else ""
        ),
        "coordinate_label": (
            'Lattice indices (recommended)' if coordinate == "lattice_indices" else 'Physical coordinates'
        ),
        "ordering": str(observable.get("ordering", "as_written")),
        "components": list(observable.get("components", ["real", "imag"])),
        "maximum_points": int(observable.get("maximum_points", 10000)),
        "constraints": "\n".join(constraints) if isinstance(constraints, list) else "",
        "state_mode": state_mode,
        "state_rows": state_rows,
        "matrix_rows": matrix_rows,
        "bra": bra_defaults,
        "ket": ket_defaults,
        "llm_provider": llm_provider,
        "llm_choice": llm if llm in llm_options else 'Custom',
        "custom_llm": llm if llm not in llm_options else "",
        "run_scope": run_scope,
        "hardware_validation": bool(config.get("hardware_validation", False)),
        "hardware_mode": str(config.get("hardware_mode", "representative")),
        # Clarification answers belong to the run that requested them.  A new
        # UI run starts from the explicit physical input instead of silently
        # replaying stale or contradictory answers.
        "clarifications": [],
    }


def launch_pipeline(config_path: str) -> Dict[str, Any]:
    """Launch the controller without a shell and retain a local diagnostic log."""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "{}_{}".format(stamp, uuid.uuid4().hex[:8])
    log_path = os.path.join(LOG_DIRECTORY, "pipeline_{}.log".format(run_id))
    command = [
        sys.executable,
        "-u",
        os.path.join(PROJECT_ROOT, "pipeline_runtime.py"),
        "run",
        "--config",
        config_path,
        "--run-id",
        run_id,
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "pid": process.pid,
        "config_path": config_path,
        "log_path": log_path,
        "run_id": run_id,
        "manifest_path": os.path.join(
            PROJECT_ROOT, "runs", run_id, "pipeline_manifest.json"
        ),
    }


def find_clarification(launch: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Find the structured Agent1 clarification for a UI-launched run."""
    run_id = launch.get("run_id")
    manifest_path = launch.get("manifest_path")
    if isinstance(run_id, str) and isinstance(manifest_path, str):
        candidate = os.path.join(
            os.path.dirname(manifest_path), "audit",
            "Agent1_Clarification_{}.json".format(run_id),
        )
        value = _load_json_object(candidate)
        if value is not None:
            value["path"] = candidate
            return value
    # Compatibility with UI runs launched before manifest_path was recorded.
    for line in reversed(_tail(str(launch.get("log_path", "")), 200).splitlines()):
        marker = "Details: "
        if marker in line:
            candidate = line.split(marker, 1)[1].strip()
            value = _load_json_object(candidate)
            if value is not None:
                value["path"] = candidate
                return value
    return None


def _load_json_object(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def apply_clarification_answers(
    config: Mapping[str, Any], questions: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
) -> Dict[str, Any]:
    """Add user-owned clarification answers without changing original inputs."""
    if len(questions) != len(answers) or not questions:
        raise UIInputError('Clarification questions and answers must have matching counts')
    additions = []
    for question, answer in zip(questions, answers):
        question_text = _text(question.get("question", ""))
        answer_text = _text(answer)
        if not question_text or not answer_text:
            raise UIInputError('Answer every clarification question before continuing')
        additions.append({"question": question_text, "answer": answer_text})
    updated = json.loads(json.dumps(config, ensure_ascii=False))
    old = updated.get("clarifications", [])
    if not isinstance(old, list):
        old = []
    replaced = {item["question"] for item in additions}
    updated["clarifications"] = [
        item for item in old
        if isinstance(item, dict) and item.get("question") not in replaced
    ] + additions
    return updated


def _records(value: Any) -> List[Dict[str, Any]]:
    if hasattr(value, "to_dict"):
        return list(value.to_dict(orient="records"))
    return [dict(item) for item in value]


def _state_text(selection: str, custom: str, bra: bool) -> str:
    del bra  # Retained only for compatibility with the former Bra/Ket wording.
    if selection in {'Global ground state / vacuum', "Vacuum", "Ground state"}:
        return "Ground state / vacuum"
    return custom.strip()


def build_sector_state(name: str, level_index: int, quantum_numbers_text: str) -> Dict[str, Any]:
    """Build the concise structured target used for a sector eigenstate."""
    try:
        quantum_numbers = json.loads(quantum_numbers_text)
    except json.JSONDecodeError as error:
        raise UIInputError('Quantum numbers must be a valid JSON object: {}'.format(error.msg)) from error
    if not isinstance(quantum_numbers, dict) or not quantum_numbers:
        raise UIInputError('Quantum numbers must be a nonempty JSON object, for example {"charge": 0}')
    if not isinstance(level_index, int) or isinstance(level_index, bool) or level_index < 0:
        raise UIInputError('The sector level index must be a nonnegative integer')
    target: Dict[str, Any] = {
        "kind": "sector_eigenstate",
        "level_index": level_index,
        "quantum_numbers": quantum_numbers,
    }
    if name.strip():
        target["name"] = name.strip()
    return target


def _tail(path: str, line_count: int = 30) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-line_count:])
    except OSError:
        return ""


def _final_run_manifests() -> List[tuple[str, str, bool]]:
    """List completed reports for the small human reuse-approval control."""
    run_root = os.path.join(PROJECT_ROOT, "runs")
    values: List[tuple[str, str, bool]] = []
    if not os.path.isdir(run_root):
        return values
    for run_id in sorted(os.listdir(run_root), reverse=True):
        path = os.path.join(run_root, run_id, "pipeline_manifest.json")
        manifest = _load_json_object(path)
        if manifest is None:
            continue
        final_stage = manifest.get("stages", {}).get("final_report", {})
        artifacts = manifest.get("artifacts", {})
        report = artifacts.get("final_report_markdown", {})
        report_path = report.get("path") if isinstance(report, Mapping) else None
        if (
            not isinstance(final_stage, Mapping)
            or final_stage.get("status") != "completed"
            or not isinstance(report_path, str)
            or not os.path.isfile(report_path)
        ):
            continue
        approval = manifest.get("program_reuse_approval", {})
        approved = (
            isinstance(approval, Mapping)
            and approval.get("status") == "approved"
        )
        values.append((run_id, path, approved))
    return values


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Jianmu-Q Pipeline", layout="wide")
    saved_default = load_default_config()
    defaults = config_to_ui_defaults(saved_default)
    st.title('Jianmu-Q Simulation Workflow')
    st.caption('Define a physics problem and generate a reproducible workflow. Agent2 selects the gauge-field treatment and truncation.')
    if saved_default is not None:
        st.info('Your saved default inputs have been loaded. You can edit them before running.')

    with st.expander('Human-approved program reuse'):
        st.write(
            'Only runs whose final reports you have reviewed and approved here can supply reusable programs. '
            'Approval is recorded in the source run manifest.'
        )
        final_runs = _final_run_manifests()
        if final_runs:
            labels = [
                "{}{}".format(run_id, ' (approved)' if approved else "")
                for run_id, _, approved in final_runs
            ]
            selected_label = st.selectbox('Run with a final report', labels)
            selected_index = labels.index(selected_label)
            selected_id, selected_manifest_path, selected_approved = final_runs[
                selected_index
            ]
            with st.container(horizontal=True):
                if st.button(
                    'Approve reuse',
                    disabled=selected_approved,
                    help='Confirm that you reviewed this run and allow its programs to be reused for the same physical target.',
                ):
                    selected_manifest = _load_json_object(selected_manifest_path)
                    if selected_manifest is not None:
                        try:
                            set_program_reuse_approval(
                                selected_manifest, selected_manifest_path, True
                            )
                        except PipelineError as error:
                            st.error('This run cannot be used as a program template: {}'.format(error))
                        else:
                            st.success('Approved: {}'.format(selected_id))
                            st.rerun()
                if st.button('Revoke approval', disabled=not selected_approved):
                    selected_manifest = _load_json_object(selected_manifest_path)
                    if selected_manifest is not None:
                        set_program_reuse_approval(
                            selected_manifest, selected_manifest_path, False
                        )
                        st.success('Approval revoked: {}'.format(selected_id))
                        st.rerun()
        else:
            st.caption('No final reports are available for review yet.')

    with st.expander('Input format guide'):
        st.markdown(
            '\n- **Parameters:** Use unique names and numerical values, JSON arrays or plain text.\n- **Lattice shape:** Use `spatial_lattice_shape` with a JSON array such as `[4, 4]`.\n- **Observable:** Describe the complete quantity, then define each insertion and position.\n- **Conventions:** Select light-cone and gamma-matrix definitions in advanced options.\n- **Positions:** Use `t_index=z_index, x_index=0` or a JSON object.\n- **Grids:** Use `0,1,2`, inclusive ranges such as `0:L-1`, or `1:L-1:2` with a step.\n- **States:** A global ground state is the vacuum; sector states need a zero-based level and quantum numbers.\n- **State labels:** Unique references such as `hadron` and `vacuum` in the matrix-element table.\n- **Linear combinations:** Each row is `coefficient × <bra|O|ket>`; use -1 for vacuum subtraction.\n- **Custom entries:** Use the complete model definition and exact API model ID.\n\nGrid symbols must match integer parameters. Both `L` and `number of lattice sites: L` allow `L` in a range.\n'
        )

    st.subheader('1. Model and parameters')
    model_choice = st.selectbox(
        'Model',
        list(MODEL_PRESETS) + ['Custom'],
        index=(list(MODEL_PRESETS) + ['Custom']).index(defaults["model_choice"]),
        help='Choose a preset or select Custom and enter the full model definition.',
    )
    custom_model = st.text_input(
        'Custom model definition',
        value=defaults["custom_model"],
        disabled=model_choice != 'Custom',
        help='Enter a model name or formula, such as the NJL model. Used only when Custom is selected.',
    )
    model = custom_model if model_choice == 'Custom' else model_choice
    preset = (
        defaults["parameter_rows"]
        if model_choice == defaults["model_choice"]
        else MODEL_PRESETS.get(
            model_choice, [{"name": "spatial_site_count", "value": "4"}]
        )
    )
    parameter_rows = st.data_editor(
        preset,
        num_rows="dynamic",
        width="stretch",
        key="model_parameters_{}".format(model_choice),
        column_config={
            "name": st.column_config.TextColumn(
                'Parameter name',
                required=True,
                help=(
                    'Use a unique name per row. Use spatial_site_count in 1D, or '
                    'spatial_lattice_shape in multiple dimensions. Use a scalar such as L in grid ranges.'
                ),
            ),
            "value": st.column_config.TextColumn(
                'Value',
                required=True,
                help=(
                    'Enter a number, a valid JSON array such as [4, 4] for a multidimensional lattice, '
                    'or plain text.'
                ),
            ),
        },
    )
    st.info('Specify the lattice size here. Agent2 selects any gauge-field truncation.')

    st.subheader("2. Observable")
    expression = st.text_area(
        'Observable expression',
        value=defaults["expression"],
        help=(
            'Describe the complete observable using text or LaTeX, for example '
            r"\bar{\psi}(z)W(z,0)\psi(0)。"
        ),
    )
    operator_rows = st.data_editor(
        defaults["operator_rows"],
        num_rows="dynamic",
        width="stretch",
        key="observable_operators",
        column_config={
            "id": st.column_config.TextColumn(
                'ID (optional)',
                help='Unique operator label. Transporter endpoints refer to these labels.',
            ),
            "operator": st.column_config.TextColumn(
                'Operator',
                required=True,
                help='Operator name or LaTeX expression, for example \\psi, \\bar{\\psi} or O_1.',
            ),
            "position": st.column_config.TextColumn(
                'Position', required=True,
                help=(
                    'Use t_index=z_index, x_index=0 or a JSON object such as '
                    '{"t_index":"z_index","x_index":0}。'
                ),
            ),
        },
    )
    grid_rows = st.data_editor(
        defaults["grid_rows"] or [{"variable": "", "values": ""}],
        num_rows="dynamic",
        width="stretch",
        key="position_grid",
        column_config={
            "variable": st.column_config.TextColumn(
                'Grid variable',
                help='Match a variable used in operator positions, for example z_index.',
            ),
            "values": st.column_config.TextColumn(
                'Values',
                help='Use a range such as 0:L-1 or discrete points such as 1,3,5. Range endpoints are inclusive.',
            ),
        },
    )
    st.caption(
        'Grid endpoints are inclusive. `0:L-1` expands using the integer model parameter L. '
        'Symbols must match parameter names or aliases following a colon. For '
        'global observables such as the Hamiltonian, use `global` as the position and leave the grid empty. '
        'Blank template rows are omitted; an empty grid gives one sample with no coordinates.'
    )

    with st.expander('Advanced observable options'):
        lightcone_choice = st.selectbox(
            'Light-cone and gamma-matrix conventions',
            list(LIGHTCONE_CONVENTIONS) + ['Agent1 defaults / custom conventions'],
            index=(list(LIGHTCONE_CONVENTIONS) + ['Agent1 defaults / custom conventions']).index(
                defaults["lightcone_choice"]
            ),
            help=(
                'If the observable uses x± or gamma±, choose a definition. Agent1 converts it into '
                'explicit downstream conventions.'
            ),
        )
        convention_notes = st.text_area(
            'Other continuum conventions',
            value=defaults["convention_notes"],
            disabled=lightcone_choice != 'Agent1 defaults / custom conventions',
            help=(
                'Describe any overrides. If left empty, Agent1 uses '
                'a mostly-minus metric in the model dimension, gamma±=gamma0±gamma1, and natural units.'
            ),
        )
        continuum_conventions = dict(
            LIGHTCONE_CONVENTIONS.get(lightcone_choice, {})
        )
        if lightcone_choice == 'Agent1 defaults / custom conventions':
            continuum_conventions = {
                "user_notes": convention_notes.strip(),
            }
        coordinate_mode_label = st.selectbox(
            'Coordinate convention',
            ['Lattice indices (recommended)', 'Physical coordinates'],
            index=['Lattice indices (recommended)', 'Physical coordinates'].index(
                defaults["coordinate_label"]
            ),
            help=(
                'Lattice indices are converted to physical coordinates using the lattice spacing. '
                'Physical coordinates are already expressed in the units of the selected model.'
            ),
        )
        coordinate_mode = (
            "lattice_indices"
            if coordinate_mode_label == 'Lattice indices (recommended)'
            else "physical_coordinates"
        )
        ordering = st.selectbox(
            'Ordering',
            ["as_written", "time_ordered", "path_ordered"],
            index=["as_written", "time_ordered", "path_ordered"].index(
                defaults["ordering"]
            ),
            help=(
                'as_written: expression order; time_ordered: time ordering; '
                'path_ordered: ordering along the specified path.'
            ),
        )
        components = st.multiselect(
            'Output components',
            ["real", "imag"],
            default=defaults["components"],
            help='Select the real part, imaginary part, or both.',
        )
        maximum_points = st.number_input(
            'Maximum sampling points',
            min_value=1,
            value=defaults["maximum_points"],
            step=1,
            help='Upper limit for the Cartesian product of grid variables.',
        )
        constraints = st.text_area(
            'Constraints (one per line)',
            value=defaults["constraints"],
            help='Optional physical or sampling constraints, such as t_index >= 0. Enter one per line.',
        )
        include_transporters = st.checkbox(
            'Include gauge transporters / Wilson lines',
            value=defaults["include_transporters"],
            help='Enable when the observable requires gauge-covariant transport between positions.',
        )
        if include_transporters:
            transporter_rows = st.data_editor(
                defaults["transporter_rows"] or [{
                    "id": "", "from": "", "to": "", "path": "",
                }],
                num_rows="dynamic",
                width="stretch",
                key="observable_transporters",
                column_config={
                    "id": st.column_config.TextColumn(
                        "ID", required=True, help='Unique transporter label.'
                    ),
                    "from": st.column_config.TextColumn(
                        'Source operator ID',
                        required=True,
                        help='Use an ID from the operator table.',
                    ),
                    "to": st.column_config.TextColumn(
                        'Target operator ID',
                        required=True,
                        help='Use an ID from the operator table.',
                    ),
                    "path": st.column_config.TextColumn(
                        'Path',
                        required=True,
                        help='Describe the path, for example straight lattice path.',
                    ),
                },
            )
            st.caption('Blank template rows are omitted. Complete all four fields to include a transporter.')
        else:
            transporter_rows = []

    st.subheader('3. External states and matrix elements')
    state_mode = st.segmented_control(
        'Calculation type',
        ['Single matrix element', 'Linear combination of matrix elements'],
        default=defaults["state_mode"],
        help=(
            'A single matrix element evaluates <bra|O|ket>. A linear combination sums '
            'terms of the form coefficient × <bra|O|ket>.'
        ),
    )
    named_state_rows = None
    matrix_element_rows = None
    bra = ket = None
    if state_mode == 'Single matrix element':
        bra_column, ket_column = st.columns(2)
        with bra_column:
            bra_choice = st.selectbox(
                "Bra", ['Global ground state / vacuum', 'Sector eigenstate', 'Custom'],
                index=['Global ground state / vacuum', 'Sector eigenstate', 'Custom'].index(
                    defaults["bra"]["choice"]
                ),
                help='Choose the bra state. Bra and ket are independent.',
            )
            custom_bra = st.text_input(
                'Custom bra', value=defaults["bra"]["custom"],
                disabled=bra_choice != 'Custom',
                help='Describe the target state and required quantum numbers. Used only for Custom.',
            )
            if bra_choice == 'Sector eigenstate':
                bra_level = int(st.number_input(
                    'Bra sector level index k', min_value=0,
                    value=defaults["bra"]["level"],
                    step=1, key="bra_sector_level",
                ))
                bra_quantum_numbers = st.text_area(
                    'Bra quantum numbers (JSON)', value=defaults["bra"]["quantum_numbers"],
                    key="bra_quantum_numbers",
                )
        with ket_column:
            ket_choice = st.selectbox(
                "Ket", ['Global ground state / vacuum', 'Sector eigenstate', 'Custom'],
                index=['Global ground state / vacuum', 'Sector eigenstate', 'Custom'].index(
                    defaults["ket"]["choice"]
                ),
                help='Choose the ket state; it can differ from the bra.',
            )
            custom_ket = st.text_input(
                'Custom ket', value=defaults["ket"]["custom"],
                disabled=ket_choice != 'Custom',
                help='Describe the target state and required quantum numbers. Used only for Custom.',
            )
            if ket_choice == 'Sector eigenstate':
                ket_level = int(st.number_input(
                    'Ket sector level index k', min_value=0,
                    value=defaults["ket"]["level"],
                    step=1, key="ket_sector_level",
                ))
                ket_quantum_numbers = st.text_area(
                    'Ket quantum numbers (JSON)', value=defaults["ket"]["quantum_numbers"],
                    key="ket_quantum_numbers",
                )
        bra = (
            build_sector_state("", bra_level, bra_quantum_numbers)
            if bra_choice == 'Sector eigenstate'
            else _state_text(bra_choice, custom_bra, True)
        )
        ket = (
            build_sector_state("", ket_level, ket_quantum_numbers)
            if ket_choice == 'Sector eigenstate'
            else _state_text(ket_choice, custom_ket, False)
        )
    else:
        st.caption(
            'The matrix-element table refers to the state labels below. The global ground state is the vacuum in this workflow. '
            'For a sector eigenstate, level_index=0 denotes the lowest state in that sector.'
        )
        named_state_rows = st.data_editor(
            defaults["state_rows"] or [
                {"label": "hadron", "kind": "Sector eigenstate",
                 "level_index": 0, "quantum_numbers": '{"charge": 0}'},
                {"label": "vacuum", "kind": "Ground state / Vacuum",
                 "level_index": 0, "quantum_numbers": ""},
            ],
            num_rows="dynamic", width="stretch", key="named_external_states",
            column_config={
                "label": st.column_config.TextColumn(
                    'State label', required=True,
                    help='Unique label, such as hadron or vacuum, referenced by the matrix-element table.',
                ),
                "kind": st.column_config.SelectboxColumn(
                    'State type',
                    options=["Ground state / Vacuum", "Sector eigenstate"],
                    required=True,
                ),
                "level_index": st.column_config.NumberColumn(
                    'Sector level k', min_value=0, step=1,
                    help='Used only for sector eigenstates; starts at zero.',
                ),
                "quantum_numbers": st.column_config.TextColumn(
                    'Quantum numbers (JSON)',
                    help='Used only for sector eigenstates, for example {"charge": 0}.',
                ),
            },
        )
        visible_labels = [
            _text(row.get("label")) for row in _records(named_state_rows)
            if _text(row.get("label"))
        ]
        matrix_element_rows = st.data_editor(
            defaults["matrix_rows"] or [
                {"id": "hadron_term", "coefficient": 1.0,
                 "bra": "hadron", "ket": "hadron"},
                {"id": "vacuum_subtraction", "coefficient": -1.0,
                 "bra": "vacuum", "ket": "vacuum"},
            ],
            num_rows="dynamic", width="stretch", key="matrix_element_combination",
            column_config={
                "id": st.column_config.TextColumn('Term ID (optional)'),
                "coefficient": st.column_config.NumberColumn(
                    'Coefficient', required=True,
                    help='Finite real number. Use -1 for a vacuum-subtraction term.',
                ),
                "bra": st.column_config.SelectboxColumn(
                    'Bra state', options=visible_labels or ['Define states first'], required=True,
                ),
                "ket": st.column_config.SelectboxColumn(
                    'Ket state', options=visible_labels or ['Define states first'], required=True,
                ),
            },
        )
    st.caption(
        'Sector levels start at zero. Quantum-number names and values depend on the model; '
        'Agent2/3 specify their lattice representation and sector basis.'
    )

    st.subheader("4. LLM")
    llm_provider = st.selectbox(
        "LLM provider", list(MODEL_PRESETS_BY_PROVIDER),
        index=list(MODEL_PRESETS_BY_PROVIDER).index(defaults["llm_provider"]),
    )
    llm_options = list(MODEL_PRESETS_BY_PROVIDER[llm_provider]) + ["Custom"]
    default_llm = defaults["llm_choice"] if llm_provider == defaults["llm_provider"] else llm_options[0]
    llm_choice = st.selectbox(
        'Workflow model',
        llm_options, index=llm_options.index(default_llm),
        key="llm_model_{}".format(llm_provider),
        help='Select the model used by all LLM stages in the workflow.',
    )
    custom_llm = st.text_input(
        'Custom model ID',
        value=defaults["custom_llm"] if llm_provider == defaults["llm_provider"] else "",
        key="custom_llm_{}".format(llm_provider),
        disabled=llm_choice != 'Custom',
        help='Enter the exact API model ID. Gemini IDs must start with gemini-.',
    )
    llm = custom_llm if llm_choice == 'Custom' else llm_choice
    st.caption(
        "Credentials: {}. Model availability depends on your API account.".format(api_key_variable(llm))
        if llm.strip() else "Enter a model ID to configure credentials."
    )
    run_scope = st.selectbox(
        'Run scope',
        [
            'Generate Agent1–5 interfaces only',
            'Run through classical cross-check',
            'Run through final academic report',
        ],
        index=[
            'Generate Agent1–5 interfaces only',
            'Run through classical cross-check',
            'Run through final academic report',
        ].index(defaults["run_scope"]),
        help=(
            'Choose the final stage. The academic report includes the quantum algorithm, classical references and available hardware results.'
        ),
    )
    through = {
        'Generate Agent1–5 interfaces only': "agent5_observable",
        'Run through classical cross-check': "crosscheck",
        'Run through final academic report': "final_report",
    }[run_scope]
    hardware_validation = st.toggle(
        'Enable IBM hardware validation (automatic submission)',
        value=defaults["hardware_validation"],
        help=(
            'Authorize planning and submission of shallow D_ans=1 demonstrations, with N_T=1 when evolution is required. '
            'No submission occurs if hardware resource checks fail.'
        ),
    )
    hardware_mode = "representative"
    if hardware_validation:
        through = "final_report"
        hardware_mode_label = st.segmented_control(
            'Hardware validation scope',
            ['Representative circuits', 'Full D=1, N_T=1 observable'],
            default=(
                'Full D=1, N_T=1 observable'
                if defaults["hardware_mode"] == "full_d1_nt1"
                else 'Representative circuits'
            ),
            help=(
                'Representative mode submits a few selected recipes. Full mode submits every position '
                'and every recipe needed for reconstruction, using more hardware time.'
            ),
        )
        hardware_mode = (
            "full_d1_nt1"
            if hardware_mode_label == 'Full D=1, N_T=1 observable'
            else "representative"
        )
        st.info(
            'Hardware validation runs through the final report. Only IBM instances with '
            'more than ten minutes of confirmed remaining allocation are eligible.'
        )

    try:
        if llm.strip() and provider_for_model(llm) != llm_provider:
            raise UIInputError("The model ID does not match the selected LLM provider.")
        named_states = None
        matrix_elements = None
        if state_mode == 'Linear combination of matrix elements':
            named_states = rows_to_states(_records(named_state_rows))
            matrix_elements = rows_to_matrix_elements(
                _records(matrix_element_rows), named_states
            )
        config = build_user_config(
            model=model,
            parameter_rows=_records(parameter_rows),
            expression=expression,
            operator_rows=_records(operator_rows),
            grid_rows=_records(grid_rows),
            transporter_rows=_records(transporter_rows),
            ordering=ordering,
            components=components,
            constraints=constraints,
            maximum_points=int(maximum_points),
            bra=bra,
            ket=ket,
            llm=llm,
            through=through,
            coordinate_mode=coordinate_mode,
            hardware_validation=hardware_validation,
            hardware_mode=hardware_mode,
            continuum_conventions=continuum_conventions,
            states=named_states,
            matrix_elements=matrix_elements,
            clarifications=defaults["clarifications"],
        )
        normalized_config = validate_config(config, PROJECT_ROOT)
        config_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        st.success('Inputs passed workflow contract validation.')
        reusable = find_approved_reuse_source_for_config(
            normalized_config, PROJECT_ROOT
        )
        if reusable is not None:
            reusable_manifest, _ = reusable
            st.info(
                'An approved run matches this physical target: {}. This run can reuse '
                'its program bundle and skip Agents 1–7. Numerical results are recalculated for the new parameters and '
                'sampling grid.'.format(
                    reusable_manifest.get("run_id", "unknown")
                )
            )
        else:
            st.caption(
                'No approved run matches this physical target. New programs will be generated.'
            )
        with st.expander('Preview configuration JSON'):
            st.code(config_text, language="json")
        st.download_button(
            'Download JSON',
            data=config_text,
            file_name="{}_pipeline.json".format(_safe_stem(model)),
            mime="application/json",
        )
        with st.container(horizontal=True):
            if st.button('Save configuration'):
                saved = save_user_config(config)
                st.session_state["saved_config_path"] = saved
                st.success('Saved: {}'.format(saved))
            if st.button(
                'Save as default inputs',
                help='Load these inputs automatically next time; all fields remain editable.',
            ):
                saved_default_path = save_default_config(config)
                st.success('Default inputs saved: {}'.format(saved_default_path))

        st.subheader('5. Launch')
        key_present = bool(api_key_for_model(llm, required=False))
        st.write("{}: {}".format(api_key_variable(llm), 'detected' if key_present else 'not detected'))
        st.caption(
            'If VQE identifies an insufficient ansatz, the controller allows up to two redesigns, '
            'for at most three ansatz attempts including the initial design.'
        )
        confirmation = (
            'I confirm paid LLM calls and authorize IBM hardware submission.'
            if hardware_validation else
            'I confirm paid LLM calls and want to launch this workflow.'
        )
        confirmed = st.checkbox(
            confirmation,
            help='Confirm API usage and any authorized hardware submission before launching.',
        )
        if st.button('Launch workflow', disabled=not confirmed or not key_present):
            saved = save_user_config(config)
            launch = launch_pipeline(saved)
            st.session_state["last_launch"] = launch
            st.success('Workflow launched in the background, PID={}'.format(launch["pid"]))
    except (UIInputError, PipelineError, ValueError) as error:
        st.error('Input validation failed: {}'.format(error))

    launch = st.session_state.get("last_launch")
    if launch:
        st.subheader('Most recent UI launch')
        st.json(launch)
        if st.button('Refresh run status', icon=":material/refresh:"):
            pass
        manifest = _load_json_object(str(launch.get("manifest_path", "")))
        if manifest is not None:
            status = manifest.get("status", "unknown")
            st.write('Workflow status: **{}**'.format(status))
        st.code(_tail(launch["log_path"]) or 'No log output yet', language="text")
        clarification = find_clarification(launch)
        if clarification is not None:
            questions = clarification.get("questions", [])
            if isinstance(questions, list) and questions:
                with st.container(border=True):
                    st.warning('The workflow needs additional physical definitions before Agent2 can start.')
                    st.write(
                        'Answer the questions below. The original inputs and your answers will be retained, '
                        'then the workflow will restart from Agent1.'
                    )
                    answers = []
                    for index, item in enumerate(questions, start=1):
                        question = _text(item.get("question", ""))
                        reason = _text(item.get("reason", ""))
                        st.markdown('**Question {}:** {}'.format(index, question))
                        if reason:
                            st.caption('Why this is needed: {}'.format(reason))
                        answers.append(st.text_area(
                            'Your answer {}'.format(index),
                            key="clarification_answer_{}_{}".format(
                                launch.get("run_id", "legacy"), index
                            ),
                            placeholder='For example: use the one-flavor 1+1D chiral NJL model, with the four-fermion coefficient defined as ...',
                        ))
                    ready = all(_text(answer) for answer in answers)
                    original = _load_json_object(str(launch.get("config_path", "")))
                    resume_model = str((original or {}).get("llm", DEFAULT_UI_MODEL))
                    resume_key_present = bool(api_key_for_model(resume_model, required=False))
                    if not resume_key_present:
                        st.caption("Set {} to continue this run.".format(api_key_variable(resume_model)))
                    if st.button(
                        'Submit answers and continue',
                        type="primary",
                        disabled=not ready or not resume_key_present,
                    ):
                        original = _load_json_object(str(launch.get("config_path", "")))
                        if original is None:
                            st.error('The original run configuration could not be found.')
                        else:
                            updated = apply_clarification_answers(
                                original, questions, answers
                            )
                            validate_config(updated, PROJECT_ROOT)
                            save_default_config(updated)
                            updated_path = save_user_config(updated)
                            continued = launch_pipeline(updated_path)
                            st.session_state["last_launch"] = continued
                            st.success(
                                'Answers saved. A new run started from Agent1, PID={}'.format(
                                    continued["pid"]
                                )
                            )


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
