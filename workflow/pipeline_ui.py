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
    "不涉及光锥坐标": {},
    "归一化：x±=(x0±x1)/√2": {
        "lightcone": {
            "coordinate_definition": "x^plus/minus=(x^0 plus/minus x^1)/sqrt(2)",
            "gamma_definition": "gamma^plus/minus=(gamma^0 plus/minus gamma^1)/sqrt(2)",
            "normalization": 0.7071067811865476,
        }
    },
    "非归一化：x±=x0±x1": {
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
            "choice": "量子数扇区本征态",
            "custom": "",
            "level": int(value.get("level_index", 0)),
            "quantum_numbers": json.dumps(
                value.get("quantum_numbers", {}), ensure_ascii=False
            ),
        }
    text = str(value or "")
    lowered = text.lower()
    if "vacuum" in lowered or "ground" in lowered:
        choice = "全空间基态／真空"
    else:
        choice = "自定义"
    return {
        "choice": choice, "custom": text if choice == "自定义" else "",
        "level": 0, "quantum_numbers": '{"charge": 0}',
    }


def config_to_ui_defaults(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Convert a saved concise config back to editable widget values."""
    config = dict(config or {})
    has_saved_config = bool(config)
    model = str(config.get("model", "Schwinger model"))
    observable = dict(config.get("observable", {}))
    model_choice = model if model in MODEL_PRESETS else "自定义"
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
        "由 Agent1 检查 / 自行说明",
    )
    coordinate = observable.get("coordinate_convention", "lattice_indices")
    if isinstance(coordinate, Mapping):
        coordinate = coordinate.get("mode", "lattice_indices")
    llm = str(config.get("llm", "gpt-5.6-sol"))
    llm_options = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    through = str(config.get("through", "crosscheck"))
    run_scope = {
        "agent5_observable": "仅验证 Agent1–5 接口",
        "crosscheck": "运行到经典 cross-check",
        "final_report": "运行到最终学术报告",
    }.get(through, "运行到经典 cross-check")
    states = config.get("states")
    if isinstance(states, Mapping) and states:
        state_mode = "矩阵元线性组合"
        state_rows = [_state_editor_row(label, value) for label, value in states.items()]
        matrix_rows = [dict(item) for item in observable.get("matrix_elements", [])]
        bra_defaults = _single_state_defaults(None)
        ket_defaults = _single_state_defaults(None)
    else:
        state_mode = "单个矩阵元"
        state_rows = []
        matrix_rows = []
        bra_defaults = _single_state_defaults(config.get("bra", "Vacuum state <Omega|"))
        ket_defaults = _single_state_defaults(config.get("ket", "Vacuum state |Omega>"))
    constraints = observable.get("constraints", [])
    return {
        "model_choice": model_choice,
        "custom_model": model if model_choice == "自定义" else "",
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
            "格点索引（推荐）" if coordinate == "lattice_indices" else "直接物理坐标"
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
        "llm_choice": llm if llm in llm_options else "自定义",
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
        raise UIInputError("澄清问题和回答数量不一致")
    additions = []
    for question, answer in zip(questions, answers):
        question_text = _text(question.get("question", ""))
        answer_text = _text(answer)
        if not question_text or not answer_text:
            raise UIInputError("请回答全部澄清问题后再继续")
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
    if selection in {"全空间基态／真空", "Vacuum", "Ground state"}:
        return "Ground state / vacuum"
    return custom.strip()


def build_sector_state(name: str, level_index: int, quantum_numbers_text: str) -> Dict[str, Any]:
    """Build the concise structured target used for a sector eigenstate."""
    try:
        quantum_numbers = json.loads(quantum_numbers_text)
    except json.JSONDecodeError as error:
        raise UIInputError("量子数必须是合法 JSON 对象：{}".format(error.msg)) from error
    if not isinstance(quantum_numbers, dict) or not quantum_numbers:
        raise UIInputError("量子数必须是非空 JSON 对象，例如 {\"charge\": 0}")
    if not isinstance(level_index, int) or isinstance(level_index, bool) or level_index < 0:
        raise UIInputError("扇区内能级序号必须是非负整数")
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
    st.title("Jianmu-Q 本地实验输入")
    st.caption("填写物理问题，生成可复现 JSON；规范场截断由 Agent2 选择。")
    if saved_default is not None:
        st.info("已自动载入上次保存的默认输入；可直接修改后再次运行。")

    with st.expander("人工批准程序复用"):
        st.write(
            "只有你审阅过最终报告并在这里批准的 run 才能成为复用来源。"
            "批准只记录在原 run 的 manifest 中，不建立额外数据库。"
        )
        final_runs = _final_run_manifests()
        if final_runs:
            labels = [
                "{}{}".format(run_id, "（已批准）" if approved else "")
                for run_id, _, approved in final_runs
            ]
            selected_label = st.selectbox("已有 Final Report 的 run", labels)
            selected_index = labels.index(selected_label)
            selected_id, selected_manifest_path, selected_approved = final_runs[
                selected_index
            ]
            with st.container(horizontal=True):
                if st.button(
                    "批准复用",
                    disabled=selected_approved,
                    help="表示你已审阅该 run，允许相同 UI 物理目标复用其程序。",
                ):
                    selected_manifest = _load_json_object(selected_manifest_path)
                    if selected_manifest is not None:
                        try:
                            set_program_reuse_approval(
                                selected_manifest, selected_manifest_path, True
                            )
                        except PipelineError as error:
                            st.error("该 run 不能作为程序模板：{}".format(error))
                        else:
                            st.success("已批准：{}".format(selected_id))
                            st.rerun()
                if st.button("撤销批准", disabled=not selected_approved):
                    selected_manifest = _load_json_object(selected_manifest_path)
                    if selected_manifest is not None:
                        set_program_reuse_approval(
                            selected_manifest, selected_manifest_path, False
                        )
                        st.success("已撤销：{}".format(selected_id))
                        st.rerun()
        else:
            st.caption("目前没有可审核的 Final Report。")

    with st.expander("输入格式速查"):
        st.markdown(
            """
- **模型参数**：每行填写唯一参数名和值；值支持数字、JSON 数组或普通文本。
- **多维格点大小**：填写 `spatial_lattice_shape`，值写成 JSON 数组，例如二维
  ` [4, 4]`；不要把 `L*L = 4*4 = 16` 同时写进参数名和值。
- **观测量**：表达式用于描述完整物理量；下方算符表逐项定义算符及位置。
- **连续理论约定**：若使用光锥坐标或 Gamma±，在高级选项选择物理定义。
- **位置**：可写 `t_index=z_index, x_index=0`，也可写 JSON 对象。
- **网格**：离散值写 `0,1,2`；参数范围写 `0:L-1`；带步长写 `1:L-1:2`。
- **态与量子数**：全空间基态就是本流程中的真空；指定扇区态需填写从 0 开始的能级和量子数 JSON。
- **状态标签**：只是矩阵元表引用该状态的唯一名称，例如 `hadron`、`vacuum`。
- **矩阵元线性组合**：每行表示 `coefficient × <bra|O|ket>`；例如强子项系数 `1`、真空项系数 `-1`。
- **自定义名称**：模型和 LLM 均填写其正式名称；自定义态应明确说明目标态。

网格中的符号必须对应一个整数模型参数。参数名可直接写 `L`，或写成
`number of lattice sites: L`，此时都可在网格中使用 `L`。
            """
        )

    st.subheader("1. 模型与参数")
    model_choice = st.selectbox(
        "模型",
        list(MODEL_PRESETS) + ["自定义"],
        index=(list(MODEL_PRESETS) + ["自定义"]).index(defaults["model_choice"]),
        help="选择预设模型，或选择“自定义”后填写正式模型名称。",
    )
    custom_model = st.text_input(
        "自定义模型名称",
        value=defaults["custom_model"],
        disabled=model_choice != "自定义",
        help="自由文本，例如 NJL model；仅在模型选择“自定义”时生效。",
    )
    model = custom_model if model_choice == "自定义" else model_choice
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
                "参数名",
                required=True,
                help=(
                    "每行唯一。一维格点可用 spatial_site_count；多维格点使用 "
                    "spatial_lattice_shape。网格范围所需的标量可使用简单符号 L。"
                ),
            ),
            "value": st.column_config.TextColumn(
                "值",
                required=True,
                help=(
                    "数字直接填写；多维格点写合法 JSON 数组，例如 [4, 4]；"
                    "其余内容按文本处理。"
                ),
            ),
        },
    )
    st.info("格点数可以在这里指定；规范场截断不要填写，由 Agent2 决定。")

    st.subheader("2. Observable")
    expression = st.text_area(
        "观测量表达式",
        value=defaults["expression"],
        help=(
            "自由文本或 LaTeX 记号，描述完整观测量；例如 "
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
                "ID（可留空）",
                help="本表内唯一的标签；transporter 的 from/to 使用该标签。",
            ),
            "operator": st.column_config.TextColumn(
                "算符",
                required=True,
                help=r"算符名称或 LaTeX 记号，例如 \psi、\bar{\psi} 或 O_1。",
            ),
            "position": st.column_config.TextColumn(
                "位置", required=True,
                help=(
                    "可写 t_index=z_index, x_index=0；也可写 "
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
                "网格变量",
                help="应与算符位置中使用的变量名一致，例如 z_index。",
            ),
            "values": st.column_config.TextColumn(
                "取值",
                help="范围可写 0:L-1；离散点可写 1,3,5。范围端点包含在内。",
            ),
        },
    )
    st.caption(
        "网格端点包含在内。`0:L-1` 会使用整数模型参数 L 自动展开；"
        "符号必须与参数名或参数名冒号后的别名一致。像 Hamiltonian 这样的"
        "全局无网格观测量，可将算符位置填写为 `global`，并把网格表留空；"
        "空白模板行不会保存，系统会把它作为唯一的空参数采样点运行。"
    )

    with st.expander("Observable 高级选项"):
        lightcone_choice = st.selectbox(
            "光锥坐标与 Gamma 矩阵约定",
            list(LIGHTCONE_CONVENTIONS) + ["由 Agent1 检查 / 自行说明"],
            index=(list(LIGHTCONE_CONVENTIONS) + ["由 Agent1 检查 / 自行说明"]).index(
                defaults["lightcone_choice"]
            ),
            help=(
                "若观测量含 x± 或 gamma±，请选择定义；这会由 Agent1 转换为"
                "下游机器参数，不需要手写归一化常数。"
            ),
        )
        convention_notes = st.text_area(
            "其他连续理论约定",
            value=defaults["convention_notes"],
            disabled=lightcone_choice != "由 Agent1 检查 / 自行说明",
            help=(
                "可用物理语言覆盖默认约定。留空时 Agent1 自动统一采用"
                "η=diag(1,-1)、γ±=γ0±γ1 和自然单位，不会因此中断询问。"
            ),
        )
        continuum_conventions = dict(
            LIGHTCONE_CONVENTIONS.get(lightcone_choice, {})
        )
        if lightcone_choice == "由 Agent1 检查 / 自行说明":
            continuum_conventions = {
                "user_notes": convention_notes.strip(),
            }
        coordinate_mode_label = st.selectbox(
            "位置坐标约定",
            ["格点索引（推荐）", "直接物理坐标"],
            index=["格点索引（推荐）", "直接物理坐标"].index(
                defaults["coordinate_label"]
            ),
            help=(
                "格点索引会用 lattice spacing 把 t_index/x_index 转成物理坐标；"
                "直接物理坐标表示网格值已经带有所选模型的物理单位。"
            ),
        )
        coordinate_mode = (
            "lattice_indices"
            if coordinate_mode_label == "格点索引（推荐）"
            else "physical_coordinates"
        )
        ordering = st.selectbox(
            "排序",
            ["as_written", "time_ordered", "path_ordered"],
            index=["as_written", "time_ordered", "path_ordered"].index(
                defaults["ordering"]
            ),
            help=(
                "as_written：按表达式顺序；time_ordered：时间排序；"
                "path_ordered：沿指定路径排序。"
            ),
        )
        components = st.multiselect(
            "输出分量",
            ["real", "imag"],
            default=defaults["components"],
            help="选择保存关联函数的实部、虚部或两者。",
        )
        maximum_points = st.number_input(
            "最大采样点数",
            min_value=1,
            value=defaults["maximum_points"],
            step=1,
            help="所有网格变量做笛卡尔积后的安全上限。",
        )
        constraints = st.text_area(
            "约束（每行一条）",
            value=defaults["constraints"],
            help="可选的物理或采样限制；一行一条，例如 t_index >= 0。",
        )
        include_transporters = st.checkbox(
            "包含规范平行输运算符 / Wilson line",
            value=defaults["include_transporters"],
            help="仅在观测量需要连接不同位置的规范协变 transporter 时开启。",
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
                        "ID", required=True, help="该 transporter 的唯一标签。"
                    ),
                    "from": st.column_config.TextColumn(
                        "起点算符 ID",
                        required=True,
                        help="填写算符表中的一个 ID。",
                    ),
                    "to": st.column_config.TextColumn(
                        "终点算符 ID",
                        required=True,
                        help="填写算符表中的一个 ID。",
                    ),
                    "path": st.column_config.TextColumn(
                        "路径",
                        required=True,
                        help="自由文本描述，例如 straight lattice path。",
                    ),
                },
            )
            st.caption("空白模板行不会保存；填写四项后才会加入观测量。")
        else:
            transporter_rows = []

    st.subheader("3. 外态与矩阵元")
    state_mode = st.segmented_control(
        "计算形式",
        ["单个矩阵元", "矩阵元线性组合"],
        default=defaults["state_mode"],
        help=(
            "单个矩阵元计算 <bra|O|ket>；线性组合计算若干个 "
            "coefficient × <bra|O|ket> 的总和。"
        ),
    )
    named_state_rows = None
    matrix_element_rows = None
    bra = ket = None
    if state_mode == "单个矩阵元":
        bra_column, ket_column = st.columns(2)
        with bra_column:
            bra_choice = st.selectbox(
                "Bra", ["全空间基态／真空", "量子数扇区本征态", "自定义"],
                index=["全空间基态／真空", "量子数扇区本征态", "自定义"].index(
                    defaults["bra"]["choice"]
                ),
                help="选择左外态类型；Bra 与 Ket 完全独立。",
            )
            custom_bra = st.text_input(
                "自定义 Bra", value=defaults["bra"]["custom"],
                disabled=bra_choice != "自定义",
                help="明确描述目标态及必要量子数；仅在选择“自定义”时生效。",
            )
            if bra_choice == "量子数扇区本征态":
                bra_level = int(st.number_input(
                    "Bra 在该扇区内的能级序号 k", min_value=0,
                    value=defaults["bra"]["level"],
                    step=1, key="bra_sector_level",
                ))
                bra_quantum_numbers = st.text_area(
                    "Bra 量子数（JSON）", value=defaults["bra"]["quantum_numbers"],
                    key="bra_quantum_numbers",
                )
        with ket_column:
            ket_choice = st.selectbox(
                "Ket", ["全空间基态／真空", "量子数扇区本征态", "自定义"],
                index=["全空间基态／真空", "量子数扇区本征态", "自定义"].index(
                    defaults["ket"]["choice"]
                ),
                help="选择右外态类型；可与 Bra 不同。",
            )
            custom_ket = st.text_input(
                "自定义 Ket", value=defaults["ket"]["custom"],
                disabled=ket_choice != "自定义",
                help="明确描述目标态及必要量子数；仅在选择“自定义”时生效。",
            )
            if ket_choice == "量子数扇区本征态":
                ket_level = int(st.number_input(
                    "Ket 在该扇区内的能级序号 k", min_value=0,
                    value=defaults["ket"]["level"],
                    step=1, key="ket_sector_level",
                ))
                ket_quantum_numbers = st.text_area(
                    "Ket 量子数（JSON）", value=defaults["ket"]["quantum_numbers"],
                    key="ket_quantum_numbers",
                )
        bra = (
            build_sector_state("", bra_level, bra_quantum_numbers)
            if bra_choice == "量子数扇区本征态"
            else _state_text(bra_choice, custom_bra, True)
        )
        ket = (
            build_sector_state("", ket_level, ket_quantum_numbers)
            if ket_choice == "量子数扇区本征态"
            else _state_text(ket_choice, custom_ket, False)
        )
    else:
        st.caption(
            "状态标签供下面的矩阵元表引用。全空间基态就是本流程中的真空；"
            "Sector eigenstate 的 level_index=0 表示指定量子数扇区的最低态。"
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
                    "状态标签", required=True,
                    help="本表内唯一，例如 hadron、vacuum。矩阵元表用该标签。",
                ),
                "kind": st.column_config.SelectboxColumn(
                    "状态类型",
                    options=["Ground state / Vacuum", "Sector eigenstate"],
                    required=True,
                ),
                "level_index": st.column_config.NumberColumn(
                    "扇区能级 k", min_value=0, step=1,
                    help="仅 Sector eigenstate 使用，从 0 开始。",
                ),
                "quantum_numbers": st.column_config.TextColumn(
                    "量子数 JSON",
                    help='仅 Sector eigenstate 使用，例如 {"charge": 0}。',
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
                "id": st.column_config.TextColumn("项 ID（可选）"),
                "coefficient": st.column_config.NumberColumn(
                    "系数", required=True,
                    help="有限实数；真空减除项填写 -1。",
                ),
                "bra": st.column_config.SelectboxColumn(
                    "Bra 状态", options=visible_labels or ["请先定义状态"], required=True,
                ),
                "ket": st.column_config.SelectboxColumn(
                    "Ket 状态", options=visible_labels or ["请先定义状态"], required=True,
                ),
            },
        )
    st.caption(
        "扇区能级采用从 0 开始的编号；量子数名称和值由具体模型决定，"
        "Agent2/3 必须给出对应离散化与计算基子空间。"
    )

    st.subheader("4. LLM")
    llm_choice = st.selectbox(
        "全流程模型",
        ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "自定义"],
        index=["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "自定义"].index(
            defaults["llm_choice"]
        ),
        help="选择 Agent1–5 使用的模型；Luna 成本较低，Sol 通常更稳健。",
    )
    custom_llm = st.text_input(
        "自定义 LLM",
        value=defaults["custom_llm"],
        disabled=llm_choice != "自定义",
        help="填写 API 可识别的完整模型 ID；仅在选择“自定义”时生效。",
    )
    llm = custom_llm if llm_choice == "自定义" else llm_choice
    run_scope = st.selectbox(
        "运行范围",
        [
            "仅验证 Agent1–5 接口",
            "运行到经典 cross-check",
            "运行到最终学术报告",
        ],
        index=[
            "仅验证 Agent1–5 接口",
            "运行到经典 cross-check",
            "运行到最终学术报告",
        ].index(defaults["run_scope"]),
        help=(
            "选择流水线停止点。最终报告会包含量子算法、经典对照和已有硬件结果。"
        ),
    )
    through = {
        "仅验证 Agent1–5 接口": "agent5_observable",
        "运行到经典 cross-check": "crosscheck",
        "运行到最终学术报告": "final_report",
    }[run_scope]
    hardware_validation = st.toggle(
        "进行 IBM 真机验证（自动提交）",
        value=defaults["hardware_validation"],
        help=(
            "开启即授权流水线自动规划并提交 D_ans=N_T=1 的最浅演示线路。"
            "若当前硬件资源门限不通过，则不会提交，也不会产生真机费用。"
        ),
    )
    hardware_mode = "representative"
    if hardware_validation:
        through = "final_report"
        hardware_mode_label = st.segmented_control(
            "真机验证范围",
            ["快速代表线路", "完整 D=1, N_T=1 关联函数"],
            default=(
                "完整 D=1, N_T=1 关联函数"
                if defaults["hardware_mode"] == "full_d1_nt1"
                else "快速代表线路"
            ),
            help=(
                "快速模式只提交信号最好的少量 recipe；完整模式提交所有位置点"
                "及重构关联函数所需的全部 recipe，机时消耗会明显增加。"
            ),
        )
        hardware_mode = (
            "full_d1_nt1"
            if hardware_mode_label == "完整 D=1, N_T=1 关联函数"
            else "representative"
        )
        st.info(
            "已选择真机验证：流水线将自动运行到最终报告。只有剩余机时严格"
            "大于十分钟的 IBM 实例才会进入候选名单。"
        )

    try:
        named_states = None
        matrix_elements = None
        if state_mode == "矩阵元线性组合":
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
        st.success("输入已通过现有 pipeline contract 验证。")
        reusable = find_approved_reuse_source_for_config(
            normalized_config, PROJECT_ROOT
        )
        if reusable is not None:
            reusable_manifest, _ = reusable
            st.info(
                "已在启动前找到人工批准的同一物理目标：{}。本轮可复用"
                " 已批准的完整程序包，直接跳过 Agent1--7；新的参数和"
                "位置网格仍会重新计算。".format(
                    reusable_manifest.get("run_id", "unknown")
                )
            )
        else:
            st.caption(
                "当前没有人工批准且物理目标相同的 run；本轮会正常生成程序。"
            )
        with st.expander("预览生成的 JSON"):
            st.code(config_text, language="json")
        st.download_button(
            "下载 JSON",
            data=config_text,
            file_name="{}_pipeline.json".format(_safe_stem(model)),
            mime="application/json",
        )
        with st.container(horizontal=True):
            if st.button("保存本次配置"):
                saved = save_user_config(config)
                st.session_state["saved_config_path"] = saved
                st.success("已保存：{}".format(saved))
            if st.button(
                "保存为下次默认输入",
                help="下次打开 UI 时自动回填当前全部输入，仍可继续编辑。",
            ):
                saved_default_path = save_default_config(config)
                st.success("已设为默认输入：{}".format(saved_default_path))

        st.subheader("5. 启动")
        key_present = bool(os.environ.get("OPENAI_API_KEY"))
        st.write("OPENAI_API_KEY：{}".format("已检测到" if key_present else "未检测到"))
        st.caption(
            "若 VQE 明确判定 ansatz 不足，总控会自动重新设计最多 2 次；"
            "包括首次方案在内，一次运行最多尝试 3 个 ansatz。"
        )
        confirmation = (
            "我确认这会调用付费 Agent，并授权提交 IBM 真机作业。"
            if hardware_validation else
            "我确认这会调用付费 Agent，并希望启动完整 pipeline。"
        )
        confirmed = st.checkbox(
            confirmation,
            help="启动前的费用与外部提交确认；未勾选时不会启动。",
        )
        if st.button("启动 Pipeline", disabled=not confirmed or not key_present):
            saved = save_user_config(config)
            launch = launch_pipeline(saved)
            st.session_state["last_launch"] = launch
            st.success("Pipeline 已在后台启动，PID={}".format(launch["pid"]))
    except (UIInputError, PipelineError, ValueError) as error:
        st.error("输入尚未通过验证：{}".format(error))

    launch = st.session_state.get("last_launch")
    if launch:
        st.subheader("最近一次 UI 启动")
        st.json(launch)
        if st.button("刷新运行状态", icon=":material/refresh:"):
            pass
        manifest = _load_json_object(str(launch.get("manifest_path", "")))
        if manifest is not None:
            status = manifest.get("status", "unknown")
            st.write("流水线状态：**{}**".format(status))
        st.code(_tail(launch["log_path"]) or "日志尚无输出", language="text")
        clarification = find_clarification(launch)
        if clarification is not None:
            questions = clarification.get("questions", [])
            if isinstance(questions, list) and questions:
                with st.container(border=True):
                    st.warning("流水线正在等待你补充物理定义，尚未进入 Agent2。")
                    st.write(
                        "请用自然语言回答下面的问题。系统会保留原输入和回答，"
                        "然后从 Agent1 重新开始并继续后续流程。"
                    )
                    answers = []
                    for index, item in enumerate(questions, start=1):
                        question = _text(item.get("question", ""))
                        reason = _text(item.get("reason", ""))
                        st.markdown("**问题 {}：** {}".format(index, question))
                        if reason:
                            st.caption("为什么需要补充：{}".format(reason))
                        answers.append(st.text_area(
                            "你的补充说明 {}".format(index),
                            key="clarification_answer_{}_{}".format(
                                launch.get("run_id", "legacy"), index
                            ),
                            placeholder="例如：采用一味 1+1 维手征 NJL2，四费米项系数定义为……",
                        ))
                    ready = all(_text(answer) for answer in answers)
                    if st.button(
                        "提交补充并继续运行",
                        type="primary",
                        disabled=not ready or not bool(os.environ.get("OPENAI_API_KEY")),
                    ):
                        original = _load_json_object(str(launch.get("config_path", "")))
                        if original is None:
                            st.error("找不到本次运行的原始输入配置，无法继续。")
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
                                "补充信息已保存，新运行已从 Agent1 启动，PID={}".format(
                                    continued["pid"]
                                )
                            )


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
