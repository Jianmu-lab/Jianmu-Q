# -*- coding: utf-8 -*-
"""Thin deterministic controller for the Jianmu-Q workflow.

This module passes explicit artifacts between existing agents/runtimes. It does
not interpret reports or repair generated content.  Its one bounded scientific
feedback loop may request at most two paid Agent4.1/Agent5 redesigns when the
deterministic VQE runtime marks an ansatz insufficient. IBM submission occurs
only when the run specification authorizes it and Agent 9 approves the actual
transpiled circuit using current backend evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Mapping, Sequence

from run_spec import build_run_spec

from pipeline_support import (
    atomic_json_write,
    load_stage_output_manifest,
)
from observable_specification import (
    resolve_observable_sampling,
    validate_observable_specification,
)
from vqe_experience_store import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_VQE_ENERGY_TOLERANCE,
    DEFAULT_VQE_VARIANCE_TOLERANCE,
)


PIPELINE_SCHEMA_VERSION = "2.0"
STAGES = (
    "agent1",
    "agent2",
    "agent3",
    "agent4_state",
    "agent4_observable",
    "agent5_state",
    "agent5_observable",
    "vqe",
    "measurement",
    "agent6",
    "scipy_reference",
    "agent7",
    "itensor_reference",
    "crosscheck",
    "agent8",
    "final_report",
)
OPTIONAL_HARDWARE_STAGES = (
    "hardware_plan", "agent9", "hardware_submit", "hardware_retrieve",
)
DEFAULT_VQE_RUNTIME = {
    "restart_count": 2,
    "max_iterations": 200,
    "max_same_depth_continuations": 2,
    "first_depth_lower_bound": -0.05,
    "first_depth_upper_bound": 0.05,
    "new_layer_value": 0.0,
    "perturbation_scale": 0.01,
    "variance_tolerance": DEFAULT_VQE_VARIANCE_TOLERANCE,
    "energy_tolerance": DEFAULT_VQE_ENERGY_TOLERANCE,
    "energy_tie_tolerance": 1.0e-10,
    "gradient_check_step": 1.0e-6,
    "gradient_check_tolerance": 1.0e-5,
    "scan_all_depths": True,
    "stop_restarts_on_convergence": True,
    "record_experience": True,
    "adaptive_depth": {
        "enabled": True,
        "plateau_window": 3,
        "minimum_relative_improvement": 0.01,
    },
}
DEFAULT_MAX_ANSATZ_RETRIES = 2
MAX_AUTOMATIC_VQE_BUDGET_EXTENSIONS = 2


class PipelineError(RuntimeError):
    """Raised when a pipeline contract or stage fails."""


class UserClarificationRequired(PipelineError):
    """Agent1 needs additional physical input; this is not a failed run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "Expected one JSON object: {}".format(path))
    return value


def _absolute(project_root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))


def _normalize_state_target(value: Any, state_key: str) -> tuple[str, Dict[str, Any]]:
    """Return an Agent-facing description and a small executable target spec."""
    if isinstance(value, str) and value.strip():
        state_text = value.strip()
        _require(
            any(token in state_text.lower() for token in ("vacuum", "ground", "omega", "ω")),
            "concise config.{} must be a ground/vacuum string or a structured "
            "sector_eigenstate object".format(state_key),
        )
        return state_text, {
            "kind": "ground",
            "level_index": 0,
            "quantum_numbers": {},
        }

    _require(
        isinstance(value, Mapping),
        "config.{} must be a nonempty state string or object".format(state_key),
    )
    kind = value.get("kind")
    _require(
        kind == "sector_eigenstate",
        "config.{}.kind must be sector_eigenstate".format(state_key),
    )
    level_index = value.get("level_index")
    _require(
        isinstance(level_index, int)
        and not isinstance(level_index, bool)
        and level_index >= 0,
        "config.{}.level_index must be a nonnegative integer".format(state_key),
    )
    quantum_numbers = value.get("quantum_numbers")
    _require(
        isinstance(quantum_numbers, Mapping) and bool(quantum_numbers),
        "config.{}.quantum_numbers must be a nonempty object".format(state_key),
    )
    name = value.get("name", "")
    _require(
        isinstance(name, str),
        "config.{}.name must be a string when supplied".format(state_key),
    )
    spec = {
        "kind": "sector_eigenstate",
        "level_index": level_index,
        "quantum_numbers": dict(quantum_numbers),
    }
    if name.strip():
        spec["name"] = name.strip()
    description = (
        "Quantum-number-sector eigenstate{named}; level_index={level}; "
        "quantum_numbers={numbers}"
    ).format(
        named=" '{}'".format(name.strip()) if name.strip() else "",
        level=level_index,
        numbers=json.dumps(dict(quantum_numbers), ensure_ascii=False, sort_keys=True),
    )
    return description, spec


def _normalize_entry_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand the five-category user entry format while retaining legacy input."""
    if "agent1" in config:
        normalized = dict(config)
        physical = config.get("physical_parameters", {})
        simulation = config.get("simulation_parameters", {})
        if isinstance(physical, dict) and isinstance(simulation, dict):
            normalized["user_parameters"] = {**physical, **simulation}
        return normalized

    required = ("model", "parameters", "observable", "llm")
    missing = [key for key in required if key not in config]
    _require(
        not missing,
        "concise pipeline config is missing: {}".format(", ".join(missing)),
    )
    for key in ("model", "llm"):
        _require(isinstance(config[key], str) and config[key].strip(),
                 "config.{} must be a nonempty string".format(key))
    parameters = config["parameters"]
    _require(isinstance(parameters, dict) and parameters,
             "config.parameters must be a nonempty object")

    has_pair = "bra" in config or "ket" in config
    has_state_set = "states" in config
    _require(
        has_pair != has_state_set,
        "concise config must use either bra/ket or states, but not both",
    )
    if has_pair:
        _require("bra" in config and "ket" in config,
                 "concise config must supply both bra and ket")
        state_inputs = {"phi": config["bra"], "psi": config["ket"]}
    else:
        _require(
            isinstance(config.get("states"), Mapping) and bool(config["states"]),
            "concise config.states must be a nonempty object",
        )
        state_inputs = dict(config["states"])
        _require(
            all(isinstance(label, str) and label.strip() for label in state_inputs),
            "config.states labels must be nonempty strings",
        )

    aliases = set(required) | {
        "bra", "ket", "states", "continuum_conventions", "clarifications",
    }
    normalized = {key: value for key, value in config.items() if key not in aliases}
    normalized.setdefault("pipeline_config_schema_version", "1.0")
    state_descriptions: Dict[str, str] = {}
    state_specs: Dict[str, Dict[str, Any]] = {}
    for label, value in state_inputs.items():
        description, specification = _normalize_state_target(value, "states.{}".format(label))
        state_descriptions[label] = description
        state_specs[label] = specification
    normalized["agent1"] = {
        "model_input": config["model"],
        "observable": config["observable"],
        "external_states": state_descriptions,
        "continuum_conventions": dict(config.get("continuum_conventions", {})),
        "user_clarifications": list(config.get("clarifications", [])),
    }
    if has_pair:
        normalized["agent1"].update({
            "initial_state": state_descriptions["psi"],
            "final_state": state_descriptions["phi"],
        })
    # Agent1's existing file interface carries this unified immutable map.  The
    # strict Agent2 contract later classifies each model-specific value into its
    # physical or simulation/discretization sector.
    normalized["physical_parameters"] = dict(parameters)
    normalized["simulation_parameters"] = {}
    normalized["user_parameters"] = dict(parameters)
    normalized["vqe_runtime"] = {
        "target_kind_by_state": {
            label: state_specs[label]["kind"] for label in state_specs
        },
        "target_spec_by_state": state_specs,
    }
    normalized["model_name"] = config["llm"]
    return normalized


def validate_config(config: Mapping[str, Any], project_root: str) -> Dict[str, Any]:
    config = _normalize_entry_config(config)
    _require(config.get("pipeline_config_schema_version") == "1.0",
             "pipeline config schema must be 1.0")
    agent1 = config.get("agent1")
    _require(isinstance(agent1, dict), "config.agent1 must be an object")
    for key in ("model_input",):
        _require(isinstance(agent1.get(key), str) and agent1[key].strip(),
                 "config.agent1.{} must be nonempty".format(key))
    external_states = agent1.get("external_states")
    if external_states is None:
        for key in ("initial_state", "final_state"):
            _require(isinstance(agent1.get(key), str) and agent1[key].strip(),
                     "config.agent1.{} must be nonempty".format(key))
        external_states = {
            "phi": agent1["final_state"], "psi": agent1["initial_state"]
        }
    _require(
        isinstance(external_states, dict)
        and external_states
        and all(isinstance(label, str) and label.strip() for label in external_states)
        and all(isinstance(value, str) and value.strip() for value in external_states.values()),
        "config.agent1.external_states must define nonempty named states",
    )
    normalized = dict(config)
    normalized["agent1"] = dict(agent1)
    normalized["agent1"]["external_states"] = dict(external_states)
    conventions = agent1.get("continuum_conventions", {})
    _require(isinstance(conventions, dict),
             "config.agent1.continuum_conventions must be an object")
    normalized["agent1"]["continuum_conventions"] = dict(conventions)
    clarifications = agent1.get("user_clarifications", [])
    _require(isinstance(clarifications, list),
             "config clarifications must be a list")
    for item in clarifications:
        _require(
            isinstance(item, dict)
            and isinstance(item.get("question"), str) and item["question"].strip()
            and isinstance(item.get("answer"), str) and item["answer"].strip(),
            "every clarification must contain a nonempty question and answer",
        )
    normalized["agent1"]["user_clarifications"] = [dict(item) for item in clarifications]
    observable = validate_observable_specification(agent1.get("observable"))
    physical_parameters = config.get("physical_parameters")
    _require(isinstance(physical_parameters, dict) and physical_parameters,
             "config.physical_parameters must be a nonempty object")
    normalized["physical_parameters"] = dict(physical_parameters)
    simulation_parameters = config.get("simulation_parameters")
    _require(isinstance(simulation_parameters, dict),
             "config.simulation_parameters must be an object")
    overlap = set(physical_parameters) & set(simulation_parameters)
    _require(not overlap,
             "physical_parameters and simulation_parameters overlap: {}".format(
                 ", ".join(sorted(overlap))
             ))
    normalized["simulation_parameters"] = dict(simulation_parameters)
    user_parameters = config.get("user_parameters", {
        **physical_parameters, **simulation_parameters,
    })
    _require(isinstance(user_parameters, dict) and user_parameters,
             "config.user_parameters must be a nonempty object")
    normalized["user_parameters"] = dict(user_parameters)
    normalized["agent1"]["observable"] = resolve_observable_sampling(
        observable, user_parameters
    )
    vqe_runtime = config.get("vqe_runtime")
    _require(isinstance(vqe_runtime, dict), "config.vqe_runtime must be an object")
    targets = vqe_runtime.get("target_kind_by_state")
    _require(
        isinstance(targets, dict)
        and set(targets) == set(external_states),
        "config.vqe_runtime.target_kind_by_state must define every named state",
    )
    matrix_elements = normalized["agent1"]["observable"].get("matrix_elements")
    if matrix_elements is not None:
        referenced = {
            label for term in matrix_elements for label in (term["bra"], term["ket"])
        }
        _require(
            referenced.issubset(set(external_states)),
            "observable.matrix_elements references an undefined state: {}".format(
                ", ".join(sorted(referenced - set(external_states)))
            ),
        )
    elif set(external_states) != {"phi", "psi"}:
        raise PipelineError(
            "named state sets require observable.matrix_elements"
        )
    normalized["vqe_runtime"] = dict(vqe_runtime)
    retries = config.get("max_ansatz_retries", DEFAULT_MAX_ANSATZ_RETRIES)
    _require(
        isinstance(retries, int) and not isinstance(retries, bool)
        and 0 <= retries <= DEFAULT_MAX_ANSATZ_RETRIES,
        "config.max_ansatz_retries must be an integer from 0 to {}".format(
            DEFAULT_MAX_ANSATZ_RETRIES
        ),
    )
    normalized["max_ansatz_retries"] = retries
    # Human reports are non-authoritative and are never compiled or repaired by
    # the controller.  Ignore the former latex_repair setting for compatibility.
    normalized.pop("latex_repair", None)
    normalized["database"] = _absolute(
        project_root,
        str(config.get("database", DEFAULT_DATABASE_PATH)),
    )
    normalized["agent2_library"] = _absolute(
        project_root, str(config.get("agent2_library", "A2_library"))
    )
    normalized["agent4_library"] = _absolute(
        project_root, str(config.get("agent4_library", "A4_library"))
    )
    _require(os.path.isdir(normalized["agent2_library"]), "A2 library is missing")
    _require(os.path.isdir(normalized["agent4_library"]), "A4 library is missing")
    normalized["model_name"] = str(config.get("model_name", "gpt-5.6-sol"))
    _require(normalized["model_name"].strip(), "config.model_name must be nonempty")
    normalized["python_executable"] = str(config.get("python_executable", sys.executable))
    normalized["julia_executable"] = str(config.get("julia_executable", "julia"))
    hardware_validation = config.get("hardware_validation", False)
    _require(
        isinstance(hardware_validation, bool),
        "config.hardware_validation must be true or false",
    )
    through = (
        "final_report" if hardware_validation
        else str(config.get("through", "crosscheck"))
    )
    _require(through in STAGES, "config.through is not a known stage")
    normalized["through"] = through
    normalized["hardware_validation"] = hardware_validation
    hardware_mode = str(config.get("hardware_mode", "representative"))
    _require(
        hardware_mode in ("representative", "full_d1_nt1"),
        "config.hardware_mode must be representative or full_d1_nt1",
    )
    normalized["hardware_mode"] = hardware_mode
    return normalized


def _new_manifest(
    config_path: str, config: Mapping[str, Any], project_root: str,
    run_id: str | None,
) -> tuple[Dict[str, Any], str]:
    if run_id is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = "{}_{}".format(stamp, uuid.uuid4().hex[:8])
    _require(run_id and all(char.isalnum() or char in "_-" for char in run_id),
             "run_id may contain only letters, digits, '_' and '-'")
    run_root = _absolute(project_root, str(config.get("run_root", "runs")))
    run_directory = os.path.join(run_root, run_id)
    _require(not os.path.exists(run_directory), "run directory already exists")
    os.makedirs(os.path.join(run_directory, "handoffs"), exist_ok=False)
    os.makedirs(os.path.join(run_directory, "artifacts"), exist_ok=False)
    for category in (
        "reports", "contracts", "adapters", "runtime", "circuits",
        "references", "analysis", "audit",
    ):
        os.makedirs(os.path.join(run_directory, category), exist_ok=False)
    manifest_path = os.path.join(run_directory, "pipeline_manifest.json")
    manifest: Dict[str, Any] = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "created",
        "project_root": project_root,
        "run_directory": run_directory,
        "config_path": os.path.abspath(config_path),
        "artifacts": {},
        "stages": {name: {"status": "pending"} for name in STAGES},
        "ansatz_search": {
            "current_attempt": 1,
            "maximum_attempts": config["max_ansatz_retries"] + 1,
            "attempts": [],
        },
    }
    inputs = os.path.join(run_directory, "inputs")
    os.makedirs(inputs, exist_ok=False)
    run_spec_path = os.path.join(inputs, "run_spec.json")
    atomic_json_write(run_spec_path, build_run_spec(run_id, config))
    _record_artifact(manifest, "run_spec", run_spec_path)
    reusable = _approved_reuse_source(
        _load_json(run_spec_path), run_root, excluded_manifest=manifest_path
    )
    if reusable is not None:
        prior, prior_manifest_path = reusable
        manifest["program_reuse"] = {
            "status": "selected",
            "source_run_id": prior.get("run_id"),
            "source_manifest": os.path.abspath(prior_manifest_path),
        }
    else:
        manifest["program_reuse"] = {"status": "no_approved_match"}
    controller_settings_path = os.path.join(inputs, "controller_settings.json")
    atomic_json_write(
        controller_settings_path,
        {
            "python_executable": config["python_executable"],
            "julia_executable": config["julia_executable"],
            "database": config["database"],
            "agent2_library": config["agent2_library"],
            "agent4_library": config["agent4_library"],
            "max_ansatz_retries": config["max_ansatz_retries"],
        },
    )
    _record_artifact(manifest, "controller_settings", controller_settings_path)
    observable_path = os.path.join(inputs, "agent1_observable.json")
    atomic_json_write(observable_path, config["agent1"]["observable"])
    _record_artifact(manifest, "agent1_observable", observable_path)
    atomic_json_write(manifest_path, manifest)
    return manifest, manifest_path


def _verify_inputs(manifest: Mapping[str, Any]) -> None:
    _artifact_path(manifest, "run_spec")


def _artifact_path(manifest: Mapping[str, Any], name: str) -> str:
    record = manifest["artifacts"].get(name)
    _require(isinstance(record, dict), "required artifact is unavailable: {}".format(name))
    path = record.get("path")
    _require(isinstance(path, str) and os.path.isfile(path),
             "artifact file is missing: {}".format(name))
    return path


def _observable_shape(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the user observable without its freely variable sample grid."""
    shaped = json.loads(json.dumps(value, ensure_ascii=False))
    shaped.pop("sampling", None)
    return shaped


def _normalize_physical_text(value: Any) -> Any:
    """Ignore harmless whitespace while preserving the user's physical content."""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalize_physical_text(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_physical_text(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return value


def _physical_reuse_target(run_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only the UI-owned physics that defines reusable program logic.

    Numerical model parameters (including m, g, a and L), the observable sample
    grid, LLM choice and execution settings deliberately do not participate.
    Internal discretization, encoding and boundary-condition choices are not UI
    facts and therefore cannot veto a match.
    """
    user = run_spec.get("user_input", {})
    _require(isinstance(user, Mapping), "RunSpec has no user_input object")
    observable = user.get("observable", {})
    states = user.get("external_states", {})
    _require(isinstance(observable, Mapping), "RunSpec observable is not an object")
    _require(isinstance(states, Mapping), "RunSpec external_states is not an object")
    return _normalize_physical_text({
        "model": user.get("model"),
        "external_states": states,
        "observable": _observable_shape(observable),
        "continuum_conventions": user.get("continuum_conventions", {}),
        "clarifications": user.get("clarifications", []),
    })


def _reusable_program_files_ready(manifest: Mapping[str, Any]) -> bool:
    try:
        state_adapter = _artifact_path(manifest, "state_adapter")
        observable_adapter = _artifact_path(manifest, "observable_adapter")
        scipy_script = _artifact_path(manifest, "scipy_script")
        itensor_script = _artifact_path(manifest, "itensor_script")
    except PipelineError:
        return False
    try:
        with open(state_adapter, "r", encoding="utf-8") as handle:
            if "def configure_runtime(" not in handle.read():
                return False
        with open(observable_adapter, "r", encoding="utf-8") as handle:
            if "def configure_runtime(" not in handle.read():
                return False
        if not all(
            _reference_accepts_runtime_inputs(path)
            for path in (scipy_script, itensor_script)
        ):
            return False
    except OSError:
        return False
    return True


def _reuse_is_human_approved(manifest: Mapping[str, Any]) -> bool:
    approval = manifest.get("program_reuse_approval", {})
    if not isinstance(approval, Mapping) or approval.get("status") != "approved":
        return False
    final_stage = manifest.get("stages", {}).get("final_report", {})
    if not isinstance(final_stage, Mapping) or final_stage.get("status") != "completed":
        return False
    try:
        _artifact_path(manifest, "final_report_markdown")
    except PipelineError:
        return False
    return _reusable_program_files_ready(manifest)


def _approved_reuse_source(
    run_spec: Mapping[str, Any], run_root: str,
    excluded_manifest: str | None = None,
) -> tuple[Dict[str, Any], str] | None:
    """Find the newest human-approved run for the same UI physics target."""
    target = _physical_reuse_target(run_spec)
    candidates: List[str] = []
    if os.path.isdir(run_root):
        for name in os.listdir(run_root):
            path = os.path.abspath(
                os.path.join(run_root, name, "pipeline_manifest.json")
            )
            if os.path.isfile(path) and path != os.path.abspath(excluded_manifest or ""):
                candidates.append(path)
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        try:
            prior = _load_json(path)
            if not _reuse_is_human_approved(prior):
                continue
            prior_spec = _load_json(_artifact_path(prior, "run_spec"))
            if _physical_reuse_target(prior_spec) == target:
                return prior, path
        except (OSError, ValueError, PipelineError, KeyError, TypeError):
            continue
    return None


def find_approved_reuse_source_for_config(
    config: Mapping[str, Any], project_root: str,
) -> tuple[Dict[str, Any], str] | None:
    """Public UI helper: decide reuse before launching any scientific Agent."""
    run_spec = build_run_spec("reuse-preview", config)
    run_root = _absolute(project_root, str(config.get("run_root", "runs")))
    return _approved_reuse_source(run_spec, run_root)


def _selected_approved_reuse_source(
    manifest: Mapping[str, Any],
) -> tuple[Dict[str, Any], str] | None:
    selection = manifest.get("program_reuse", {})
    if not isinstance(selection, Mapping) or selection.get("status") not in {
        "selected", "materialized"
    }:
        return None
    path = selection.get("source_manifest")
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    prior = _load_json(path)
    if not _reuse_is_human_approved(prior):
        return None
    current_spec = _load_json(_artifact_path(manifest, "run_spec"))
    prior_spec = _load_json(_artifact_path(prior, "run_spec"))
    if _physical_reuse_target(prior_spec) != _physical_reuse_target(current_spec):
        return None
    return prior, os.path.abspath(path)


def _runtime_parameter_payload(run_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only user-owned runtime parameter values from one RunSpec."""
    user_input = run_spec.get("user_input", {})
    payload = dict(user_input.get("parameters", {}))
    groups = user_input.get("parameter_groups", {})
    simulation = groups.get("simulation", {}) if isinstance(groups, Mapping) else {}
    if isinstance(simulation, Mapping):
        payload.update(simulation)
    return payload


def _mapping_has_parameterized_hamiltonian(
    mapping: Mapping[str, Any], parameter_contract: Mapping[str, Any],
) -> bool:
    """Whether at least one Hamiltonian coefficient names a runtime parameter."""
    tokens = set()
    for entry in parameter_contract.get("parameters", []):
        if not isinstance(entry, Mapping):
            continue
        for key in ("name", "symbol"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                tokens.add(value.strip())
    expressions = [
        str(term.get("coefficient", ""))
        for term in mapping.get("hamiltonian", {}).get("terms", [])
        if isinstance(term, Mapping)
    ]
    return any(
        re.search(r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])".format(re.escape(token)), expression)
        for token in tokens
        for expression in expressions
    )


def _require_safe_mapping_for_partial_regeneration(
    stage: str, manifest: Mapping[str, Any],
) -> None:
    """Reject only the unsafe hybrid: new parameters plus an old numeric map."""
    if stage not in {
        "agent4_state", "agent4_observable", "agent5_state", "agent5_observable",
    }:
        return
    selected = _selected_approved_reuse_source(manifest)
    if selected is None:
        return
    prior, _ = selected
    current_spec = _load_json(_artifact_path(manifest, "run_spec"))
    prior_spec = _load_json(_artifact_path(prior, "run_spec"))
    if _runtime_parameter_payload(current_spec) == _runtime_parameter_payload(prior_spec):
        return
    mapping = _load_json(_artifact_path(manifest, "agent3_mapping_contract"))
    parameters = _load_json(_artifact_path(manifest, "agent2_parameter_contract"))
    _require(
        _mapping_has_parameterized_hamiltonian(mapping, parameters),
        "Cannot regenerate Agent4/5 after changing runtime parameters because "
        "the reused legacy Agent3 Hamiltonian contains only old numerical "
        "coefficients. Keep the approved parameterized adapters, or regenerate "
        "Agent3 first so its Pauli coefficients retain Agent2 parameter expressions.",
    )


def _rebind_reused_parameter_contract(
    source: Mapping[str, Any], run_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Replace only UI-owned numerical values in one approved contract."""
    rebound = json.loads(json.dumps(source, ensure_ascii=False))
    user_parameters = run_spec["user_input"]["parameters"]
    simulation_parameters = (
        run_spec["user_input"].get("parameter_groups", {}).get("simulation", {})
    )
    _require(
        isinstance(simulation_parameters, Mapping),
        "RunSpec simulation parameters must be an object",
    )
    supplied = dict(user_parameters)
    supplied.update(simulation_parameters)
    aliases: Dict[str, Any] = {}
    for name, value in supplied.items():
        aliases[str(name).strip()] = value
        if ":" in str(name):
            aliases[str(name).rsplit(":", 1)[1].strip()] = value

    changed = set()
    for entry in rebound.get("parameters", []):
        if not isinstance(entry, dict):
            continue
        source_ref = entry.get("source_ref")
        if isinstance(source_ref, str) and source_ref.startswith(
            "/user_input/parameters/"
        ):
            key = source_ref[len("/user_input/parameters/"):]
            if key in user_parameters:
                entry["selected_value"] = user_parameters[key]
                changed.add(key)
                continue
        if isinstance(source_ref, str) and source_ref.startswith(
            "/user_input/parameter_groups/simulation/"
        ):
            key = source_ref[len("/user_input/parameter_groups/simulation/"):]
            if key in simulation_parameters:
                entry["selected_value"] = simulation_parameters[key]
                continue
        symbol = str(entry.get("symbol", "")).strip()
        name = str(entry.get("name", "")).strip()
        if symbol in aliases:
            entry["selected_value"] = aliases[symbol]
        elif name in aliases:
            entry["selected_value"] = aliases[name]

    missing = sorted(set(user_parameters).difference(changed))
    _require(
        not missing,
        "Approved parameter program has no runtime binding for: {}".format(
            ", ".join(missing)
        ),
    )

    by_name = {
        item.get("name"): item
        for item in rebound.get("parameters", []) if isinstance(item, dict)
    }
    site_entry = by_name.get("spatial_site_count")
    mode_entry = by_name.get("fermionic_mode_count")
    if isinstance(site_entry, dict) and isinstance(mode_entry, dict):
        site_count = site_entry.get("selected_value")
        if isinstance(site_count, int) and not isinstance(site_count, bool):
            mode_entry["selected_value"] = 2 * site_count
    return rebound


def _copy_reused_artifact(
    source_manifest: Mapping[str, Any], artifact_name: str,
    destination_directory: str, output_tag: str,
) -> str:
    source = _artifact_path(source_manifest, artifact_name)
    extension = os.path.splitext(source)[1]
    destination = os.path.join(
        destination_directory,
        "{}_Reused_{}{}".format(artifact_name, output_tag, extension),
    )
    shutil.copyfile(source, destination)
    return destination


def _materialize_approved_program_reuse(
    manifest: Dict[str, Any], manifest_path: str,
) -> bool:
    """Install one approved program bundle without invoking Agents 1--7."""
    selection = manifest.get("program_reuse", {})
    if isinstance(selection, Mapping) and selection.get("status") == "materialized":
        return True
    reusable = _selected_approved_reuse_source(manifest)
    if reusable is None:
        return False
    prior, prior_manifest_path = reusable
    run_directory = manifest["run_directory"]
    output_tag = str(manifest["run_id"])
    run_spec = _load_json(_artifact_path(manifest, "run_spec"))

    copied = {
        "agent1_report": ("reports",),
        "agent1_theory_contract": ("contracts",),
        "agent2_report": ("reports",),
        "agent2_discretization_contract": ("contracts",),
        "agent3_report": ("reports",),
        "agent3_mapping_contract": ("contracts",),
        "agent4_state_report": ("reports",),
        "agent4_state_design_contract": ("contracts",),
        "agent4_observable_report": ("reports",),
        "agent4_observable_design_contract": ("contracts",),
        "scipy_script": ("references",),
        "itensor_script": ("references",),
    }
    for artifact_name, (subdirectory,) in copied.items():
        path = _copy_reused_artifact(
            prior, artifact_name, os.path.join(run_directory, subdirectory), output_tag
        )
        _record_artifact(manifest, artifact_name, path)

    parameter_contract = _rebind_reused_parameter_contract(
        _load_json(_artifact_path(prior, "agent2_parameter_contract")), run_spec
    )
    parameter_path = os.path.join(
        run_directory, "contracts",
        "Agent2_ParameterContract_Reused_{}.json".format(output_tag),
    )
    atomic_json_write(parameter_path, parameter_contract)
    _record_artifact(manifest, "agent2_parameter_contract", parameter_path)

    runtime_settings = _generate_runtime_settings(manifest)
    _record_artifact(manifest, "runtime_settings", runtime_settings)

    from agent5_test_slim_qiskit import (
        ensure_runtime_configuration,
        ensure_state_runtime_configuration,
        inject_authoritative_constants,
        validate_generated_adapter,
        MODE_OBSERVABLE,
        MODE_STATE_PREPARATION,
    )

    state_source = open(
        _artifact_path(prior, "state_adapter"), "r", encoding="utf-8"
    ).read()
    state_source = inject_authoritative_constants(
        state_source,
        {
            "PARAMETER_CONTRACT": parameter_contract,
            "STATE_PREPARATION_CONTRACT": _load_json(
                _artifact_path(manifest, "agent4_state_design_contract")
            ),
        },
    )
    state_source = ensure_state_runtime_configuration(state_source)
    state_source = validate_generated_adapter(state_source, MODE_STATE_PREPARATION)
    state_path = os.path.join(
        run_directory, "adapters",
        "StatePreparationAdapter_{}.py".format(output_tag),
    )
    with open(state_path, "w", encoding="utf-8") as handle:
        handle.write(state_source.rstrip() + "\n")
    _record_artifact(manifest, "state_adapter", state_path)

    observable_spec = _load_json(_artifact_path(manifest, "agent1_observable"))
    observable_source = open(
        _artifact_path(prior, "observable_adapter"), "r", encoding="utf-8"
    ).read()
    observable_source = inject_authoritative_constants(
        observable_source,
        {
            "PARAMETER_CONTRACT": parameter_contract,
            "OBSERVABLE_CONTRACT": _load_json(
                _artifact_path(manifest, "agent4_observable_design_contract")
            ),
            "OBSERVABLE_SPECIFICATION": observable_spec,
        },
    )
    observable_source = ensure_runtime_configuration(observable_source)
    observable_source = validate_generated_adapter(
        observable_source, MODE_OBSERVABLE
    )
    observable_path = os.path.join(
        run_directory, "adapters", "ObservableAdapter_{}.py".format(output_tag)
    )
    with open(observable_path, "w", encoding="utf-8") as handle:
        handle.write(observable_source.rstrip() + "\n")
    _record_artifact(manifest, "observable_adapter", observable_path)

    source_vqe = _load_json(_artifact_path(prior, "vqe_config"))
    runtime = dict(DEFAULT_VQE_RUNTIME)
    runtime.update(_execution_config(manifest)["vqe_runtime"])
    source_vqe.update(runtime)
    source_vqe["experience_database"] = _execution_config(manifest)["database"]
    source_vqe["checkpoint_file"] = os.path.join(
        run_directory, "runtime", "VQE_Runtime_Checkpoint_{}.json".format(output_tag)
    )
    source_vqe["result_file"] = os.path.join(
        run_directory, "runtime", "VQE_Runtime_Result_{}.json".format(output_tag)
    )
    source_vqe["parameter_bundle_file"] = os.path.join(
        run_directory, "runtime", "VQE_ParameterBundle_{}.json".format(output_tag)
    )
    vqe_path = os.path.join(
        run_directory, "runtime", "VQE_RunConfig_{}.json".format(output_tag)
    )
    atomic_json_write(vqe_path, source_vqe)
    _record_artifact(manifest, "vqe_config", vqe_path)

    stage_artifacts = {
        "agent1": ("agent1_report", "agent1_theory_contract"),
        "agent2": (
            "agent2_report", "agent2_parameter_contract",
            "agent2_discretization_contract", "runtime_settings",
        ),
        "agent3": ("agent3_report", "agent3_mapping_contract"),
        "agent4_state": (
            "agent4_state_report", "agent4_state_design_contract",
        ),
        "agent4_observable": (
            "agent4_observable_report", "agent4_observable_design_contract",
        ),
        "agent5_state": ("state_adapter", "vqe_config"),
        "agent5_observable": ("observable_adapter",),
        "agent6": ("scipy_script",),
        "agent7": ("itensor_script",),
    }
    for stage, artifacts in stage_artifacts.items():
        manifest["stages"][stage] = {
            "status": "completed",
            "artifacts": list(artifacts),
            "reused_from": prior_manifest_path,
        }
    manifest["program_reuse"] = {
        "status": "materialized",
        "source_run_id": prior.get("run_id"),
        "source_manifest": prior_manifest_path,
    }
    manifest["status"] = "running"
    _save(manifest_path, manifest)
    print(
        "[reuse] installed approved program bundle from {}; Agents 1--7 skipped"
        .format(prior.get("run_id", prior_manifest_path))
    )
    return True


def _find_reusable_run(
    manifest: Mapping[str, Any], stage: str,
) -> tuple[Dict[str, Any], str] | None:
    """Return the UI-matched, human-approved source for one reusable stage."""
    requirements = {
        "agent4_observable": ("agent4_observable_design_contract", "observable_adapter"),
        "agent5_observable": ("observable_adapter",),
        "agent6": ("scipy_script",),
        "agent7": ("itensor_script",),
    }
    if stage not in requirements:
        return None
    selection = manifest.get("program_reuse", {})
    if not isinstance(selection, Mapping) or selection.get("status") != "selected":
        return None
    path = selection.get("source_manifest")
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    try:
        prior = _load_json(path)
        if not _reuse_is_human_approved(prior):
            return None
        current_spec = _load_json(_artifact_path(manifest, "run_spec"))
        prior_spec = _load_json(_artifact_path(prior, "run_spec"))
        if _physical_reuse_target(prior_spec) != _physical_reuse_target(current_spec):
            return None
        for artifact in requirements[stage]:
            _artifact_path(prior, artifact)
        if stage in ("agent4_observable", "agent5_observable"):
            source = _artifact_path(prior, "observable_adapter")
            with open(source, "r", encoding="utf-8") as handle:
                if "def configure_runtime(" not in handle.read():
                    return None
        elif not _reference_accepts_runtime_inputs(
            _artifact_path(prior, requirements[stage][0])
        ):
            return None
        return prior, path
    except (OSError, ValueError, PipelineError, KeyError, TypeError):
        return None


def _execution_config(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Load operational settings without duplicating scientific input in manifest."""
    legacy = manifest.get("config")
    if isinstance(legacy, dict):
        return dict(legacy)
    settings = _load_json(_artifact_path(manifest, "controller_settings"))
    run_spec = _load_json(_artifact_path(manifest, "run_spec"))
    request = run_spec.get("execution_request", {})
    settings.update({
        "model_name": request["llm"],
        "through": request["through"],
        "hardware_validation": bool(request.get("hardware_validation", False)),
        "hardware_mode": str(request.get("hardware_mode", "representative")),
        "vqe_runtime": request["vqe_runtime"],
        "max_ansatz_retries": request.get(
            "max_ansatz_retries",
            settings.get("max_ansatz_retries", DEFAULT_MAX_ANSATZ_RETRIES),
        ),
    })
    return settings


def _record_artifact(
    manifest: Dict[str, Any], name: str, path: str,
) -> None:
    absolute = os.path.abspath(path)
    _require(os.path.isfile(absolute), "stage did not produce {}".format(absolute))
    manifest["artifacts"][name] = {"path": absolute}


def _stage_manifest_path(manifest: Mapping[str, Any], stage: str) -> str:
    attempt = int(manifest.get("ansatz_search", {}).get("current_attempt", 1))
    suffix = (
        "_ansatz{}".format(attempt)
        if attempt > 1 and stage in ("agent4_state", "agent5_state") else ""
    )
    return os.path.join(
        manifest["run_directory"], "handoffs", stage + suffix + ".json"
    )


def _stage_output_tag(manifest: Mapping[str, Any], stage: str) -> str:
    attempt = int(manifest.get("ansatz_search", {}).get("current_attempt", 1))
    if attempt > 1 and stage in ("agent4_state", "agent5_state"):
        return "{}_ansatz{}".format(manifest["run_id"], attempt)
    return str(manifest["run_id"])


def _generate_runtime_settings(manifest: Dict[str, Any]) -> str:
    contract = _load_json(_artifact_path(manifest, "agent2_parameter_contract"))
    parameters = contract.get("parameters")
    _require(isinstance(parameters, list) and parameters,
             "Agent2 parameter contract contains no parameters")
    unresolved_required = [
        item.get("name") for item in parameters
        if item.get("required_for_run") is True and item.get("selected_value") is None
    ]
    _require(not unresolved_required,
             "Agent2 left required run parameters unresolved: {}".format(
                 ", ".join(str(name) for name in unresolved_required)
             ))
    resolved_values = {
        item["name"]: item["selected_value"]
        for item in parameters if item.get("selected_value") is not None
    }
    _require(resolved_values, "Agent2 resolved no simulation parameters")
    by_name = {item["name"]: item for item in parameters}
    run_spec = _load_json(_artifact_path(manifest, "run_spec"))
    user_input = run_spec["user_input"]
    supplied = user_input["parameters"]
    parameter_groups = user_input.get("parameter_groups", {})
    simulation_inputs = parameter_groups.get("simulation", {})
    missing_user_parameters = sorted(set(supplied) - set(by_name))
    _require(
        not missing_user_parameters,
        "Agent2 omitted user parameters: {}".format(
            ", ".join(missing_user_parameters)
        ),
    )
    changed_user_parameters = sorted(
        name for name, value in supplied.items()
        if by_name[name].get("selected_value") != value
    )
    _require(
        not changed_user_parameters,
        "Agent2 changed user parameters: {}".format(
            ", ".join(changed_user_parameters)
        ),
    )
    missing_controls = sorted(
        set(simulation_inputs) - set(by_name)
    )
    _require(not missing_controls,
             "Agent2 omitted user simulation controls: {}".format(
                 ", ".join(missing_controls)
             ))
    incorrect_controls = sorted(
        name for name, value in simulation_inputs.items()
        if by_name[name].get("selected_value") != value
    )
    _require(not incorrect_controls,
             "Agent2 did not preserve user simulation controls: {}".format(
                 ", ".join(incorrect_controls)
             ))
    external_states = contract.get("external_states")
    _require(isinstance(external_states, dict),
             "Agent2 contract must contain explicit external_states")
    config = _execution_config(manifest)
    targets = config["vqe_runtime"]["target_kind_by_state"]
    labels = external_states.get("labels")
    _require(isinstance(labels, list) and set(targets) == set(labels),
             "vqe target_kind_by_state must match Agent2 external-state labels")
    runtime = dict(DEFAULT_VQE_RUNTIME)
    runtime.update(config["vqe_runtime"])
    runtime.setdefault("experience_database", config["database"])
    runtime.setdefault("random_seed", 12345)
    document = {
        "schema_version": "1.0",
        "source": "pipeline_resolved",
        "model": contract["model"]["name"],
        "user_inputs": supplied,
        "simulation_inputs": simulation_inputs,
        "resolved_values": resolved_values,
        "external_states": external_states,
        "vqe_runtime": runtime,
        "sources": {
            "user_parameters": "run_spec.user_input.parameters",
            "simulation_parameters": "run_spec.user_input.parameter_groups.simulation",
            "resolved_values": "agent2_parameter_contract.selected_value",
            "external_states": "agent2_parameter_contract.external_states",
            "agent2_parameter_contract": _artifact_path(
                manifest, "agent2_parameter_contract"
            ),
        },
    }
    inputs = os.path.join(manifest["run_directory"], "inputs")
    if not os.path.isdir(inputs):
        inputs = os.path.join(manifest["run_directory"], "internal")
    os.makedirs(inputs, exist_ok=True)
    path = os.path.join(inputs, "resolved_runtime_settings.json")
    atomic_json_write(path, document)
    return path


def _python(config: Mapping[str, Any], project_root: str, script: str) -> List[str]:
    return [str(config["python_executable"]), os.path.join(project_root, script)]


def _agent_command(
    stage: str, manifest: Mapping[str, Any], regenerate: bool = False,
) -> tuple[List[str], str, Mapping[str, str]]:
    _require_safe_mapping_for_partial_regeneration(stage, manifest)
    config = _execution_config(manifest)
    root = manifest["project_root"]
    output_manifest = _stage_manifest_path(manifest, stage)
    output_tag = _stage_output_tag(manifest, stage)
    model = str(config["model_name"])
    if stage == "agent1":
        command = _python(config, root, "agent1_formulation.py") + [
            "--run-spec", _artifact_path(manifest, "run_spec"),
            "--model", model, "--output-manifest", output_manifest,
            "--output-tag", output_tag,
            "--output-root", manifest["run_directory"],
        ]
        mapping = {
            "report": "agent1_report",
            "theory_contract": "agent1_theory_contract",
        }
    elif stage == "agent2":
        command = _python(config, root, "agent2_Discretization.py") + [
            "--run-spec", _artifact_path(manifest, "run_spec"),
            "--theory-contract", _artifact_path(manifest, "agent1_theory_contract"),
            "--library", config["agent2_library"],
            "--experience-db", config["database"],
            "--model", model, "--output-manifest", output_manifest,
            "--output-tag", output_tag,
            "--output-root", manifest["run_directory"],
        ]
        mapping = {
            "report": "agent2_report",
            "parameter_contract": "agent2_parameter_contract",
            "discretization_contract": "agent2_discretization_contract",
        }
    elif stage == "agent3":
        command = _python(config, root, "agent3_translator.py") + [
            "--discretization-contract", _artifact_path(
                manifest, "agent2_discretization_contract"
            ),
            "--parameter-contract", _artifact_path(manifest, "agent2_parameter_contract"),
            "--model", model, "--output-manifest", output_manifest,
            "--output-tag", output_tag,
            "--output-root", manifest["run_directory"],
        ]
        mapping = {
            "report": "agent3_report",
            "mapping_contract": "agent3_mapping_contract",
        }
    elif stage in ("agent4_state", "agent4_observable"):
        script = "agent4_1_state_preparation.py" if stage == "agent4_state" else "agent4_2_observable.py"
        command = _python(config, root, script) + [
            "--mapping-contract", _artifact_path(manifest, "agent3_mapping_contract"),
            "--parameter-contract", _artifact_path(manifest, "agent2_parameter_contract"),
            "--library", config["agent4_library"],
            "--model", model, "--output-manifest", output_manifest,
            "--output-tag", output_tag,
            "--output-root", manifest["run_directory"],
        ]
        if stage == "agent4_state":
            runtime = dict(DEFAULT_VQE_RUNTIME)
            runtime.update(config["vqe_runtime"])
            command += ["--experience-db", config["database"]]
            command += [
                "--search-attempt",
                str(manifest.get("ansatz_search", {}).get("current_attempt", 1)),
                "--energy-tolerance",
                str(runtime["energy_tolerance"]),
                "--variance-tolerance",
                str(runtime["variance_tolerance"]),
            ]
            prefix = "agent4_state"
        else:
            command += ["--observable-spec", _artifact_path(manifest, "agent1_observable")]
            prefix = "agent4_observable"
        mapping = {
            "report": prefix + "_report",
            "design_contract": prefix + "_design_contract",
        }
    elif stage in ("agent5_state", "agent5_observable"):
        state = stage == "agent5_state"
        command = _python(config, root, "agent5_test_slim_qiskit.py") + [
            "--mode", "state_preparation" if state else "observable",
            "--mapping-contract", _artifact_path(manifest, "agent3_mapping_contract"),
            "--parameter-contract", _artifact_path(manifest, "agent2_parameter_contract"),
            "--design-contract", _artifact_path(
                manifest,
                "agent4_state_design_contract" if state
                else "agent4_observable_design_contract",
            ),
            "--model", model, "--output-manifest", output_manifest,
            "--output-tag", output_tag,
            "--output-root", manifest["run_directory"],
        ]
        if state:
            command += ["--runtime-settings", _artifact_path(manifest, "runtime_settings")]
            mapping = {"adapter": "state_adapter", "vqe_config": "vqe_config"}
        else:
            command += ["--observable-spec", _artifact_path(manifest, "agent1_observable")]
            current_adapter = os.path.join(
                manifest["run_directory"], "adapters",
                "ObservableAdapter_{}.py".format(output_tag),
            )
            if os.path.isfile(current_adapter):
                # A failed smoke test may leave a complete adapter checkpoint.
                # Revalidate that checkpoint on resume instead of paying for a
                # second model translation.
                command += ["--reuse-adapter", current_adapter]
            else:
                reusable = _find_reusable_run(manifest, stage)
                if reusable is not None:
                    prior, _ = reusable
                    command += [
                        "--reuse-adapter", _artifact_path(prior, "observable_adapter")
                    ]
            mapping = {"adapter": "observable_adapter"}
    else:
        raise PipelineError("not an Agent1-5 stage: {}".format(stage))
    if regenerate:
        _require(
            stage in ("agent4_state", "agent4_observable"),
            "explicit regeneration is currently supported only for Agent4",
        )
        command.append("--regenerate")
    return command, output_manifest, mapping


def _reference_accepts_runtime_inputs(path: str) -> bool:
    """Keep old generated references runnable while enabling reusable ones."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return "--parameter-contract" in source and "--observable-spec" in source


def _known_output_command(
    stage: str, manifest: Mapping[str, Any]
) -> tuple[List[str], Mapping[str, str]]:
    config = _execution_config(manifest)
    root = manifest["project_root"]
    runtime_directory = os.path.join(manifest["run_directory"], "runtime")
    reference_directory = os.path.join(manifest["run_directory"], "references")
    analysis_directory = os.path.join(manifest["run_directory"], "analysis")
    model = str(config["model_name"])
    if stage == "vqe":
        config_path = _artifact_path(manifest, "vqe_config")
        vqe_config = _load_json(config_path)
        result_path = _absolute(root, str(vqe_config["result_file"]))
        default_bundle_path = os.path.join(
            os.path.dirname(result_path),
            os.path.basename(result_path).replace(
                "Runtime_Result", "ParameterBundle", 1
            ),
        )
        bundle_path = _absolute(
            root,
            str(
                vqe_config.get(
                    "parameter_bundle_file",
                    default_bundle_path,
                )
            ),
        )
        return _python(config, root, "vqe_runtime.py") + [
            "--adapter", _artifact_path(manifest, "state_adapter"),
            "--config", config_path,
        ], {"vqe_result": result_path, "vqe_parameter_bundle": bundle_path}
    if stage == "measurement":
        output = os.path.join(runtime_directory, "measurement_result.json")
        circuits = os.path.join(manifest["run_directory"], "circuits")
        return _python(config, root, "measurement_runtime.py") + [
            "--state-adapter", _artifact_path(manifest, "state_adapter"),
            "--observable-adapter", _artifact_path(manifest, "observable_adapter"),
            "--vqe-result", _artifact_path(manifest, "vqe_result"),
            "--observable-spec", _artifact_path(manifest, "agent1_observable"),
            "--output", output, "--circuit-directory", circuits,
            "--database", config["database"],
        ], {"measurement_result": output}
    if stage in ("agent6", "agent7"):
        kind = "scipy" if stage == "agent6" else "itensor"
        suffix = "py" if kind == "scipy" else "jl"
        script = os.path.join(reference_directory, "reference_{}.{}".format(kind, suffix))
        return _python(config, root, "agent678_test_crosscheck.py") + [
            kind,
            "--discretization-contract", _artifact_path(
                manifest, "agent2_discretization_contract"
            ),
            "--parameter-contract", _artifact_path(manifest, "agent2_parameter_contract"),
            "--observable-spec", _artifact_path(manifest, "agent1_observable"),
            "--run-spec", _artifact_path(manifest, "run_spec"),
            "--output-script", script, "--model", model,
        ] + (["--julia-executable", config["julia_executable"]] if stage == "agent7" else []), {kind + "_script": script}
    if stage == "scipy_reference":
        output = os.path.join(reference_directory, "scipy_reference.json")
        script = _artifact_path(manifest, "scipy_script")
        command = [config["python_executable"], script]
        if _reference_accepts_runtime_inputs(script):
            command += [
                "--parameter-contract", _artifact_path(
                    manifest, "agent2_parameter_contract"
                ),
                "--observable-spec", _artifact_path(manifest, "agent1_observable"),
            ]
        command += ["--output", output]
        return command, {"scipy_reference": output}
    if stage == "itensor_reference":
        output = os.path.join(reference_directory, "itensor_reference.json")
        script = _artifact_path(manifest, "itensor_script")
        command = [config["julia_executable"], script]
        if _reference_accepts_runtime_inputs(script):
            command += [
                "--parameter-contract", _artifact_path(
                    manifest, "agent2_parameter_contract"
                ),
                "--observable-spec", _artifact_path(manifest, "agent1_observable"),
            ]
        command += ["--output", output]
        return command, {"itensor_reference": output}
    if stage == "crosscheck":
        output = os.path.join(analysis_directory, "crosscheck_result.json")
        return _python(config, root, "crosscheck_runtime.py") + [
            "--measurement", _artifact_path(manifest, "measurement_result"),
            "--scipy", _artifact_path(manifest, "scipy_reference"),
            "--itensor", _artifact_path(manifest, "itensor_reference"),
            "--output", output, "--database", config["database"],
        ], {"crosscheck_result": output}
    if stage == "agent8":
        output = os.path.join(analysis_directory, "agent8_review.md")
        return _python(config, root, "agent678_test_crosscheck.py") + [
            "review", "--crosscheck-result", _artifact_path(manifest, "crosscheck_result"),
            "--output-report", output, "--model", model,
        ], {"agent8_review": output}
    if stage == "final_report":
        report_directory = os.path.join(manifest["run_directory"], "reports")
        audit_directory = os.path.join(manifest["run_directory"], "audit")
        stem = "Final_Academic_Report_{}".format(manifest["run_id"])
        markdown = os.path.join(report_directory, stem + ".md")
        latex = os.path.join(report_directory, stem + ".tex")
        status = os.path.join(audit_directory, stem + "_status.json")
        evidence = os.path.join(audit_directory, stem + "_evidence.json")
        command = _python(config, root, "agent10_final_report.py") + [
            "--pipeline-manifest",
            os.path.join(manifest["run_directory"], "pipeline_manifest.json"),
            "--markdown-output", markdown,
            "--latex-output", latex,
            "--status-output", status,
            "--evidence-output", evidence,
            "--model", model,
        ]
        hardware_result = _artifact_path(manifest, "hardware_result") \
            if "hardware_result" in manifest.get("artifacts", {}) else None
        if hardware_result:
            command += ["--hardware-result", hardware_result]
        return command, {
            "final_report_markdown": markdown,
            "final_report_latex": latex,
            "final_report_status": status,
            "final_report_evidence": evidence,
        }
    raise PipelineError("unknown stage: {}".format(stage))


def _save(manifest_path: str, manifest: Mapping[str, Any]) -> None:
    atomic_json_write(manifest_path, manifest)


def _next_stage_log_path(manifest: Mapping[str, Any], stage: str) -> str:
    directory = os.path.join(manifest["run_directory"], "audit", "stage_logs")
    os.makedirs(directory, exist_ok=True)
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_attempt{}".format(attempt)
        path = os.path.join(directory, "{}{}.log".format(stage, suffix))
        if not os.path.exists(path):
            return path
        attempt += 1


def _run_logged_command(command: Sequence[str], cwd: str, log_path: str) -> None:
    """Stream a stage to the terminal while retaining complete append-only evidence."""
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _prune_superseded_qpy(
    project_root: str, current_run_directory: str,
) -> Dict[str, int]:
    """Delete only QPY files superseded by a successful measurement run.

    Scientific JSON, reports, logs, parameters, and database rows are retained.
    Old run directories must contain a pipeline manifest before they are
    eligible, which prevents a broadly configured run root from becoming a
    general recursive deletion target.
    """
    project_root = os.path.abspath(project_root)
    current_run_directory = os.path.abspath(current_run_directory)
    run_root = os.path.dirname(current_run_directory)
    eligible_directories = []
    if os.path.isdir(run_root):
        for name in os.listdir(run_root):
            candidate = os.path.abspath(os.path.join(run_root, name))
            if (
                candidate != current_run_directory
                and os.path.isdir(candidate)
                and os.path.isfile(os.path.join(candidate, "pipeline_manifest.json"))
            ):
                eligible_directories.append(candidate)
    legacy_outputs = os.path.join(project_root, "outputs")
    if os.path.isdir(legacy_outputs):
        eligible_directories.append(legacy_outputs)

    removed_files = 0
    removed_bytes = 0
    failed_files = 0
    for root in eligible_directories:
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".qpy"):
                    continue
                path = os.path.join(directory, filename)
                try:
                    removed_bytes += os.path.getsize(path)
                    os.unlink(path)
                    removed_files += 1
                except OSError:
                    failed_files += 1
    return {
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "failed_files": failed_files,
    }


def _reuse_program_stage(
    stage: str, manifest: Dict[str, Any], manifest_path: str,
) -> bool:
    reusable = _find_reusable_run(manifest, stage)
    if reusable is None:
        return False
    prior, prior_manifest_path = reusable
    run_directory = manifest["run_directory"]
    output_tag = _stage_output_tag(manifest, stage)
    if stage == "agent4_observable":
        design = os.path.join(
            run_directory, "contracts",
            "Agent4_ObservableContract_{}.json".format(output_tag),
        )
        report = os.path.join(
            run_directory, "reports",
            "Agent4_Observable_Reused_{}.md".format(output_tag),
        )
        shutil.copyfile(
            _artifact_path(prior, "agent4_observable_design_contract"), design
        )
        with open(report, "w", encoding="utf-8") as handle:
            handle.write(
                "# Reused Agent4.2 measurement design\n\n"
                "The model/discretization, qubit layout, state sectors, and "
                "observable structure match a previous run. The measurement "
                "algorithm was reused; current numerical parameters are bound "
                "by the current Agent5 adapter and runtimes.\n"
            )
        outputs = {
            "agent4_observable_report": report,
            "agent4_observable_design_contract": design,
        }
    elif stage in ("agent6", "agent7"):
        kind = "scipy" if stage == "agent6" else "itensor"
        suffix = "py" if kind == "scipy" else "jl"
        script = os.path.join(
            run_directory, "references", "reference_{}.{}".format(kind, suffix)
        )
        shutil.copyfile(_artifact_path(prior, kind + "_script"), script)
        outputs = {kind + "_script": script}
    else:
        return False
    for name, path in outputs.items():
        _record_artifact(manifest, name, path)
    manifest["stages"][stage] = {
        "status": "completed",
        "artifacts": list(outputs),
        "reused_from": os.path.abspath(prior_manifest_path),
    }
    _save(manifest_path, manifest)
    print("[reuse] {} from {}".format(stage, prior.get("run_id", prior_manifest_path)))
    return True


def _run_stage(
    stage: str, manifest: Dict[str, Any], manifest_path: str,
    regenerate: bool = False,
) -> None:
    record = manifest["stages"][stage]
    if record["status"] == "completed":
        _require(
            not regenerate,
            "Cannot regenerate a completed stage in place; start a new run to "
            "avoid stale downstream artifacts.",
        )
        for artifact in record.get("artifacts", []):
            _artifact_path(manifest, artifact)
        print("[skip] {} already completed".format(stage))
        return
    if not regenerate and stage in ("agent4_observable", "agent6", "agent7"):
        if _reuse_program_stage(stage, manifest, manifest_path):
            return
    if stage in {
        "agent1", "agent2", "agent3", "agent4_state", "agent4_observable",
        "agent5_state", "agent5_observable",
    }:
        command, output_manifest, mapping = _agent_command(
            stage, manifest, regenerate=regenerate
        )
        known_outputs = None
    else:
        command, known_outputs = _known_output_command(stage, manifest)
        output_manifest = None
        mapping = None
    stage_log = _next_stage_log_path(manifest, stage)
    record.update({
        "status": "running",
        "stage_log": stage_log,
    })
    manifest["status"] = "running"
    _save(manifest_path, manifest)
    print("[run] {}".format(stage))
    try:
        _run_logged_command(command, manifest["project_root"], stage_log)
        produced = []
        if output_manifest is not None:
            handoff = load_stage_output_manifest(output_manifest, stage)
            for output_name, artifact_name in mapping.items():
                output_path = handoff["outputs"][output_name]["path"]
                _record_artifact(
                    manifest,
                    artifact_name,
                    output_path,
                )
                produced.append(artifact_name)
            if stage == "agent2":
                runtime_settings = _generate_runtime_settings(manifest)
                _record_artifact(
                    manifest,
                    "runtime_settings",
                    runtime_settings,
                )
                produced.append("runtime_settings")
        else:
            for artifact_name, output_path in known_outputs.items():
                _record_artifact(
                    manifest,
                    artifact_name,
                    output_path,
                )
                produced.append(artifact_name)
        record["artifacts"] = produced
        record["status"] = "completed"
        record.pop("error", None)
        _save(manifest_path, manifest)
        if stage == "measurement":
            cleanup = _prune_superseded_qpy(
                manifest["project_root"], manifest["run_directory"]
            )
            print(
                "[qpy retention] removed {removed_files} superseded files "
                "({removed_bytes} bytes); failed={failed_files}".format(**cleanup)
            )
    except Exception as error:
        if stage == "agent1":
            clarification_path = os.path.join(
                manifest["run_directory"],
                "audit",
                "Agent1_Clarification_{}.json".format(manifest["run_id"]),
            )
            if os.path.isfile(clarification_path):
                record["status"] = "needs_user_clarification"
                record["artifacts"] = ["agent1_clarification"]
                record.pop("error", None)
                _record_artifact(
                    manifest, "agent1_clarification", clarification_path
                )
                manifest["status"] = "needs_user_clarification"
                _save(manifest_path, manifest)
                raise UserClarificationRequired(
                    "Agent1 needs additional physical input: {}".format(
                        clarification_path
                    )
                ) from None
        record["status"] = "failed"
        record["error"] = "{}: {}".format(type(error).__name__, error)
        manifest["status"] = "failed"
        _save(manifest_path, manifest)
        raise


def _run_auxiliary_stage(
    stage: str,
    command: Sequence[str],
    outputs: Mapping[str, str],
    manifest: Dict[str, Any],
    manifest_path: str,
) -> None:
    """Run one optional deterministic/hardware stage with normal checkpoints."""
    record = manifest["stages"].setdefault(stage, {"status": "pending"})
    if record.get("status") == "completed":
        for artifact in record.get("artifacts", []):
            _artifact_path(manifest, artifact)
        print("[skip] {} already completed".format(stage))
        return
    stage_log = _next_stage_log_path(manifest, stage)
    record.update({"status": "running", "stage_log": stage_log})
    record.pop("reason", None)
    manifest["status"] = "running"
    _save(manifest_path, manifest)
    print("[run] {}".format(stage))
    try:
        _run_logged_command(command, manifest["project_root"], stage_log)
        produced = []
        for artifact_name, output_path in outputs.items():
            _record_artifact(manifest, artifact_name, output_path)
            produced.append(artifact_name)
        record.update({"status": "completed", "artifacts": produced})
        record.pop("error", None)
        _save(manifest_path, manifest)
    except Exception as error:
        record.update({
            "status": "failed",
            "error": "{}: {}".format(type(error).__name__, error),
        })
        manifest["status"] = "failed"
        _save(manifest_path, manifest)
        raise


def _skip_auxiliary_stage(
    stage: str, reason: str, manifest: Dict[str, Any], manifest_path: str,
) -> None:
    manifest["stages"].setdefault(stage, {}).update({
        "status": "skipped",
        "reason": reason,
        "artifacts": [],
    })
    _save(manifest_path, manifest)


def _run_hardware_validation(
    manifest: Dict[str, Any], manifest_path: str,
) -> None:
    """Collect hardware evidence, let Agent 9 decide, then execute that decision."""
    config = _execution_config(manifest)
    if not config.get("hardware_validation", False):
        return
    root = manifest["project_root"]
    analysis = os.path.join(manifest["run_directory"], "analysis")
    circuits = os.path.join(manifest["run_directory"], "circuits", "hardware_isa")
    plan_path = os.path.join(analysis, "hardware_plan.json")
    plan_command = _python(config, root, "hardware_runtime.py") + [
        "plan",
        "--crosscheck", _artifact_path(manifest, "crosscheck_result"),
        "--output", plan_path,
        "--isa-directory", circuits,
        "--database", config["database"],
        "--demo-mode", str(config.get("hardware_mode", "representative")),
    ]
    _run_auxiliary_stage(
        "hardware_plan", plan_command, {"hardware_plan": plan_path},
        manifest, manifest_path,
    )

    review_path = os.path.join(analysis, "agent9_hardware_review.md")
    decision_path = os.path.join(analysis, "agent9_hardware_decision.json")
    review_command = _python(config, root, "Agent9_hardware.py") + [
        "--hardware-plan", _artifact_path(manifest, "hardware_plan"),
        "--output-decision", decision_path,
        "--output-report", review_path,
        "--model", str(config["model_name"]),
    ]
    _run_auxiliary_stage(
        "agent9", review_command, {
            "agent9_decision": decision_path,
            "agent9_review": review_path,
        },
        manifest, manifest_path,
    )

    decision = _load_json(_artifact_path(manifest, "agent9_decision"))
    if decision.get("decision") == "do_not_submit":
        reason = "Agent9 rejected hardware submission: {}".format(
            decision.get("reason", "the expected result is not scientifically meaningful")
        )
        _skip_auxiliary_stage("hardware_submit", reason, manifest, manifest_path)
        _skip_auxiliary_stage("hardware_retrieve", reason, manifest, manifest_path)
        print("[hardware] {}".format(reason))
        return
    _require(
        decision.get("decision") == "submit"
        and bool(decision.get("selected_deployments")),
        "Agent9 must explicitly approve and select a deployment before submission.",
    )

    run_path = os.path.join(analysis, "hardware_run.json")
    submit_command = _python(config, root, "hardware_runtime.py") + [
        "submit", "--plan", _artifact_path(manifest, "hardware_plan"),
        "--agent9-decision", _artifact_path(manifest, "agent9_decision"),
        "--output", run_path,
        "--shots", "4096",
        "--resilience-level", "2",
        "--database", config["database"],
    ]
    _run_auxiliary_stage(
        "hardware_submit", submit_command, {"hardware_run": run_path},
        manifest, manifest_path,
    )

    result_path = os.path.join(analysis, "hardware_result.json")
    retrieve_command = _python(config, root, "hardware_runtime.py") + [
        "retrieve", "--run-record", _artifact_path(manifest, "hardware_run"),
        "--output", result_path,
        "--database", config["database"],
    ]
    _run_auxiliary_stage(
        "hardware_retrieve", retrieve_command, {"hardware_result": result_path},
        manifest, manifest_path,
    )


_ANSATZ_STAGE_ARTIFACTS = {
    "agent4_state": (
        "agent4_state_report",
        "agent4_state_design_contract",
    ),
    "agent5_state": ("state_adapter", "vqe_config"),
    "vqe": ("vqe_result", "vqe_parameter_bundle"),
}


def _record_ansatz_attempt(
    manifest: Dict[str, Any], manifest_path: str,
) -> str:
    result_path = _artifact_path(manifest, "vqe_result")
    result = _load_json(result_path)
    status = str(result.get("status", "unknown"))
    search = manifest.setdefault("ansatz_search", {
        "current_attempt": 1,
        "maximum_attempts": 1,
        "attempts": [],
    })
    attempts = search.setdefault("attempts", [])
    artifact_names = tuple(
        name for names in _ANSATZ_STAGE_ARTIFACTS.values() for name in names
    )
    record = {
        "attempt": int(search.get("current_attempt", 1)),
        "status": status,
        "vqe_result": result_path,
        "artifacts": {
            name: manifest["artifacts"][name]["path"]
            for name in artifact_names if name in manifest["artifacts"]
        },
        "stage_logs": {
            stage: manifest["stages"][stage]["stage_log"]
            for stage in _ANSATZ_STAGE_ARTIFACTS
            if manifest["stages"].get(stage, {}).get("stage_log")
        },
    }
    existing = next(
        (
            item for item in attempts
            if isinstance(item, dict) and item.get("vqe_result") == result_path
        ),
        None,
    )
    if existing is None:
        attempts.append(record)
    else:
        existing.update(record)
    _save(manifest_path, manifest)
    return status


def _extend_promising_vqe_budget(
    manifest: Dict[str, Any], manifest_path: str,
) -> None:
    """Append one warm-start depth without regenerating an Agent or adapter."""
    result = _load_json(_artifact_path(manifest, "vqe_result"))
    schedule = result.get("config", {}).get("depth_schedule")
    _require(
        isinstance(schedule, list)
        and bool(schedule)
        and all(isinstance(depth, int) and depth > 0 for depth in schedule),
        "Promising VQE result is missing its resolved depth schedule.",
    )
    config_path = _artifact_path(manifest, "vqe_config")
    config = _load_json(config_path)
    config["depth_schedule"] = list(schedule)
    atomic_json_write(config_path, config)

    old_record = manifest["stages"].get("vqe", {})
    for artifact_name in old_record.get("artifacts", []):
        manifest["artifacts"].pop(artifact_name, None)
    manifest["stages"]["vqe"] = {"status": "pending"}
    manifest["status"] = "running"
    _save(manifest_path, manifest)


def _start_next_ansatz_attempt(
    manifest: Dict[str, Any], manifest_path: str,
) -> None:
    search = manifest["ansatz_search"]
    attempts = search.get("attempts", [])
    current = int(search["current_attempt"])
    _require(
        isinstance(attempts, list)
        and bool(attempts)
        and isinstance(attempts[-1], dict)
        and attempts[-1].get("status") == "ansatz_insufficient"
        and int(attempts[-1].get("attempt", -1)) == current,
        "A new ansatz attempt may start only after the current VQE result is "
        "explicitly recorded as ansatz_insufficient.",
    )
    search["current_attempt"] = int(search["current_attempt"]) + 1
    for stage, artifact_names in _ANSATZ_STAGE_ARTIFACTS.items():
        manifest["stages"][stage] = {"status": "pending"}
        for name in artifact_names:
            manifest["artifacts"].pop(name, None)
    manifest["status"] = "running"
    _save(manifest_path, manifest)


def _complete_ansatz_search(
    manifest: Dict[str, Any], manifest_path: str,
) -> None:
    """Run bounded state-ansatz redesigns after scientific insufficiency only."""
    budget_extensions = 0
    while True:
        result = _load_json(_artifact_path(manifest, "vqe_result"))
        status = str(result.get("status", "unknown"))
        while (
            status == "promising_state"
            and budget_extensions < MAX_AUTOMATIC_VQE_BUDGET_EXTENSIONS
        ):
            resolved_schedule = result.get("config", {}).get("depth_schedule", [])
            previous_depth = (
                resolved_schedule[-1] if isinstance(resolved_schedule, list)
                and resolved_schedule else "unknown"
            )
            print(
                "[VQE budget extension] promising state at depth {}; "
                "continuing the same ansatz with one deeper warm start ({}/{})"
                .format(
                    previous_depth,
                    budget_extensions + 1,
                    MAX_AUTOMATIC_VQE_BUDGET_EXTENSIONS,
                )
            )
            _extend_promising_vqe_budget(manifest, manifest_path)
            _run_stage("vqe", manifest, manifest_path)
            budget_extensions += 1
            result = _load_json(_artifact_path(manifest, "vqe_result"))
            status = str(result.get("status", "unknown"))

        status = _record_ansatz_attempt(manifest, manifest_path)
        if status == "promising_state":
            manifest["status"] = "vqe_budget_exhausted"
            _save(manifest_path, manifest)
            raise PipelineError(
                "VQE is still improving but has not met the accuracy thresholds. "
                "Two automatic warm-start depth extensions were exhausted, so "
                "measurement and cross-check were not started."
            )
        if status != "ansatz_insufficient":
            return
        search = manifest["ansatz_search"]
        current = int(search["current_attempt"])
        maximum = int(search["maximum_attempts"])
        if current >= maximum:
            manifest["status"] = "ansatz_search_exhausted"
            _save(manifest_path, manifest)
            raise PipelineError(
                "VQE rejected all {} ansatz attempts; see ansatz_search.attempts "
                "in the pipeline manifest.".format(maximum)
            )
        print(
            "[ansatz redesign] attempt {} was insufficient; starting attempt {} of {}"
            .format(current, current + 1, maximum)
        )
        _start_next_ansatz_attempt(manifest, manifest_path)
        for stage in ("agent4_state", "agent5_state", "vqe"):
            _run_stage(stage, manifest, manifest_path)
        budget_extensions = 0


def execute_pipeline(
    manifest: Dict[str, Any], manifest_path: str, through: str,
    regenerate_stages: Sequence[str] = (),
) -> Dict[str, Any]:
    _require(through in STAGES, "unknown through stage")
    _verify_inputs(manifest)
    # Older manifests predate optional tail stages such as final_report.
    for stage in STAGES:
        manifest["stages"].setdefault(stage, {"status": "pending"})
    _materialize_approved_program_reuse(manifest, manifest_path)
    target_index = STAGES.index(through)
    for stage in STAGES[:target_index + 1]:
        _run_stage(
            stage, manifest, manifest_path,
            regenerate=stage in regenerate_stages,
        )
        if stage == "vqe":
            _complete_ansatz_search(manifest, manifest_path)
        if stage == "crosscheck":
            _run_hardware_validation(manifest, manifest_path)
    manifest["status"] = "completed_through_{}".format(through)
    manifest["completed_through"] = through
    _save(manifest_path, manifest)
    return manifest


def _load_pipeline_manifest(path: str) -> Dict[str, Any]:
    manifest = _load_json(path)
    _require(manifest.get("pipeline_schema_version") in ("1.0", PIPELINE_SCHEMA_VERSION),
             "pipeline manifest schema must be 1.0 or 2.0")
    return manifest


def set_program_reuse_approval(
    manifest: Dict[str, Any], manifest_path: str, approved: bool,
) -> None:
    """Record or revoke the human decision; no extra registry is maintained."""
    if approved:
        final_stage = manifest.get("stages", {}).get("final_report", {})
        _require(
            isinstance(final_stage, Mapping)
            and final_stage.get("status") == "completed",
            "Only a run with a completed final report can be approved for reuse.",
        )
        _artifact_path(manifest, "final_report_markdown")
        _require(
            _reusable_program_files_ready(manifest),
            "This run does not contain parameterized observable, SciPy and "
            "ITensor programs required for reuse.",
        )
    manifest["program_reuse_approval"] = {
        "status": "approved" if approved else "not_approved",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _save(manifest_path, manifest)


def plan_hardware(
    manifest: Dict[str, Any], manifest_path: str, backends: Sequence[str]
) -> None:
    crosscheck = _artifact_path(manifest, "crosscheck_result")
    output = os.path.join(manifest["run_directory"], "analysis", "hardware_plan.json")
    isa = os.path.join(manifest["run_directory"], "circuits", "hardware_isa")
    config = _execution_config(manifest)
    command = _python(config, manifest["project_root"], "hardware_runtime.py") + [
        "plan", "--crosscheck", crosscheck, "--output", output,
        "--isa-directory", isa, "--database", config["database"],
    ]
    for backend in backends:
        command += ["--backend", backend]
    record = manifest["stages"].setdefault("hardware_plan", {})
    record.update({"status": "running"})
    _save(manifest_path, manifest)
    try:
        subprocess.run(command, cwd=manifest["project_root"], check=True)
        _record_artifact(manifest, "hardware_plan", output)
        record.update({"status": "completed", "artifacts": ["hardware_plan"]})
        _save(manifest_path, manifest)
    except Exception as error:
        record.update({"status": "failed", "error": "{}: {}".format(type(error).__name__, error)})
        _save(manifest_path, manifest)
        raise


def refresh_final_report(
    manifest: Dict[str, Any], manifest_path: str,
) -> None:
    """Explicitly spend one report call, for example after hardware retrieval."""
    manifest["stages"]["final_report"] = {"status": "pending"}
    for name in (
        "final_report_markdown", "final_report_latex",
        "final_report_status", "final_report_evidence",
    ):
        manifest["artifacts"].pop(name, None)
    _save(manifest_path, manifest)
    _run_stage("final_report", manifest, manifest_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin Jianmu-Q pipeline controller.")
    children = parser.add_subparsers(dest="mode", required=True)
    run = children.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id")
    run.add_argument("--through", choices=STAGES)
    resume = children.add_parser("resume")
    resume.add_argument("--manifest", required=True)
    resume.add_argument("--through", choices=STAGES)
    resume.add_argument(
        "--regenerate-stage",
        action="append",
        default=[],
        choices=("agent4_state", "agent4_observable"),
        help="Explicitly make one fresh paid Agent4 call instead of reusing raw output.",
    )
    status = children.add_parser("status")
    status.add_argument("--manifest", required=True)
    hardware = children.add_parser("plan-hardware")
    hardware.add_argument("--manifest", required=True)
    hardware.add_argument("--backend", action="append", default=[])
    report = children.add_parser("report")
    report.add_argument("--manifest", required=True)
    approval = children.add_parser("approve-reuse")
    approval.add_argument("--manifest", required=True)
    approval.add_argument(
        "--revoke", action="store_true",
        help="Remove this run from the human-approved reuse pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.mode == "run":
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = _absolute(project_root, args.config)
        config = validate_config(_load_json(config_path), project_root)
        manifest, manifest_path = _new_manifest(
            config_path, config, project_root, args.run_id
        )
        through = args.through or config["through"]
        try:
            execute_pipeline(manifest, manifest_path, through)
        except UserClarificationRequired as error:
            print(error)
            return
        print("Pipeline manifest: {}".format(manifest_path))
    else:
        manifest_path = os.path.abspath(args.manifest)
        manifest = _load_pipeline_manifest(manifest_path)
        if args.mode == "resume":
            through = args.through or _execution_config(manifest)["through"]
            try:
                execute_pipeline(
                    manifest, manifest_path, through,
                    regenerate_stages=args.regenerate_stage,
                )
            except UserClarificationRequired as error:
                print(error)
                return
        elif args.mode == "plan-hardware":
            plan_hardware(manifest, manifest_path, args.backend)
        elif args.mode == "report":
            refresh_final_report(manifest, manifest_path)
        elif args.mode == "approve-reuse":
            set_program_reuse_approval(
                manifest, manifest_path, approved=not args.revoke
            )
            print(
                "Program reuse approval for {}: {}".format(
                    manifest["run_id"],
                    manifest["program_reuse_approval"]["status"],
                )
            )
        else:
            print("run_id={} status={}".format(manifest["run_id"], manifest["status"]))
            ordered = []
            for stage in STAGES:
                ordered.append(stage)
                if stage == "crosscheck":
                    ordered.extend(
                        item for item in OPTIONAL_HARDWARE_STAGES
                        if item in manifest["stages"]
                    )
            ordered.extend(
                stage for stage in manifest["stages"]
                if stage not in ordered
            )
            for stage in ordered:
                record = manifest["stages"].get(stage, {"status": "pending"})
                print("{:<20} {}".format(stage, record["status"]))


if __name__ == "__main__":
    main()
