# -*- coding: utf-8 -*-
"""Small controller-owned snapshot of one run's user inputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from observable_specification import validate_observable_specification


class RunSpecError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunSpecError(message)


def build_run_spec(run_id: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    agent1 = config["agent1"]
    external_states = agent1.get("external_states")
    if external_states is None:
        external_states = {
            "phi": agent1["final_state"], "psi": agent1["initial_state"]
        }
    combined_parameters = dict(config["user_parameters"])
    if "physical_parameters" in config or "simulation_parameters" in config:
        physical_parameters = dict(config.get("physical_parameters", {}))
        simulation_parameters = dict(config.get("simulation_parameters", {}))
    else:
        # Backward-compatible interpretation for small internal/manual callers.
        physical_parameters = dict(combined_parameters)
        simulation_parameters = {}
    document = {
        "schema_version": "1.0",
        "run_id": run_id,
        "user_input": {
            "model": agent1["model_input"],
            "parameters": combined_parameters,
            "parameter_groups": {
                "physical": physical_parameters,
                "simulation": simulation_parameters,
            },
            "observable": agent1["observable"],
            "continuum_conventions": dict(
                agent1.get("continuum_conventions", {})
            ),
            "clarifications": [
                dict(item) for item in agent1.get("user_clarifications", [])
            ],
            "external_states": dict(external_states),
        },
        "execution_request": {
            "llm": config["model_name"],
            "through": config["through"],
            "hardware_validation": bool(config.get("hardware_validation", False)),
            "hardware_mode": str(config.get("hardware_mode", "representative")),
            "vqe_runtime": config["vqe_runtime"],
            "max_ansatz_retries": config.get("max_ansatz_retries", 2),
        },
    }
    return validate_run_spec(document)


def validate_run_spec(document: Any) -> Dict[str, Any]:
    _require(isinstance(document, dict), "RunSpec must be an object.")
    _require(
        document.get("schema_version", document.get("run_spec_schema_version")) == "1.0",
        "RunSpec schema must be 1.0.",
    )
    _require(isinstance(document.get("run_id"), str) and document["run_id"],
             "RunSpec run_id is required.")
    user = document.get("user_input")
    _require(isinstance(user, dict), "RunSpec user_input must be an object.")
    _require(isinstance(user.get("model"), str) and user["model"].strip(),
             "RunSpec model is required.")
    _require(isinstance(user.get("parameters"), dict) and user["parameters"],
             "RunSpec parameters must be nonempty.")
    groups = user.get("parameter_groups")
    if groups is not None:
        _require(
            isinstance(groups, dict)
            and isinstance(groups.get("physical"), dict)
            and isinstance(groups.get("simulation"), dict),
            "RunSpec parameter_groups must contain physical and simulation objects.",
        )
        _require(
            not (set(groups["physical"]) & set(groups["simulation"])),
            "RunSpec physical and simulation parameter groups must not overlap.",
        )
        _require(
            {**groups["physical"], **groups["simulation"]} == user["parameters"],
            "RunSpec parameter groups must reproduce the combined parameters.",
        )
    _require(isinstance(user.get("observable"), dict),
             "RunSpec observable must be an object.")
    _require(isinstance(user.get("continuum_conventions", {}), dict),
             "RunSpec continuum_conventions must be an object.")
    clarifications = user.get("clarifications", [])
    _require(isinstance(clarifications, list),
             "RunSpec clarifications must be a list.")
    for item in clarifications:
        _require(
            isinstance(item, dict)
            and isinstance(item.get("question"), str) and item["question"].strip()
            and isinstance(item.get("answer"), str) and item["answer"].strip(),
            "Every RunSpec clarification needs a question and answer.",
        )
    states = user.get("external_states")
    _require(
        isinstance(states, dict)
        and states
        and all(isinstance(label, str) and label.strip() for label in states)
        and all(isinstance(value, str) and value.strip() for value in states.values()),
        "RunSpec must define one or more named external states.",
    )
    _require(isinstance(document.get("execution_request"), dict),
             "RunSpec execution_request must be an object.")
    hardware_validation = document["execution_request"].get(
        "hardware_validation", False
    )
    _require(
        isinstance(hardware_validation, bool),
        "RunSpec hardware_validation must be true or false.",
    )
    document["execution_request"]["hardware_validation"] = hardware_validation
    hardware_mode = str(document["execution_request"].get(
        "hardware_mode", "representative"
    ))
    _require(
        hardware_mode in ("representative", "full_d1_nt1"),
        "RunSpec hardware_mode must be representative or full_d1_nt1.",
    )
    document["execution_request"]["hardware_mode"] = hardware_mode
    retries = document["execution_request"].get("max_ansatz_retries", 2)
    _require(
        isinstance(retries, int) and not isinstance(retries, bool)
        and 0 <= retries <= 2,
        "RunSpec max_ansatz_retries must be an integer from 0 to 2.",
    )
    normalized = json.loads(json.dumps(document, ensure_ascii=False))
    normalized.pop("run_spec_schema_version", None)
    normalized["schema_version"] = "1.0"
    normalized["user_input"]["observable"] = validate_observable_specification(
        normalized["user_input"]["observable"]
    )
    terms = normalized["user_input"]["observable"].get("matrix_elements")
    if terms is not None:
        referenced = {
            label for term in terms for label in (term["bra"], term["ket"])
        }
        _require(
            referenced.issubset(set(states)),
            "RunSpec matrix elements reference undefined states.",
        )
    else:
        _require(
            set(states) == {"phi", "psi"},
            "RunSpec named state sets require observable.matrix_elements.",
        )
    normalized["user_input"].setdefault("continuum_conventions", {})
    normalized["user_input"].setdefault("clarifications", [])
    return normalized


def load_run_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return validate_run_spec(json.load(handle))
