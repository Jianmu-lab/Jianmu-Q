# -*- coding: utf-8 -*-
"""Validated scientific machine handoffs, independent of human LaTeX audits."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, Mapping, Sequence

from parameter_contract import (
    ParameterContractError,
    requires_physical_sector,
    validate_parameter_contract,
)
from model_output_ingestion import (
    ModelOutputIngestionError,
    json_object_candidates,
    select_semantic_object,
)


MACHINE_CONTRACT_SCHEMA_VERSION = "1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ParameterContractError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise ParameterContractError("Machine contract must be JSON serializable.") from error


def _header(contract: Any, contract_type: str) -> Dict[str, Any]:
    _require(isinstance(contract, dict), "Machine contract must be a JSON object.")
    _require(
        contract.get("schema_version") == MACHINE_CONTRACT_SCHEMA_VERSION,
        "Machine contract schema_version must be '1.0'.",
    )
    _require(contract.get("contract_type") == contract_type, "Unexpected contract_type.")
    return contract


def _definitions(value: Any, *, required_kinds: Sequence[str] = ()) -> None:
    _require(isinstance(value, list) and value, "definitions must be a nonempty list.")
    identifiers = []
    kinds = set()
    for index, item in enumerate(value):
        _require(isinstance(item, dict), "definitions[{}] must be an object.".format(index))
        for key in ("id", "kind", "canonical_expression"):
            _require(
                _text(item.get(key)),
                "definitions[{}].{} must be nonempty.".format(index, key),
            )
        dependencies = item.get("dependencies")
        if dependencies is not None:
            _require(
                isinstance(dependencies, list)
                and all(_text(entry) for entry in dependencies),
                "definition dependencies must be strings when provided.",
            )
        identifiers.append(item["id"])
        kinds.add(item["kind"])
    _require(len(identifiers) == len(set(identifiers)), "definition IDs must be unique.")
    missing = sorted(set(required_kinds) - kinds)
    _require(not missing, "definitions are missing kinds: {}".format(", ".join(missing)))


def _external_states(value: Any) -> None:
    _require(isinstance(value, dict), "external_states must be an object.")
    labels = value.get("labels")
    definitions = value.get("definitions")
    groups = value.get("equivalent_groups")
    _require(
        isinstance(labels, list)
        and labels
        and len(labels) == len(set(labels))
        and all(_text(label) for label in labels),
        "external_states.labels must contain unique nonempty strings.",
    )
    _require(
        isinstance(definitions, dict)
        and set(definitions) == set(labels)
        and all(_text(item) for item in definitions.values()),
        "external_states.definitions must define every state label.",
    )
    _require(isinstance(groups, list), "external_states.equivalent_groups must be a list.")


def validate_theory_contract(
    contract: Any,
    *,
    expected_model: str | None = None,
    expected_observable: Mapping[str, Any] | None = None,
    expected_initial_state: str | None = None,
    expected_final_state: str | None = None,
    expected_external_states: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    contract = _header(contract, "continuum_theory")
    model = contract.get("model")
    _require(isinstance(model, dict) and _text(model.get("name")), "model.name is required.")
    _require(_text(model.get("gauge_structure")), "model.gauge_structure is required.")
    if expected_model is not None:
        _require(model["name"] == expected_model, "Theory contract changed the user model.")
    del expected_observable
    _external_states(contract.get("external_states"))
    states = contract["external_states"]["definitions"]
    if expected_external_states is not None:
        _require(
            states == dict(expected_external_states),
            "Theory contract changed the named external states.",
        )
    if expected_final_state is not None:
        _require(states.get("phi") == expected_final_state, "Theory contract changed phi.")
    if expected_initial_state is not None:
        _require(states.get("psi") == expected_initial_state, "Theory contract changed psi.")
    _definitions(
        contract.get("definitions"),
        required_kinds=("hamiltonian", "observable"),
    )
    conventions = contract.get("resolved_conventions", {})
    _require(isinstance(conventions, dict), "resolved_conventions must be an object.")
    clarifications = contract.get("clarifications_required", [])
    _require(isinstance(clarifications, list),
             "clarifications_required must be a list.")
    for item in clarifications:
        _require(
            isinstance(item, dict) and _text(item.get("question")),
            "Each clarification must contain a question.",
        )
        if "reason" in item:
            _require(_text(item.get("reason")),
                     "A supplied clarification reason must be text.")
    return _json_copy(contract)


def validate_discretization_contract(
    contract: Any,
    *,
    expected_theory_contract: Mapping[str, Any] | None = None,
    expected_parameter_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = _header(contract, "lattice_discretization")
    if expected_parameter_contract is not None:
        validate_parameter_contract(expected_parameter_contract)
    lattice = contract.get("lattice")
    _require(isinstance(lattice, dict), "lattice must be an object.")
    for key in ("geometry", "spatial_boundary_condition"):
        _require(_text(lattice.get(key)), "lattice.{} is required.".format(key))
    fields = contract.get("field_encodings")
    _require(isinstance(fields, list) and fields, "field_encodings must be nonempty.")
    for field in fields:
        _require(isinstance(field, dict), "Each field encoding must be an object.")
        for key in ("sector", "scheme", "degrees_of_freedom", "truncation"):
            _require(key in field, "Field encoding is missing {}.".format(key))
    _definitions(
        contract.get("definitions"),
        required_kinds=("lattice_hamiltonian", "lattice_observable"),
    )
    del expected_theory_contract
    return _json_copy(contract)


def _pauli_sum(value: Any, qubit_count: int, name: str) -> None:
    _require(isinstance(value, dict), "{} must be an object.".format(name))
    _require(value.get("representation") == "pauli_sum", "{} must use pauli_sum.".format(name))
    terms = value.get("terms")
    _require(isinstance(terms, list) and terms, "{}.terms must be nonempty.".format(name))
    for index, term in enumerate(terms):
        _require(isinstance(term, dict), "{} term must be an object.".format(name))
        coefficient = term.get("coefficient")
        _require(
            (isinstance(coefficient, (int, float)) and not isinstance(coefficient, bool))
            or _text(coefficient),
            "{} term coefficient must be numeric or symbolic text.".format(name),
        )
        pauli = term.get("pauli")
        _require(
            isinstance(pauli, str)
            and len(pauli) == qubit_count
            and set(pauli).issubset(set("IXYZ")),
            "{} term {} has an invalid Qiskit-order Pauli label.".format(name, index),
        )


def validate_mapping_contract(
    contract: Any,
    *,
    expected_discretization_contract: Mapping[str, Any] | None = None,
    expected_parameter_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = _header(contract, "qubit_mapping")
    del expected_discretization_contract
    if expected_parameter_contract is not None:
        validate_parameter_contract(expected_parameter_contract)
    qubits = contract.get("qubits")
    _require(isinstance(qubits, dict), "qubits must be an object.")
    count = qubits.get("count")
    _require(
        isinstance(count, int) and not isinstance(count, bool) and count > 0,
        "qubits.count must be a positive integer.",
    )
    ordering = qubits.get("ordering")
    _require(
        isinstance(ordering, list)
        and len(ordering) == count
        and len(ordering) == len(set(ordering))
        and all(_text(item) for item in ordering),
        "qubits.ordering must name every qubit exactly once.",
    )
    _require(
        qubits.get("pauli_label_order") == "qiskit_little_endian",
        "pauli_label_order must be qiskit_little_endian.",
    )
    _pauli_sum(contract.get("hamiltonian"), count, "hamiltonian")
    observable = contract.get("observable")
    _require(isinstance(observable, dict), "observable must be an object.")
    _require(_text(observable.get("representation")), "observable.representation is required.")
    _require(isinstance(observable.get("terms"), list) and observable["terms"],
             "observable.terms must be nonempty.")
    sector = contract.get("physical_sector")
    _require(isinstance(sector, dict), "physical_sector must be an object.")
    _require(isinstance(sector.get("required"), bool), "physical_sector.required must be boolean.")
    _require(_text(sector.get("basis_rule")), "physical_sector.basis_rule is required.")
    tests = sector.get("validation_tests")
    _require(isinstance(tests, list), "physical_sector.validation_tests must be a list.")
    for test in tests:
        _require(isinstance(test, dict) and _text(test.get("id")) and _text(test.get("rule")),
                 "Every physical-sector test needs id and rule.")
    if expected_parameter_contract is not None:
        sector_required = requires_physical_sector(expected_parameter_contract)
        _require(
            sector["required"] == sector_required,
            "physical_sector.required disagrees with the requested state sector.",
        )
    definitions = contract.get("circuit_definitions", [])
    _require(isinstance(definitions, list), "circuit_definitions must be a list.")
    return _json_copy(contract)


def extract_machine_contract(text: str, validator: Callable[..., Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """Find exactly one ingested JSON object accepted by the strict validator."""
    accepted = []
    errors = []
    for candidate in json_object_candidates(text):
        try:
            accepted.append(validator(candidate, **kwargs))
        except ParameterContractError as error:
            errors.append(str(error))
    _require(
        len(accepted) == 1,
        "Expected exactly one valid machine contract; found {}.{}".format(
            len(accepted), " Last error: " + errors[-1] if errors else ""
        ),
    )
    return accepted[0]


def extract_json_contract_candidate(text: str, contract_type: str) -> Dict[str, Any]:
    """Recover a model payload by identity or semantic structure.

    ``contract_type`` is controller-owned. It is useful when present, but a
    stochastic model is not required to reproduce it exactly.
    """
    fingerprints = {
        "continuum_theory": (
            "model", "definitions", "symmetries", "assumptions",
            "external_states",
        ),
        "lattice_discretization": (
            "lattice", "field_encodings", "definitions",
            "validation_obligations", "external_states",
        ),
        "qubit_mapping": (
            "qubits", "hamiltonian", "observable", "physical_sector",
            "circuit_definitions",
        ),
    }
    try:
        candidate = select_semantic_object(
            text,
            label=contract_type,
            fingerprint_keys=fingerprints.get(contract_type, ("contract_type",)),
            discriminator_key="contract_type",
            discriminator_value=contract_type,
        )
    except ModelOutputIngestionError as error:
        raise ParameterContractError(str(error)) from error
    return _json_copy(candidate)


def machine_contract_digest(
    contract: Mapping[str, Any], validator: Callable[..., Dict[str, Any]], **kwargs: Any
) -> str:
    validated = validator(contract, **kwargs)
    canonical = json.dumps(
        validated, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_machine_contract(
    contract: Mapping[str, Any], path: str,
    validator: Callable[..., Dict[str, Any]], **kwargs: Any
) -> None:
    validated = validator(contract, **kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(validated, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_machine_contract(
    path: str, validator: Callable[..., Dict[str, Any]], **kwargs: Any
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return validator(json.load(handle), **kwargs)


def format_machine_contract(
    contract: Mapping[str, Any], validator: Callable[..., Dict[str, Any]], **kwargs: Any
) -> str:
    return json.dumps(
        validator(contract, **kwargs), ensure_ascii=False, indent=2, sort_keys=True
    )
