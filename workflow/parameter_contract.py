# -*- coding: utf-8 -*-
"""Shared helpers for the model-specific simulation parameter contract."""

import json
import hashlib
import copy
import logging
from model_output_ingestion import (
    ModelOutputIngestionError,
    select_semantic_object,
)


PARAMETER_CONTRACT_SCHEMA_GUIDANCE = r'''
Use this stable envelope, but choose the parameter names and contents dynamically for the actual model:
{
  "schema_version": "1.1",
  "model": {"name": "...", "spacetime_dimension": "...", "gauge_structure": "..."},
  "external_states": {
    "labels": ["phi", "psi"],
    "definitions": {"phi": "final-bra state definition", "psi": "initial-ket state definition"},
    "equivalent_groups": []
  },
  "discretizations": [
    {
      "sector": "fermion|boson|gauge|spacetime|other",
      "scheme": "model-specific scheme name",
      "settings": {"model_specific_setting": "value"},
      "parameter_references": ["optional index of declared parameter names used by this scheme"],
      "truncation": {
        "scheme": "none or model-specific truncation name",
        "settings": {"model_specific_cutoff": "value"},
        "reason": "optional human explanation"
      }
    }
  ],
  "parameters": [
    {
      "name": "model_specific_name",
      "symbol": "...",
      "type": "integer|real|boolean|string|enum",
      "domain": {"model_specific_rule": "..."},
      "selected_value": null,
      "unit": null,
      "required_for_run": true,
      "role": "..."
    }
  ],
  "constraints": [
    {
      "id": "unique_id",
      "applies_when": "always or a precise model-specific condition",
      "requirement": "machine-implementable requirement",
      "parameters": ["affected_parameter_names"],
      "reason": "physical or mathematical reason",
      "severity": "fatal|warning"
    }
  ],
  "validation_obligations": [
    {
      "id": "unique_id",
      "description": "runtime check that downstream code must implement",
      "stage": "pre_simulation|post_construction",
      "severity": "fatal|warning"
    }
  ]
}

Minimum executable-core rule:
- model.name and model.gauge_structure are physical identity;
- every parameter needs a unique name and an explicit selected_value (null means unresolved);
- discretization sector/scheme/settings and every gauge truncation scheme/settings are authoritative;
- parameter_references, symbol, unit, domain, role, reasons, citations and field order are metadata.
  Omit uncertain metadata instead of inventing it. Unknown parameter_references are ignored without
  changing the corresponding discretization or truncation settings.
'''.strip()


class ParameterContractError(ValueError):
    """Raised when a parameter contract is missing or structurally invalid."""


def _inferred_parameter_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "unspecified"


def normalize_parameter_contract(contract):
    """Canonicalize optional metadata while preserving scientific content.

    The contract has a deliberately small fatal core: model identity, selected
    parameter values, external states, and the actual discretization/truncation
    choices.  Cross-reference lists, prose, units, domains, and validation-note
    bookkeeping are useful audit metadata, but they must not make an otherwise
    executable calculation fail.
    """
    if not isinstance(contract, dict):
        return contract
    normalized = copy.deepcopy(contract)
    normalized.setdefault("constraints", [])
    normalized.setdefault("validation_obligations", [])

    parameters = normalized.get("parameters")
    known_names = set()
    if isinstance(parameters, list):
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue
            name = parameter.get("name")
            if isinstance(name, str) and name.strip():
                known_names.add(name)
                parameter.setdefault("symbol", name)
            if "selected_value" in parameter and not isinstance(parameter.get("type"), str):
                parameter["type"] = _inferred_parameter_type(
                    parameter["selected_value"]
                )
            parameter.setdefault("domain", {})
            parameter.setdefault("unit", None)
            if not isinstance(parameter.get("required_for_run"), bool):
                parameter["required_for_run"] = True
            parameter.setdefault("role", "")

    states = normalized.get("external_states")
    if isinstance(states, dict):
        states.setdefault("equivalent_groups", [])

    discretizations = normalized.get("discretizations")
    if isinstance(discretizations, list):
        for index, component in enumerate(discretizations):
            if not isinstance(component, dict):
                continue
            component.setdefault("settings", {})
            references = component.get("parameter_references", [])
            if not isinstance(references, list):
                references = []
            cleaned = []
            removed = []
            for reference in references:
                if (
                    isinstance(reference, str)
                    and reference in known_names
                    and reference not in cleaned
                ):
                    cleaned.append(reference)
                elif isinstance(reference, str):
                    removed.append(reference)
            component["parameter_references"] = cleaned
            if removed:
                logging.getLogger("ContractNormalizer").warning(
                    "Ignored non-authoritative parameter reference(s) in "
                    "discretizations[%d]: %s. Discretization settings were preserved.",
                    index,
                    ", ".join(removed),
                )
            truncation = component.get("truncation")
            if isinstance(truncation, dict):
                truncation.setdefault("settings", {})
                truncation.setdefault("reason", "")

    # Constraint/obligation IDs and severities are audit bookkeeping.  Their
    # scientific statements remain untouched; only stale parameter indexes are
    # removed because they are not an executable dependency graph.
    constraints = normalized.get("constraints")
    if not isinstance(constraints, list):
        normalized["constraints"] = []
    else:
        retained = []
        used_ids = set()
        for index, item in enumerate(constraints):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("requirement"), str)
                or not item["requirement"].strip()
            ):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip() or item_id in used_ids:
                item_id = "constraint_{}".format(index + 1)
            while item_id in used_ids:
                item_id += "_"
            item["id"] = item_id
            used_ids.add(item_id)
            if item.get("severity") not in ("fatal", "warning"):
                item["severity"] = "warning"
            references = item.get("parameters", [])
            if not isinstance(references, list):
                references = []
            item["parameters"] = [
                name for name in references
                if isinstance(name, str) and name in known_names
            ]
            retained.append(item)
        normalized["constraints"] = retained
    obligations = normalized.get("validation_obligations")
    if not isinstance(obligations, list):
        normalized["validation_obligations"] = []
    else:
        retained = []
        used_ids = set()
        for index, item in enumerate(obligations):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("description"), str)
                or not item["description"].strip()
            ):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip() or item_id in used_ids:
                item_id = "validation_{}".format(index + 1)
            while item_id in used_ids:
                item_id += "_"
            item["id"] = item_id
            used_ids.add(item_id)
            if item.get("severity") not in ("fatal", "warning"):
                item["severity"] = "warning"
            if item.get("stage") not in ("pre_simulation", "post_construction"):
                item["stage"] = "post_construction"
            retained.append(item)
        normalized["validation_obligations"] = retained
    return normalized


def validate_parameter_contract(contract):
    """Validate the minimum scientific/executable core after normalization."""
    contract = normalize_parameter_contract(contract)
    if not isinstance(contract, dict):
        raise ParameterContractError("Parameter contract must be a JSON object.")

    required = ("schema_version", "model", "parameters")
    missing = [key for key in required if key not in contract]
    if missing:
        raise ParameterContractError(f"Parameter contract is missing fields: {', '.join(missing)}")
    if not isinstance(contract["schema_version"], str) or not contract["schema_version"].strip():
        raise ParameterContractError("schema_version must be a non-empty string.")
    if not isinstance(contract["model"], dict) or not contract["model"].get("name"):
        raise ParameterContractError("model must be an object containing a non-empty name.")
    if not isinstance(contract["model"].get("gauge_structure"), str) or not contract["model"]["gauge_structure"].strip():
        raise ParameterContractError("model.gauge_structure must name the local gauge group or be 'none'.")

    external_states = contract.get("external_states")
    if external_states is not None:
        if not isinstance(external_states, dict):
            raise ParameterContractError("external_states must be an object.")
        labels = external_states.get("labels")
        definitions = external_states.get("definitions")
        equivalent_groups = external_states.get("equivalent_groups")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ParameterContractError(
                "external_states.labels must contain unique non-empty strings."
            )
        if (
            not isinstance(definitions, dict)
            or set(definitions) != set(labels)
            or any(not isinstance(value, str) or not value.strip() for value in definitions.values())
        ):
            raise ParameterContractError(
                "external_states.definitions must define every label exactly once."
            )
        if not isinstance(equivalent_groups, list):
            raise ParameterContractError("external_states.equivalent_groups must be a list.")
        grouped_labels = []
        for index, group in enumerate(equivalent_groups):
            if (
                not isinstance(group, list)
                or len(group) < 2
                or len(group) != len(set(group))
                or any(label not in labels for label in group)
            ):
                raise ParameterContractError(
                    f"external_states.equivalent_groups[{index}] is invalid."
                )
            grouped_labels.extend(group)
        if len(grouped_labels) != len(set(grouped_labels)):
            raise ParameterContractError(
                "An external-state label cannot occur in multiple equivalence groups."
            )

    if not isinstance(contract["parameters"], list):
        raise ParameterContractError("parameters must be a list.")
    names = []
    for index, parameter in enumerate(contract["parameters"]):
        if not isinstance(parameter, dict):
            raise ParameterContractError(f"parameters[{index}] must be an object.")
        name = parameter.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ParameterContractError(f"parameters[{index}].name must be a non-empty string.")
        if "type" not in parameter or not isinstance(parameter["type"], str):
            raise ParameterContractError(f"parameters[{index}].type must be a string.")
        if "selected_value" not in parameter:
            raise ParameterContractError(f"parameters[{index}] must explicitly contain selected_value (use null if unresolved).")
        if not isinstance(parameter.get("required_for_run"), bool):
            raise ParameterContractError(
                f"parameters[{index}].required_for_run must be boolean."
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise ParameterContractError("Parameter names must be unique.")

    discretizations = contract.get("discretizations")
    if contract["schema_version"] == "1.1" and discretizations is None:
        raise ParameterContractError(
            "Parameter Contract schema 1.1 requires discretizations."
        )
    if discretizations is not None:
        if not isinstance(discretizations, list) or not discretizations:
            raise ParameterContractError(
                "discretizations must be a nonempty list when provided."
            )
        component_keys = []
        for index, component in enumerate(discretizations):
            if not isinstance(component, dict):
                raise ParameterContractError(
                    f"discretizations[{index}] must be an object."
                )
            sector = component.get("sector")
            scheme = component.get("scheme")
            settings = component.get("settings")
            references = component.get("parameter_references")
            if not isinstance(sector, str) or not sector.strip():
                raise ParameterContractError(
                    f"discretizations[{index}].sector must be a non-empty string."
                )
            if not isinstance(scheme, str) or not scheme.strip():
                raise ParameterContractError(
                    f"discretizations[{index}].scheme must be a non-empty string."
                )
            if not isinstance(settings, dict):
                raise ParameterContractError(
                    f"discretizations[{index}].settings must be an object."
                )
            if not isinstance(references, list):
                raise ParameterContractError(
                    f"discretizations[{index}].parameter_references must be a list."
                )
            truncation = component.get("truncation")
            if sector.strip().lower() == "gauge" and truncation is None:
                raise ParameterContractError(
                    f"discretizations[{index}] is a gauge component and must declare truncation."
                )
            if sector.strip().lower() == "gauge" and not settings:
                raise ParameterContractError(
                    f"discretizations[{index}].settings must describe the gauge "
                    "representation, fixing/elimination, and boundary choice."
                )
            if truncation is not None:
                if not isinstance(truncation, dict):
                    raise ParameterContractError(
                        f"discretizations[{index}].truncation must be an object."
                    )
                if (
                    not isinstance(truncation.get("scheme"), str)
                    or not truncation["scheme"].strip()
                ):
                    raise ParameterContractError(
                        f"discretizations[{index}].truncation.scheme must be a non-empty string."
                    )
                if not isinstance(truncation.get("settings"), dict):
                    raise ParameterContractError(
                        f"discretizations[{index}].truncation.settings must be an object."
                    )
            component_keys.append((sector.strip(), scheme.strip()))
        if (
            contract["schema_version"] == "1.1"
            and contract["model"]["gauge_structure"].strip().lower() != "none"
            and not any(
                component["sector"].strip().lower() == "gauge"
                for component in discretizations
            )
        ):
            raise ParameterContractError(
                "A gauge-theory schema 1.1 contract requires a gauge discretization component."
            )

    for collection_name in ("constraints", "validation_obligations"):
        collection = contract[collection_name]
        if not isinstance(collection, list):
            raise ParameterContractError(f"{collection_name} must be a list.")
        ids = []
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                raise ParameterContractError(f"{collection_name}[{index}] must be an object.")
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ParameterContractError(f"{collection_name}[{index}].id must be a non-empty string.")
            if item.get("severity") not in ("fatal", "warning"):
                raise ParameterContractError(f"{collection_name}[{index}].severity must be fatal or warning.")
            if collection_name == "constraints":
                if not isinstance(item.get("requirement"), str) or not item["requirement"].strip():
                    raise ParameterContractError(f"constraints[{index}].requirement must be a non-empty string.")
                if not isinstance(item.get("parameters"), list):
                    raise ParameterContractError(f"constraints[{index}].parameters must be a list.")
            else:
                if not isinstance(item.get("description"), str) or not item["description"].strip():
                    raise ParameterContractError(
                        f"validation_obligations[{index}].description must be a non-empty string."
                    )
                if item.get("stage") not in ("pre_simulation", "post_construction"):
                    raise ParameterContractError(
                        f"validation_obligations[{index}].stage must be pre_simulation or post_construction."
                    )
            ids.append(item_id)
        if len(ids) != len(set(ids)):
            raise ParameterContractError(f"IDs in {collection_name} must be unique.")

    return contract


def requires_physical_sector(contract):
    """Whether execution must remain in a user-requested physical sector.

    Gauge theories always require their Gauss-law sector.  A non-gauge theory
    requires a restricted sector only when the user explicitly requested a
    quantum-number-sector eigenstate; the actual eigenvalues are carried by
    the state definitions and are not assumed to be zero.
    """
    model = contract.get("model", {}) if isinstance(contract, dict) else {}
    gauge = model.get("gauge_structure", "")
    if isinstance(gauge, str) and gauge.strip() and gauge.strip().lower() != "none":
        return True
    states = contract.get("external_states", {}) if isinstance(contract, dict) else {}
    definitions = states.get("definitions", {}) if isinstance(states, dict) else {}
    return any(
        isinstance(definition, str)
        and "quantum-number-sector eigenstate" in definition.lower()
        for definition in definitions.values()
    )


def bind_immutable_run_inputs(contract, user_parameters, external_states):
    """Deterministically attach immutable RunSpec values.

    User-owned values always win. A model disagreement is a recoverable
    normalization event, not a fatal contract error; the proposed value is
    retained only in the persistent stage log and can never reach downstream
    physics.
    """
    contract = validate_parameter_contract(contract)
    if not isinstance(user_parameters, dict) or not user_parameters:
        raise ParameterContractError("RunSpec user parameters must be nonempty.")
    if (
        not isinstance(external_states, dict)
        or not external_states
        or any(not isinstance(label, str) or not label.strip() for label in external_states)
        or any(not isinstance(value, str) or not value.strip()
               for value in external_states.values())
    ):
        raise ParameterContractError("RunSpec external states must define named states.")
    bound = copy.deepcopy(contract)
    by_name = {item["name"]: item for item in bound["parameters"]}

    for name, value in user_parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise ParameterContractError("RunSpec parameter names must be nonempty strings.")
        if name in by_name:
            proposed_value = by_name[name].get("selected_value")
            if proposed_value != value:
                event = {
                    "path": "/parameters/{}/selected_value".format(name),
                    "action": "restore_immutable_run_spec_value",
                    "model_proposed_value": copy.deepcopy(proposed_value),
                    "authoritative_value": copy.deepcopy(value),
                }
                logging.getLogger("ContractNormalizer").warning(
                    "Recoverable controller normalization: %s",
                    json.dumps(event, ensure_ascii=False, sort_keys=True),
                )
            by_name[name]["selected_value"] = copy.deepcopy(value)
            by_name[name]["source_ref"] = "/user_input/parameters/{}".format(name)
            by_name[name]["immutable"] = True
        else:
            item = {
                "name": name,
                "symbol": name,
                "type": _inferred_parameter_type(value),
                "domain": {"source": "immutable RunSpec value"},
                "selected_value": copy.deepcopy(value),
                "unit": None,
                "required_for_run": True,
                "role": "Immutable user input; downstream stages must reference this path.",
                "source_ref": "/user_input/parameters/{}".format(name),
                "immutable": True,
            }
            bound["parameters"].append(item)
            by_name[name] = item
    labels = list(external_states)
    previous_groups = bound.get("external_states", {}).get("equivalent_groups", [])
    previous_groups = [
        group for group in previous_groups
        if isinstance(group, list)
        and len(group) >= 2
        and len(group) == len(set(group))
        and set(group).issubset(labels)
    ]
    bound["external_states"] = {
        "labels": labels,
        "definitions": copy.deepcopy(external_states),
        "equivalent_groups": previous_groups,
        "source_ref": "/user_input/external_states",
        "immutable": True,
    }
    return validate_parameter_contract(bound)


def extract_parameter_contract(text):
    """Recover a parameter payload, then apply the strict internal schema."""
    try:
        candidate = select_semantic_object(
            text,
            label="parameter contract",
            fingerprint_keys=(
                "model", "parameters", "constraints", "validation_obligations",
                "discretizations", "external_states",
            ),
            minimum_fingerprint_matches=3,
        )
    except ModelOutputIngestionError as error:
        raise ParameterContractError(str(error)) from error
    # Version identity is controller-owned; scientific fields remain strict.
    candidate["schema_version"] = "1.1" if "discretizations" in candidate else "1.0"
    return validate_parameter_contract(candidate)


def format_parameter_contract(contract):
    normalized = validate_parameter_contract(contract)
    return json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)


def parameter_contract_digest(contract):
    canonical = format_parameter_contract(contract).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parameter_contract_semantic_view(contract):
    """Return only fields that can change the scientific calculation."""
    validated = validate_parameter_contract(contract)
    parameters = []
    for item in validated.get("parameters", []):
        parameters.append({
            key: copy.deepcopy(item.get(key))
            for key in (
                "name", "type", "selected_value", "unit",
            )
        })
    parameters.sort(key=lambda item: item["name"])
    discretizations = []
    for item in validated.get("discretizations", []):
        component = {
            key: copy.deepcopy(item.get(key))
            for key in ("sector", "scheme", "settings", "parameter_references")
        }
        truncation = item.get("truncation")
        if isinstance(truncation, dict):
            component["truncation"] = {
                key: copy.deepcopy(truncation.get(key))
                for key in ("scheme", "settings")
            }
        discretizations.append(component)
    discretizations.sort(
        key=lambda item: (str(item.get("sector", "")), str(item.get("scheme", "")))
    )
    return {
        "model": {
            key: copy.deepcopy(validated.get("model", {}).get(key))
            for key in ("name", "spacetime_dimension", "gauge_structure")
        },
        "external_states": copy.deepcopy(validated.get("external_states")),
        "discretizations": discretizations,
        "parameters": parameters,
    }


def parameter_contracts_semantically_equal(left, right):
    return parameter_contract_semantic_view(left) == parameter_contract_semantic_view(right)


def load_parameter_contract(path):
    with open(path, "r", encoding="utf-8") as handle:
        return validate_parameter_contract(json.load(handle))


def save_parameter_contract(contract, path):
    content = format_parameter_contract(contract)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content + "\n")


def resolve_parameter_contract(contract, override_document):
    """Fill unresolved run parameters from an auditable user-override document."""
    contract = validate_parameter_contract(contract)
    if not isinstance(override_document, dict):
        raise ParameterContractError("Parameter override document must be a JSON object.")
    if override_document.get("schema_version") != "1.0":
        raise ParameterContractError("Parameter override schema_version must be '1.0'.")
    if override_document.get("source") not in ("user", "pipeline_resolved"):
        raise ParameterContractError(
            "Parameter overrides source must be 'user' or 'pipeline_resolved'."
        )
    values = override_document.get("resolved_values")
    if not isinstance(values, dict) or not values:
        raise ParameterContractError("Parameter overrides must contain a nonempty resolved_values object.")

    resolved = copy.deepcopy(contract)
    parameters_by_name = {item["name"]: item for item in resolved["parameters"]}
    unknown = sorted(set(values) - set(parameters_by_name))
    if unknown:
        raise ParameterContractError(f"Parameter overrides reference unknown names: {', '.join(unknown)}")

    newly_resolved = []
    for name, value in values.items():
        parameter = parameters_by_name[name]
        previous = parameter["selected_value"]
        if previous is not None and previous != value:
            raise ParameterContractError(
                f"Parameter override may not change resolved value {name}={previous!r} to {value!r}."
            )
        if previous is None:
            parameter["selected_value"] = value
            newly_resolved.append(name)

    # A pipeline settings document normally repeats values that Agent2 already
    # resolved. Repeating identical values is a semantic no-op and must not add
    # provenance to the physics contract: doing so would invalidate the hash
    # already embedded in Agent3's mapping contract.
    if not newly_resolved:
        return validate_parameter_contract(resolved)

    # Optimizer/runtime controls are deliberately not part of the physical
    # Parameter Contract. They may change without invalidating Agent2-4 physics.
    resolution_record = copy.deepcopy(override_document)
    resolution_record.pop("vqe_runtime", None)
    resolution_record["newly_resolved_parameters"] = sorted(newly_resolved)
    resolved["parameter_resolution"] = resolution_record
    return validate_parameter_contract(resolved)


def load_parameter_overrides(path):
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ParameterContractError("Parameter override file must contain a JSON object.")
    return document
