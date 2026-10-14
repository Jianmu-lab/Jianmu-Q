# -*- coding: utf-8 -*-
"""SQLite archive for complete simulation campaigns and reusable VQE experience.

This is deterministic support code, not an Agent. It records completed
simulation campaigns and exposes compact, parameter-aware summaries for
Agent2 discretization selection and Agent4.1 state-preparation selection.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Mapping, Optional

from parameter_contract import parameter_contract_digest, validate_parameter_contract
from validation_policy import provenance_warning


DATABASE_SCHEMA_VERSION = "3.1"
DEFAULT_DATABASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "local_data", "simulation_knowledge.sqlite3"
))
# ``energy_tolerance`` is the normalized relative-energy tolerance used by the
# runtime: |E-E_target| / max(|E_target|, 1). The unit floor keeps E_target=0
# well defined without adding a second user-facing threshold.
DEFAULT_VQE_ENERGY_TOLERANCE = 1.0e-3
DEFAULT_VQE_VARIANCE_TOLERANCE = 1.0e-2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _selected_values(parameter_contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        item["name"]: item.get("selected_value")
        for item in parameter_contract.get("parameters", [])
    }


def _legacy_discretizations(
    parameter_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Classify schema-1.0 contracts from explicit parameter names only."""
    values = _selected_values(parameter_contract)
    names = set(values)
    components: List[Dict[str, Any]] = []

    spacetime_references = [
        name for name in (
            "spatial_site_count", "lattice_spacing", "spatial_boundary_condition"
        ) if name in names
    ]
    if spacetime_references:
        components.append({
            "sector": "spacetime",
            "scheme": "finite_spatial_lattice",
            "settings": {name: values[name] for name in spacetime_references},
            "parameter_references": spacetime_references,
            "source": "legacy_parameter_inference",
        })

    fermion_scheme_parameter = next((
        name for name in (
            "fermion_discretization_scheme", "fermion_scheme",
            "matter_discretization_scheme",
        ) if name in names
    ), None)
    if fermion_scheme_parameter is not None:
        fermion_scheme = str(values[fermion_scheme_parameter])
    elif "wilson_parameter" in names:
        fermion_scheme = "Wilson"
    else:
        fermion_scheme = None
    fermion_references = [
        name for name in (
            fermion_scheme_parameter,
            "wilson_parameter", "fermionic_mode_count", "fermion_mass",
            "dimensionless_mass",
        ) if name is not None and name in names
    ]
    if fermion_scheme is not None:
        components.append({
            "sector": "fermion",
            "scheme": fermion_scheme,
            "settings": {name: values[name] for name in fermion_references},
            "parameter_references": fermion_references,
            "source": "legacy_parameter_inference",
        })

    gauge_structure = str(
        parameter_contract.get("model", {}).get("gauge_structure", "none")
    )
    if gauge_structure.strip().lower() != "none":
        gauge_scheme_parameter = next((
            name for name in (
                "gauge_digitization_scheme", "gauge_discretization_scheme",
                "gauge_scheme",
            ) if name in names
        ), None)
        gauge_scheme = (
            str(values[gauge_scheme_parameter])
            if gauge_scheme_parameter is not None
            else "unspecified_gauge_discretization"
        )
        gauge_references = [
            name for name in (
                gauge_scheme_parameter, "gauge_fixing", "gauge_coupling",
                "dimensionless_gauge_coupling", "left_boundary_electric_flux",
                "right_boundary_electric_flux", "total_charge_sector",
            ) if name is not None and name in names
        ]
        normalized_scheme = gauge_scheme.strip().lower()
        if "gauss" in normalized_scheme and "elimination" in normalized_scheme:
            truncation = {
                "scheme": "none",
                "settings": {},
                "reason": "Independent gauge degrees of freedom are exactly eliminated.",
            }
        else:
            truncation_parameter = next((
                name for name in (
                    "gauge_truncation_scheme", "electric_field_cutoff",
                    "representation_cutoff", "link_spin",
                ) if name in names
            ), None)
            truncation = {
                "scheme": (
                    str(values[truncation_parameter])
                    if truncation_parameter is not None
                    else "unspecified"
                ),
                "settings": (
                    {truncation_parameter: values[truncation_parameter]}
                    if truncation_parameter is not None else {}
                ),
                "reason": "Recovered conservatively from a legacy schema-1.0 contract.",
            }
        components.append({
            "sector": "gauge",
            "scheme": gauge_scheme,
            "settings": {name: values[name] for name in gauge_references},
            "parameter_references": gauge_references,
            "truncation": truncation,
            "source": "legacy_parameter_inference",
        })
    return components


def classify_discretizations(
    parameter_contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return explicit schema-1.1 metadata or a marked legacy classification."""
    explicit = parameter_contract.get("discretizations")
    if explicit:
        return [dict(component, source="parameter_contract") for component in explicit]
    return _legacy_discretizations(parameter_contract)


def _runtime_setting_category(name: str) -> str:
    if name in {
        "state_labels", "equivalent_state_groups", "target_kind_by_state",
        "target_spec_by_state",
    }:
        return "target_state"
    if name in {"depth_schedule", "adaptive_depth", "nested_by_depth", "scan_all_depths"}:
        return "depth_policy"
    if name in {"restart_count", "max_iterations", "stop_restarts_on_convergence"}:
        return "optimizer"
    if name in {
        "first_depth_lower_bound", "first_depth_upper_bound", "new_layer_value",
        "perturbation_scale", "initial_parameters_by_depth",
    }:
        return "initialization"
    if "tolerance" in name or name.startswith("gradient_check"):
        return "accuracy"
    if name in {"random_seed"}:
        return "reproducibility"
    if name.endswith("_file") or name.endswith("_database"):
        return "artifact_path"
    if name in {"record_experience"}:
        return "archive_policy"
    return "other"


def _connect(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vqe_runs (
            run_id TEXT PRIMARY KEY,
            recorded_at TEXT NOT NULL,
            parameter_contract_sha256 TEXT NOT NULL,
            model_name TEXT NOT NULL,
            spacetime_dimension TEXT NOT NULL,
            gauge_structure TEXT NOT NULL,
            ansatz_family_id TEXT NOT NULL,
            target_state_labels_json TEXT NOT NULL,
            target_kinds_json TEXT NOT NULL,
            status TEXT NOT NULL,
            termination_reason TEXT NOT NULL,
            selected_depth INTEGER,
            adapter_sha256 TEXT NOT NULL,
            parameter_contract_json TEXT NOT NULL,
            state_contract_json TEXT NOT NULL,
            run_config_json TEXT NOT NULL,
            result_path TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_vqe_runs_parameter
            ON vqe_runs(parameter_contract_sha256, recorded_at);
        CREATE INDEX IF NOT EXISTS idx_vqe_runs_model
            ON vqe_runs(model_name, gauge_structure, recorded_at);
        CREATE TABLE IF NOT EXISTS vqe_depth_results (
            run_id TEXT NOT NULL REFERENCES vqe_runs(run_id) ON DELETE CASCADE,
            depth INTEGER NOT NULL,
            state_label TEXT NOT NULL,
            status TEXT NOT NULL,
            energy REAL NOT NULL,
            energy_error REAL NOT NULL,
            variance REAL NOT NULL,
            optimizer_success INTEGER NOT NULL,
            function_evaluations INTEGER NOT NULL,
            gradient_evaluations INTEGER NOT NULL,
            parameter_count INTEGER NOT NULL,
            parameters_json TEXT NOT NULL,
            PRIMARY KEY (run_id, depth, state_label)
        );
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            spacetime_dimension TEXT NOT NULL,
            gauge_structure TEXT NOT NULL,
            model_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parameter_contracts (
            parameter_contract_sha256 TEXT PRIMARY KEY,
            model_id TEXT NOT NULL REFERENCES models(model_id),
            schema_version TEXT NOT NULL,
            external_states_json TEXT NOT NULL,
            contract_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS discretization_components (
            parameter_contract_sha256 TEXT NOT NULL
                REFERENCES parameter_contracts(parameter_contract_sha256)
                ON DELETE CASCADE,
            component_index INTEGER NOT NULL,
            sector TEXT NOT NULL,
            scheme TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            parameter_references_json TEXT NOT NULL,
            truncation_scheme TEXT,
            truncation_settings_json TEXT,
            truncation_reason TEXT,
            source TEXT NOT NULL,
            PRIMARY KEY (parameter_contract_sha256, component_index)
        );
        CREATE INDEX IF NOT EXISTS idx_discretization_sector_scheme
            ON discretization_components(sector, scheme);
        CREATE INDEX IF NOT EXISTS idx_discretization_truncation
            ON discretization_components(sector, truncation_scheme);
        CREATE TABLE IF NOT EXISTS physical_parameters (
            parameter_contract_sha256 TEXT NOT NULL
                REFERENCES parameter_contracts(parameter_contract_sha256)
                ON DELETE CASCADE,
            parameter_name TEXT NOT NULL,
            symbol TEXT,
            value_type TEXT NOT NULL,
            selected_value_json TEXT NOT NULL,
            value_resolved INTEGER NOT NULL,
            unit_json TEXT NOT NULL,
            required_for_run INTEGER NOT NULL,
            role TEXT NOT NULL,
            domain_json TEXT NOT NULL,
            PRIMARY KEY (parameter_contract_sha256, parameter_name)
        );
        CREATE INDEX IF NOT EXISTS idx_physical_parameter_name
            ON physical_parameters(parameter_name);
        CREATE TABLE IF NOT EXISTS ansatz_families (
            ansatz_id TEXT PRIMARY KEY,
            family_id TEXT NOT NULL,
            depth_symbol TEXT,
            nested_by_depth INTEGER,
            ansatz_json TEXT NOT NULL,
            state_contract_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ansatz_family_id
            ON ansatz_families(family_id);
        CREATE TABLE IF NOT EXISTS run_catalog (
            run_id TEXT PRIMARY KEY REFERENCES vqe_runs(run_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL REFERENCES models(model_id),
            parameter_contract_sha256 TEXT NOT NULL
                REFERENCES parameter_contracts(parameter_contract_sha256),
            ansatz_id TEXT NOT NULL REFERENCES ansatz_families(ansatz_id)
        );
        CREATE TABLE IF NOT EXISTS run_target_states (
            run_id TEXT NOT NULL REFERENCES vqe_runs(run_id) ON DELETE CASCADE,
            state_label TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            state_definition TEXT NOT NULL,
            equivalent_group_json TEXT,
            PRIMARY KEY (run_id, state_label)
        );
        CREATE TABLE IF NOT EXISTS run_settings (
            run_id TEXT NOT NULL REFERENCES vqe_runs(run_id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            setting_name TEXT NOT NULL,
            setting_value_json TEXT NOT NULL,
            PRIMARY KEY (run_id, setting_name)
        );
        CREATE INDEX IF NOT EXISTS idx_run_settings_category
            ON run_settings(category, setting_name);
        CREATE TABLE IF NOT EXISTS observable_protocols (
            observable_id TEXT PRIMARY KEY,
            parameter_contract_sha256 TEXT NOT NULL
                REFERENCES parameter_contracts(parameter_contract_sha256),
            protocol_id TEXT NOT NULL,
            time_evolution_method TEXT,
            observable_contract_sha256 TEXT NOT NULL,
            observable_contract_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_observable_protocol
            ON observable_protocols(protocol_id, time_evolution_method);
        CREATE TABLE IF NOT EXISTS simulation_campaigns (
            campaign_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            parameter_contract_sha256 TEXT NOT NULL
                REFERENCES parameter_contracts(parameter_contract_sha256),
            vqe_run_id TEXT REFERENCES vqe_runs(run_id),
            observable_id TEXT NOT NULL REFERENCES observable_protocols(observable_id),
            campaign_identity_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_campaign_contract
            ON simulation_campaigns(parameter_contract_sha256, updated_at);
        CREATE TABLE IF NOT EXISTS measurement_runs (
            measurement_run_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            recorded_at TEXT NOT NULL,
            status TEXT NOT NULL,
            depth_schedule_json TEXT NOT NULL,
            trotter_schedule_json TEXT NOT NULL,
            state_adapter_sha256 TEXT NOT NULL,
            observable_adapter_sha256 TEXT NOT NULL,
            result_path TEXT,
            result_sha256 TEXT,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS measurement_grid_points (
            measurement_run_id TEXT NOT NULL
                REFERENCES measurement_runs(measurement_run_id) ON DELETE CASCADE,
            candidate_id TEXT NOT NULL,
            ansatz_depth INTEGER NOT NULL,
            trotter_steps INTEGER NOT NULL,
            state_preparation_validated INTEGER NOT NULL,
            observable_real REAL NOT NULL,
            observable_imag REAL NOT NULL,
            state_quality_json TEXT NOT NULL,
            PRIMARY KEY (measurement_run_id, candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_measurement_grid_coordinates
            ON measurement_grid_points(ansatz_depth, trotter_steps);
        CREATE TABLE IF NOT EXISTS measurement_recipes (
            measurement_run_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            component TEXT NOT NULL,
            noiseless_expectation REAL NOT NULL,
            readout_pauli TEXT NOT NULL,
            logical_qubits INTEGER,
            logical_depth INTEGER,
            logical_two_qubit_gates INTEGER,
            logical_operation_count INTEGER,
            qpy_path TEXT,
            qpy_sha256 TEXT,
            resources_json TEXT NOT NULL,
            PRIMARY KEY (measurement_run_id, candidate_id, recipe_id),
            FOREIGN KEY (measurement_run_id, candidate_id)
                REFERENCES measurement_grid_points(measurement_run_id, candidate_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS classical_references (
            reference_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            method TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            ground_state_energy REAL NOT NULL,
            energy_variance REAL NOT NULL,
            observable_real REAL NOT NULL,
            observable_imag REAL NOT NULL,
            uncertainty_real REAL NOT NULL,
            uncertainty_imag REAL NOT NULL,
            convergence_passed INTEGER NOT NULL,
            result_path TEXT,
            result_sha256 TEXT,
            result_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_classical_reference_method
            ON classical_references(campaign_id, method);
        CREATE TABLE IF NOT EXISTS crosscheck_runs (
            crosscheck_run_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            measurement_run_id TEXT NOT NULL
                REFERENCES measurement_runs(measurement_run_id),
            scipy_reference_id TEXT NOT NULL
                REFERENCES classical_references(reference_id),
            itensor_reference_id TEXT NOT NULL
                REFERENCES classical_references(reference_id),
            recorded_at TEXT NOT NULL,
            status TEXT NOT NULL,
            thresholds_json TEXT NOT NULL,
            result_path TEXT,
            result_sha256 TEXT,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hardware_candidates (
            crosscheck_run_id TEXT NOT NULL
                REFERENCES crosscheck_runs(crosscheck_run_id) ON DELETE CASCADE,
            candidate_id TEXT NOT NULL,
            ansatz_depth INTEGER NOT NULL,
            trotter_steps INTEGER NOT NULL,
            algorithm_error REAL NOT NULL,
            candidate_json TEXT NOT NULL,
            PRIMARY KEY (crosscheck_run_id, candidate_id)
        );
        CREATE TABLE IF NOT EXISTS hardware_plans (
            hardware_plan_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            crosscheck_run_id TEXT NOT NULL REFERENCES crosscheck_runs(crosscheck_run_id),
            recorded_at TEXT NOT NULL,
            status TEXT NOT NULL,
            plan_path TEXT,
            plan_sha256 TEXT,
            plan_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hardware_deployments (
            hardware_plan_id TEXT NOT NULL
                REFERENCES hardware_plans(hardware_plan_id) ON DELETE CASCADE,
            deployment_index INTEGER NOT NULL,
            selected INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            ansatz_depth INTEGER NOT NULL,
            trotter_steps INTEGER NOT NULL,
            backend TEXT NOT NULL,
            score REAL NOT NULL,
            deployment_json TEXT NOT NULL,
            PRIMARY KEY (hardware_plan_id, deployment_index)
        );
        CREATE TABLE IF NOT EXISTS hardware_jobs (
            hardware_run_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            hardware_plan_id TEXT NOT NULL REFERENCES hardware_plans(hardware_plan_id),
            candidate_id TEXT NOT NULL,
            ansatz_depth INTEGER NOT NULL,
            trotter_steps INTEGER NOT NULL,
            backend TEXT NOT NULL,
            shots INTEGER NOT NULL,
            status TEXT NOT NULL,
            submitted_record_path TEXT,
            job_json TEXT NOT NULL,
            PRIMARY KEY (hardware_run_id, job_id)
        );
        CREATE TABLE IF NOT EXISTS hardware_results (
            hardware_run_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            backend TEXT NOT NULL,
            observable_real REAL NOT NULL,
            observable_imag REAL NOT NULL,
            standard_error_real REAL NOT NULL,
            standard_error_imag REAL NOT NULL,
            hardware_error_real REAL NOT NULL,
            hardware_error_imag REAL NOT NULL,
            result_path TEXT,
            result_json TEXT NOT NULL,
            PRIMARY KEY (hardware_run_id, job_id),
            FOREIGN KEY (hardware_run_id, job_id)
                REFERENCES hardware_jobs(hardware_run_id, job_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS final_analyses (
            analysis_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL
                REFERENCES simulation_campaigns(campaign_id) ON DELETE CASCADE,
            crosscheck_run_id TEXT NOT NULL REFERENCES crosscheck_runs(crosscheck_run_id),
            recorded_at TEXT NOT NULL,
            crosscheck_status TEXT NOT NULL,
            summary_path TEXT,
            summary_sha256 TEXT,
            summary_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_artifacts (
            analysis_id TEXT NOT NULL
                REFERENCES final_analyses(analysis_id) ON DELETE CASCADE,
            artifact_name TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            PRIMARY KEY (analysis_id, artifact_name)
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", DATABASE_SCHEMA_VERSION),
    )
    return connection


def record_vqe_result(
    database_path: str,
    adapter: Any,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    result_path: Optional[str] = None,
) -> str:
    """Insert or replace one completed VQE run and all depth/state records."""
    parameter_contract = adapter.PARAMETER_CONTRACT
    state_contract = adapter.STATE_PREPARATION_CONTRACT
    validate_parameter_contract(parameter_contract)
    parameter_digest = parameter_contract_digest(parameter_contract)
    if result.get("adapter_sha256") is None:
        raise ValueError("VQE result is missing adapter_sha256.")
    run_material = {
        "adapter_sha256": result["adapter_sha256"],
        "config": config,
        "parameter_contract_sha256": parameter_digest,
    }
    run_id = hashlib.sha256(_json(run_material).encode("utf-8")).hexdigest()
    model = parameter_contract["model"]
    ansatz = state_contract["ansatz"]
    model_id = _digest(model)
    ansatz_id = _digest(state_contract)
    selected_depth = result.get("shallowest_validated_depth")
    if selected_depth is None and result.get("tested_depths"):
        selected_depth = result["tested_depths"][-1]["depth"]
    recorded_at = _datetime.datetime.now(_datetime.timezone.utc).isoformat()

    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO models(
                model_id, name, spacetime_dimension, gauge_structure, model_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                name=excluded.name,
                spacetime_dimension=excluded.spacetime_dimension,
                gauge_structure=excluded.gauge_structure,
                model_json=excluded.model_json
            """,
            (
                model_id,
                str(model.get("name", "")),
                str(model.get("spacetime_dimension", "")),
                str(model.get("gauge_structure", "")),
                _json(model),
            ),
        )
        connection.execute(
            """
            INSERT INTO parameter_contracts(
                parameter_contract_sha256, model_id, schema_version,
                external_states_json, contract_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(parameter_contract_sha256) DO UPDATE SET
                model_id=excluded.model_id,
                schema_version=excluded.schema_version,
                external_states_json=excluded.external_states_json,
                contract_json=excluded.contract_json
            """,
            (
                parameter_digest,
                model_id,
                str(parameter_contract["schema_version"]),
                _json(parameter_contract.get("external_states", {})),
                _json(parameter_contract),
            ),
        )
        connection.execute(
            "DELETE FROM discretization_components WHERE parameter_contract_sha256 = ?",
            (parameter_digest,),
        )
        for index, component in enumerate(classify_discretizations(parameter_contract)):
            truncation = component.get("truncation") or {}
            connection.execute(
                """
                INSERT INTO discretization_components(
                    parameter_contract_sha256, component_index, sector, scheme,
                    settings_json, parameter_references_json, truncation_scheme,
                    truncation_settings_json, truncation_reason, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parameter_digest,
                    index,
                    str(component["sector"]),
                    str(component["scheme"]),
                    _json(component.get("settings", {})),
                    _json(component.get("parameter_references", [])),
                    str(truncation["scheme"]) if truncation else None,
                    _json(truncation.get("settings", {})) if truncation else None,
                    str(truncation.get("reason", "")) if truncation else None,
                    str(component.get("source", "parameter_contract")),
                ),
            )
        connection.execute(
            "DELETE FROM physical_parameters WHERE parameter_contract_sha256 = ?",
            (parameter_digest,),
        )
        for parameter in parameter_contract.get("parameters", []):
            selected_value = parameter.get("selected_value")
            connection.execute(
                """
                INSERT INTO physical_parameters(
                    parameter_contract_sha256, parameter_name, symbol,
                    value_type, selected_value_json, value_resolved, unit_json,
                    required_for_run, role, domain_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parameter_digest,
                    parameter["name"],
                    parameter.get("symbol"),
                    parameter["type"],
                    _json(selected_value),
                    int(selected_value is not None),
                    _json(parameter.get("unit")),
                    int(bool(parameter.get("required_for_run", False))),
                    str(parameter.get("role", "")),
                    _json(parameter.get("domain", {})),
                ),
            )
        connection.execute(
            """
            INSERT INTO ansatz_families(
                ansatz_id, family_id, depth_symbol, nested_by_depth,
                ansatz_json, state_contract_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ansatz_id) DO UPDATE SET
                family_id=excluded.family_id,
                depth_symbol=excluded.depth_symbol,
                nested_by_depth=excluded.nested_by_depth,
                ansatz_json=excluded.ansatz_json,
                state_contract_json=excluded.state_contract_json
            """,
            (
                ansatz_id,
                str(ansatz.get("family_id", "")),
                ansatz.get("depth_symbol"),
                (
                    int(ansatz["nested_by_depth"])
                    if isinstance(ansatz.get("nested_by_depth"), bool) else None
                ),
                _json(ansatz),
                _json(state_contract),
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO vqe_runs(
                run_id, recorded_at, parameter_contract_sha256, model_name,
                spacetime_dimension, gauge_structure, ansatz_family_id,
                target_state_labels_json, target_kinds_json, status,
                termination_reason, selected_depth, adapter_sha256,
                parameter_contract_json, state_contract_json, run_config_json,
                result_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                recorded_at,
                parameter_digest,
                str(model.get("name", "")),
                str(model.get("spacetime_dimension", "")),
                str(model.get("gauge_structure", "")),
                str(ansatz.get("family_id", "")),
                _json(state_contract["state_labels"]),
                _json(config["target_kind_by_state"]),
                str(result.get("status", "unknown")),
                str(result.get("termination_reason", "unknown")),
                selected_depth,
                str(result["adapter_sha256"]),
                _json(parameter_contract),
                _json(state_contract),
                _json(config),
                os.path.abspath(result_path) if result_path else None,
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO run_catalog(
                run_id, model_id, parameter_contract_sha256, ansatz_id
            ) VALUES (?, ?, ?, ?)
            """,
            (run_id, model_id, parameter_digest, ansatz_id),
        )
        connection.execute("DELETE FROM run_target_states WHERE run_id = ?", (run_id,))
        external_states = parameter_contract.get("external_states", {})
        definitions = external_states.get("definitions", {})
        groups = external_states.get("equivalent_groups", [])
        for label in state_contract["state_labels"]:
            equivalent_group = next((group for group in groups if label in group), None)
            connection.execute(
                """
                INSERT INTO run_target_states(
                    run_id, state_label, target_kind, state_definition,
                    equivalent_group_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    label,
                    str(config["target_kind_by_state"][label]),
                    str(definitions.get(label, "")),
                    _json(equivalent_group) if equivalent_group else None,
                ),
            )
        connection.execute("DELETE FROM run_settings WHERE run_id = ?", (run_id,))
        for setting_name, setting_value in config.items():
            connection.execute(
                """
                INSERT INTO run_settings(
                    run_id, category, setting_name, setting_value_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    _runtime_setting_category(setting_name),
                    setting_name,
                    _json(setting_value),
                ),
            )
        connection.execute("DELETE FROM vqe_depth_results WHERE run_id = ?", (run_id,))
        for entry in result.get("tested_depths", []):
            for label, state in entry["states"].items():
                connection.execute(
                    """
                    INSERT INTO vqe_depth_results(
                        run_id, depth, state_label, status, energy, energy_error,
                        variance, optimizer_success, function_evaluations,
                        gradient_evaluations, parameter_count, parameters_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(entry["depth"]),
                        label,
                        str(state["status"]),
                        float(state["energy"]),
                        float(state["energy_error"]),
                        float(state["variance"]),
                        int(bool(state.get("optimizer_success", False))),
                        int(state.get("function_evaluations", 0)),
                        int(state.get("gradient_evaluations", 0)),
                        len(state.get("parameters", [])),
                        _json(state.get("parameters", [])),
                    ),
                )
    return run_id


def query_vqe_experience(
    database_path: str,
    parameter_contract: Mapping[str, Any],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Return exact-parameter records first, followed by same-model records."""
    if not os.path.isfile(database_path):
        return []
    validate_parameter_contract(parameter_contract)
    digest = parameter_contract_digest(parameter_contract)
    model = parameter_contract["model"]
    with _connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *, CASE WHEN parameter_contract_sha256 = ? THEN 1 ELSE 0 END AS exact_match
            FROM vqe_runs
            WHERE parameter_contract_sha256 = ?
               OR (model_name = ? AND gauge_structure = ?)
            ORDER BY exact_match DESC, recorded_at DESC
            LIMIT ?
            """,
            (
                digest,
                digest,
                str(model.get("name", "")),
                str(model.get("gauge_structure", "")),
                int(limit),
            ),
        ).fetchall()
        output = []
        for row in rows:
            record = dict(row)
            depths = connection.execute(
                """
                SELECT depth, state_label, status, energy, energy_error, variance,
                       parameter_count
                FROM vqe_depth_results WHERE run_id = ?
                ORDER BY depth, state_label
                """,
                (record["run_id"],),
            ).fetchall()
            record["depth_results"] = [dict(item) for item in depths]
            target_energies: Dict[str, float] = {}
            result_path = record.get("result_path")
            if isinstance(result_path, str) and os.path.isfile(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as handle:
                        stored_result = json.load(handle)
                    target_energies = {
                        str(label): float(value)
                        for label, value in stored_result.get(
                            "target_energies_by_state", {}
                        ).items()
                    }
                    if not target_energies and stored_result.get(
                        "target_ground_energy"
                    ) is not None:
                        ground = float(stored_result["target_ground_energy"])
                        target_energies = {
                            str(label): ground
                            for label in json.loads(record["target_state_labels_json"])
                        }
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    target_energies = {}
            for item in record["depth_results"]:
                target = target_energies.get(str(item["state_label"]))
                item["relative_energy_error"] = (
                    float(item["energy_error"]) / max(abs(target), 1.0)
                    if target is not None
                    else float(item["energy_error"])
                    / max(abs(float(item["energy"])), 1.0)
                )
            components = connection.execute(
                """
                SELECT sector, scheme, settings_json, parameter_references_json,
                       truncation_scheme, truncation_settings_json,
                       truncation_reason, source
                FROM discretization_components
                WHERE parameter_contract_sha256 = ?
                ORDER BY component_index
                """,
                (record["parameter_contract_sha256"],),
            ).fetchall()
            record["discretizations"] = []
            for item in components:
                component = dict(item)
                component["settings"] = json.loads(component.pop("settings_json"))
                component["parameter_references"] = json.loads(
                    component.pop("parameter_references_json")
                )
                truncation_scheme = component.pop("truncation_scheme")
                truncation_settings_json = component.pop("truncation_settings_json")
                truncation_reason = component.pop("truncation_reason")
                if truncation_scheme is not None:
                    component["truncation"] = {
                        "scheme": truncation_scheme,
                        "settings": json.loads(truncation_settings_json),
                        "reason": truncation_reason,
                    }
                record["discretizations"].append(component)
            parameters = connection.execute(
                """
                SELECT parameter_name, selected_value_json, value_resolved
                FROM physical_parameters
                WHERE parameter_contract_sha256 = ?
                ORDER BY parameter_name
                """,
                (record["parameter_contract_sha256"],),
            ).fetchall()
            record["physical_parameters"] = {
                item["parameter_name"]: json.loads(item["selected_value_json"])
                for item in parameters
            }
            output.append(record)
        return output


def format_vqe_experience_context(
    database_path: str,
    parameter_contract: Mapping[str, Any],
    *,
    limit: int = 8,
    energy_tolerance: float = DEFAULT_VQE_ENERGY_TOLERANCE,
    variance_tolerance: float = DEFAULT_VQE_VARIANCE_TOLERANCE,
    plateau_window: int = 3,
    minimum_relative_improvement: float = 0.01,
) -> str:
    """Format history after evaluating raw metrics under the current policy.

    Stored statuses remain immutable audit facts.  Agent4.1 instead receives a
    current-policy status computed from the archived per-depth metrics, so a
    run that missed an older, stricter threshold is not permanently blacklisted.
    """
    if energy_tolerance <= 0.0 or variance_tolerance <= 0.0:
        raise ValueError("VQE experience tolerances must be positive.")
    if plateau_window < 2:
        raise ValueError("VQE plateau_window must be at least 2.")
    records = query_vqe_experience(database_path, parameter_contract, limit=limit)
    if not records:
        return "No matching VQE experience record is available."
    compact = []
    for record in records:
        try:
            required_labels = set(json.loads(record["target_state_labels_json"]))
        except (TypeError, json.JSONDecodeError):
            required_labels = set()
        rows_by_depth: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for row in record["depth_results"]:
            rows_by_depth.setdefault(int(row["depth"]), {})[str(row["state_label"])] = row
        qualifying_depths = []
        residual_history = []
        for depth, states in rows_by_depth.items():
            if not required_labels or not required_labels.issubset(states):
                if required_labels:
                    continue
            labels_at_depth = required_labels or set(states)
            residual_history.append({
                "depth": depth,
                "residual": max(
                    max(
                        float(states[label]["relative_energy_error"])
                        / energy_tolerance,
                        float(states[label]["variance"]) / variance_tolerance,
                    )
                    for label in labels_at_depth
                ),
            })
            if all(
                states[label].get("relative_energy_error") is not None
                and math.isfinite(float(states[label]["relative_energy_error"]))
                and math.isfinite(float(states[label]["variance"]))
                and float(states[label]["relative_energy_error"]) < energy_tolerance
                and float(states[label]["variance"]) < variance_tolerance
                for label in labels_at_depth
            ):
                qualifying_depths.append(depth)
        residual_history.sort(key=lambda item: item["depth"])
        recent = residual_history[-plateau_window:]
        relative_improvement = None
        plateau = False
        if len(recent) == plateau_window:
            first_residual = recent[0]["residual"]
            last_residual = recent[-1]["residual"]
            relative_improvement = (
                (first_residual - last_residual)
                / max(abs(first_residual), 1.0e-300)
            )
            plateau = (
                last_residual > 1.0
                and relative_improvement < minimum_relative_improvement
            )
        if qualifying_depths:
            current_status = "converged_state"
            current_depth = min(qualifying_depths)
        elif plateau:
            current_status = "ansatz_insufficient"
            current_depth = None
        elif rows_by_depth:
            current_status = "promising_state"
            current_depth = max(rows_by_depth)
        else:
            current_status = str(record["status"])
            current_depth = record["selected_depth"]
        item = {
            "exact_parameter_match": bool(record["exact_match"]),
            "model_name": record["model_name"],
            "gauge_structure": record["gauge_structure"],
            "discretizations": record["discretizations"],
            "physical_parameters": record["physical_parameters"],
            "ansatz_family_id": record["ansatz_family_id"],
            "current_policy_status": current_status,
            "current_policy_selected_depth": current_depth,
            "current_policy_thresholds": {
                "relative_energy_error": energy_tolerance,
                "variance": variance_tolerance,
            },
            "current_policy_trend": {
                "residual_history": residual_history,
                "recent_relative_improvement": relative_improvement,
                "plateau": plateau,
            },
            "historical_status": record["status"],
            "historical_termination_reason": record["termination_reason"],
            "historical_selected_depth": record["selected_depth"],
            "depth_results": record["depth_results"],
        }
        # A family name is enough to avoid a known failure, but not enough to
        # reproduce a successful circuit.  Supply the actual prior design for
        # exact-parameter records and for successful same-model transfers.
        if bool(record["exact_match"]) or current_status == "converged_state":
            try:
                prior_contract = json.loads(record["state_contract_json"])
                prior_ansatz = prior_contract.get("ansatz")
                if isinstance(prior_ansatz, dict):
                    item["ansatz_design"] = prior_ansatz
            except (TypeError, json.JSONDecodeError):
                pass
        compact.append(item)
    return json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True)


def _model_terms(value: Any) -> set[str]:
    """Return stable, non-generic terms for fuzzy model-family matching."""
    generic = {
        "and", "field", "fermion", "fermions", "gauge", "lattice", "model",
        "quantum", "theory", "the", "with",
    }
    return {
        term for term in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(term) > 1 and term not in generic
    }


def _parameter_alias(name: Any) -> str:
    """Normalize harmless symbol suffixes without imposing a model schema."""
    tokens = [token for token in re.findall(r"[a-z0-9]+", str(name).lower()) if token]
    if len(tokens) > 1 and len(tokens[-1]) == 1:
        tokens.pop()
    return "_".join(tokens)


def _same_selected_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(1.0, abs(float(left)), abs(float(right)))
        return abs(float(left) - float(right)) <= 1.0e-12 * scale
    return left == right


def _normalized_gauge_structure(value: Any) -> str:
    terms = [
        term for term in re.findall(r"[a-z0-9]+", str(value).lower())
        if term not in {"gauge", "group", "local", "symmetry"}
    ]
    return "".join(terms)


def summarize_discretization_problem(
    observable: Mapping[str, Any],
    external_states: Mapping[str, Any],
    *,
    gauge_structure: Optional[str] = None,
    theory_definitions: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize the physics facts Agent2 needs when reusing experience.

    This is advisory context, not a validator. In particular, charged field
    insertions raise an intermediate-sector question without hard-coding a
    charge convention or rejecting a discretization.
    """
    expression = str(observable.get("expression", ""))
    insertions = observable.get("insertions", [])
    insertion_labels = [
        str(item.get("operator", item.get("id", "")))
        for item in insertions if isinstance(item, Mapping)
    ]
    operator_text = " ".join([expression] + insertion_labels).lower()
    fermionic_insertions = sum(
        1 for label in insertion_labels
        if any(token in label.lower() for token in ("psi", "fermion", "quark"))
    )
    if not fermionic_insertions and any(
        token in operator_text for token in (r"\psi", "psi", "fermion", "quark")
    ):
        fermionic_insertions = 1

    transporters = observable.get("gauge_transporters", [])
    transporter_required = bool(transporters) or any(
        token in operator_text for token in ("wilson", "transporter", " w(")
    )

    def has_nonzero_time(position: Any) -> bool:
        if not isinstance(position, Mapping):
            return False
        for key, value in position.items():
            if str(key).lower() not in {"t", "time", "t_index", "time_index"}:
                continue
            if isinstance(value, (int, float)):
                if float(value) != 0.0:
                    return True
            elif str(value).strip().lower() not in {"", "0", "0.0", "none"}:
                return True
        return False

    real_time_required = any(
        has_nonzero_time(item.get("position"))
        for item in insertions if isinstance(item, Mapping)
    )
    gauge_theory = _normalized_gauge_structure(gauge_structure) not in {"", "none"}
    symmetry_ids = []
    for definition in theory_definitions or []:
        if not isinstance(definition, Mapping):
            continue
        identifier = str(definition.get("id", ""))
        kind = str(definition.get("kind", ""))
        searchable = f"{identifier} {kind}".lower()
        if any(token in searchable for token in (
            "gauss", "gauge", "charge", "conserved", "symmetry", "current",
        )):
            symmetry_ids.append(identifier or kind)

    required_symmetries = list(dict.fromkeys(symmetry_ids))
    if gauge_theory and "local_gauge_invariance" not in required_symmetries:
        required_symmetries.append("local_gauge_invariance")
    if gauge_theory and fermionic_insertions:
        required_symmetries.append("charge_sector_consistency")

    return {
        "observable": {
            "ordering": observable.get("ordering", "as_written"),
            "fermionic_insertion_count": fermionic_insertions,
            "gauge_transporter_required": transporter_required,
            "real_time_evolution_required": real_time_required,
        },
        "external_states": {
            "definitions": dict(external_states),
            "must_be_preserved_independently": True,
        },
        "required_symmetries": list(dict.fromkeys(required_symmetries)),
        "intermediate_sector_check_required": bool(
            gauge_theory and fermionic_insertions
        ),
    }


def query_discretization_experience(
    database_path: str,
    model_name: str,
    user_parameters: Mapping[str, Any],
    *,
    gauge_structure: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Rank reusable *complete discretization bundles* from successful campaigns.

    This lookup deliberately does not require a Parameter Contract from the new
    run: Agent2 is the stage that creates it.  Full-pipeline evidence ranks above
    a VQE-only success, and a failed cross-check is retained as negative evidence.
    """
    if not os.path.isfile(database_path):
        return []
    requested_terms = _model_terms(model_name)
    requested_parameters = {
        _parameter_alias(name): value for name, value in user_parameters.items()
    }
    with _connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT pc.parameter_contract_sha256, pc.contract_json,
                   m.name AS model_name, m.spacetime_dimension, m.gauge_structure
            FROM parameter_contracts AS pc
            JOIN models AS m ON m.model_id = pc.model_id
            """
        ).fetchall()
        ranked: List[Dict[str, Any]] = []
        for row in rows:
            candidate_terms = _model_terms(row["model_name"])
            overlap = requested_terms & candidate_terms
            if requested_terms and not overlap:
                continue
            model_similarity = (
                len(overlap) / max(1, min(len(requested_terms), len(candidate_terms)))
            )
            if model_similarity < 0.5:
                continue
            candidate_gauge = str(row["gauge_structure"] or "")
            gauge_match = not gauge_structure or (
                _normalized_gauge_structure(candidate_gauge)
                == _normalized_gauge_structure(gauge_structure)
            )
            if gauge_structure and not gauge_match:
                continue

            digest = row["parameter_contract_sha256"]
            campaign_statuses = sorted({item[0] for item in connection.execute(
                "SELECT status FROM simulation_campaigns "
                "WHERE parameter_contract_sha256 = ?", (digest,)
            ).fetchall()})
            crosscheck_statuses = sorted({item[0] for item in connection.execute(
                """
                SELECT cr.status
                FROM crosscheck_runs AS cr
                JOIN simulation_campaigns AS sc
                  ON sc.campaign_id = cr.campaign_id
                WHERE sc.parameter_contract_sha256 = ?
                """,
                (digest,),
            ).fetchall()})
            vqe_statuses = sorted({item[0] for item in connection.execute(
                "SELECT status FROM vqe_runs WHERE parameter_contract_sha256 = ?",
                (digest,),
            ).fetchall()})
            if not campaign_statuses and not vqe_statuses:
                continue
            evidence_score = max(
                [0]
                + [
                    {
                        "analysis_complete": 120,
                        "crosscheck_approved": 110,
                        "measurement_complete": 70,
                        "crosscheck_not_approved": -40,
                    }.get(status, 0)
                    for status in campaign_statuses
                ]
                + [
                    {"converged_state": 40, "best_effort_state": 10}.get(status, 0)
                    for status in vqe_statuses
                ]
                + [
                    {"approved": 110, "not_approved": -40}.get(status, 0)
                    for status in crosscheck_statuses
                ]
            )
            if (
                (
                    "crosscheck_not_approved" in campaign_statuses
                    or "not_approved" in crosscheck_statuses
                )
                and "crosscheck_approved" not in campaign_statuses
                and "approved" not in crosscheck_statuses
                and "analysis_complete" not in campaign_statuses
            ):
                evidence_score -= 40

            parameter_rows = connection.execute(
                """
                SELECT parameter_name, selected_value_json
                FROM physical_parameters
                WHERE parameter_contract_sha256 = ?
                ORDER BY parameter_name
                """,
                (digest,),
            ).fetchall()
            physical_parameters = {
                item["parameter_name"]: json.loads(item["selected_value_json"])
                for item in parameter_rows
            }
            candidate_parameters = {
                _parameter_alias(name): value
                for name, value in physical_parameters.items()
                if value is not None
            }
            shared = sorted(set(requested_parameters) & set(candidate_parameters))
            matching = [
                name for name in shared
                if _same_selected_value(requested_parameters[name], candidate_parameters[name])
            ]
            conflicting = [name for name in shared if name not in matching]

            component_rows = connection.execute(
                """
                SELECT sector, scheme, settings_json, parameter_references_json,
                       truncation_scheme, truncation_settings_json,
                       truncation_reason, source
                FROM discretization_components
                WHERE parameter_contract_sha256 = ?
                ORDER BY component_index
                """,
                (digest,),
            ).fetchall()
            components = []
            for item in component_rows:
                component = {
                    "sector": item["sector"],
                    "scheme": item["scheme"],
                    "settings": json.loads(item["settings_json"]),
                    "parameter_references": json.loads(item["parameter_references_json"]),
                    "source": item["source"],
                }
                if item["truncation_scheme"] is not None:
                    component["truncation"] = {
                        "scheme": item["truncation_scheme"],
                        "settings": json.loads(item["truncation_settings_json"] or "{}"),
                        "reason": item["truncation_reason"] or "",
                    }
                components.append(component)
            if not components:
                continue

            historical_observables = connection.execute(
                """
                SELECT DISTINCT op.protocol_id, op.time_evolution_method,
                                op.observable_contract_json
                FROM observable_protocols AS op
                JOIN simulation_campaigns AS sc
                  ON sc.observable_id = op.observable_id
                WHERE sc.parameter_contract_sha256 = ?
                """,
                (digest,),
            ).fetchall()
            real_time_observed = False
            protocol_ids = []
            for item in historical_observables:
                protocol_ids.append(item["protocol_id"])
                try:
                    observable_contract = json.loads(item["observable_contract_json"])
                except (TypeError, json.JSONDecodeError):
                    observable_contract = {}
                real_time_observed = real_time_observed or bool(
                    observable_contract.get("time_evolution", {}).get("required")
                    or item["time_evolution_method"]
                )

            try:
                source_contract = json.loads(row["contract_json"])
            except (TypeError, json.JSONDecodeError):
                source_contract = {}

            rank_score = (
                evidence_score
                + round(30 * model_similarity)
                + (10 if gauge_match else 0)
                + 4 * len(matching)
                - 2 * len(conflicting)
            )
            ranked.append({
                "source_parameter_contract_sha256": digest,
                "model": {
                    "name": row["model_name"],
                    "spacetime_dimension": row["spacetime_dimension"],
                    "gauge_structure": row["gauge_structure"],
                },
                "discretizations": components,
                "comparable_physical_parameters": {
                    name: value for name, value in physical_parameters.items()
                    if _parameter_alias(name) in shared
                },
                "evidence": {
                    "campaign_statuses": campaign_statuses,
                    "crosscheck_statuses": crosscheck_statuses,
                    "vqe_statuses": vqe_statuses,
                },
                "historical_scope": {
                    "external_states": source_contract.get("external_states", {}),
                    "observable_protocol_ids": sorted(set(protocol_ids)),
                    "real_time_evolution_observed": real_time_observed,
                },
                "parameter_compatibility": {
                    "matching_aliases": matching,
                    "conflicting_aliases": conflicting,
                    "uncompared_user_parameters": sorted(
                        set(requested_parameters) - set(candidate_parameters)
                    ),
                },
                "reuse_recommended": (
                    "crosscheck_approved" in campaign_statuses
                    or "approved" in crosscheck_statuses
                    or "analysis_complete" in campaign_statuses
                ),
                "rank_score": rank_score,
            })
    ranked.sort(
        key=lambda item: (
            item["rank_score"],
            item["source_parameter_contract_sha256"],
        ),
        reverse=True,
    )
    return ranked[:max(0, int(limit))]


def format_discretization_experience_context(
    database_path: str,
    model_name: str,
    user_parameters: Mapping[str, Any],
    *,
    gauge_structure: Optional[str] = None,
    current_problem: Optional[Mapping[str, Any]] = None,
    limit: int = 3,
) -> str:
    """Return a compact Agent2 prompt payload, never the entire database."""
    records = query_discretization_experience(
        database_path,
        model_name,
        user_parameters,
        gauge_structure=gauge_structure,
        limit=limit,
    )
    if not records:
        if current_problem:
            return json.dumps({
                "current_problem_requirements": current_problem,
                "historical_candidates": [],
            }, ensure_ascii=False, indent=2, sort_keys=True)
        return "No compatible historical discretization experience is available."
    compact = []
    has_validated_bundle = any(record["reuse_recommended"] for record in records)
    for index, record in enumerate(records):
        if record["reuse_recommended"] or (index == 0 and not has_validated_bundle):
            item = dict(record)
        else:
            # Failed or VQE-only records are useful warnings, but their full
            # schemes should not consume prompt space or look like templates.
            item = {
                key: record[key] for key in (
                    "source_parameter_contract_sha256", "model", "evidence",
                    "historical_scope", "parameter_compatibility",
                    "reuse_recommended", "rank_score",
                )
            }
        item["selection_role"] = "candidate_evidence_not_mandatory_choice"
        compact.append(item)
    payload = {
        "current_problem_requirements": current_problem or {},
        "historical_candidates": compact,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _artifact_fields(path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not path:
        return None, None
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        return absolute, None
    digest = hashlib.sha256()
    with open(absolute, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return absolute, digest.hexdigest()


def _prefixed_id(prefix: str, material: Any) -> str:
    return "{}_{}".format(prefix, _digest(material))


def record_measurement_result(
    database_path: str,
    state_adapter: Any,
    observable_adapter: Any,
    vqe_result: Mapping[str, Any],
    measurement: Mapping[str, Any],
    *,
    result_path: Optional[str] = None,
) -> Dict[str, str]:
    """Archive one complete noiseless ``(D_ans, N_T)`` measurement campaign."""
    parameter_contract = state_adapter.PARAMETER_CONTRACT
    state_contract = state_adapter.STATE_PREPARATION_CONTRACT
    observable_contract = observable_adapter.OBSERVABLE_CONTRACT
    validate_parameter_contract(parameter_contract)
    parameter_digest = parameter_contract_digest(parameter_contract)
    provenance_warning(
        "parameter_contract_hash_mismatch",
        "Measurement provenance hash differs from the adapter; the record is retained with explicit provenance.",
        parameter_digest,
        measurement.get("parameter_contract_sha256"),
    )
    config = vqe_result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(
            "VQE result must contain its validated config before campaign archiving."
        )
    vqe_run_id = record_vqe_result(
        database_path,
        state_adapter,
        config,
        vqe_result,
        result_path=measurement.get("vqe_result_path"),
    )
    observable_id = _prefixed_id("observable", observable_contract)
    campaign_identity = {
        "parameter_contract_sha256": parameter_digest,
        "vqe_run_id": vqe_run_id,
        "state_preparation_contract_sha256": measurement[
            "state_preparation_contract_sha256"
        ],
        "observable_contract_sha256": measurement["observable_contract_sha256"],
        "state_adapter_sha256": measurement["state_adapter_sha256"],
        "observable_adapter_sha256": measurement["observable_adapter_sha256"],
        "depth_schedule": measurement["depth_schedule"],
        "trotter_schedule": measurement["trotter_schedule"],
    }
    campaign_id = measurement.get("campaign_id") or _prefixed_id(
        "campaign", campaign_identity
    )
    measurement_identity = {
        "campaign_id": campaign_id,
        "grid": [
            {
                "candidate_id": point["candidate_id"],
                "D_ans": point["D_ans"],
                "N_T": point["N_T"],
                "observable_real": point["observable_real"],
                "observable_imag": point["observable_imag"],
                "recipe_hashes": [recipe.get("qpy_sha256") for recipe in point["recipes"]],
            }
            for point in measurement["validation_grid"]
        ],
    }
    measurement_run_id = measurement.get("measurement_run_id") or _prefixed_id(
        "measurement", measurement_identity
    )
    now = _utc_now()
    result_absolute, result_sha = _artifact_fields(result_path)
    ansatz_id = _digest(state_contract)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO observable_protocols(
                observable_id, parameter_contract_sha256, protocol_id,
                time_evolution_method, observable_contract_sha256,
                observable_contract_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(observable_id) DO UPDATE SET
                protocol_id=excluded.protocol_id,
                time_evolution_method=excluded.time_evolution_method,
                observable_contract_sha256=excluded.observable_contract_sha256,
                observable_contract_json=excluded.observable_contract_json
            """,
            (
                observable_id,
                parameter_digest,
                str(observable_contract.get("measurement", {}).get("protocol_id", "")),
                str(observable_contract.get("time_evolution", {}).get("method", "")),
                str(measurement["observable_contract_sha256"]),
                _json(observable_contract),
            ),
        )
        existing = connection.execute(
            "SELECT created_at FROM simulation_campaigns WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        created_at = existing[0] if existing else now
        connection.execute(
            """
            INSERT INTO simulation_campaigns(
                campaign_id, created_at, updated_at, status,
                parameter_contract_sha256, vqe_run_id, observable_id,
                campaign_identity_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=CASE
                    WHEN simulation_campaigns.status = 'measurement_complete'
                    THEN excluded.status
                    ELSE simulation_campaigns.status
                END,
                vqe_run_id=excluded.vqe_run_id,
                observable_id=excluded.observable_id,
                campaign_identity_json=excluded.campaign_identity_json
            """,
            (
                campaign_id, created_at, now, "measurement_complete",
                parameter_digest, vqe_run_id, observable_id, _json(campaign_identity),
            ),
        )
        connection.execute(
            """
            INSERT INTO measurement_runs(
                measurement_run_id, campaign_id, recorded_at, status,
                depth_schedule_json, trotter_schedule_json,
                state_adapter_sha256, observable_adapter_sha256,
                result_path, result_sha256, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(measurement_run_id) DO UPDATE SET
                recorded_at=excluded.recorded_at,
                status=excluded.status,
                result_path=excluded.result_path,
                result_sha256=excluded.result_sha256,
                result_json=excluded.result_json
            """,
            (
                measurement_run_id, campaign_id, now,
                str(measurement.get("status", "unknown")),
                _json(measurement["depth_schedule"]),
                _json(measurement["trotter_schedule"]),
                str(measurement["state_adapter_sha256"]),
                str(measurement["observable_adapter_sha256"]),
                result_absolute, result_sha, _json(measurement),
            ),
        )
        connection.execute(
            "DELETE FROM measurement_grid_points WHERE measurement_run_id = ?",
            (measurement_run_id,),
        )
        for point in measurement["validation_grid"]:
            connection.execute(
                """
                INSERT INTO measurement_grid_points(
                    measurement_run_id, candidate_id, ansatz_depth,
                    trotter_steps, state_preparation_validated,
                    observable_real, observable_imag, state_quality_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement_run_id, point["candidate_id"], int(point["D_ans"]),
                    int(point["N_T"]), int(bool(point["state_preparation_validated"])),
                    float(point["observable_real"]), float(point["observable_imag"]),
                    _json(point.get("state_quality", {})),
                ),
            )
            for recipe in point["recipes"]:
                resources = recipe.get("resources", {})
                connection.execute(
                    """
                    INSERT INTO measurement_recipes(
                        measurement_run_id, candidate_id, recipe_id, component,
                        noiseless_expectation, readout_pauli, logical_qubits,
                        logical_depth, logical_two_qubit_gates,
                        logical_operation_count, qpy_path, qpy_sha256,
                        resources_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        measurement_run_id, point["candidate_id"], recipe["recipe_id"],
                        str(recipe.get("component", "")),
                        float(recipe.get("noiseless_expectation", 0.0)),
                        str(recipe.get("readout_pauli", "")),
                        resources.get("num_qubits"), resources.get("logical_depth"),
                        resources.get("logical_two_qubit_gates"),
                        resources.get("logical_operation_count"),
                        recipe.get("qpy_path"), recipe.get("qpy_sha256"),
                        _json(resources),
                    ),
                )
    return {
        "campaign_id": str(campaign_id),
        "measurement_run_id": str(measurement_run_id),
        "vqe_run_id": str(vqe_run_id),
        "observable_id": str(observable_id),
        "ansatz_id": str(ansatz_id),
    }


def record_classical_reference(
    database_path: str,
    campaign_id: str,
    reference: Mapping[str, Any],
    *,
    result_path: Optional[str] = None,
) -> str:
    reference_id = _prefixed_id("reference", {
        "campaign_id": campaign_id,
        "method": reference["method"],
        "result": reference,
    })
    absolute, sha = _artifact_fields(result_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO classical_references(
                reference_id, campaign_id, method, recorded_at,
                ground_state_energy, energy_variance, observable_real,
                observable_imag, uncertainty_real, uncertainty_imag,
                convergence_passed, result_path, result_sha256, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reference_id) DO UPDATE SET
                recorded_at=excluded.recorded_at,
                result_path=excluded.result_path,
                result_sha256=excluded.result_sha256,
                result_json=excluded.result_json
            """,
            (
                reference_id, campaign_id, str(reference["method"]), _utc_now(),
                float(reference["ground_state_energy"]),
                float(reference["energy_variance"]),
                float(reference["observable_real"]), float(reference["observable_imag"]),
                float(reference["uncertainty_real"]), float(reference["uncertainty_imag"]),
                int(bool(reference["convergence"]["passed"])), absolute, sha,
                _json(reference),
            ),
        )
    return reference_id


def record_crosscheck_result(
    database_path: str,
    measurement: Mapping[str, Any],
    scipy_reference: Mapping[str, Any],
    itensor_reference: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    *,
    result_path: Optional[str] = None,
    scipy_path: Optional[str] = None,
    itensor_path: Optional[str] = None,
) -> Dict[str, str]:
    campaign_id = str(measurement["campaign_id"])
    measurement_run_id = str(measurement["measurement_run_id"])
    scipy_id = record_classical_reference(
        database_path, campaign_id, scipy_reference, result_path=scipy_path
    )
    itensor_id = record_classical_reference(
        database_path, campaign_id, itensor_reference, result_path=itensor_path
    )
    crosscheck_run_id = crosscheck.get("crosscheck_run_id") or _prefixed_id(
        "crosscheck",
        {
            "campaign_id": campaign_id,
            "measurement_run_id": measurement_run_id,
            "scipy_reference_id": scipy_id,
            "itensor_reference_id": itensor_id,
            "thresholds": crosscheck["thresholds"],
        },
    )
    absolute, sha = _artifact_fields(result_path)
    now = _utc_now()
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO crosscheck_runs(
                crosscheck_run_id, campaign_id, measurement_run_id,
                scipy_reference_id, itensor_reference_id, recorded_at,
                status, thresholds_json, result_path, result_sha256, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(crosscheck_run_id) DO UPDATE SET
                recorded_at=excluded.recorded_at,
                status=excluded.status,
                result_path=excluded.result_path,
                result_sha256=excluded.result_sha256,
                result_json=excluded.result_json
            """,
            (
                crosscheck_run_id, campaign_id, measurement_run_id, scipy_id,
                itensor_id, now, str(crosscheck["status"]),
                _json(crosscheck["thresholds"]), absolute, sha, _json(crosscheck),
            ),
        )
        connection.execute(
            "DELETE FROM hardware_candidates WHERE crosscheck_run_id = ?",
            (crosscheck_run_id,),
        )
        for candidate in crosscheck.get("hardware_eligible_candidates", []):
            connection.execute(
                """
                INSERT INTO hardware_candidates(
                    crosscheck_run_id, candidate_id, ansatz_depth,
                    trotter_steps, algorithm_error, candidate_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    crosscheck_run_id, candidate["candidate_id"],
                    int(candidate["D_ans"]), int(candidate["N_T"]),
                    float(candidate["algorithm_error_vs_scipy"]), _json(candidate),
                ),
            )
        connection.execute(
            "UPDATE simulation_campaigns SET status = ?, updated_at = ? WHERE campaign_id = ?",
            ("crosscheck_{}".format(crosscheck["status"]), now, campaign_id),
        )
    return {
        "campaign_id": campaign_id,
        "measurement_run_id": measurement_run_id,
        "scipy_reference_id": scipy_id,
        "itensor_reference_id": itensor_id,
        "crosscheck_run_id": str(crosscheck_run_id),
    }


def record_hardware_plan(
    database_path: str,
    plan: Mapping[str, Any],
    *,
    plan_path: Optional[str] = None,
) -> str:
    campaign_id = str(plan["campaign_id"])
    crosscheck_run_id = str(plan["crosscheck_run_id"])
    plan_id = plan.get("hardware_plan_id") or _prefixed_id(
        "hardware_plan",
        {"campaign_id": campaign_id, "policy": plan["planning_policy"],
         "deployments": plan["all_deployments"]},
    )
    absolute, sha = _artifact_fields(plan_path)
    selected = {
        (item["candidate_id"], item["backend"])
        for item in plan.get("selected_deployments", [])
    }
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO hardware_plans(
                hardware_plan_id, campaign_id, crosscheck_run_id, recorded_at,
                status, plan_path, plan_sha256, plan_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hardware_plan_id) DO UPDATE SET
                recorded_at=excluded.recorded_at,
                status=excluded.status,
                plan_path=excluded.plan_path,
                plan_sha256=excluded.plan_sha256,
                plan_json=excluded.plan_json
            """,
            (
                plan_id, campaign_id, crosscheck_run_id, _utc_now(),
                str(plan["status"]), absolute, sha, _json(plan),
            ),
        )
        connection.execute(
            "DELETE FROM hardware_deployments WHERE hardware_plan_id = ?", (plan_id,)
        )
        for index, deployment in enumerate(plan["all_deployments"]):
            connection.execute(
                """
                INSERT INTO hardware_deployments(
                    hardware_plan_id, deployment_index, selected, candidate_id,
                    ansatz_depth, trotter_steps, backend, score, deployment_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id, index,
                    int((deployment["candidate_id"], deployment["backend"]) in selected),
                    deployment["candidate_id"], int(deployment["D_ans"]),
                    int(deployment["N_T"]), deployment["backend"],
                    float(deployment["score"]), _json(deployment),
                ),
            )
    return str(plan_id)


def record_hardware_run(
    database_path: str,
    run_record: Mapping[str, Any],
    *,
    run_record_path: Optional[str] = None,
) -> str:
    run_id = run_record.get("hardware_run_id") or _prefixed_id(
        "hardware_run",
        {"hardware_plan_id": run_record["hardware_plan_id"], "jobs": run_record["jobs"]},
    )
    absolute, _ = _artifact_fields(run_record_path)
    with _connect(database_path) as connection:
        for job in run_record["jobs"]:
            connection.execute(
                """
                INSERT INTO hardware_jobs(
                    hardware_run_id, job_id, campaign_id, hardware_plan_id,
                    candidate_id, ansatz_depth, trotter_steps, backend, shots,
                    status, submitted_record_path, job_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hardware_run_id, job_id) DO UPDATE SET
                    status=excluded.status,
                    submitted_record_path=excluded.submitted_record_path,
                    job_json=excluded.job_json
                """,
                (
                    run_id, job["job_id"], run_record["campaign_id"],
                    run_record["hardware_plan_id"], job["candidate_id"],
                    int(job["D_ans"]), int(job["N_T"]), job["backend"],
                    int(job["shots"]), str(run_record["status"]), absolute, _json(job),
                ),
            )
    return str(run_id)


def record_hardware_results(
    database_path: str,
    run_record: Mapping[str, Any],
    hardware_result: Mapping[str, Any],
    *,
    result_path: Optional[str] = None,
) -> None:
    absolute, _ = _artifact_fields(result_path)
    run_id = str(hardware_result["hardware_run_id"])
    with _connect(database_path) as connection:
        for item in hardware_result["results"]:
            connection.execute(
                """
                INSERT OR REPLACE INTO hardware_results(
                    hardware_run_id, job_id, candidate_id, backend,
                    observable_real, observable_imag, standard_error_real,
                    standard_error_imag, hardware_error_real,
                    hardware_error_imag, result_path, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, item["job_id"], item["candidate_id"], item["backend"],
                    float(item["hardware_observable_real"]),
                    float(item["hardware_observable_imag"]),
                    float(item["hardware_standard_error_real"]),
                    float(item["hardware_standard_error_imag"]),
                    float(item["hardware_error_real"]),
                    float(item["hardware_error_imag"]), absolute, _json(item),
                ),
            )
        connection.execute(
            "UPDATE simulation_campaigns SET status='hardware_retrieved', updated_at=? WHERE campaign_id=?",
            (_utc_now(), run_record["campaign_id"]),
        )


def record_final_analysis(
    database_path: str,
    summary: Mapping[str, Any],
    *,
    summary_path: Optional[str] = None,
) -> str:
    analysis_id = summary.get("analysis_id") or _prefixed_id(
        "analysis",
        {"campaign_id": summary["campaign_id"],
         "crosscheck_run_id": summary["crosscheck_run_id"],
         "plots": summary["plots"]},
    )
    absolute, sha = _artifact_fields(summary_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO final_analyses(
                analysis_id, campaign_id, crosscheck_run_id, recorded_at,
                crosscheck_status, summary_path, summary_sha256, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(analysis_id) DO UPDATE SET
                recorded_at=excluded.recorded_at,
                summary_path=excluded.summary_path,
                summary_sha256=excluded.summary_sha256,
                summary_json=excluded.summary_json
            """,
            (
                analysis_id, summary["campaign_id"], summary["crosscheck_run_id"],
                _utc_now(), summary["crosscheck_status"], absolute, sha, _json(summary),
            ),
        )
        connection.execute(
            "DELETE FROM analysis_artifacts WHERE analysis_id = ?", (analysis_id,)
        )
        for name, artifact in summary["plots"].items():
            connection.execute(
                """
                INSERT INTO analysis_artifacts(
                    analysis_id, artifact_name, artifact_type,
                    artifact_path, artifact_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    analysis_id, name, "plot", artifact["path"], artifact["sha256"],
                ),
            )
        connection.execute(
            "UPDATE simulation_campaigns SET status='analysis_complete', updated_at=? WHERE campaign_id=?",
            (_utc_now(), summary["campaign_id"]),
        )
    return str(analysis_id)


def campaign_summary(database_path: str, campaign_id: str) -> Dict[str, Any]:
    """Return compact counts and identifiers for one complete simulation campaign."""
    with _connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        campaign = connection.execute(
            "SELECT * FROM simulation_campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if campaign is None:
            raise KeyError("Unknown campaign_id: {}".format(campaign_id))
        summary = dict(campaign)
        for key, table in (
            ("measurement_runs", "measurement_runs"),
            ("classical_references", "classical_references"),
            ("crosscheck_runs", "crosscheck_runs"),
            ("hardware_plans", "hardware_plans"),
            ("hardware_jobs", "hardware_jobs"),
            ("final_analyses", "final_analyses"),
        ):
            summary[key] = connection.execute(
                "SELECT COUNT(*) FROM {} WHERE campaign_id = ?".format(table),
                (campaign_id,),
            ).fetchone()[0]
        summary["measurement_grid_points"] = connection.execute(
            """
            SELECT COUNT(*) FROM measurement_grid_points AS point
            JOIN measurement_runs AS run
              ON run.measurement_run_id = point.measurement_run_id
            WHERE run.campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()[0]
        summary["measurement_recipes"] = connection.execute(
            """
            SELECT COUNT(*) FROM measurement_recipes AS recipe
            JOIN measurement_runs AS run
              ON run.measurement_run_id = recipe.measurement_run_id
            WHERE run.campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()[0]
        summary["hardware_candidates"] = connection.execute(
            """
            SELECT COUNT(*) FROM hardware_candidates AS candidate
            JOIN crosscheck_runs AS checkrun
              ON checkrun.crosscheck_run_id = candidate.crosscheck_run_id
            WHERE checkrun.campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()[0]
        summary["hardware_results"] = connection.execute(
            """
            SELECT COUNT(*) FROM hardware_results AS result
            JOIN hardware_jobs AS job
              ON job.hardware_run_id = result.hardware_run_id
             AND job.job_id = result.job_id
            WHERE job.campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()[0]
        summary["analysis_artifacts"] = connection.execute(
            """
            SELECT COUNT(*) FROM analysis_artifacts AS artifact
            JOIN final_analyses AS analysis
              ON analysis.analysis_id = artifact.analysis_id
            WHERE analysis.campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()[0]
        return summary
