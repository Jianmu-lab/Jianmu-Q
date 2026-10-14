# -*- coding: utf-8 -*-
"""Deterministic support code for the two circuit-design agents."""

import glob
import hashlib
import json
import logging
import os
import re
import sys

from parameter_contract import (
    ParameterContractError,
    load_parameter_contract,
    requires_physical_sector,
    validate_parameter_contract,
)
from machine_contracts import (
    format_machine_contract,
    load_machine_contract,
    validate_mapping_contract,
)
from model_output_ingestion import json_object_candidates


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def extract_single_latex_block(text):
    """Extract a non-authoritative audit view without blocking machine data."""
    blocks = re.findall(r"```(?:latex|tex)\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "Agent 4 produced a validated machine contract. The optional human "
        "audit block was absent.\n\\end{document}"
    )


def extract_design_response(text, validator, *validator_arguments, normalizer=None):
    """Return independent human-audit and machine-contract artifacts."""
    report = extract_single_latex_block(text)
    root_keys = {
        "schema_version", "contract_type", "state_labels",
        "literature_sources", "adapter_interface", "ansatz", "measurement",
        "time_evolution", "reconstruction", "operator_recipes",
    }
    candidates = list(json_object_candidates(text))
    # Raw decoding also finds every nested object. Rank complete envelopes
    # first so a nested object's unrelated error cannot hide the actionable
    # top-level contract error.
    candidates.sort(
        key=lambda item: (
            sum(key in item for key in root_keys),
            len(item),
        ),
        reverse=True,
    )
    accepted = []
    errors = []
    for candidate in candidates:
        try:
            if normalizer is not None:
                candidate = normalizer(candidate)
            accepted.append(validator(candidate, *validator_arguments))
        except (ParameterContractError, TypeError, ValueError) as error:
            errors.append(str(error))
    if not accepted:
        detail = " Top-level error: {}".format(errors[0]) if errors else ""
        raise ParameterContractError(
            "No valid circuit-design JSON contract was found.{}".format(
                detail
            )
        )
    # Model responses sometimes repeat the same corrected contract. The final
    # valid semantic object is authoritative; strict scientific validation has
    # already run on every accepted candidate.
    return report, accepted[-1]


def normalize_literature_sources(candidate):
    """Normalize optional audit metadata without inventing circuit content."""
    sources = candidate.get("literature_sources")
    if isinstance(sources, str):
        sources = [sources]
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        candidate["literature_sources"] = []
        return candidate
    normalized = []
    for source in sources:
        if isinstance(source, str) and source.strip():
            normalized.append({
                "source_path": source.strip(),
                "locator": "unspecified",
                "applicability": "reported by Agent 4",
            })
            continue
        if not isinstance(source, dict):
            continue
        path = source.get("source_path", source.get("path", source.get("source")))
        if not isinstance(path, str) or not path.strip():
            continue
        normalized.append({
            "source_path": path.strip(),
            "locator": str(source.get("locator") or "unspecified"),
            "applicability": str(
                source.get("applicability") or "reported by Agent 4"
            ),
        })
    candidate["literature_sources"] = normalized
    return candidate


def call_model_once(
    api_key, base_url, model_name, system_instruction, prompt,
    raw_response_path=None,
):
    """Reuse a raw checkpoint, or make and checkpoint one paid call."""
    if raw_response_path and os.path.isfile(raw_response_path):
        with open(raw_response_path, "r", encoding="utf-8") as handle:
            return handle.read()
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    if raw_response_path:
        with open(raw_response_path, "w", encoding="utf-8") as handle:
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
    return content


def generator_raw_response_path(
    audit_directory, stage_name, output_tag, regenerate=False
):
    """Return a reusable raw path, or a fresh attempt on explicit regeneration."""
    base = os.path.join(
        audit_directory,
        "{}_GeneratorRaw_{}.txt".format(stage_name, output_tag),
    )
    if not regenerate or not os.path.exists(base):
        return base
    attempt = 2
    while True:
        path = os.path.join(
            audit_directory,
            "{}_GeneratorRaw_{}_Attempt{}.txt".format(
                stage_name, output_tag, attempt
            ),
        )
        if not os.path.exists(path):
            return path
        attempt += 1


def _expand_search_phrase(value, include_components=True):
    """Return useful literal phrases/tokens without interpreting the physics."""
    if not isinstance(value, str) or not value.strip():
        return []
    identifiers = [
        item.replace("_", " ")
        for item in re.findall(
            r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", value
        )
    ]
    normalized = re.sub(r"\s+", " ", value.replace("_", " ")).strip()
    if normalized.lower() in {"none", "null", "unspecified"}:
        return []
    phrases = [normalized] + identifiers
    # Full contract phrases are often more specific than a paper's spelling.
    # Retain distinctive component words as a deterministic recall fallback.
    stopwords = {
        "about", "after", "always", "before", "between", "current",
        "exact", "finite", "from", "into", "model", "none", "only",
        "same", "selected", "state", "that", "theory", "this", "with",
    }
    if include_components:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]*", normalized):
            lower = token.lower().strip(".-")
            if lower in stopwords:
                continue
            if (
                len(lower) >= 8
                or "-" in token
                or (len(token) >= 2 and token.isupper())
            ):
                phrases.append(token)
    return phrases


def _context_search_phrases(value):
    """Extract bounded searchable wording from an existing task specification."""
    phrases = []
    if isinstance(value, dict):
        for key, item in value.items():
            phrases.extend(_expand_search_phrase(str(key)))
            phrases.extend(_context_search_phrases(item))
    elif isinstance(value, list):
        for item in value:
            phrases.extend(_context_search_phrases(item))
    elif isinstance(value, str) and len(value) <= 240:
        phrases.extend(_expand_search_phrase(value))
    return phrases


def model_priority_keywords(parameter_contract, task_context=None):
    """Build task-specific literature terms from already-authoritative inputs."""
    keywords = []
    model = parameter_contract.get("model", {})
    for value in model.values():
        keywords.extend(_expand_search_phrase(value))

    for discretization in parameter_contract.get("discretizations", []):
        keywords.extend(_expand_search_phrase(discretization.get("sector")))
        keywords.extend(_expand_search_phrase(discretization.get("scheme")))
        truncation = discretization.get("truncation", {})
        if isinstance(truncation, dict):
            keywords.extend(_expand_search_phrase(truncation.get("scheme")))
        settings = discretization.get("settings", {})
        if isinstance(settings, dict):
            for key, value in settings.items():
                key_text = str(key).lower()
                if any(term in key_text for term in (
                    "boundary", "dimension", "encoding", "gauge", "group",
                    "map", "ordering", "symmetry", "translation",
                )):
                    keywords.extend(_expand_search_phrase(str(key)))
                    keywords.extend(_expand_search_phrase(value))

    external_states = parameter_contract.get("external_states", {})
    if isinstance(external_states, dict):
        keywords.extend(_context_search_phrases(
            external_states.get("definitions", {})
        ))
    for parameter in parameter_contract.get("parameters", []):
        keywords.extend(_expand_search_phrase(
            parameter.get("name"), include_components=False
        ))
        value = parameter.get("selected_value")
        if isinstance(value, str):
            keywords.extend(_expand_search_phrase(value))
    if task_context is not None:
        keywords.extend(_context_search_phrases(task_context))

    return list(dict.fromkeys(
        keyword for keyword in keywords if isinstance(keyword, str) and keyword.strip()
    ))


_LATEX_SECTION = re.compile(
    r"(?m)^\s*\\(?P<level>section|subsection|subsubsection)\*?"
    r"\{(?P<title>[^{}]*)\}"
)


def _source_sections(text):
    """Keep LaTeX method sections intact, including equations and captions."""
    matches = list(_LATEX_SECTION.finditer(text))
    if matches:
        sections = []
        prefix = text[:matches[0].start()].strip()
        if len(prefix) >= 80:
            sections.append(("preamble", prefix))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.start():end].strip()
            if len(body) >= 80:
                sections.append((match.group("title").strip(), body))
        return sections

    # Plain-text sources have no structural LaTeX headings. Preserve the old
    # bounded paragraph behavior for them.
    sections = []
    for raw in re.split(r"\n\s*\n", text):
        for start in range(0, len(raw), 4000):
            chunk = raw[start:start + 4000].strip()
            if len(chunk) >= 80:
                sections.append(("text", chunk))
    return sections


def _section_score(title, text, targets, priorities, focus):
    lower = text.lower()
    title_lower = title.lower()
    score = sum(min(lower.count(term), 3) for term in targets)
    score += 4 * sum(term in lower for term in priorities)
    score += 5 * sum(min(lower.count(term), 3) for term in focus)
    # A method term in a heading is much stronger evidence than a passing
    # mention in an introduction. Equations/captions keep executable details
    # ahead of prose-only background sections.
    score += 18 * sum(title_lower.count(term) for term in focus)
    score += 24 * sum(
        marker in title_lower
        for marker in (
            "algorithm", "ansatz", "circuit", "measurement", "preparation",
            "simulation",
        )
    )
    score += 5 * sum(
        marker in lower
        for marker in (
            r"\begin{align", r"\begin{equation", r"\caption{",
            "gate count", "circuit depth", "complexity",
        )
    )
    return score


def _keyword_specificity_weight(term):
    """Give exact compound task phrases more weight than generic single words."""
    if " " in term or "-" in term or "+" in term:
        return 8
    if len(term) >= 8:
        return 5
    return 1


def extract_relevant_sections_from_library(
    *,
    library_path,
    target_keywords,
    priority_keywords,
    focus_terms,
    max_chars=30000,
    max_papers=5,
):
    """Retrieve complete method sections from a few task-relevant papers."""
    logger = setup_logger("Agent4_Library_Search")
    files = sorted(
        glob.glob(os.path.join(library_path, "**", "*.txt"), recursive=True)
        + glob.glob(os.path.join(library_path, "**", "*.tex"), recursive=True)
    )
    if not files:
        logger.warning("No .txt/.tex sources found under %s", library_path)
        return ""

    focus = tuple(term.lower() for term in focus_terms)
    targets = tuple(term.lower() for term in target_keywords)
    priorities = tuple(term.lower() for term in priority_keywords)
    seen = set()
    by_paper = {}
    paper_text = {}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeError) as exc:
            logger.warning("Skipping unreadable source %s: %s", path, exc)
            continue
        relative = os.path.relpath(path, library_path)
        paper = relative.split(os.sep, 1)[0]
        paper_text[paper] = paper_text.get(paper, "") + "\n" + text
        for title, section in _source_sections(text):
            normalized = re.sub(r"\s+", " ", section)
            if normalized in seen:
                continue
            seen.add(normalized)
            score = _section_score(title, section, targets, priorities, focus)
            if score > 0:
                by_paper.setdefault(paper, []).append(
                    (score, relative, title, section)
                )

    ranked_papers = []
    for paper, sections in by_paper.items():
        sections.sort(key=lambda item: item[0], reverse=True)
        full_text = paper_text.get(paper, "").lower()
        compatibility_score = sum(
            _keyword_specificity_weight(term)
            for term in priorities
            if term in full_text
        )
        paper_score = sections[0][0] + compatibility_score
        ranked_papers.append(
            (paper_score, compatibility_score, paper, sections)
        )
    ranked_papers.sort(key=lambda item: item[0], reverse=True)
    # Balance method quality with physical compatibility. This prevents a very
    # generic ansatz review from crowding out a slightly lower-scoring paper
    # that matches the current discretization, encoding, and target sector.
    selected = ranked_papers[:min(3, max_papers)]
    selected_names = {item[2] for item in selected}
    by_compatibility = sorted(
        ranked_papers, key=lambda item: (item[1], item[0]), reverse=True
    )
    for item in by_compatibility:
        if len(selected) >= max_papers:
            break
        if item[2] not in selected_names:
            selected.append(item)
            selected_names.add(item[2])
    selected.sort(key=lambda item: item[0], reverse=True)

    output = "### EXTRACTED LOCAL REFERENCES ###\n\n"
    if selected:
        per_paper_budget = max(2000, (max_chars - len(output)) // len(selected))
    else:
        per_paper_budget = 0
    for paper_score, compatibility_score, paper, sections in selected:
        paper_output = (
            "=== Paper: {}; relevance: {}; compatibility: {} ===\n"
        ).format(
            paper, paper_score, compatibility_score
        )
        for score, relative, title, section in sections:
            addition = (
                "--- Source: {}; section: {}; score: {} ---\n{}\n\n"
            ).format(relative, title, score, section)
            if len(paper_output) + len(addition) <= per_paper_budget:
                paper_output += addition
                continue
            if "--- Source:" not in paper_output:
                remaining = per_paper_budget - len(paper_output)
                if remaining > 300:
                    paper_output += (
                        addition[:remaining - 45]
                        + "\n...[section truncated by retriever]...\n"
                    )
            break
        if len(output) + len(paper_output) > max_chars:
            break
        output += paper_output + "\n"
    logger.info(
        "Retrieved %d characters from %d selected papers (%d matched)",
        len(output), len(selected), len(by_paper),
    )
    return output


def load_agent3_inputs(parameter_contract_path, mapping_contract_path):
    if not mapping_contract_path or not parameter_contract_path:
        raise ParameterContractError(
            "Agent4 requires explicit mapping and parameter contracts."
        )
    contract = load_parameter_contract(parameter_contract_path)
    if contract.get("external_states") is None:
        raise ParameterContractError(
            "The parameter contract has no external_states; regenerate Agent1-3."
        )
    mapping = load_machine_contract(
        mapping_contract_path,
        validate_mapping_contract,
        expected_parameter_contract=contract,
    )
    return format_machine_contract(
        mapping,
        validate_mapping_contract,
        expected_parameter_contract=contract,
    ), contract


LATEX_SAFETY_RULES = r"""
Human-audit report rules:
1. Return one concise, complete LaTeX document in one fenced latex block, followed
   by one independent fenced JSON machine contract.
2. Do not use quantikz, tikzpicture, figures, boxed multi-line alignments, or embed
   JSON/verbatim data. Circuit recipes belong only in the JSON contract.
3. Keep only equations needed to audit the selected design. Do not repeat upstream
   derivations, rejected alternatives, implementation code, or runtime policy.
4. The LaTeX report is explanatory and non-authoritative. The JSON contract is the
   sole machine interface and must contain every fact Agent 5 needs.
""".strip()


# Machine-readable handoffs.  These validators intentionally fix only the
# cross-stage interface; all model-specific physics remains dynamic.
STATE_PREPARATION_MARKER = "AGENT4_STATE_PREPARATION_CONTRACT"
OBSERVABLE_MARKER = "AGENT4_OBSERVABLE_CONTRACT"


def _require(condition, message):
    if not condition:
        raise ParameterContractError(message)


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_header(contract, contract_type, expected_parameter_contract):
    _require(isinstance(contract, dict), "Circuit-design handoff must be a JSON object.")
    _require(
        contract.get("schema_version") in ("1.0", "1.1"),
        "schema_version must be '1.0' or '1.1'.",
    )
    _require(contract.get("contract_type") == contract_type, "Unexpected contract_type.")
    if expected_parameter_contract is not None:
        validate_parameter_contract(expected_parameter_contract)


def _validate_labels(labels, expected_parameter_contract):
    _require(
        isinstance(labels, list)
        and labels
        and len(labels) == len(set(labels))
        and all(_nonempty_string(label) for label in labels),
        "state_labels must contain unique nonempty strings.",
    )
    if expected_parameter_contract is not None:
        expected = expected_parameter_contract.get("external_states", {}).get("labels")
        _require(labels == expected, "state_labels must preserve the parameter-contract order.")


def _validate_sources(sources):
    # Literature provenance belongs to the human audit, not the executable
    # interface.  Normalizers retain usable entries, but malformed or absent
    # metadata never blocks circuit construction.
    if not isinstance(sources, list):
        return
    for source in sources:
        if not isinstance(source, dict):
            continue


def normalize_state_preparation_contract(contract):
    """Fill only controller-owned adapter conventions and harmless metadata."""
    if not isinstance(contract, dict):
        return contract
    normalized = json.loads(json.dumps(contract, ensure_ascii=False))
    normalize_literature_sources(normalized)
    ansatz = normalized.get("ansatz")
    if isinstance(ansatz, dict):
        ansatz["depth_symbol"] = "D_ans"
        depths = ansatz.get("recommended_test_depths")
        if isinstance(depths, list):
            ansatz["recommended_test_depths"] = sorted({
                depth for depth in depths
                if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0
            })
        # False is the conservative runtime choice: it disables parameter
        # inheritance unless the designer explicitly guarantees nesting.
        ansatz.setdefault("nested_by_depth", False)
    interface = normalized.get("adapter_interface")
    if not isinstance(interface, dict):
        interface = {}
        normalized["adapter_interface"] = interface
    interface["builder"] = "build_state_preparation"
    interface["returns"] = ["QuantumCircuit", "ordered_parameters"]
    # This flag records exact constructibility, not hardware efficiency. Every
    # finite unitary gate-level adapter required by this contract can be lifted
    # to a controlled Qiskit gate; backend cost is checked downstream.
    if interface.get("unitary") is True:
        interface["supports_controlled"] = True
    return normalized


def normalize_observable_contract(contract):
    """Normalize names owned by the Python adapter API, not physics choices."""
    if not isinstance(contract, dict):
        return contract
    normalized = json.loads(json.dumps(contract, ensure_ascii=False))
    normalize_literature_sources(normalized)
    state_interface = normalized.get("state_preparation_interface")
    if isinstance(state_interface, dict):
        state_interface["treat_as_black_box"] = True
    measurement = normalized.get("measurement")
    if isinstance(measurement, dict):
        recipes = measurement.get("circuit_recipes")
        if not isinstance(measurement.get("circuit_recipe_ids"), list) and isinstance(recipes, list):
            measurement["circuit_recipe_ids"] = [
                item.get("recipe_id") for item in recipes if isinstance(item, dict)
            ]
    evolution = normalized.get("time_evolution")
    if isinstance(evolution, dict) and evolution.get("required") is True:
        evolution["step_symbol"] = "N_T"
        steps = evolution.get("recommended_test_steps")
        if isinstance(steps, list):
            evolution["recommended_test_steps"] = sorted({
                step for step in steps
                if isinstance(step, int) and not isinstance(step, bool) and step > 0
            })
    reconstruction = normalized.get("reconstruction")
    if (
        isinstance(reconstruction, dict)
        and isinstance(measurement, dict)
        and isinstance(measurement.get("components"), list)
    ):
        reconstruction.setdefault(
            "output_components", list(measurement["components"])
        )
    interface = normalized.get("adapter_interface")
    if not isinstance(interface, dict):
        interface = {}
        normalized["adapter_interface"] = interface
    interface["builder"] = "build_observable_measurements"
    return normalized


def validate_state_preparation_contract(contract, expected_parameter_contract=None):
    """Validate only the model-independent state-preparation interface."""
    contract = normalize_state_preparation_contract(contract)
    _validate_header(contract, "state_preparation", expected_parameter_contract)
    _validate_labels(contract.get("state_labels"), expected_parameter_contract)
    _validate_sources(contract.get("literature_sources"))

    ansatz = contract.get("ansatz")
    _require(isinstance(ansatz, dict), "ansatz must be an object.")
    for key in (
        "family_id",
        "depth_symbol",
        "depth_extension_rule",
        "parameter_order_rule",
        "reference_state_rule",
    ):
        _require(
            _nonempty_string(ansatz.get(key)),
            "ansatz.{} must be nonempty.".format(key),
        )
    _require(ansatz["depth_symbol"] == "D_ans", "ansatz.depth_symbol must be D_ans.")
    depths = ansatz.get("recommended_test_depths")
    _require(
        isinstance(depths, list)
        and depths
        and depths == sorted(set(depths))
        and all(
            isinstance(depth, int) and not isinstance(depth, bool) and depth > 0
            for depth in depths
        ),
        "recommended_test_depths must be increasing unique positive integers.",
    )
    _require(
        isinstance(ansatz.get("nested_by_depth"), bool),
        "ansatz.nested_by_depth must be boolean.",
    )

    sector = ansatz.get("physical_sector")
    _require(isinstance(sector, dict), "ansatz.physical_sector must be an object.")
    for key in ("required", "preserved_exactly"):
        _require(
            isinstance(sector.get(key), bool),
            "ansatz.physical_sector.{} must be boolean.".format(key),
        )
    _require(
        _nonempty_string(sector.get("rule")),
        "ansatz.physical_sector.rule must be nonempty.",
    )
    if expected_parameter_contract is not None:
        gauge = expected_parameter_contract["model"]["gauge_structure"].strip().lower()
        gauge_theory = gauge != "none"
        sector_required = requires_physical_sector(expected_parameter_contract)
        _require(
            sector["required"] == sector_required,
            "physical_sector.required disagrees with the requested state sector.",
        )
        if sector_required:
            _require(
                sector["preserved_exactly"] is True,
                "The ansatz must preserve the requested physical sector exactly.",
            )

    interface = contract.get("adapter_interface")
    _require(isinstance(interface, dict), "adapter_interface must be an object.")
    _require(
        interface.get("builder") == "build_state_preparation",
        "adapter_interface.builder must be build_state_preparation.",
    )
    _require(
        interface.get("returns") == ["QuantumCircuit", "ordered_parameters"],
        "adapter_interface.returns has the wrong interface.",
    )
    for key in ("unitary", "supports_inverse", "supports_controlled"):
        _require(
            isinstance(interface.get(key), bool),
            "adapter_interface.{} must be boolean.".format(key),
        )
    _require(
        interface["unitary"] and interface["supports_inverse"],
        "State-preparation adapters must be unitary and invertible.",
    )
    if contract["schema_version"] == "1.1":
        reference_states = ansatz.get("reference_states")
        _require(
            isinstance(reference_states, dict)
            and set(reference_states) == set(contract["state_labels"]),
            "schema 1.1 ansatz.reference_states must define every state label.",
        )
        for label, recipe in reference_states.items():
            _require(
                isinstance(recipe, dict)
                and isinstance(recipe.get("operations"), list)
                and _nonempty_string(recipe.get("basis_state")),
                "reference state {} needs basis_state and operations.".format(label),
            )
        operations = ansatz.get("layer_operations")
        _require(
            isinstance(operations, list) and operations,
            "schema 1.1 ansatz.layer_operations must be nonempty.",
        )
        for operation in operations:
            _require(
                isinstance(operation, dict)
                and _nonempty_string(operation.get("operation")),
                "Every layer operation needs an operation field.",
            )
    return contract


def validate_observable_contract(
    contract, expected_parameter_contract=None,
    expected_observable_specification=None,
):
    """Validate only the model-independent observable-measurement interface."""
    contract = normalize_observable_contract(contract)
    _validate_header(contract, "observable_measurement", expected_parameter_contract)
    _validate_labels(contract.get("state_labels"), expected_parameter_contract)
    _validate_sources(contract.get("literature_sources"))

    state_interface = contract.get("state_preparation_interface")
    _require(
        isinstance(state_interface, dict),
        "state_preparation_interface must be an object.",
    )
    _require(
        state_interface.get("treat_as_black_box") is True,
        "Observable design must treat state preparation as a black box.",
    )
    operations = state_interface.get("required_operations")
    _require(
        isinstance(operations, list)
        and operations
        and all(_nonempty_string(item) for item in operations),
        "required_operations must be a nonempty string list.",
    )

    measurement = contract.get("measurement")
    _require(isinstance(measurement, dict), "measurement must be an object.")
    _require(
        _nonempty_string(measurement.get("protocol_id")),
        "measurement.protocol_id must be nonempty.",
    )
    components = measurement.get("components")
    _require(
        isinstance(components, list)
        and components
        and set(components).issubset({"real", "imag"}),
        "measurement.components must contain real and/or imag.",
    )
    if expected_observable_specification is not None:
        _require(
            components == expected_observable_specification["requested_components"],
            "measurement.components must preserve requested_components order.",
        )
    ancillas = measurement.get("ancilla_qubits")
    _require(
        isinstance(ancillas, int) and not isinstance(ancillas, bool) and ancillas >= 0,
        "measurement.ancilla_qubits must be a nonnegative integer.",
    )
    recipes = measurement.get("circuit_recipe_ids")
    _require(
        isinstance(recipes, list)
        and recipes
        and all(_nonempty_string(item) for item in recipes)
        and len(recipes) == len(set(recipes)),
        "measurement.circuit_recipe_ids must contain unique nonempty strings.",
    )

    evolution = contract.get("time_evolution")
    _require(isinstance(evolution, dict), "time_evolution must be an object.")
    _require(
        isinstance(evolution.get("required"), bool),
        "time_evolution.required must be boolean.",
    )
    if evolution["required"]:
        _require(
            _nonempty_string(evolution.get("method")),
            "time_evolution.method must be nonempty when required.",
        )
        _require(
            evolution.get("step_symbol") == "N_T",
            "time_evolution.step_symbol must be N_T.",
        )
        steps = evolution.get("recommended_test_steps")
        _require(
            isinstance(steps, list)
            and steps == sorted(set(steps))
            and steps
            and all(
                isinstance(step, int) and not isinstance(step, bool) and step > 0
                for step in steps
            ),
            "Executable time evolution requires nonempty increasing positive "
            "recommended_test_steps; exact-mode sentinel 0 and opaque ideal "
            "evolution are not runtime interfaces.",
        )

    reconstruction = contract.get("reconstruction")
    _require(isinstance(reconstruction, dict), "reconstruction must be an object.")
    _require(
        _nonempty_string(reconstruction.get("rule")),
        "reconstruction.rule must be nonempty.",
    )
    _require(
        reconstruction.get("output_components") == components,
        "reconstruction.output_components must match measurement.components.",
    )

    interface = contract.get("adapter_interface")
    _require(isinstance(interface, dict), "adapter_interface must be an object.")
    _require(
        interface.get("builder") == "build_observable_measurements",
        "adapter_interface.builder must be build_observable_measurements.",
    )
    if contract["schema_version"] == "1.1":
        operator_recipes = contract.get("operator_recipes")
        _require(
            isinstance(operator_recipes, list) and operator_recipes,
            "schema 1.1 operator_recipes must be nonempty.",
        )
        for recipe in operator_recipes:
            _require(
                isinstance(recipe, dict)
                and _nonempty_string(recipe.get("id"))
                and isinstance(recipe.get("operations"), list),
                "Every operator recipe needs id and operations.",
            )
        circuit_recipes = measurement.get("circuit_recipes")
        _require(
            isinstance(circuit_recipes, list)
            and [item.get("recipe_id") for item in circuit_recipes] == recipes,
            "measurement.circuit_recipes must define every recipe ID in order.",
        )
        for recipe in circuit_recipes:
            _require(
                isinstance(recipe, dict)
                and recipe.get("component") in components
                and isinstance(recipe.get("operations"), list),
                "Every circuit recipe needs a requested component and operations.",
            )
        if evolution["required"]:
            _require(
                isinstance(evolution.get("step_operations"), list)
                and evolution["step_operations"],
                "Required schema 1.1 evolution needs step_operations.",
            )
        _require(
            isinstance(reconstruction.get("terms"), list)
            and reconstruction["terms"],
            "schema 1.1 reconstruction.terms must be nonempty.",
        )
    return contract


def _extract(text, marker, validator, *validator_arguments):
    pattern = re.compile(
        re.escape(marker + "_BEGIN")
        + r"\s*(.*?)\s*"
        + re.escape(marker + "_END"),
        re.DOTALL,
    )
    errors = []
    for block in pattern.findall(text):
        try:
            return validator(json.loads(block), *validator_arguments)
        except (json.JSONDecodeError, ParameterContractError) as exc:
            errors.append(str(exc))
    detail = " Last error: {}".format(errors[-1]) if errors else ""
    raise ParameterContractError(
        "No valid {} block was found.{}".format(marker, detail)
    )


def extract_state_preparation_contract(text, expected_parameter_contract=None):
    return _extract(
        text,
        STATE_PREPARATION_MARKER,
        validate_state_preparation_contract,
        expected_parameter_contract,
    )


def extract_observable_contract(
    text, expected_parameter_contract=None, expected_observable_specification=None
):
    return _extract(
        text,
        OBSERVABLE_MARKER,
        validate_observable_contract,
        expected_parameter_contract,
        expected_observable_specification,
    )


def save_contract(contract, path, validator, expected_parameter_contract=None):
    contract = validator(contract, expected_parameter_contract)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_state_preparation_contract(path, expected_parameter_contract=None):
    with open(path, "r", encoding="utf-8") as handle:
        return validate_state_preparation_contract(
            json.load(handle), expected_parameter_contract
        )


def load_observable_contract(
    path, expected_parameter_contract=None, expected_observable_specification=None
):
    with open(path, "r", encoding="utf-8") as handle:
        return validate_observable_contract(
            json.load(handle), expected_parameter_contract,
            expected_observable_specification,
        )


def format_state_preparation_contract(contract, expected_parameter_contract=None):
    contract = validate_state_preparation_contract(contract, expected_parameter_contract)
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def format_observable_contract(
    contract, expected_parameter_contract=None, expected_observable_specification=None
):
    contract = validate_observable_contract(
        contract, expected_parameter_contract, expected_observable_specification
    )
    return json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True)


def contract_sha256(contract, validator, expected_parameter_contract=None):
    contract = validator(contract, expected_parameter_contract)
    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
