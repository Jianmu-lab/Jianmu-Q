# -*- coding: utf-8 -*-
import os
import sys
import logging
import re
import time
import argparse
import json
import copy
import glob
# 导入标准的 OpenAI 库
from openai import OpenAI
from audit_report_support import publish_audit_report
from machine_contracts import (
    extract_machine_contract,
    extract_json_contract_candidate,
    format_machine_contract,
    load_machine_contract,
    save_machine_contract,
    validate_discretization_contract,
    validate_mapping_contract,
)
from parameter_contract import (
    ParameterContractError,
    format_parameter_contract,
    load_parameter_contract,
    requires_physical_sector,
)
from pipeline_support import (
    atomic_json_write,
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)
from review_protocol import (
    AUTOMATIC_REVISION_POLICY_VERSION,
    automatic_revision_count,
    extract_single_artifact_review,
    next_review_paths,
    render_review_audit,
)

# ==========================================
# 通用日志与辅助函数
# ==========================================
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def extract_latex(text):
    tb = chr(96) * 3
    pattern = re.compile(f'{tb}(?:latex|tex)?\\s*(.*?)\\s*{tb}', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def reusable_or_new_mapping_draft_paths(report_directory, contract_directory, output_tag):
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_Attempt{}".format(attempt)
        paths = (
            os.path.join(report_directory, "Agent3_Draft_{}{}.tex".format(output_tag, suffix)),
            os.path.join(
                contract_directory,
                "Agent3_Draft_MappingContract_{}{}.json".format(output_tag, suffix),
            ),
        )
        existence = [os.path.isfile(path) for path in paths]
        if all(existence) or not any(existence):
            return paths
        attempt += 1


def next_generator_raw_path(audit_directory, output_tag):
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_Attempt{}".format(attempt)
        path = os.path.join(
            audit_directory,
            "Agent3_GeneratorRaw_{}{}.txt".format(output_tag, suffix),
        )
        if not os.path.exists(path):
            return path
        attempt += 1


def latest_generator_raw_path(audit_directory, output_tag):
    paths = glob.glob(os.path.join(
        audit_directory, "Agent3_GeneratorRaw_{}*.txt".format(output_tag)
    ))
    return max(paths, key=os.path.getmtime) if paths else None


def prepare_mapping_contract(mapping, parameter_contract):
    mapping = copy.deepcopy(mapping)
    mapping["schema_version"] = "1.0"
    mapping["contract_type"] = "qubit_mapping"
    mapping.pop("discretization_contract_sha256", None)
    mapping.pop("parameter_contract_sha256", None)
    # ParameterContract remains the sole authority for external-state identity.
    mapping.pop("external_states", None)
    sector = mapping.get("physical_sector")
    if isinstance(sector, dict):
        sector["required"] = requires_physical_sector(parameter_contract)
    return mapping


def normalize_mapping_contract(mapping, parameter_contract, discretization_contract):
    mapping = prepare_mapping_contract(mapping, parameter_contract)
    return validate_mapping_contract(
        mapping,
        expected_discretization_contract=discretization_contract,
        expected_parameter_contract=parameter_contract,
    )


def _merge_identified_entries(base, corrections, identity_keys):
    merged = copy.deepcopy(base) if isinstance(base, list) else []
    if not isinstance(corrections, list):
        return merged
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        identity_key = next(
            (key for key in identity_keys if correction.get(key) is not None), None
        )
        match = next((
            index for index, existing in enumerate(merged)
            if isinstance(existing, dict)
            and identity_key is not None
            and existing.get(identity_key) == correction.get(identity_key)
        ), None)
        if match is None:
            merged.append(copy.deepcopy(correction))
        else:
            merged[match] = _merge_mapping(merged[match], correction)
    return merged


def _merge_mapping(base, correction):
    """Recursively merge one local correction without erasing sibling facts."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in correction.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _replace_indexed_terms(base, corrections, label, value_key="term"):
    terms = copy.deepcopy(base) if isinstance(base, list) else []
    if not isinstance(corrections, list):
        raise ParameterContractError("{} corrections must be a list.".format(label))
    for correction in corrections:
        if not isinstance(correction, dict):
            raise ParameterContractError("{} correction must be an object.".format(label))
        index = correction.get("index")
        term = correction.get(value_key)
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(term, dict):
            raise ParameterContractError(
                "{} correction requires integer index and complete {}.".format(
                    label, value_key
                )
            )
        if index == len(terms):
            terms.append(copy.deepcopy(term))
        elif 0 <= index < len(terms):
            terms[index] = copy.deepcopy(term)
        else:
            raise ParameterContractError("{} correction index is out of range.".format(label))
    return terms


def apply_mapping_correction(
    mapping_contract, correction_text, parameter_contract, discretization_contract
):
    """Merge only identified Agent3 entries, then validate the whole mapping."""
    correction = extract_json_contract_candidate(
        correction_text, "qubit_mapping_correction"
    )
    allowed = {
        "contract_type", "hamiltonian", "hamiltonian_terms", "observable_terms",
        "observable", "conserved_quantities", "circuit_definitions", "qubits",
        "physical_sector",
    }
    unsupported = sorted(set(correction) - allowed)
    if unsupported:
        raise ParameterContractError(
            "Unsupported qubit-mapping correction fields: {}.".format(
                ", ".join(unsupported)
            )
        )
    merged = copy.deepcopy(mapping_contract)
    if "hamiltonian" in correction:
        hamiltonian_correction = correction["hamiltonian"]
        if not isinstance(hamiltonian_correction, dict):
            raise ParameterContractError("hamiltonian correction must be an object.")
        if "terms" in hamiltonian_correction:
            raise ParameterContractError(
                "Correct Pauli terms through indexed hamiltonian_terms entries."
            )
        merged["hamiltonian"] = _merge_mapping(
            merged.get("hamiltonian"), hamiltonian_correction
        )
    if "hamiltonian_terms" in correction:
        merged["hamiltonian"]["terms"] = _replace_indexed_terms(
            merged["hamiltonian"].get("terms"),
            correction["hamiltonian_terms"],
            "hamiltonian_terms",
        )
    if "observable_terms" in correction:
        merged["observable"]["terms"] = _replace_indexed_terms(
            merged["observable"].get("terms"),
            correction["observable_terms"],
            "observable_terms",
        )
    if "observable" in correction:
        observable_correction = correction["observable"]
        if not isinstance(observable_correction, dict) or set(observable_correction) != {"ordered_factorizations"}:
            raise ParameterContractError(
                "observable correction may contain only ordered_factorizations."
            )
        merged["observable"]["ordered_factorizations"] = _replace_indexed_terms(
            merged["observable"].get("ordered_factorizations"),
            observable_correction["ordered_factorizations"],
            "observable.ordered_factorizations",
            value_key="factorization",
        )
    if "conserved_quantities" in correction:
        merged["conserved_quantities"] = _merge_identified_entries(
            merged.get("conserved_quantities"),
            correction["conserved_quantities"],
            ("id",),
        )
    if "circuit_definitions" in correction:
        merged["circuit_definitions"] = _merge_identified_entries(
            merged.get("circuit_definitions"),
            correction["circuit_definitions"],
            ("id", "name", "type"),
        )
    for key in ("qubits", "physical_sector"):
        if isinstance(correction.get(key), dict):
            merged.setdefault(key, {}).update(correction[key])
    return normalize_mapping_contract(
        merged, parameter_contract, discretization_contract
    )


def discretization_mapping_projection(discretization_contract, parameter_contract):
    """Return the canonical-only Agent2 payload consumed by Agent3.

    ``display_latex`` is deliberately excluded: it is a human mirror and must
    never become an independent source of physics for a downstream agent.
    """
    validated = validate_discretization_contract(
        discretization_contract,
        expected_parameter_contract=parameter_contract,
    )
    projection = copy.deepcopy(validated)
    for definition in projection.get("definitions", []):
        if isinstance(definition, dict):
            definition.pop("display_latex", None)
    return json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True)


def render_mapping_audit(mapping):
    """Render a short deterministic human audit from validated machine data."""
    qubits = mapping["qubits"]["count"]
    h_terms = len(mapping["hamiltonian"]["terms"])
    observable_terms = len(mapping["observable"]["terms"])
    sector_required = "yes" if mapping["physical_sector"]["required"] else "no"
    validation_tests = len(mapping["physical_sector"].get("validation_tests", []))
    return r"""\documentclass{article}
\begin{document}
\section*{Agent 3 Qubit-Mapping Audit}
The authoritative result is the separately validated JSON mapping contract.
This human audit contains no independent formula mirror.
\begin{itemize}
\item Logical qubits: %d.
\item Explicit full-width Hamiltonian Pauli terms: %d.
\item Observable terms: %d.
\item Physical-sector restriction required: %s.
\item Physical-sector validation tests: %d.
\end{itemize}
\end{document}
""" % (qubits, h_terms, observable_terms, sector_required, validation_tests)


def extract_translation_response(text, parameter_contract, discretization_contract):
    mapping = extract_json_contract_candidate(text, "qubit_mapping")
    mapping = normalize_mapping_contract(
        mapping, parameter_contract, discretization_contract
    )
    return render_mapping_audit(mapping), mapping


# ==========================================
# Agent 3: 量子语言翻译智能体
# ==========================================
class QuantumTranslatorAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol", max_retries=2):
        # 实例化标准的 OpenAI 客户端，支持传入自定义的 API 网关地址
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_retries = max_retries
        self.logger = setup_logger("Agent3_QuantumTranslator")
        
        # 将 Agent 3 塑造成精通底层量子算法的专家
        self.system_instruction = (
            "You are a top-tier Quantum Computing Algorithm Engineer and Physicist. "
            "Your expertise is mapping lattice field theories to qubit representations (Jordan-Wigner, Bravyi-Kitaev, block encoding, and so on)."
            "You DO NOT use high-level abstractions like 'black box block encoding'. "
        )
        
    def _call_model_with_retry(self, prompt, system_instruction):
        """处理大模型的重试逻辑与消息构建 (兼容 OpenAI 格式)"""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                # 统一使用流式输出防止超时，移除 temperature 限制
                response_stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True
                )
                full_content = ""
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
            except Exception as e:
                self.logger.warning(f"网络或服务器异常 (可能是 503 拥挤): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = 5 * (attempt + 1)
                    self.logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("达到最大重试次数，调用失败。")
                    sys.exit(1)

    def translate_to_quantum_language(
        self, agent2_report, parameter_contract, discretization_contract,
        raw_response_path=None,
    ):
        self.logger.info("接收到 Agent 2 的格点化报告，开始翻译为量子计算机可执行语言 (Pauli 算符与线路)...")
        
        tb = chr(96) * 3 
        labels = parameter_contract["external_states"]["labels"]
        mapping_example = {
            "schema_version": "1.0",
            "contract_type": "qubit_mapping",
            "qubits": {
                "count": "replace with positive integer",
                "ordering": ["replace with every logical qubit name in order"],
                "pauli_label_order": "qiskit_little_endian",
            },
            "hamiltonian": {
                "representation": "pauli_sum",
                "terms": [
                    {
                        "coefficient": "exact symbolic or numeric coefficient",
                        "pauli": "exact full-length Pauli label",
                    }
                ],
            },
            "conserved_quantities": [
                {
                    "id": "same identifier as Agent2 lattice_symmetries",
                    "lattice_symmetry_status": "exact|broken_by_boundary_or_discretization|continuum_only",
                    "representation": "physically correct operator or generator representation",
                    "operators_or_generators": "explicit data appropriate to a single operator, an operator family, a qubit permutation, or an antiunitary symmetry",
                    "target_eigenvalues": {label: "mapped exact eigenvalue or unspecified" for label in labels},
                }
            ],
            "observable": {
                "representation": "explicit_operator_sum",
                "terms": [{"coefficient": "exact coefficient", "operations": []}],
            },
            "external_states": {
                "labels": labels,
                "definitions": parameter_contract["external_states"]["definitions"],
                "equivalent_groups": parameter_contract["external_states"]["equivalent_groups"],
            },
            "physical_sector": {
                "required": requires_physical_sector(parameter_contract),
                "basis_rule": "exact implementation-ready subspace construction, using computational-basis indices only when the sector is diagonal in that basis",
                "validation_tests": [
                    {"id": item["id"], "rule": "exact qubit-space pass criterion"}
                    for item in parameter_contract.get("validation_obligations", [])
                ],
            },
            "circuit_definitions": [],
        }
        
        translation_prompt = rf"""
Here is the authoritative finite-dimensional lattice-discretization contract generated by Agent 2:
{tb}json
{agent2_report}
{tb}

Here is Agent 2's AUTHORITATIVE model-specific parameter contract:
{tb}json
{format_parameter_contract(parameter_contract)}
{tb}

The contract is immutable. Use its parameter values, domains, and constraints in
the qubit mapping. Do not rename parameters, change selected values, weaken
constraints, or invent replacements. Record implementation data only; do not
return hidden reasoning or a prose derivation.

Agent3 owns the exact qubit mapping, not the later simulation algorithm. Keep
the exact target operator, continuous path, orientation and ordering, but do
not turn an upstream statement that Agent2 did not choose a time step or
product formula into `product_formula: forbidden` or another ban on Agent4.
Runtime quadrature, Trotter/product-formula choice and convergence controls
belong to Agent4, provided their limiting target is this exact mapped operator.

Your task is to translate this lattice theory strictly into the language of quantum computer (qubit mappings).

Systematically execute the following steps and encode their results in JSON:

a) Qubit Mapping Strategy: Explicitly define how you map the discretized fermionic degrees of freedom (e.g., using Jordan-Wigner, Bravyi-Kitaev transformations and so on) and the truncated gauge field degrees of freedom to physical qubits. Clearly state the total number of qubits required.
b) Hamiltonian Translation ($H_{{qc}}$): Express the lattice Hamiltonian purely as a linear combination of Pauli tensor products (Pauli strings) over the assigned qubits.
For every Pauli coefficient that depends on an Agent2 runtime parameter, keep
that dependency as an explicit algebraic expression in `coefficient`, using
the exact Parameter Contract name (for example `bare_quark_mass`,
`bare_four_fermion_coupling`, `lattice_spacing`, or `wilson_parameter`). Do not
evaluate such an expression at the current selected values. A plain numerical
coefficient is allowed only when it is genuinely independent of all runtime
parameters. Agent5 must translate these expressions literally and must not be
asked to reconstruct missing physics from old numerical coefficients.
c) Conserved Quantities Translation ($Q_{{qc}}$): Map every lattice conserved
quantity using its physically correct qubit representation. A single additive
operator may be a Pauli sum, but a family of local constraints must remain a
family, a spatial symmetry may be a qubit permutation/circuit, and an
antiunitary symmetry must include complex conjugation. Never replace any of
these by an identity, zero operator, or one summed Pauli operator merely to fit
the example envelope. Include only the representation-specific fields needed
to define the actual operator or generators explicitly.
c.1) Preserve Agent2's lattice-symmetry classification and external-state
target eigenvalues exactly in each `conserved_quantities` entry. Independently
verify the mapped operator relation with H_qc. Never promote a
boundary/discretization-broken or continuum-only symmetry to an exact qubit
constraint, and never infer a discrete-symmetry eigenvalue merely from another
charge quantum number.
d) Observable and External-State Translation: Map $O_{{qc}}$, preserve every named state, and preserve every coefficient, bra label, ket label, and sign in observable.matrix_elements. Use the Parameter Contract labels as the only machine state identifiers; a lattice decoration such as `hadron_lat` in an upstream formula denotes the existing `hadron` state and must not create a new label. When that list is absent, preserve the legacy $\langle\phi|O|\psi\rangle$ target. Do not identify state encodings or preparation requirements unless Agent 1 explicitly defined identical states.
d.1) For every external state that specifies quantum numbers and a zero-based
sector level index, map the relevant conserved operators and eigenvalues exactly.
Make `physical_sector.basis_rule` sufficient to construct that label's requested
subspace exactly. Use computational-basis enumeration only when the defining
constraints are diagonal in that basis; otherwise specify an exact projector,
stabilizer construction, symmetry-adapted basis, or orthonormal basis rule.
Never substitute the unrestricted ground sector.
e) If you use Block Encoding method to map gauge fields to qubits, you are STRICTLY FORBIDDEN from treating it as a mathematical black box. You MUST explicitly describe the quantum circuit operations. Define the ancilla state preparation unitaries (PREP), the select operators (SEL) mapped to controlled-Pauli gates, and how they combine to encode the gauge fields as well as the quantities that are the function of gauge fields.
f) Runtime Physical-Sector Specification: Give the mapped constraint operators,
target eigenvalues or allowed subspace, and a precise nonempty/pass criterion
needed to construct the physical basis. Do not copy every Parameter Contract
validation obligation into this mapping; that contract remains the sole source.
Do not write Qiskit code or invent missing physics.

OUTPUT FORMAT REQUIREMENTS:
Output exactly one fenced JSON block and no LaTeX or conversational text. The
controller renders the short human audit deterministically after validation.

This JSON is the sole authoritative machine interface. Replace every scientific
placeholder in the envelope. The controller injects schema/type and immutable
external states, so do not reason about or compare identity metadata.

For every Hamiltonian term, `pauli` MUST be one explicit string of exactly
`qubits.count` characters drawn from I/X/Y/Z in Qiskit little-endian order.
Fully expand all repeated local terms for the current lattice size in
`hamiltonian.terms`; symbolic Pauli labels, loops and templates are forbidden
inside that list. Separately, when the lattice contract supplies an indexed
construction, record its exact parameterized mapping in `circuit_definitions`:
site/link index ranges, mode-to-qubit rule, boundary terms, Pauli-support rule
and parameter-dependent coefficients. This definition must reproduce the
expanded current-size list exactly and let Agent5 instantiate the same mapping
at another supported size without inferring physics. If no such construction
is determined, state the fixed-size limitation explicitly; do not invent one.
If the scientific mapping is genuinely underdetermined,
state the missing information in an otherwise schema-valid contract rather
than inventing it.

Required envelope:
{tb}json
{json.dumps(mapping_example, ensure_ascii=False, indent=2)}
{tb}

"""
        final_latex_content = self._call_model_with_retry(translation_prompt, self.system_instruction)
        if raw_response_path:
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(final_latex_content)
                if final_latex_content and not final_latex_content.endswith("\n"):
                    handle.write("\n")
        
        self.logger.info("Agent 3 量子语言翻译完成！")
        return extract_translation_response(
            final_latex_content, parameter_contract, discretization_contract
        )

    def revise_from_review(
        self,
        agent2_report,
        mapping_contract,
        parameter_contract,
        discretization_contract,
        findings,
        raw_response_path=None,
    ):
        """Apply one bounded correction to identified mapping entries."""
        tb = chr(96) * 3
        prompt = rf"""
Correct only the Agent3 entries named by the fatal findings and their necessary
dependent entries. Do not regenerate the complete mapping.

Authoritative lattice-discretization reference:
{tb}json
{agent2_report}
{tb}

Immutable Parameter Contract:
{tb}json
{format_parameter_contract(parameter_contract)}
{tb}

Existing qubit-mapping candidate (preserve every unaffected entry):
{tb}json
{json.dumps(mapping_contract, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Fatal review findings to correct:
{tb}json
{json.dumps(findings, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Return exactly one JSON correction object with
`contract_type: qubit_mapping_correction`. Use `hamiltonian_terms` or
`observable_terms` entries of the form `{{"index": N, "term": COMPLETE_TERM}}`;
when a finding names non-term metadata inside `hamiltonian`, include only that
local nested `hamiltonian` object and never place `terms` inside it;
for a corrected ordered observable factorization use
`"observable": {{"ordered_factorizations": [{{"index": N,
"factorization": COMPLETE_FACTORIZATION}}]}}`;
identify corrected conserved quantities by `id` and corrected circuit
definitions by their existing `id`, `name`, or `type`. Include `qubits` or
`physical_sector` only when those fields are directly implicated. Omit every
unchanged entry and section. This is a small stage-specific correction
envelope, not a general JSON Patch.
Correct every fatal finding while preserving all valid mapping decisions,
full-width Pauli terms, named-state definitions, parameter values, and
physical-sector validation tests.
Preserve every lattice-symmetry classification and mapped target eigenvalue;
change one only when the reviewer identified a concrete error in it.
"""
        content = self._call_model_with_retry(prompt, self.system_instruction)
        if raw_response_path:
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(content)
                if content and not content.endswith("\n"):
                    handle.write("\n")
        corrected = apply_mapping_correction(
            mapping_contract, content, parameter_contract, discretization_contract
        )
        return render_mapping_audit(corrected), corrected

# ==========================================
# Agent 3.5: 量子翻译审查智能体 (Reviewer)
# ==========================================
class QuantumReviewerAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.logger = setup_logger("Agent3.5_QuantumReviewer")
        
        self.system_instruction = (
            "You are a meticulous Quantum Algorithm Reviewer. Your job is to catch algebraic hallucinations "
            "in qubit mappings (e.g., Jordan-Wigner string errors), Pauli algebra, and block encoding circuit logic. "
            "Review scientific and mathematical correctness only; never review formatting, provenance, "
            "workflow authorization, historical compliance, or stage ownership."
        )

    def review(
        self, agent2_report, mapping_contract, parameter_contract,
        discretization_contract, raw_response_path, review_status_path,
        deterministic_findings=None,
    ):
        tb = chr(96) * 3
        deterministic_findings = list(deterministic_findings or [])
        prompt = rf"""
Review the complete Agent3 mapping candidate. Do not regenerate the
mapping and do not return LaTeX. Return only one compact JSON review.

Authoritative lattice reference:
{tb}json
{agent2_report}
{tb}

Qubit-mapping candidate:
{tb}json
{json.dumps(mapping_contract, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Deterministic executable-interface findings already known to the controller:
{tb}json
{json.dumps(deterministic_findings, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}
These findings are part of the same review. Continue through every scientific
check and report all other errors; do not stop after confirming the first one.

The immutable model-specific parameter contract from Agent 2:
{tb}json
{format_parameter_contract(parameter_contract)}
{tb}

Please strictly verify the following:
1. Qubit Mapping: If Jordan-Wigner is used, are the Z-strings correctly applied to preserve fermionic anti-commutation? Check the indices strictly.
2. Pauli Algebra: Does the linear combination of Pauli strings exactly reconstruct the original lattice Hamiltonian? 
3. If Block Encoding/LCU is actually used, verify its circuit logic, including
PREP coefficient amplitudes and SEL controlled operations. Otherwise mark this
check not applicable and do not demand such a construction.
4. Parameter Consistency: Verify that the qubit mapping and Pauli operators obey every fatal parameter domain and compatibility constraint.
For every Hamiltonian coefficient that depends on a runtime parameter, verify
that the mapping retains the explicit algebraic dependence using exact Agent2
Parameter Contract names. Treat premature substitution of the current
numerical values as fatal because it makes an otherwise reusable mapping stale
when only physical parameters change. Numerical literals remain valid for
genuinely parameter-independent coefficients.
5. Runtime Validation Mapping: Verify that every validation-obligation ID has an exact, physically correct qubit-space mathematical test.
6. Independent External States: Verify that all named states remain separate unless the parameter contract explicitly identifies them, and that every matrix-element coefficient and bra/ket label is preserved.
7. Observable Specialization: Verify term-by-term that no Wilson transporter, Jordan--Wigner string, time-evolution factor, coefficient, bra, or ket is dropped.
8. Symmetry Mapping: Verify that every Agent2 lattice symmetry has the correct
qubit operator and unchanged classification, and that target eigenvalues are
present only for states whose authoritative definitions fix them. Distinguish
a single Pauli-sum operator from a family of constraint generators, a qubit
permutation, and an antiunitary action; reject identity/zero placeholders that
do not implement the stated symmetry.

This is a read-only review. Never rewrite or patch the mapping contract. Use
verdict `needs_revision` only for a concrete scientific mapping error and mark
that finding `fatal`. Use `warning` only for a nonblocking scientific
limitation. Do not report formatting, schema style, metadata, provenance,
authorization, ownership, or historical-policy findings.
For every fatal finding, identify the smallest affected object explicitly:
give the Hamiltonian/observable term index, conserved-quantity `id`, circuit
definition identifier, or physical-sector field, plus only the dependent
entries that must change with it.

Finish all eight checks before returning. Report every independently discovered
scientific error in this one response, including errors in other terms of the
same operator family. Do not return early after the first failure.

Return one compact JSON object and no other text. The controller owns and
injects schema/version/type metadata; provide only the semantic decision:
{tb}json
{{
  "verdict": "accept",
  "findings": []
}}
{tb}
"""
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        self.logger.info("正在将 mapping contract 送交只读 Reviewer（单次 review）...")
        full_content = ""
        model_calls = 1
        start_time = time.time()
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name, messages=messages, stream=True
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(full_content)
                if full_content and not full_content.endswith("\n"):
                    handle.write("\n")
            decision = extract_single_artifact_review(full_content, "agent3_review")
            combined_findings = list(deterministic_findings)
            known_messages = {
                item.get("message") for item in combined_findings
                if isinstance(item, dict)
            }
            for finding in decision["findings"]:
                if finding.get("message") not in known_messages:
                    combined_findings.append(finding)
                    known_messages.add(finding.get("message"))
            decision["findings"] = combined_findings
            if deterministic_findings:
                decision["verdict"] = "needs_revision"
                reviewed = mapping_contract
            else:
                reviewed = normalize_mapping_contract(
                    mapping_contract, parameter_contract, discretization_contract
                )
            blocked = decision["verdict"] == "needs_revision"
            record = {
                "review_status_schema_version": "1.0", "stage": "agent3",
                "status": "accepted" if not blocked else "needs_revision",
                "verdict": decision["verdict"], "findings": decision["findings"],
                "model_calls": model_calls, "raw_response_path": os.path.abspath(raw_response_path),
                "machine_chain_blocked": blocked,
            }
            atomic_json_write(review_status_path, record)
            if blocked:
                self.logger.error("Reviewer 发现 fatal 科学问题；mapping 保持不变。")
                return render_review_audit(decision), reviewed, record
            self.logger.info("Reviewer 只读审查完成，耗时: %.2f 秒", time.time() - start_time)
            return render_review_audit(decision), reviewed, record
        except Exception as error:
            if not os.path.isfile(raw_response_path):
                with open(raw_response_path, "w", encoding="utf-8") as handle:
                    handle.write(full_content)
            if deterministic_findings:
                decision = {
                    "verdict": "needs_revision",
                    "findings": deterministic_findings,
                }
                record = {
                    "review_status_schema_version": "1.0", "stage": "agent3",
                    "status": "needs_revision", "verdict": "needs_revision",
                    "findings": deterministic_findings,
                    "error": "{}: {}".format(type(error).__name__, error),
                    "model_calls": model_calls,
                    "raw_response_path": os.path.abspath(raw_response_path),
                    "machine_chain_blocked": True,
                }
                atomic_json_write(review_status_path, record)
                return render_review_audit(decision), mapping_contract, record
            record = {
                "review_status_schema_version": "1.0", "stage": "agent3",
                "status": "review_inconclusive_nonblocking",
                "error": "{}: {}".format(type(error).__name__, error),
                "model_calls": model_calls, "raw_response_path": os.path.abspath(raw_response_path),
                "machine_chain_blocked": False,
            }
            atomic_json_write(review_status_path, record)
            self.logger.warning("Reviewer 输出无法结构化；保留已验证 mapping 并继续: %s", error)
            fallback = {
                "verdict": "accept",
                "findings": [{
                    "severity": "warning",
                    "message": "Reviewer response could not be parsed; raw response was retained.",
                }],
            }
            return render_review_audit(fallback), mapping_contract, record


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Agent3 qubit translation and review.")
    parser.add_argument("--discretization-contract", required=True)
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    return parser.parse_args()


# ==========================================
# 独立测试工作流
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    report_directory = categorized_output_directory(args.output_root, "reports")
    contract_directory = categorized_output_directory(args.output_root, "contracts")
    audit_directory = categorized_output_directory(args.output_root, "audit")
    YOUR_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not YOUR_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY before running this agent.")
    parameter_contract = load_parameter_contract(args.parameter_contract)
    discretization_contract = load_machine_contract(
        args.discretization_contract,
        validate_discretization_contract,
        expected_parameter_contract=parameter_contract,
    )
    agent2_report_text = discretization_mapping_projection(
        discretization_contract, parameter_contract
    )
    print("成功读取 Agent 2 格点离散化机器契约")

    draft_file, draft_mapping_file = reusable_or_new_mapping_draft_paths(
        report_directory, contract_directory, output_tag
    )
    generator_raw_file = next_generator_raw_path(audit_directory, output_tag)

    # A valid mapping draft is a paid-stage checkpoint.
    agent3 = QuantumTranslatorAgent(
        api_key=YOUR_API_KEY,
        base_url=args.base_url,
        model_name=args.model
    )
    draft_preflight_findings = []
    if os.path.isfile(draft_file) and os.path.isfile(draft_mapping_file):
        with open(draft_file, "r", encoding="utf-8") as handle:
            agent3_draft_latex = handle.read()
        with open(draft_mapping_file, "r", encoding="utf-8") as handle:
            draft_mapping_contract = normalize_mapping_contract(
                json.load(handle), parameter_contract, discretization_contract
            )
        # Persist deterministic controller-owned normalization so subsequent
        # resumes see the same checkpoint without another compatibility step.
        save_machine_contract(
            draft_mapping_contract,
            draft_mapping_file,
            validate_mapping_contract,
            expected_discretization_contract=discretization_contract,
            expected_parameter_contract=parameter_contract,
        )
        agent3.logger.info("读取已验证 Agent3 mapping checkpoint；跳过付费翻译。")
    else:
        replayed_raw = latest_generator_raw_path(audit_directory, output_tag)
        recovered = None
        invalid_replayed_mapping = None
        invalid_replayed_error = None
        if replayed_raw:
            try:
                with open(replayed_raw, "r", encoding="utf-8") as handle:
                    recovered = extract_translation_response(
                        handle.read(), parameter_contract, discretization_contract
                    )
                agent3.logger.info(
                    "零成本重新验证已保存的 Agent3 generator 原始响应: %s",
                    replayed_raw,
                )
            except ParameterContractError as error:
                with open(replayed_raw, "r", encoding="utf-8") as handle:
                    invalid_replayed_mapping = prepare_mapping_contract(
                        extract_json_contract_candidate(
                            handle.read(), "qubit_mapping"
                        ),
                        parameter_contract,
                    )
                invalid_replayed_error = error
                agent3.logger.warning(
                    "已保存的 Agent3 响应未通过预检；将只修正错误及关联项: %s",
                    error,
                )
        if recovered is not None:
            agent3_draft_latex, draft_mapping_contract = recovered
        elif invalid_replayed_mapping is not None:
            draft_mapping_contract = invalid_replayed_mapping
            agent3_draft_latex = render_mapping_audit(draft_mapping_contract)
            draft_preflight_findings = [{
                "severity": "fatal",
                "message": str(invalid_replayed_error),
            }]
        else:
            try:
                agent3_draft_latex, draft_mapping_contract = (
                    agent3.translate_to_quantum_language(
                        agent2_report_text,
                        parameter_contract,
                        discretization_contract,
                        raw_response_path=generator_raw_file,
                    )
                )
            except ParameterContractError as error:
                # Preserve the complete candidate. The scientific reviewer sees
                # this finding together with the whole mapping before Agent3
                # performs one consolidated local correction.
                with open(generator_raw_file, "r", encoding="utf-8") as handle:
                    draft_mapping_contract = prepare_mapping_contract(
                        extract_json_contract_candidate(
                            handle.read(), "qubit_mapping"
                        ),
                        parameter_contract,
                    )
                agent3_draft_latex = render_mapping_audit(draft_mapping_contract)
                draft_preflight_findings = [{
                    "severity": "fatal",
                    "message": str(error),
                }]
                agent3.logger.warning(
                    "Agent3 可执行性预检发现问题；将与完整科学审查合并: %s",
                    error,
                )
        if not draft_preflight_findings:
            with open(draft_file, "w", encoding="utf-8") as handle:
                handle.write(agent3_draft_latex)
            save_machine_contract(
                draft_mapping_contract,
                draft_mapping_file,
                validate_mapping_contract,
                expected_discretization_contract=discretization_contract,
                expected_parameter_contract=parameter_contract,
            )
            agent3.logger.info("Agent3 已验证 mapping checkpoint 已保存。")

    # 2. 运行 Agent 3.5 (Reviewer). At most two bounded local corrections
    # are allowed; the complete mapping is never regenerated here.
    reviewer = QuantumReviewerAgent(
        api_key=YOUR_API_KEY,
        base_url=args.base_url,
        model_name=args.model
    )
    reviewer_raw_file, reviewer_status_file = next_review_paths(
        audit_directory, "Agent3", output_tag
    )
    translate_final_latex, final_mapping_contract, reviewer_record = reviewer.review(
        agent2_report_text,
        draft_mapping_contract,
        parameter_contract,
        discretization_contract,
        reviewer_raw_file,
        reviewer_status_file,
        deterministic_findings=draft_preflight_findings,
    )
    max_local_revisions = 2
    revision_count = automatic_revision_count(
        audit_directory, "Agent3", output_tag
    )
    while reviewer_record.get("machine_chain_blocked"):
        if revision_count >= max_local_revisions:
            raise SystemExit(
                "Agent3 reviewer still reports a fatal scientific issue, and "
                "this run already used {} local revisions. Inspect {}."
                .format(max_local_revisions, reviewer_status_file)
            )
        initial_reviewer_status_file = reviewer_status_file
        revision_raw_file = next_generator_raw_path(audit_directory, output_tag)
        revision_number = revision_count + 1
        reviewer_record["automatic_revision"] = {
            "attempted": True,
            "policy_version": AUTOMATIC_REVISION_POLICY_VERSION,
            "attempt_number": revision_number,
            "status": "started",
            "revision_raw_response_path": os.path.abspath(revision_raw_file),
        }
        atomic_json_write(initial_reviewer_status_file, reviewer_record)
        agent3.logger.warning(
            "Agent3 reviewer 发现 fatal 问题；执行第 %d/%d 次局部关联项修正。",
            revision_number, max_local_revisions,
        )
        try:
            agent3_draft_latex, draft_mapping_contract = agent3.revise_from_review(
                agent2_report_text,
                draft_mapping_contract,
                parameter_contract,
                discretization_contract,
                reviewer_record.get("findings", []),
                raw_response_path=revision_raw_file,
            )
        except Exception as error:
            raise SystemExit(
                "Agent3 automatic revision failed deterministic validation: {}. "
                "Inspect {} and {}.".format(
                    error, initial_reviewer_status_file, revision_raw_file
                )
            )
        # The corrected, validated mapping becomes the paid-stage checkpoint.
        with open(draft_file, "w", encoding="utf-8") as handle:
            handle.write(agent3_draft_latex)
        save_machine_contract(
            draft_mapping_contract,
            draft_mapping_file,
            validate_mapping_contract,
            expected_discretization_contract=discretization_contract,
            expected_parameter_contract=parameter_contract,
        )
        reviewer_raw_file, reviewer_status_file = next_review_paths(
            audit_directory, "Agent3", output_tag
        )
        translate_final_latex, final_mapping_contract, reviewer_record = (
            reviewer.review(
                agent2_report_text,
                draft_mapping_contract,
                parameter_contract,
                discretization_contract,
                reviewer_raw_file,
                reviewer_status_file,
            )
        )
        reviewer_record["automatic_revision"] = {
            "attempted": True,
            "policy_version": AUTOMATIC_REVISION_POLICY_VERSION,
            "attempt_number": revision_number,
            "status": "completed",
            "initial_review_status_path": os.path.abspath(
                initial_reviewer_status_file
            ),
            "revision_raw_response_path": os.path.abspath(revision_raw_file),
        }
        atomic_json_write(reviewer_status_file, reviewer_record)
        revision_count = revision_number
    mapping_contract_file = os.path.join(
        contract_directory, f"Agent3_MappingContract_{output_tag}.json"
    )
    save_machine_contract(
        final_mapping_contract,
        mapping_contract_file,
        validate_mapping_contract,
        expected_discretization_contract=discretization_contract,
        expected_parameter_contract=parameter_contract,
    )
    audit_record = publish_audit_report(
        translate_final_latex,
        stage_name="Agent3_Verified",
        output_tag=output_tag,
        report_directory=report_directory,
        audit_directory=audit_directory,
        compiled_filename=f"Agent3_Verified_{output_tag}.tex",
        rejected_filename=f"Agent3_Rejected_{output_tag}.tex",
        status_filename=f"Agent3_AuditStatus_{output_tag}.json",
    )
    final_file = audit_record["report_path"]
    reviewer.logger.info(f"格点化终稿已保存至: {final_file}")
    reviewer.logger.info("参数契约保持由 Agent2 单独拥有；Agent3 不再复制。")
    write_stage_output_manifest(
        args.output_manifest,
        "agent3",
        {
            "draft_report": draft_file,
            "draft_mapping_contract": draft_mapping_file,
            "report": final_file,
            "audit_status": audit_record["status_path"],
            "review_status": reviewer_status_file,
            "review_raw_response": reviewer_raw_file,
            "mapping_contract": mapping_contract_file,
        },
    )
