# -*- coding: utf-8 -*-
import os
import sys
import logging
import re
import time
import json
import argparse
import copy
# 引入标准的 OpenAI 库
from openai import OpenAI
from llm_support import api_key_for_model, llm_connection_options
from audit_report_support import publish_audit_report
from machine_contracts import (
    extract_json_contract_candidate,
    format_machine_contract,
    save_machine_contract,
    validate_theory_contract,
)
from pipeline_support import (
    atomic_json_write,
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)
from run_spec import load_run_spec


# Defaults are conventions, not user questions.  Agent1 may replace these only
# when the user explicitly supplies a different convention.
DEFAULT_CONTINUUM_CONVENTIONS = {
    "metric": {
        "signature": "mostly_minus",
        "matrix_rule": "diag(1,-1,...,-1) in the model's spacetime dimension",
        "display_latex": r"\eta_{\mu\nu}=\operatorname{diag}(1,-1,\ldots,-1)",
    },
    "gamma_algebra": {
        "clifford_relation": "{gamma^mu,gamma^nu}=2*eta^(mu nu)",
        "representation": "representation independent",
    },
    "lightcone": {
        "coordinate_definition": "x^plus/minus=x^0 plus/minus x^1",
        "gamma_definition": "gamma^plus/minus=gamma^0 plus/minus gamma^1",
        "normalization": 1.0,
    },
    "units": {"c": 1, "description": "natural units with c=1"},
}

# ==========================================
# 通用日志配置
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
    """从大模型回复中精准提取 LaTeX 代码块"""
    tb = chr(96) * 3  # 生成 ``` 符号以避免排版错乱
    pattern = re.compile(f'{tb}(?:latex|tex)?\\s*(.*?)\\s*{tb}', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize_clarifications(value):
    """Normalize harmless JSON variations without inventing user questions."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str):
            question = item.strip()
            reason = ""
        elif isinstance(item, dict):
            question = next(
                (
                    item.get(key, "").strip()
                    for key in ("question", "prompt", "request", "clarification")
                    if isinstance(item.get(key), str) and item.get(key).strip()
                ),
                "",
            )
            reason = next(
                (
                    item.get(key, "").strip()
                    for key in ("reason", "rationale")
                    if isinstance(item.get(key), str) and item.get(key).strip()
                ),
                "",
            )
        else:
            continue
        # An unrecognized formatting fragment is not a physical question and
        # must never be forwarded to the user.
        if not question:
            continue
        entry = {"question": question}
        if reason:
            entry["reason"] = reason
        normalized.append(entry)
    return normalized


def _merge_reviewed_entries(base_entries, reviewed_entries, identity_keys):
    """Apply identified reviewer corrections without touching omitted entries."""
    if not isinstance(reviewed_entries, list):
        return copy.deepcopy(base_entries)
    merged = copy.deepcopy(base_entries) if isinstance(base_entries, list) else []
    for reviewed in reviewed_entries:
        if not isinstance(reviewed, dict):
            continue
        identity_key = next(
            (key for key in identity_keys if reviewed.get(key) is not None), None
        )
        match = next((
            index for index, existing in enumerate(merged)
            if isinstance(existing, dict)
            and identity_key is not None
            and existing.get(identity_key) == reviewed.get(identity_key)
        ), None)
        if match is None:
            merged.append(copy.deepcopy(reviewed))
        else:
            merged[match] = copy.deepcopy(reviewed)
    return merged


def _merge_reviewed_definitions(base_definitions, reviewed_definitions):
    """Apply partial reviewer corrections without deleting omitted definitions."""
    if not isinstance(reviewed_definitions, list):
        return copy.deepcopy(base_definitions)
    merged = copy.deepcopy(base_definitions) if isinstance(base_definitions, list) else []
    core_kinds = {"lagrangian", "hamiltonian", "observable"}
    for reviewed in reviewed_definitions:
        if not isinstance(reviewed, dict):
            continue
        match = next(
            (
                index for index, existing in enumerate(merged)
                if isinstance(existing, dict)
                and existing.get("id") == reviewed.get("id")
                and reviewed.get("id")
            ),
            None,
        )
        if match is None and reviewed.get("kind") in core_kinds:
            match = next(
                (
                    index for index, existing in enumerate(merged)
                    if isinstance(existing, dict)
                    and existing.get("kind") == reviewed.get("kind")
                ),
                None,
            )
        if match is None:
            merged.append(copy.deepcopy(reviewed))
        else:
            merged[match] = copy.deepcopy(reviewed)
    return merged


def extract_theory_response(
    text, model_input, observable, initial_state, final_state,
    base_contract=None, continuum_conventions=None, external_states=None,
):
    states = dict(external_states or {"phi": final_state, "psi": initial_state})
    candidate = extract_json_contract_candidate(text, "continuum_theory")
    if base_contract is None:
        contract = candidate
    else:
        # A reviewer supplies a bounded correction to an already valid draft.
        # Omitting fields must not erase prior validated scientific content.
        contract = copy.deepcopy(base_contract)
        if "definitions" in candidate:
            contract["definitions"] = _merge_reviewed_definitions(
                contract.get("definitions"), candidate["definitions"]
            )
        if "symmetries" in candidate:
            contract["symmetries"] = _merge_reviewed_entries(
                contract.get("symmetries"), candidate["symmetries"], ("id", "name")
            )
        for key in ("assumptions", "clarifications_required"):
            if key in candidate:
                contract[key] = candidate[key]
        if isinstance(candidate.get("resolved_conventions"), dict):
            contract.setdefault("resolved_conventions", {}).update(
                candidate["resolved_conventions"]
            )
        if isinstance(candidate.get("model"), dict):
            model = contract.setdefault("model", {})
            model.update(candidate["model"])
        candidate_states = candidate.get("external_states")
        if (
            isinstance(candidate_states, dict)
            and isinstance(candidate_states.get("equivalent_groups"), list)
        ):
            contract.setdefault("external_states", {})["equivalent_groups"] = (
                candidate_states["equivalent_groups"]
            )
    # These fields describe immutable controller input, not Agent1's scientific
    # contribution.  Never require a stochastic model (or reviewer) to copy a
    # model name or external-state definition byte-for-byte.
    contract["schema_version"] = "1.0"
    contract["contract_type"] = "continuum_theory"
    contract.pop("observable_specification_sha256", None)
    if not isinstance(contract.get("model"), dict):
        contract["model"] = {}
    contract["model"]["name"] = model_input
    generated_states = contract.get("external_states")
    equivalent_groups = (
        generated_states.get("equivalent_groups", [])
        if isinstance(generated_states, dict)
        else []
    )
    if not isinstance(equivalent_groups, list):
        equivalent_groups = []
    equivalent_groups = [
        group for group in equivalent_groups
        if isinstance(group, list)
        and len(group) >= 2
        and len(group) == len(set(group))
        and set(group).issubset(states)
    ]
    contract["external_states"] = {
        "labels": list(states),
        "definitions": states,
        "equivalent_groups": equivalent_groups,
    }
    supplied_conventions = copy.deepcopy(continuum_conventions or {})
    contract["user_continuum_conventions"] = supplied_conventions
    resolved_conventions = contract.setdefault("resolved_conventions", {})
    for key, value in DEFAULT_CONTINUUM_CONVENTIONS.items():
        resolved_conventions.setdefault(key, copy.deepcopy(value))
    # Explicit UI selections are already deterministic physical definitions.
    # Agent1 may explain or extend them, but may not erase or numerically alter them.
    if isinstance(supplied_conventions.get("lightcone"), dict):
        resolved_conventions["lightcone"] = copy.deepcopy(
            supplied_conventions["lightcone"]
        )
    contract["clarifications_required"] = _normalize_clarifications(
        contract.get("clarifications_required", [])
    )
    return (
        extract_latex(text),
        validate_theory_contract(
            contract,
            expected_model=model_input,
            expected_observable=observable,
            expected_external_states=states,
        ),
    )


def bind_theory_to_run_spec(contract, run_spec):
    """Keep user data in RunSpec; theory owns definitions and references only."""
    bound = copy.deepcopy(contract)
    bound.pop("physical_parameters", None)
    bound.pop("simulation_discretization_controls", None)
    bound.pop("user_continuum_conventions", None)
    bound["user_input_refs"] = {
        "model": "/user_input/model",
        "parameters": "/user_input/parameters",
        "observable": "/user_input/observable",
        "external_states": "/user_input/external_states",
        "continuum_conventions": "/user_input/continuum_conventions",
    }
    return validate_theory_contract(bound)


# ==========================================
# Agent 1: 理论推导智能体
# ==========================================
class TheoryProcessorAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol"):
        # 实例化标准的 OpenAI 客户端，支持传入自定义的 API 网关地址
        self.client = OpenAI(**llm_connection_options(model_name, api_key=api_key, base_url=base_url))
        self.model_name = model_name
        self.logger = setup_logger("Agent1_TheoryProcessor")
        
        # 将系统指令保存为字符串
        self.system_instruction = (
            "You are an elite theoretical physicist specializing in Quantum Field Theory (QFT), "
            "Gauge Theories (like QCD, Schwinger model), and Quantum Simulation. "
            "Your task is to process user inputs for a quantum simulation workflow. "
            "An explicit user-supplied Lagrangian or Hamiltonian is authoritative; "
            "a conventional model name is only a literature/search label and must "
            "never be used to add, remove, or renormalize explicit terms. "
            "You have access to a vast internal knowledge base of physics literature."
        )
        
    def process_user_input(
        self, model_input, observable, initial_state, final_state,
        independent_physical_parameters, simulation_parameters,
        continuum_conventions, external_states=None, user_clarifications=None,
    ):
        self.logger.info(f"接收到用户输入 -> 模型: {model_input}")
        
        tb = chr(96) * 3 
        states = dict(external_states or {"phi": final_state, "psi": initial_state})
        by_definition = {}
        for label, definition in states.items():
            by_definition.setdefault(definition, []).append(label)
        equivalent_groups = [group for group in by_definition.values() if len(group) > 1]
        theory_example = {
            "schema_version": "1.0",
            "contract_type": "continuum_theory",
            "model": {
                "name": model_input,
                "gauge_structure": "replace with exact local gauge group or none",
                "spacetime_dimension": "exact dimension",
            },
            "external_states": {
                "labels": list(states),
                "definitions": states,
                "equivalent_groups": equivalent_groups,
            },
            "definitions": [
                {
                    "id": "continuum_hamiltonian",
                    "kind": "hamiltonian",
                    "canonical_expression": "exact machine-readable symbolic expression",
                    "display_latex": "exact LaTeX expression",
                    "dependencies": [],
                },
                {
                    "id": "requested_observable",
                    "kind": "observable",
                    "canonical_expression": "exact ordered n-point operator expression",
                    "display_latex": "exact LaTeX expression",
                    "dependencies": [],
                },
            ],
            "symmetries": [],
            "assumptions": [],
            "user_continuum_conventions": continuum_conventions,
            "resolved_conventions": {},
            "clarifications_required": [],
        }
        prompt = rf"""
Here are the initial inputs for a quantum simulation workflow:
1. QFT Model: {model_input}
2. Observable specification (arbitrary n-point operator insertions and sampling):
{json.dumps(observable, ensure_ascii=False, indent=2, sort_keys=True)}
3. Named external states (each label denotes one physical state that may appear
as a bra, a ket, or both in observable.matrix_elements):
{json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True)}
5. User-specified independent physical parameters (model-specific names; immutable):
{json.dumps(independent_physical_parameters, ensure_ascii=False, indent=2)}
6. User-specified simulation/discretization controls (immutable, not continuum inputs):
{json.dumps(simulation_parameters, ensure_ascii=False, indent=2)}
7. User-facing continuous-theory conventions:
{json.dumps(continuum_conventions, ensure_ascii=False, indent=2)}
8. User answers to earlier clarification questions (authoritative additions):
{json.dumps(user_clarifications or [], ensure_ascii=False, indent=2)}

Please perform the following strict theoretical processing and literature cross-referencing.
SOURCE PRECEDENCE: an explicit user formula and its written coefficients are the
authoritative theory. A model name is only a classification/search hint. If the
formula is complete, do not ask whether a differently named standard variant,
extra interaction term, flavor factor, 1/2, 1/N, or alternative coupling symbol
was intended. Preserve the formula literally and note naming differences only
as nonblocking context. The current explicit formula also outranks an older
clarification answer that described a different conventional model. A user
answer delegating a regulator or discretization choice means Agent2 may choose
it; Agent1 must not turn that delegation into a continuum-theory prescription.

a) Model Validation: If the user supplied only a model name, retrieve its
standard continuous theory. If the user supplied a Lagrangian or Hamiltonian,
derive from that exact expression and use literature only to check the algebra;
never replace it with a different conventional variant.
a1) Target Realizability: Any chosen realization of the model must actually
define and support the user's requested external states, observable, and
physical parameters. Do not silently replace the requested theory by an
effective theory, special limit, phase, or restricted sector if that replacement
makes a requested state absent, the observable undefined, or the requested
quantity trivially fixed unless the user explicitly requested that replacement.
If multiple physically inequivalent realizations remain and the supplied input
cannot distinguish them, ask one concise physical clarification question.
Write that question in clear, academically standard English. Do not use an
undefined abbreviation, a newly coined model name, or jargon as the sole
description of an option. Describe each alternative by its defining physical
content and the consequence relevant to the requested calculation; introduce
the standard technical name in parentheses only when useful. The user must be
able to answer from the question without already knowing your terminology.
b) Conserved Quantities: Based on the verified model, identify its conserved quantities (both continuous and discrete symmetries) and provide their explicit operator expressions in continuous spacetime.
b1) Sampling-Grid Symmetry Relevance: Inspect the physical coordinates generated
by the user-supplied observable sampling grid. If its time coordinates occur in
pairs related by t -> -t, treat time reversal T as relevant to the downstream
discretization only when the verified continuum theory at the supplied physical
parameters actually has T. If its spatial coordinates occur in pairs related by
x -> -x about the stated origin, analogously treat reflection/parity P as
relevant only when the continuum theory has P. If both coordinates reverse,
also identify PT when it is an actual continuum symmetry. Record these as
relevant continuum symmetries using the existing `symmetries` entries; add no
new contract field. A symmetric grid alone neither creates a symmetry nor
proves an even/odd or complex-conjugation identity for the matrix element:
external states, the complete ordered observable and any transporter must also
transform appropriately. Do not infer an unspecified state eigenvalue.
c) Observable Check: Determine if the model possesses local gauge symmetry. If it does, strictly check if the provided observable is gauge-invariant. If it does not have local gauge symmetry, explicitly state this.
c1) N-Point Structure: Preserve every operator insertion, its position, ordering, gauge transporter, requested component, sampling variable, and sample assignment exactly. The number of insertions is arbitrary; never reduce an n-point function to a two-point function.
d) External States Check: Check every named state separately against the conserved quantities identified in step (b). A state may be a vacuum or a bound state in a specified quantum-number sector. Never identify two labels unless their input definitions are exactly identical.
e) Matrix-Element Combination: Preserve and write the complete requested linear combination $\sum_k c_k\langle a_k|O|b_k\rangle$ from observable.matrix_elements. When matrix_elements is absent, use the legacy single matrix element $\langle\phi|O|\psi\rangle$. Never replace a subtraction by one transition matrix element.
f) Physical-Parameter Preservation: Preserve every user-specified model-specific parameter name and value exactly. Distinguish continuum parameters from downstream discretization controls such as a requested lattice spacing. You may calculate only direct algebraic consequences of the supplied values (for example, $m=(m/g)g$); do not choose any missing parameter or algorithmic setting.
g) Simulation-Control Preservation: Reproduce every user-specified simulation/discretization control exactly in a separate subsection. Do not reinterpret these controls as continuum physics. Explicitly state direct compatibility consequences that Agent 2 must enforce, but do not invent missing controls.
h) Convention Resolution: Resolve unspecified notation conventions yourself and
record them consistently in `resolved_conventions`. Unless explicitly overridden,
use a mostly-minus metric in the model's actual spacetime dimension and natural
units c=1. Apply gamma-matrix and light-cone conventions only when the theory or
observable uses them; then use {{gamma^mu,gamma^nu}}=2 eta^(mu nu) and
x^plus/minus=x^0 plus/minus x^1 with gamma^plus/minus=gamma^0 plus/minus
gamma^1 (no sqrt(2)). Never request user clarification for metric signature,
gamma-matrix representation, light-cone normalization, units, index placement,
notation, operator normalization convention, vacuum-energy zero, boundary
conditions, gauge fixing, regulator, discretization, truncation, mapping,
ansatz, optimizer, Trotterization, shots, or hardware. Agent2 owns unprovided
discretization and regulator choices.

Add `clarifications_required` only when missing or contradictory PHYSICAL input
leaves inequivalent target theories or observables: for example contradictory
explicit interactions, a required physical parameter that cannot remain
symbolic, an undefined operator/insertion, or an external state lacking the
quantum-number sector and level needed to identify it. Treat supplied answers
as authoritative and never repeat an answered question unless the answer is
itself contradictory.

Agent1's statement that a boundary condition, gauge fixing, regulator or
truncation was not supplied means only that Agent1 does not choose it. Do not
turn that absence into a downstream prohibition: Agent2 owns scientifically
justified discretization choices that the user left unspecified.

Do not impose a zero-vacuum-energy convention and do not subtract an unknown
vacuum expectation value from the Hamiltonian. Derive and preserve the
unshifted continuum Hamiltonian. Energy differences may be formed only as
downstream result analysis without redefining the simulated Hamiltonian.

OUTPUT FORMAT REQUIREMENTS:
Output exactly two independent fenced blocks and no conversational text. First
return a concise complete compilable LaTeX audit document. Do not use quantikz,
tikz, figures, verbatim, embedded JSON, or boxed multi-line alignments.
The LaTeX document must contain the following specific sections:
(1) The supplied theory's defining action/Lagrangian when one exists, and its
unshifted Hamiltonian. If the authoritative target is specified directly as a
Hamiltonian model with no unique Lagrangian, say so instead of inventing one.
The Hamiltonian must be given without gauge fixing or vacuum-energy subtraction.
(2) Result of the gauge invariance check for the observable.
(3) Conserved quantities and their operator definitions.
(4) Separate consistency-check results for every named external state.
(4a) Explicitly list only state labels whose input definitions are identical; otherwise keep them independent. Never infer identity merely from matching quantum numbers.
(5) Strict operator expression of the complete matrix-element linear combination.
(6) A section named ``User-Specified Independent Physical Parameters'' that reproduces all supplied names and values, distinguishes continuum inputs from downstream discretization controls, and lists only directly implied derived relations.
(7) A section named ``User-Specified Simulation and Discretization Controls'' that reproduces the supplied simulation controls exactly.
(8) A section named ``Continuous-Theory Conventions'' giving every resolved
definition and any genuinely necessary clarification.

Then return one fenced JSON contract using the following envelope. Replace all
formula placeholders with exact content. This JSON is the authoritative machine
interface; LaTeX is only a human audit view and must not contain the JSON.
{tb}json
{json.dumps(theory_example, ensure_ascii=False, indent=2)}
{tb}
"""
        # 构建标准的消息体格式
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt}
        ]

        self.logger.info(f"正在呼叫 {self.model_name} 进行物理理论推导 (流式生成中)...")
        start_time = time.time()
        
        # 调用标准的 chat completions 接口
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
#temperature=0.1,
            stream=True
        )
        
        full_content = ""
        for chunk in response_stream:
            # 解析 OpenAI 格式的流式数据
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                
        duration = time.time() - start_time
        self.logger.info(f"Agent 1 理论分析完成！耗时: {duration:.2f} 秒")
        
        return extract_theory_response(
            full_content, model_input, observable, initial_state, final_state,
            continuum_conventions=continuum_conventions,
            external_states=states,
        )

# ==========================================
# Agent 1.5: 理论审查智能体 (Reviewer)
# ==========================================
class TheoryReviewerAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol"):
        self.client = OpenAI(**llm_connection_options(model_name, api_key=api_key, base_url=base_url))
        self.model_name = model_name
        self.logger = setup_logger("Agent1.5_TheoryReviewer")
        
        self.system_instruction = (
            "You are a rigorous Senior Theoretical Physicist acting as a peer reviewer. "
            "Your job is to verify and correct scientific derivations in Quantum Field Theory. "
            "Explicit user formulas outrank conventional model names. "
            "Do not review formatting, provenance, workflow authorization, agent ownership, "
            "or downstream policy."
        )

    def review_and_correct(
        self, latex_report, theory_contract, model_input, observable, initial_state, final_state,
        independent_physical_parameters, simulation_parameters,
        continuum_conventions, external_states=None, user_clarifications=None,
    ):
        self.logger.info("正在将 Agent 1 的报告送交 Reviewer Agent 进行公式及逻辑审查...")
        
        tb = chr(96) * 3
        states = dict(external_states or {"phi": final_state, "psi": initial_state})
        prompt = rf"""
Please review the following LaTeX report generated by a junior physicist regarding a Quantum Simulation workflow.

The immutable original user inputs are:
- QFT Model: {model_input}
- Observable specification (immutable):
{json.dumps(observable, ensure_ascii=False, indent=2, sort_keys=True)}
- Named external states:
{json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True)}
- User-specified independent physical parameters (immutable, model-specific):
{json.dumps(independent_physical_parameters, ensure_ascii=False, indent=2)}
- User-specified simulation/discretization controls (immutable):
{json.dumps(simulation_parameters, ensure_ascii=False, indent=2)}
- User-facing continuous-theory conventions (immutable):
{json.dumps(continuum_conventions, ensure_ascii=False, indent=2)}
- User answers to earlier clarification questions (authoritative additions):
{json.dumps(user_clarifications or [], ensure_ascii=False, indent=2)}

Here is the report:
{tb}latex
{latex_report}
{tb}

Here is the authoritative continuum-theory contract to verify and correct:
{tb}json
{format_machine_contract(theory_contract, validate_theory_contract, expected_model=model_input, expected_observable=observable, expected_external_states=states)}
{tb}

Your task is to strictly verify the theoretical correctness:
0. Does the chosen realization of the model actually support every requested
   external state and define the requested observable using the supplied
   physical parameters? If an effective theory, special limit, phase, or
   restricted sector makes a requested state absent, the observable undefined,
   or the requested quantity trivially fixed without an explicit user request,
   do not conditionally approve it. Correct the theory selection when the user
   input determines one; otherwise retain one concise physical clarification.
1. Is the unshifted Hamiltonian derived correctly from the user's explicit
   Lagrangian/action when supplied, or preserved directly when the Hamiltonian
   defines the model, without inventing a Lagrangian, adding an absent
   interaction, or imposing a zero-vacuum-energy subtraction?
2. Are the operator expressions for the conserved quantities mathematically sound?
2a. For every time- or space-reflection pair actually present in the supplied
   sampling grid, did Agent1 identify T, P or PT as relevant only when it is an
   exact symmetry of the continuum theory at the supplied parameters? Reject a
   symmetry invented from grid shape alone, and reject an even/odd conclusion
   that was asserted without checking the external states, complete ordered
   observable and transporter. Do not require a state eigenvalue absent from
   the authoritative state definition.
3. Is the logic for the gauge invariance check of the observable flawless, and are all n-point insertions, positions, ordering, transporters and sampling assignments preserved?
4. Are all named external-state quantum numbers consistent with the identified symmetries, and are all matrix-element coefficients, bra labels and ket labels preserved without silent identification?
5. Do the continuum equations use every supplied physical parameter correctly,
   including valid direct algebraic consequences?
6. Are supplied simulation/discretization controls kept out of continuum
   equations except where they define the requested observable coordinates?
7. Were user-facing continuum conventions translated consistently? Remove every
clarification request about notation or conventions (metric signature, gamma
representation, light-cone normalization, units, normalization convention,
vacuum-energy zero, boundary condition, regulator or downstream algorithm).
Retain a concise question only for missing or contradictory PHYSICAL input that
leaves inequivalent target theories, observables, or externally requested states.
Do not repeat a question already answered by the user unless the answer is
internally contradictory. Write any remaining clarification question and reason
in concise, academically standard English for direct display in the local UI.
Rewrite any undefined abbreviation, invented shorthand, or jargon-only option.
Each alternative must instead be distinguished by its defining physical content
and its consequence for the requested states or observable, with a standard
technical name added in parentheses only when it helps the user.

Review scientific content only. Do not report document formatting, JSON field
order, provenance, workflow authorization, historical compliance, or stage
ownership as findings. The absence of a user-selected boundary condition,
gauge fixing, regulator or truncation means Agent1 must leave it undecided; it
does not authorize this reviewer to forbid Agent2 from choosing it later.

Complete every check above before writing the response. Do not stop after the
first error. In the LaTeX audit, list every concrete scientific error you found,
then correct all of those errors together in the single partial JSON response.
If you find ANY errors (even minor typos in indices or signs), correct them.
If it is completely correct, return an empty correction envelope containing
only `contract_type: continuum_theory`.

OUTPUT FORMAT REQUIREMENTS:
Return exactly two independent fenced blocks and no other text: a concise final
LaTeX audit document followed by a partial corrected JSON theory contract.
The JSON must include `contract_type: continuum_theory` and ONLY the erroneous
definitions, symmetries, conventions, assumptions or clarification content and
their scientifically necessary dependent entries. Identify corrected
definitions by `id` and corrected symmetries by `id` or `name`. Omit every
unchanged entry and top-level section; the controller merges these corrections
into the validated draft and preserves everything omitted.
Do not use quantikz, tikz, figures, verbatim or embedded JSON in the report.
The JSON contract is authoritative. Correct only scientific content; controller
code preserves the immutable user inputs.
"""
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
#temperature=0.1,
            stream=True
        )
        
        full_content = ""
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                
        duration = time.time() - start_time
        self.logger.info(f"Reviewer 审查完成！耗时: {duration:.2f} 秒")

        correction = extract_json_contract_candidate(
            full_content, "continuum_theory"
        )
        correction_keys = {
            "definitions", "symmetries", "assumptions",
            "clarifications_required", "resolved_conventions", "model",
            "external_states",
        }
        has_scientific_corrections = any(
            key in correction for key in correction_keys
        )
        corrected_report, corrected_contract = extract_theory_response(
            full_content, model_input, observable, initial_state, final_state,
            base_contract=theory_contract,
            continuum_conventions=continuum_conventions,
            external_states=states,
        )
        return corrected_report, corrected_contract, has_scientific_corrections

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Agent1 formulation and review.")
    parser.add_argument("--run-spec", required=True, help="Immutable controller-owned RunSpec.")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    return parser.parse_args()


# ==========================================
# 工作流总控
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    report_directory = categorized_output_directory(args.output_root, "reports")
    contract_directory = categorized_output_directory(args.output_root, "contracts")
    audit_directory = categorized_output_directory(args.output_root, "audit")
    YOUR_API_KEY = api_key_for_model(args.model, required=False)
    if not YOUR_API_KEY:
        api_key_for_model(args.model)
    run_spec = load_run_spec(args.run_spec)
    user_input = run_spec["user_input"]
    parameter_groups = user_input.get("parameter_groups", {})
    user_independent_physical_parameters = dict(
        parameter_groups.get("physical", user_input["parameters"])
    )
    user_simulation_parameters = dict(parameter_groups.get("simulation", {}))
    user_model = user_input["model"]
    user_observable = user_input["observable"]
    user_continuum_conventions = dict(
        user_input.get("continuum_conventions", {})
    )
    user_external_states = dict(user_input["external_states"])
    user_clarifications = list(user_input.get("clarifications", []))
    user_initial_state = user_external_states.get("psi")
    user_final_state = user_external_states.get("phi")

    # 1. 启动 Agent 1 进行初稿推导，传入 base_url
    agent1 = TheoryProcessorAgent(
        api_key=YOUR_API_KEY, 
        base_url=args.base_url,
        model_name=args.model
    )
    draft_latex, draft_theory_contract = agent1.process_user_input(
        model_input=user_model,
        observable=user_observable,
        initial_state=user_initial_state,
        final_state=user_final_state,
        independent_physical_parameters=user_independent_physical_parameters,
        simulation_parameters=user_simulation_parameters,
        continuum_conventions=user_continuum_conventions,
        external_states=user_external_states,
        user_clarifications=user_clarifications,
    )
    draft_theory_contract = bind_theory_to_run_spec(draft_theory_contract, run_spec)
    
    draft_file = os.path.join(report_directory, f"Agent1_Draft_{output_tag}.tex")
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(draft_latex)
    agent1.logger.info(f"初稿已保存至: {draft_file}")

    # 2. 启动 Reviewer Agent 进行审查
    reviewer = TheoryReviewerAgent(
        api_key=YOUR_API_KEY,
        base_url=args.base_url,
        model_name=args.model
    )
    final_latex = draft_latex
    final_theory_contract = draft_theory_contract
    max_local_revisions = 2
    for review_number in range(1, max_local_revisions + 1):
        (
            reviewed_latex,
            reviewed_theory_contract,
            has_scientific_corrections,
        ) = reviewer.review_and_correct(
            final_latex,
            final_theory_contract,
            user_model,
            user_observable,
            user_initial_state,
            user_final_state,
            user_independent_physical_parameters,
            user_simulation_parameters,
            user_continuum_conventions,
            user_external_states,
            user_clarifications,
        )
        final_latex = reviewed_latex
        final_theory_contract = reviewed_theory_contract
        # A genuine physical ambiguity belongs to the user, not another model
        # rewrite. Otherwise stop as soon as a review finds nothing to change.
        if final_theory_contract.get("clarifications_required"):
            break
        if not has_scientific_corrections:
            break
        reviewer.logger.warning(
            "Agent1 reviewer 已执行第 %d/%d 次局部科学修正%s。",
            review_number,
            max_local_revisions,
            "；将再审查一次" if review_number < max_local_revisions else "",
        )
    clarifications = final_theory_contract.get("clarifications_required", [])
    if clarifications:
        clarification_file = os.path.join(
            audit_directory, f"Agent1_Clarification_{output_tag}.json"
        )
        atomic_json_write(
            clarification_file,
            {
                "status": "needs_user_clarification",
                "questions": clarifications,
            },
        )
        questions = "; ".join(item["question"] for item in clarifications)
        raise SystemExit(
            "Agent1 requires user clarification before discretization: "
            + questions + ". Details: " + clarification_file
        )
    final_theory_contract = bind_theory_to_run_spec(final_theory_contract, run_spec)
    theory_contract_file = os.path.join(
        contract_directory, f"Agent1_TheoryContract_{output_tag}.json"
    )
    save_machine_contract(
        final_theory_contract,
        theory_contract_file,
        validate_theory_contract,
        expected_model=user_model,
        expected_observable=user_observable,
        expected_external_states=user_external_states,
    )
    audit_record = publish_audit_report(
        final_latex,
        stage_name="Agent1_Verified",
        output_tag=output_tag,
        report_directory=report_directory,
        audit_directory=audit_directory,
        compiled_filename=f"Agent1_Verified_Report_{output_tag}.tex",
        rejected_filename=f"Agent1_Rejected_{output_tag}.tex",
        status_filename=f"Agent1_AuditStatus_{output_tag}.json",
    )
    final_file = audit_record["report_path"]
    reviewer.logger.info(f"审查完毕，最终验证版报告已保存至: {final_file}")
    write_stage_output_manifest(
        args.output_manifest,
        "agent1",
        {
            "report": final_file,
            "audit_status": audit_record["status_path"],
            "theory_contract": theory_contract_file,
        },
    )
