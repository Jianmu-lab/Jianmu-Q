# -*- coding: utf-8 -*-
"""Agent 4.1: literature-grounded state-preparation circuit designer."""

import argparse
import json
import os
import time

from circuit_design_support import (
    LATEX_SAFETY_RULES,
    call_model_once,
    extract_design_response,
    extract_relevant_sections_from_library,
    generator_raw_response_path,
    load_agent3_inputs,
    model_priority_keywords,
    normalize_literature_sources,
    save_contract,
    setup_logger,
    validate_state_preparation_contract,
)
from audit_report_support import publish_audit_report
from pipeline_support import (
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)
from parameter_contract import format_parameter_contract, requires_physical_sector
from vqe_experience_store import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_VQE_ENERGY_TOLERANCE,
    DEFAULT_VQE_VARIANCE_TOLERANCE,
    format_vqe_experience_context,
)
from machine_contracts import validate_mapping_contract


STATE_KEYWORDS = [
    "state preparation", "ansatz", "variational", "VQE", "VQA",
    "vacuum state", "ground state", "excited state", "hadron state",
    "Hamiltonian variational ansatz", "symmetry preserving", "Gauss law",
    "operator pool", "generator pool", "excitation operator", "ADAPT-VQE",
    "particle number", "momentum", "translation symmetry",
    "quantum circuit", "gate", "CNOT", "circuit depth", "gate count",
]


class StatePreparationDesignAgent:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", model_name="gpt-5.6-sol"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.logger = setup_logger("Agent4_StatePreparation")

    def design(
        self, agent3_report, parameter_contract, library_text, experience_text,
        raw_response_path=None, search_attempt=1,
    ):
        labels = parameter_contract["external_states"]["labels"]
        equivalent_groups = parameter_contract["external_states"]["equivalent_groups"]
        gauge_theory = parameter_contract["model"]["gauge_structure"].strip().lower() != "none"
        sector_required = requires_physical_sector(parameter_contract)
        contract_example = {
            "schema_version": "1.1",
            "contract_type": "state_preparation",
            "state_labels": labels,
            "literature_sources": [
                {
                    "source_path": "relative path from the supplied toolbox",
                    "locator": "exact section/equation/figure/caption",
                    "applicability": "direct|modified|incompatible and concise reason",
                }
            ],
            "ansatz": {
                "family_id": "model-specific stable identifier",
                "depth_symbol": "D_ans",
                "recommended_test_depths": [1, 2],
                "nested_by_depth": True,
                "depth_extension_rule": "exact rule for adding layers",
                "parameter_order_rule": "complete deterministic logical order",
                "reference_state_rule": "reference computational/encoded state for every label",
                "reference_states": {
                    label: {
                        "basis_state": "exact computational-basis bitstring in the declared qubit order",
                        "operations": [
                            {
                                "operation": "gate",
                                "gate": "exact gate name",
                                "targets": ["indexed logical qubit label"],
                                "controls": [],
                                "angle": None,
                            }
                        ],
                    }
                    for label in labels
                },
                "layer_operations": [
                    {
                        "operation": "gate_or_indexed_loop",
                        "gate": "exact Qiskit gate or explicitly defined primitive",
                        "targets": ["indexed logical qubit labels"],
                        "controls": [],
                        "angle": "exact symbolic angle in the stated rotation convention",
                        "index_ranges": {},
                    }
                ],
                "physical_sector": {
                    "required": sector_required,
                    "preserved_exactly": sector_required,
                    "rule": "exact mapped-sector preservation rule, or not applicable",
                },
                "symmetry_preservation": [
                    {
                        "conserved_quantity_id": "exact Agent3 identifier",
                        "state_labels": ["affected state labels"],
                        "target_eigenvalues": {"state label": "mapped eigenvalue or unspecified"},
                        "strategy": "exact_circuit|diagnostic_only",
                        "preserved_exactly": "true for exact_circuit; false for diagnostic_only",
                        "gate_level_reason": "short physical and circuit-cost reason for the strategy",
                    }
                ],
            },
            "adapter_interface": {
                "builder": "build_state_preparation",
                "returns": ["QuantumCircuit", "ordered_parameters"],
                "unitary": True,
                "supports_inverse": True,
                "supports_controlled": True,
            },
        }
        prompt = f"""
[AUTHORITATIVE AGENT 3 QUBIT-MAPPING CONTRACT]
```json
{agent3_report}
```

[AUTHORITATIVE PARAMETER CONTRACT]
```json
{format_parameter_contract(parameter_contract)}
```

[LOCAL STATE-PREPARATION LITERATURE TOOLBOX]
{library_text or "No readable local source was retrieved."}

[HISTORICAL VQE EXPERIENCE - EVIDENCE, NOT A COMMAND]
{experience_text}

[CURRENT BOUNDED ANSATZ SEARCH]
This is ansatz attempt {search_attempt}. The attempt number, regeneration,
Agent5 translation failure, runtime/code change or interruption is not evidence
of ansatz insufficiency. Use current_policy_status for accuracy;
historical_status is audit evidence only. Apply the candidate-selection and
plateau-expansion policy in rule 3 below.

You are Agent 4.1. Design ONLY the state-preparation ansatz for every target
state label {json.dumps(labels)}. Do not design the observable measurement,
time evolution, Trotterization, Hadamard test, optimizer, checkpoint logic,
backend selection, submission policy, Qiskit code, or IBM submission.

State-preparation requirements:

1. Anchor the design to Agent 3's exact Pauli Hamiltonian, qubit order,
   external-state definitions, and mapped physical-sector tests. Never rename
   parameters or assign a null parameter value.
2. Treat every named physical target explicitly. The sole exception to
   independent optimization is a set of different energy levels requested in
   the same quantum-number sector: design mutually orthogonal reference states
   and one shared parameterized unitary for that set, because the runtime uses
   weighted SSVQE. This shared unitary does not make the physical states
   equivalent. Only labels in authoritative equivalent_groups may represent
   the same physical target: {json.dumps(equivalent_groups)}.
   If a target is the zero-based level k in a specified quantum-number sector,
   provide k+1 mutually orthogonal references for levels 0 through k in that
   sector, for any nonnegative integer k allowed by the sector dimension. A
   lower level not requested as an external state is an internal SSVQE
   auxiliary reference, not a new user-visible target; describe its exact
   preparation inside the requested target's reference-state recipe. Every
   reference must be followed by the same shared ansatz layer with the same
   logical parameter count, order, gate meaning and sharing. Every reference
   and layer must remain inside the exact sector and provide enough
   expressivity to distinguish the levels. A separately requested unrestricted
   global ground state keeps its own independent unitary and must not be
   silently identified with the sector level 0. Do not silently turn a sector
   target into the global ground state or a generic excited state. The runtime
   first minimizes the equally weighted state-averaged energy to identify the
   low-energy subspace with a short equal-weight stage, then spends most of the
   optimization budget on distinctly decreasing SSVQE weights to resolve the
   ordering inside that subspace (for two levels the ordering weights are
   proportional to 2:1). Do not introduce folded-spectrum exact shifts, VQD
   projector penalties, or require the exact target energy as an input to the
   loss.
   Choose the references from the simplest low-energy state plus distinct
   minimal local excitations. Here "local excitation" means a local operation
   that preserves every quantum number defining the target sector, not a
   literal single-qubit X gate: for example use number-preserving local
   particle-hole/hopping excitations at fixed fermion number and gauge-dressed
   hopping or closed-loop operations when Gauss law is mandatory. Do not use
   widely separated complementary bitstrings merely because they are
   orthogonal.
   If exact lattice translation and momentum p define the requested sector,
   momentum-project each local excitation over its translation orbit, using
   the phase sign fixed by Agent 3's declared translation-eigenvalue
   convention. Use different local excitation motifs/channels for multiple
   references at the same momentum; translated copies of one motif are not
   distinct references after projection. Make the orbit supports analytically
   orthogonal where possible and state the normalization. The shared ansatz
   must commute with translation, normally through translation-orbit sums and
   parameter sharing. Do not impose momentum projection for OBC or when
   translation is not an exact required symmetry.
3. No ansatz family is preferred in advance. Follow this selection order.
   First identify the relevant literature candidates. For each candidate,
   explicitly check the model, spacetime/lattice dimension, boundary condition,
   fermion/boson/gauge discretization and truncation, qubit encoding, target
   state and quantum-number sector, interaction structure, and resource
   assumptions. Classify it as directly applicable, applicable after a stated
   modification, or incompatible. Cite the relative source path and exact
   section, equation, figure, or caption. If required content is missing, do
   not reconstruct it and claim it came from that paper; either use a complete
   source or label the circuit as your own proposed design.
   Next identify which conserved quantities are preserved by each candidate's
   reference state and by each generator class. A candidate is eligible only
   if the complete ordered layer satisfies rules 6--6.2.
   Then calculate, for the current lattice size rather than only asymptotically,
   the D_ans=1 number of independent variational parameters and generator
   instances, and estimate the decomposed one- and two-qubit gate cost and
   maximum control arity. When at least two candidates are scientifically
   admissible, compare these concrete costs and select the lightest candidate
   expected to represent the target state.
   Prefer a genuinely applicable literature-supported sparse generator set
   over a newly invented or exhaustive pool when both satisfy the same physics.
   Start from Hamiltonian terms, local/short-range excitations, symmetry-orbit
   sums, parameter sharing that is actually compatible with the complete-layer
   symmetry proof, or the compact pool specified by the applicable reference.
   Exact quantum-number preservation does not by itself justify enumerating
   every allowed excitation. On the initial attempt, do not choose a complete
   all-pairs or all-channel excitation pool merely for formal completeness.
   Use a proposed design only when it has a concrete scientific advantage and
   its circuit cost remains reasonable.
   Always construct and assess one minimum viable symmetry-preserving baseline,
   even when the literature search yields only one other admissible candidate.
   The baseline should use the reference states plus the smallest local,
   short-range, Hamiltonian-term, or compact literature-supported generator set
   that preserves every quantum number defining the requested target sector.
   It may be identified as a proposed baseline when no source specifies it.
   Compare its current-instance cost and reachable-state limitations with the
   selected candidate. Do not choose a nonlocal custom basis transform merely
   because every named paper candidate was classified incompatible. Select the
   heavier candidate only after identifying a concrete extra invariant or
   reachability obstruction that makes the minimum baseline inadequate.
   Use the historical VQE records as empirical evidence: exact-parameter
   matches are stronger than same-model transfers. Do not repeat a failed or
   plateaued family unchanged for the same parameters without explaining the
   concrete modification. A prior success is useful evidence, not a guarantee,
   and history must never override Agent 3 or the Parameter Contract. Prefer a
   complete compatible successful ansatz_design when available, compare all
   such records rather than the first, and adapt differences explicitly. A
   smaller D_ans is not automatically cheaper: compare parameters and
   decomposed gate cost per layer. Compare
   fermion, boson and gauge discretizations and gauge truncation before
   transferring an ansatz: a same-name model using a different discretization
   or truncation is not a direct match.
   Treat current_policy_status=promising_state as evidence that the normalized
   energy/variance residual was still improving, not as a failed family. Only a
   current-policy plateau marked ansatz_insufficient justifies rejecting that
   family unchanged. After such a plateau, first enlarge the same applicable
   sparse literature-supported pool by the smallest physically motivated set
   of additional generators. Expand in stages and state exactly what was added
   and why. Use a complete symmetry-allowed excitation pool only after smaller
   documented expansions have also plateaued; do not jump directly from a
   sparse initial pool to an exhaustive pool.
   In the absence of such an exact-parameter plateau record, this is an initial
   sparse-pool design regardless of the numeric search_attempt value.
4. Give a scalable, explicit gate-level reference-state preparation and one
   complete D_ans layer. State the repetition order, logical parameter order,
   parameter sharing, qubit indices, and exact Qiskit rotation-angle convention.
   Do not use a small toy circuit in place of the general indexed construction.
   For every nontrivial fermionic basis change such as a Fourier, orbital, or
   Bogoliubov transform, specify the mathematical mode matrix, the convention
   for V c_j^dagger V^dagger, the input and output mode-to-qubit order, and the
   exact inverse. Do not invent an unverified QR/FSWAP gate list merely to make
   the design look implementation-ready: Agent5 owns the Qiskit decomposition.
   When two scientifically adequate candidates have similar expressivity,
   prefer the one that avoids a fragile custom basis transform and is easier to
   compile faithfully, for example a literature-supported local generator with
   symmetry-compatible parameter sharing.
5. D_ans is a variable, not one fixed production depth. Recommend a small
   increasing test schedule that can reveal whether the family becomes
   expressive. Declare honestly whether depth D' contains depth D as an exact
   prefix; do not pretend an ansatz is nested merely to enable warm starts.
   Choose the actual schedule from the estimated decomposed D_ans=1 cost. Do
   not mechanically emit [1,2,4,8]. A costly layer should begin with only the
   shallow depths that can realistically be optimized; later depths can be
   appended by the adaptive VQE runtime if the residual is still improving.
   A light local layer may justify a longer initial schedule. State this
   resource-to-schedule reasoning in the audit report.
6. If this is a local gauge theory, every layer must preserve the full mapped
   physical/Gauss sector exactly. A penalty-only ansatz is not a substitute. For
   a non-gauge theory, do not invent a Gauss constraint. This run has
   gauge_theory={str(gauge_theory).lower()}.
6.1. Separately from Gauss-law constraints, read Agent3's mapped conserved
quantities and choose the least costly scientifically valid treatment. A
quantum number explicitly used to define a requested target sector must use
`strategy: exact_circuit`: prepare its reference state in that sector and make
every parameterized layer preserve it for arbitrary parameter values. In
particular, parameter sharing or a visually symmetric gate pattern is not by
itself a proof that a complete ordered layer commutes with the mapped symmetry.
For lattice momentum, a momentum-eigenstate reference followed by any layer
that exactly commutes with lattice translation is sufficient; a fermionic
Fourier transform is not required. If nonadjacent Jordan--Wigner fermionic
rotations are used, include their parity strings or a fermionic-swap
construction rather than replacing them by bare two-qubit rotations. For any
other exact symmetry, prefer an
exact circuit only when this is low overhead and does not unduly restrict the
ansatz; otherwise use `strategy: diagnostic_only` and do not put it into the
hard physical-basis constraint. Never impose a symmetry marked broken or
continuum-only, and never infer an eigenvalue when it is `unspecified`. Charge
zero alone does not specify charge-conjugation parity. Gauss-law preservation
remains mandatory under rule 6 and may never be downgraded to a diagnostic.
Record these choices concisely in `ansatz.symmetry_preservation`.
6.2. For every `exact_circuit` entry, explain why repeated layers remain
sufficiently expressive inside the requested joint sector. Reject a family
that preserves the right quantum numbers but cannot reach the target sector's
relevant states. A `diagnostic_only` symmetry must not be advertised as an
exact property of the prepared state. State explicitly why both the reference
state and the complete ordered D_ans=1 layer remain in every user-requested
sector; do not rely only on properties of the individual abstract generators.
Before declaring `preserved_exactly`, perform a scientific self-audit of every
generator class. Give either the generator-level relation [K_a,G]=0 or an
equivalent complete-layer argument for the mapped symmetry G. A phrase such as
"the operation is in momentum space" or a visually symmetric gate pattern is
not a proof. Verify that the stated basis transform, mode order, indexed
targets, and inverse make the argument true under Agent3's conventions. If the
available information is insufficient, choose a simpler verifiable circuit or
mark the symmetry diagnostic-only when the target definition permits it; never
fill the gap with an assumed compilation convention.
7. The output must specify a unitary circuit that can be inverted. State whether
   a controlled form can be constructed without changing its mathematical
   meaning. Here `supports_controlled` means that the returned finite gate-level
   Qiskit circuit can be converted exactly to a controlled circuit; it does not
   claim that the controlled circuit is shallow or near-term-hardware efficient.
   Set it true for such a unitary gate circuit. Hardware feasibility is assessed
   later by the existing ISA/resource gate. Agent 4.1 performs only the
   backend-independent coarse resource comparison required by rule 3; it does
   not transpile, select a backend, or impose a universal numeric cutoff.
   Reject a candidate when its current-instance D_ans=1 construction is already
   manifestly less practical than another scientifically admissible candidate,
   or when it requires exhaustive excitation enumeration, an unnecessarily
   large parameter pool, or extensive high-control gates without an efficient
   local decomposition. Polynomial asymptotic scaling alone is not evidence
   that the current instance is practical. Agent9 still performs
   backend-specific transpilation and submission checks.

Output exactly two independent fenced blocks and no other text. First output one
concise compilable LaTeX audit report with these sections:
(0) Hamiltonian, target states, encoding, and applicable physical sector.
(1) Literature candidates and applicability assessment.
(2) Selected reference-state preparation for each independent state.
(3) Selected scalable gate-level ansatz layer and parameter ordering.
(3a) Compilation and symmetry self-audit: for each custom basis transform give
its mode matrix, operator convention, mode order and inverse, and for each
exactly preserved target symmetry give the generator- or complete-layer proof.
(4) Current-instance D_ans=1 parameter/generator/gate estimates, followed by
asymptotic gate/depth scaling and the depth-extension rule.
(4a) Finite-lattice symmetry strategy for each target state, including which
constraints are exact and which are diagnostic only.
(5) Short human-readable summary of the adapter handoff.

Then output one separate fenced JSON block using the following envelope. The
JSON is the authoritative machine interface: replace every descriptive
placeholder with exact model-specific, indexed and executable design facts,
include only sources actually used, and keep the field names unchanged. Do not
embed this JSON in the LaTeX document.

```json
{json.dumps(contract_example, ensure_ascii=False, indent=2)}
```

The handoff contains design facts only. Do not add optimizer settings,
convergence thresholds, hardware selection, or measurement instructions.

{LATEX_SAFETY_RULES}
"""
        system = (
            "You are a quantum state-preparation circuit architect. Select an ansatz "
            "from supplied literature when genuinely applicable, and specify its exact "
            "scalable gate-level structure. You do not design measurements or execution runtimes."
        )
        response = call_model_once(
            self.api_key, self.base_url, self.model_name, system, prompt,
            raw_response_path=raw_response_path,
        )
        def normalize_contract(candidate):
            candidate["schema_version"] = "1.1"
            candidate["contract_type"] = "state_preparation"
            candidate.pop("parameter_contract_sha256", None)
            candidate.pop("mapping_contract_sha256", None)
            candidate["state_labels"] = list(labels)
            normalize_literature_sources(candidate)
            sector = candidate.get("ansatz", {}).get("physical_sector")
            if isinstance(sector, dict):
                sector["required"] = requires_physical_sector(parameter_contract)
            return candidate
        return extract_design_response(
            response,
            validate_state_preparation_contract,
            parameter_contract,
            normalizer=normalize_contract,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the focused Agent-4 state-preparation designer.")
    parser.add_argument("--mapping-contract", required=True)
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--library", default="A4_library")
    parser.add_argument(
        "--experience-db", default=DEFAULT_DATABASE_PATH
    )
    parser.add_argument(
        "--energy-tolerance", type=float, default=DEFAULT_VQE_ENERGY_TOLERANCE
    )
    parser.add_argument(
        "--variance-tolerance", type=float, default=DEFAULT_VQE_VARIANCE_TOLERANCE
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    parser.add_argument("--search-attempt", type=int, default=1)
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Explicitly spend a new model call instead of reusing saved raw output.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    if args.search_attempt < 1:
        raise SystemExit("--search-attempt must be a positive integer.")
    if args.energy_tolerance <= 0.0 or args.variance_tolerance <= 0.0:
        raise SystemExit("VQE tolerances must be positive.")
    report_directory = categorized_output_directory(args.output_root, "reports")
    contract_directory = categorized_output_directory(args.output_root, "contracts")
    audit_directory = categorized_output_directory(args.output_root, "audit")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before calling Agent 4.1.")
    agent3_report, parameter_contract = load_agent3_inputs(
        args.parameter_contract, args.mapping_contract
    )
    references = extract_relevant_sections_from_library(
        library_path=args.library,
        target_keywords=STATE_KEYWORDS,
        priority_keywords=model_priority_keywords(parameter_contract),
        focus_terms=(
            "ansatz", "state preparation", "variational", "operator pool",
            "generator", "symmetry", "momentum", "particle number",
            "circuit depth", "gate count",
        ),
    )
    experience = format_vqe_experience_context(
        args.experience_db,
        parameter_contract,
        energy_tolerance=args.energy_tolerance,
        variance_tolerance=args.variance_tolerance,
    )
    agent = StatePreparationDesignAgent(
        api_key=api_key, base_url=args.base_url, model_name=args.model
    )
    raw_response_path = generator_raw_response_path(
        audit_directory, "Agent4_StatePreparation", output_tag,
        regenerate=args.regenerate,
    )
    report, contract = agent.design(
        agent3_report, parameter_contract, references, experience,
        raw_response_path=raw_response_path,
        search_attempt=args.search_attempt,
    )
    design_path = os.path.join(
        contract_directory,
        "Agent4_StatePreparationContract_{}.json".format(output_tag),
    )
    save_contract(
        contract, design_path, validate_state_preparation_contract, parameter_contract
    )
    audit_record = publish_audit_report(
        report,
        stage_name="Agent4_StatePreparation",
        output_tag=output_tag,
        report_directory=report_directory,
        audit_directory=audit_directory,
        compiled_filename="Agent4_StatePreparation_{}.tex".format(output_tag),
        rejected_filename="Agent4_StatePreparation_Rejected_{}.tex".format(output_tag),
        status_filename="Agent4_StatePreparation_AuditStatus_{}.json".format(output_tag),
    )
    report_path = audit_record["report_path"]
    agent.logger.info(
        "State-preparation audit report saved to %s (%s; nonblocking)",
        report_path, audit_record["status"],
    )
    agent.logger.info("State-preparation contract saved to %s", design_path)
    write_stage_output_manifest(
        args.output_manifest,
        "agent4_state",
        {
            "report": report_path,
            "audit_status": audit_record["status_path"],
            "design_contract": design_path,
            "generator_raw_response": raw_response_path,
        },
    )


if __name__ == "__main__":
    main()
