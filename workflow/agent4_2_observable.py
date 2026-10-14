# -*- coding: utf-8 -*-
"""Agent 4.2: observable-measurement and time-evolution circuit designer."""

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
    validate_observable_contract,
)
from audit_report_support import publish_audit_report
from pipeline_support import (
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)
from parameter_contract import format_parameter_contract
from observable_specification import (
    load_observable_specification,
)
from machine_contracts import validate_mapping_contract


OBSERVABLE_KEYWORDS = [
    "correlation function", "observable", "Wilson line", "Hadamard test",
    "interferometry", "real time", "time evolution", "Trotter",
    "Suzuki product formula", "Pauli string", "LCU", "ancilla",
    "controlled unitary", "uncontrolled time evolution", "unequal-time correlator",
    "same-state expectation value", "measurement", "quantum circuit",
    "CNOT sharing", "product formula ordering", "resource estimation",
]


def normalize_observable_design_contract(candidate, labels, equivalent_groups=None):
    """Canonicalize controller-owned interface spelling, not scientific facts."""
    candidate = json.loads(json.dumps(candidate, ensure_ascii=False))
    candidate["schema_version"] = "1.1"
    candidate["contract_type"] = "observable_measurement"
    candidate.pop("parameter_contract_sha256", None)
    candidate.pop("mapping_contract_sha256", None)
    candidate.pop("observable_specification_sha256", None)
    candidate["state_labels"] = list(labels)
    normalize_literature_sources(candidate)

    evolution = candidate.get("time_evolution")
    if isinstance(evolution, dict) and evolution.get("required") is True:
        # N_T is the runtime API's canonical variable name, not a physical
        # choice made by the model.
        evolution["step_symbol"] = "N_T"
        for target, aliases in (
            ("recommended_test_steps", (
                "recommended_steps", "test_steps", "trotter_steps",
            )),
            ("step_operations", (
                "single_step_operations", "one_step_operations",
                "trotter_step_operations",
            )),
        ):
            if target not in evolution:
                for alias in aliases:
                    if alias in evolution:
                        evolution[target] = evolution.pop(alias)
                        break

    measurement = candidate.get("measurement")
    if isinstance(measurement, dict):
        recipes = measurement.get("circuit_recipes")
        if (
            "circuit_recipe_ids" not in measurement
            and isinstance(recipes, list)
        ):
            measurement["circuit_recipe_ids"] = [
                item.get("recipe_id") for item in recipes
                if isinstance(item, dict)
            ]
        recipe_ids = measurement.get("circuit_recipe_ids")
        if isinstance(recipe_ids, list) and isinstance(recipes, list):
            # circuit_recipe_ids names only circuits returned by the adapter.
            # Models occasionally place reusable helper templates in the same
            # list. Preserve those definitions under operator_recipes and keep
            # the executable recipes in their declared order.
            by_id = {
                item.get("recipe_id"): item for item in recipes
                if isinstance(item, dict)
                and isinstance(item.get("recipe_id"), str)
            }
            if len(by_id) == len(recipes) and all(
                recipe_id in by_id for recipe_id in recipe_ids
            ):
                extra_ids = [
                    recipe_id for recipe_id in by_id
                    if recipe_id not in set(recipe_ids)
                ]
                if extra_ids:
                    operator_recipes = candidate.get("operator_recipes")
                    if not isinstance(operator_recipes, list):
                        operator_recipes = []
                        candidate["operator_recipes"] = operator_recipes
                    existing_ids = {
                        item.get("id") for item in operator_recipes
                        if isinstance(item, dict)
                    }
                    for recipe_id in extra_ids:
                        if recipe_id not in existing_ids:
                            operator_recipes.append({
                                "id": recipe_id,
                                "operations": by_id[recipe_id].get(
                                    "operations", []
                                ),
                            })
                    measurement["circuit_recipes"] = [
                        by_id[recipe_id] for recipe_id in recipe_ids
                    ]

    state_interface = candidate.get("state_preparation_interface")
    if isinstance(state_interface, dict) and not state_interface.get(
        "required_operations"
    ):
        groups = equivalent_groups or []
        one_reusable_state = any(
            isinstance(group, list) and set(labels).issubset(set(group))
            for group in groups
        )
        selected_labels = [labels[-1]] if one_reusable_state else list(labels)
        state_interface["required_operations"] = [
            "prepare_{}".format(label) for label in selected_labels
        ]

    reconstruction = candidate.get("reconstruction")
    if (
        isinstance(reconstruction, dict)
        and isinstance(measurement, dict)
        and "output_components" not in reconstruction
        and isinstance(measurement.get("components"), list)
    ):
        reconstruction["output_components"] = list(measurement["components"])

    interface = candidate.get("adapter_interface")
    if not isinstance(interface, dict):
        interface = {}
        candidate["adapter_interface"] = interface
    interface["builder"] = "build_observable_measurements"
    return candidate


class ObservableDesignAgent:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", model_name="gpt-5.6-sol"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.logger = setup_logger("Agent4_Observable")

    def design(
        self, agent3_report, parameter_contract, observable_specification,
        library_text, raw_response_path=None,
    ):
        labels = parameter_contract["external_states"]["labels"]
        equivalent_groups = parameter_contract["external_states"]["equivalent_groups"]
        contract_example = {
            "schema_version": "1.1",
            "contract_type": "observable_measurement",
            "state_labels": labels,
            "literature_sources": [
                {
                    "source_path": "relative path from the supplied toolbox",
                    "locator": "exact section/equation/figure/caption",
                    "applicability": "direct|modified|incompatible and concise reason",
                }
            ],
            "state_preparation_interface": {
                "treat_as_black_box": True,
                "required_operations": [
                    "prepare_{}".format(label) for label in labels
                ] + ["inverse"],
            },
            "operator_recipes": [
                {
                    "id": "exact_stable_operator_id",
                    "operations": [
                        {
                            "operation": "gate_or_indexed_loop",
                            "gate": "structured mathematical primitive or exact gate family",
                            "targets": ["indexed logical qubit labels"],
                            "controls": [],
                            "angle": "exact symbolic angle or null",
                        }
                    ],
                }
            ],
            "measurement": {
                "protocol_id": "model-specific stable identifier",
                "components": ["real", "imag"],
                "ancilla_qubits": 1,
                "circuit_recipe_ids": [
                    "trajectory_0_real",
                    "trajectory_0_imag",
                ],
                "circuit_recipes": [
                    {
                        "recipe_id": "trajectory_0_real",
                        "component": "real",
                        "operations": [
                            {
                                "operation": "compose",
                                "reference": "exact operator/state/evolution recipe id",
                            }
                        ],
                    },
                    {
                        "recipe_id": "trajectory_0_imag",
                        "component": "imag",
                        "operations": [
                            {
                                "operation": "compose",
                                "reference": "exact operator/state/evolution recipe id",
                            }
                        ],
                    },
                ],
            },
            "time_evolution": {
                "required": True,
                "method": "selected model-specific product formula or exact method",
                "step_symbol": "N_T",
                "recommended_test_steps": [1, 2, 4, 8, 16],
                "symmetry_preservation_rule": "exact mandatory sector rule plus any disclosed finite-step breaking of other symmetries",
                "step_operations": [
                    {
                        "operation": "gate_or_indexed_loop",
                        "gate": "structured mathematical primitive or exact gate family",
                        "targets": ["indexed logical qubit labels"],
                        "angle": "exact symbolic angle as a function of N_T",
                    }
                ],
            },
            "reconstruction": {
                "output_components": ["real", "imag"],
                "rule": "complete coefficient/trajectory reconstruction rule",
                "terms": [
                    {
                        "recipe_id": "trajectory_0_real",
                        "coefficient": "exact symbolic coefficient",
                    }
                ],
            },
            "adapter_interface": {
                "builder": "build_observable_measurements",
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

[AUTHORITATIVE USER OBSERVABLE SPECIFICATION]
```json
{json.dumps(observable_specification, ensure_ascii=False, indent=2, sort_keys=True)}
```

[LOCAL OBSERVABLE-MEASUREMENT LITERATURE TOOLBOX]
{library_text or "No readable local source was retrieved."}

You are Agent 4.2. Design ONLY the observable-measurement protocol and the
structured finite-evolution mathematics needed to evaluate Agent 3's exact
observable. State preparation is supplied
later through one abstract unitary U_label for each named state. Do not choose an
ansatz family, ansatz gates, ansatz depth, VQE optimizer, checkpoint rule,
hardware backend, Qiskit implementation, or IBM submission policy.

[CONTROLLER ROLE BOUNDARY]
Agent2 and Agent3 own the exact target operator, path geometry, ordering and
qubit mapping; they do not own the runtime approximation algorithm. Upstream
phrases such as "no numerical time step", "no Trotterization",
"temporal_discretization: none", or "product_formula: forbidden" mean only
that a discretized algorithm must not replace the exact mathematical
definition at those stages. They are not a prohibition on Agent4.2 selecting a
controlled convergent implementation. Preserve the authoritative path geometry,
orientation and ordered exponential as the N_T -> infinity target, but when no finite
exact gate decomposition exists, discretize along that same path and choose an
executable approximation with an explicit positive N_T and error/convergence
interpretation. Never replace it by a physically inequivalent path merely for
circuit convenience.
This controller policy supersedes any upstream algorithm-prohibition wording.
Do not return an "unbuildable"/empty measurement contract solely because of
such wording: select and fully specify the convergent finite-gate
implementation owned by this stage.

Observable-design requirements:

1. Anchor every operator, coefficient, time argument, Wilson transporter,
   Jordan-Wigner string, qubit index, and rotation angle to Agent 3. Never rename
   a parameter or silently assign a null value.
   Preserve the arbitrary number of operator insertions and every sampling
   variable. Do not reduce an n-point function to a two-point function.
2. Preserve all named states and every term in observable.matrix_elements as
   independent interfaces. Their authoritative reuse groups are
   {json.dumps(equivalent_groups)}. For each term preserve its coefficient, bra,
   ket, and sign. Do not replace the requested sum or subtraction by one matrix
   element. A diagonal term may use an exact same-state fast path; an off-diagonal
   term must use the appropriate transition-amplitude protocol. Put all circuits
   needed for the complete current linear combination into circuit_recipe_ids.
3. Treat every U_label as a legal black box in macroscopic diagrams. State
   exactly whether the protocol requires preparation, inverse, controlled, or
   multiplexed forms so measurement_runtime can check compatibility later. Never
   assume that a controlled or multiplexed form of an opaque unitary is free;
   require it only when no exact lower-resource construction is available.
4. Compare applicable literature protocols and cite relative source path plus
   exact section/equation/figure/caption. If a source omits the needed circuit,
   do not invent it and attribute it to that source. A new proposal must be
   labelled as such and derived independently. Include resource-aware alternatives
   for unequal-time correlators, not only the first applicable Hadamard test.
   First decide from the requested operator and time arguments whether any time
   evolution is required; never introduce N_T merely because the envelope has a
   time_evolution field. Before using any interferometric or LCU protocol, apply this simple-observable
   fast path: if the requested observable is the mapped Hamiltonian itself, or
   another explicit Hermitian Pauli sum, and every requested matrix element is
   diagonal (bra=ket), measure each required Pauli expectation directly on the
   prepared state and reconstruct the weighted sum classically. Use no ancilla,
   LCU, block encoding, Hadamard test, controlled-Hamiltonian, or time evolution
   for this case. Add identity contributions classically without a circuit.
   First combine algebraically identical Pauli terms. The current adapter's
   stable scalar interface returns one expectation for each recipe_id with one
   readout_pauli, so represent the remaining scalar expectations through that
   interface. You may identify commuting groups or shared-basis execution as a
   resource optimization, but do not claim the current adapter performs grouped
   readout unless its executable recipe actually implements it. Treat
   `readout_pauli` as the Pauli observable that the
   runtime Estimator applies after the returned circuit. For a direct Pauli
   expectation, return the unrotated logical circuit with `readout_pauli=P`;
   do not append H or S-dagger/H gates merely to convert X or Y readout to Z.
   This rule does not remove gates intrinsic to an interferometric protocol:
   a Hadamard/SWAP test retains its required ancilla gates and uses a Z-only
   ancilla `readout_pauli`. Never encode the same readout basis change in both
   the circuit and `readout_pauli`. Use an interferometric fallback only for genuinely
   off-diagonal matrix elements or when direct Pauli measurement is
   mathematically inapplicable, and state the reason. When this fast path
   applies, set time_evolution.required=false; the interface may retain
   trotter_steps, but the runtime passes the existing sentinel value 0 and the
   recipes must not use it.
5. Decompose observable/Wilson/Pauli operations into finite executable recipes
   with exact indexed angles and rotation conventions. For time evolution,
   hand off the mathematical structure rather than acting as a Qiskit compiler:
   identify every ordered generator as an individual Pauli term, a mutually
   commuting Pauli group, a quadratic number/charge-preserving block, or a
   genuinely general local block. Preserve coefficients, supports, grouping and
   order explicitly. Do not prescribe a native hardware basis, routing, QSD, or
   dense-matrix circuit when an equivalent structured representation exists.
   Pauli reordering or shared evolution is allowed only after proving the
   relevant commutation and preserving the declared target unitary.
   Apply the following model-independent optimization hierarchy before falling
   back to independent Pauli ladders:
   (i) combine proved commuting terms and repeated identical rotations;
   (ii) recognize paired Pauli strings that are the two quadratures of the
   same mapped quadratic fermion generator, including Jordan--Wigner strings,
   and represent them as one number/charge-preserving Givens, matchgate, fSim,
   or equivalent structured block when the equivalence is derived explicitly;
   (iii) expose shared parity support across consecutive strings so Agent5 can
   compute and uncompute that parity once rather than once per term; and
   (iv) fuse equal half steps at a symmetric-formula midpoint and between
   adjacent product-formula steps. Compare the resulting symbolic logical
   two-qubit count with the independent-Pauli baseline and select the lighter
   exact realization. None of these structures is mandatory: when its algebraic
   conditions are not proved, retain the ordinary structured Pauli exponential.
   Never label a genuinely interacting or gauge-field block as Gaussian,
   number-preserving, or matchgate merely to reduce a resource estimate.
   Preserve Agent3's parameter expressions literally throughout this grouping;
   do not replace them with generation-run numerical values.
   For every controlled operation, prefer an explicit factor-by-factor local
   decomposition when it is algebraically equivalent; do not recommend a
   controlled opaque whole-state or whole-evolution block merely for diagrammatic
   convenience. Report concrete logical depth and two-qubit-gate scaling so an
   obviously infeasible hardware realization can be rejected before submission.
   Give per-recipe symbolic counts including state preparation, evolution,
   controlled endpoint operations, and ancillas; do not hide state-interface cost.
6. Do not hard-code second-order Suzuki-Trotter. Select the exact, product-formula,
   or other method justified for this Hamiltonian and observable. If a product
   formula is selected, give the full single-step ordering, declare N_T as the
   variable step count, and recommend an increasing noiseless test schedule.
   Prefer a decomposition into individual Pauli terms, proved commuting groups,
   or known exactly synthesizable structured blocks. Do not demand the exact
   synthesis of a generic dense local matrix merely for convenience. If no
   scalable exact structured realization exists, place the required
   approximation explicitly in the selected product formula and expose its
   convergence through N_T; never hide a second undeclared Trotterization inside
   a nominally exact block. Agent 5 chooses the locally available Qiskit
   implementation of this declared mathematics and may not change its error
   order or repetition count.
   Compare applicable grouped/commutator-aware second-order and higher-order
   alternatives by the total two-qubit cost required at equal target accuracy;
   write the chosen step schedule after all proved midpoint and inter-step
   boundary fusions, so the reduction does not depend on a transpiler noticing
   opaque cancellations. Merge adjacent identical rotations whenever the equality is proved. If the
   supplied evidence cannot establish a winner, say so and specify the noiseless
   comparison to run instead of inventing performance numbers.
   Evolution must preserve mandatory physical/Gauss constraints and commute
   with conserved quantum numbers defining each target at every finite N_T.
   Operator insertions may change a global charge or other quantum number:
   retain and evolve every intermediate sector required by the ordered
   observable. Never project an intermediate state back into the external-state
   sector. For gauge-charged insertions, preserve the full transporter-dressed
   physical construction and the intermediate constraints declared upstream.
   For other Agent3 symmetries
   classified `exact`, prefer symmetry-preserving blocks when practical; if a
   selected finite-step formula breaks one, quantify/disclose that artifact and
   test its convergence rather than pretending every block commutes. Do not
   impose broken or continuum-only symmetries. Record the result in
   time_evolution.symmetry_preservation_rule.
7. Explicitly test whether common uncontrolled evolution gives each required
   overlap. In particular, for a diagonal term with bra=ket=chi, derive
   whether the sequence prepare chi once, controlled-B, uncontrolled U,
   controlled-A yields
   <U chi|A U B|chi>=<chi|U^dagger A U B|chi>. If it does, use this exact fast
   path: do not add controlled-U, U^dagger, controlled state preparation, or
   controlled inverse preparation. If the identity does not apply, state why and
   use the general independent-state fallback without pretending it is cheap.
8. Derive all circuits required for the requested real and/or imaginary parts,
   and give the exact classical reconstruction formula, including every
   matrix-element coefficient, Pauli/LCU coefficient, trajectory weight, sign, normalization, and measurement
   convention. For every interferometric branch, define the complex overlap c
   explicitly and algebraically derive the sign of the measured expectation.
   Under the standard (|0>|psi>+|1>V|psi>)/sqrt(2) branch convention, an
   S-dagger then H ancilla basis rotation before Z readout measures +Im(c),
   whereas S then H measures -Im(c); the component label and reconstruction
   signs must match the chosen circuit. Do not assume every observable requires both components; decide
   from the requested matrix element and explain the choice.
   Generate only components requested by the authoritative observable input.
   Never drop a requested component or sampling point merely because another
   component has a larger signal. Symmetry may reduce executed recipes or grid
   points only when Agent2/3 classify that symmetry as exact, the external
   states have the required eigenvalues, the finite-step circuit preserves it,
   and the omitted values are reconstructed by an explicit proved rule.
9. The adapter has exactly this model-independent interface:
   build_observable_measurements(state_circuits, trotter_steps, sample_values).
   It is called once for one selected sampling assignment and step count and
   must return every circuit needed for the
   reconstruction. Do not require extra component or trajectory arguments.
   Enumerate every returned circuit in measurement.circuit_recipe_ids using a
   unique stable ID that distinguishes its component and, when applicable, its
   trajectory/operator term. The list must be exhaustive: for example, eight
   trajectories measured in real and imaginary bases require sixteen IDs, not
   two generic component IDs. measurement.circuit_recipes must contain exactly
   those executable IDs in the same order. Put reusable helper/template
   definitions in operator_recipes, never in measurement.circuit_recipes.
   reconstruct_observable receives a mapping from
   these exact IDs to expectation values; reconstruct_observable also receives
   the same sample_values and performs the full reconstruction.
10. Treat sample_values as one immutable assignment from the authoritative
    sampling request. Give exact rules for substituting it into every insertion
    position, transporter path and evolution time, and require the adapter to
    reject missing, extra, or physically incompatible assignments. The runtime,
    not Agent 4.2 or Agent 5, owns the loop over sampling assignments.
11. Every selected operation must be translatable by Agent 5 into a finite
    Qiskit gate construction. Do not declare an ideal TIME_ORDERED_EXP,
    analog-Hamiltonian channel, or other opaque exact-evolution primitive. If a
    nontrivial evolution has no justified finite exact gate decomposition,
    choose a literature-supported scalable approximation, expose its positive
    integer step count as N_T, provide increasing positive
    recommended_test_steps, and give the executable one-step operation order.
    Stop at this structured mathematical interface: Agent 4.2 owns the grouping,
    order, approximation and symmetry argument, while Agent 5 owns Qiskit
    synthesis and the downstream transpiler owns basis translation and routing.
12. Separate the reusable measurement algorithm from the current numerical
    parameter point. Express every coupling-dependent angle, Hamiltonian
    coefficient, normalization, reconstruction weight, lattice size `L`, qubit
    count, link/site loop and valid sampling assignment in terms of runtime
    inputs. Do not turn the current `L`, current Pauli-string length or current
    sample grid into a whitelist. The same physical model, external states and
    observable form must use this one design when `m`, `g`, `a`, `L` or the grid
    values change. The previously approved boundary/discretization/encoding
    method remains the program's method; those internal choices are not a
    second reuse-matching interface.

Output exactly two independent fenced blocks and no other text. First output one
concise compilable LaTeX audit report with these sections:
(0) Exact qubit-space matrix element and external-state interfaces.
(1) Literature protocols, same-state fast path, and independent-state fallback.
(2) Operator/Wilson-line finite recipes.
(3) Time-evolution method and scalable one-step circuit, if required.
(4) Complete real/imaginary measurement circuit recipes.
(5) Reconstruction formula and concrete plus asymptotic resource comparison.
(6) Short human-readable summary of the adapter handoff.

Then output one separate fenced JSON block using the following envelope. The
JSON is the authoritative machine interface: replace descriptive placeholders
with exact indexed/executable recipes, set time_evolution.required accurately,
include only sources actually used, and keep the field names unchanged. Do not
embed this JSON in the LaTeX document.

```json
{json.dumps(contract_example, ensure_ascii=False, indent=2)}
```

The handoff contains measurement-design facts only. Do not add ansatz,
optimizer, convergence, or hardware-selection settings. Its
state_preparation_interface.required_operations must list only operations used
by the selected current protocol; do not retain controlled/inverse operations
from an unselected fallback.

{LATEX_SAFETY_RULES}
"""
        system = (
            "You are a quantum observable-measurement circuit architect. Treat state "
            "preparation as a declared unitary interface and design the exact scalable "
            "operator, evolution, interferometry, and reconstruction recipes only."
        )
        response = call_model_once(
            self.api_key, self.base_url, self.model_name, system, prompt,
            raw_response_path=raw_response_path,
        )
        def normalize_contract(candidate):
            return normalize_observable_design_contract(
                candidate, labels, equivalent_groups
            )
        return extract_design_response(
            response,
            validate_observable_contract,
            parameter_contract,
            observable_specification,
            normalizer=normalize_contract,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the focused Agent-4 observable designer.")
    parser.add_argument("--mapping-contract", required=True)
    parser.add_argument("--parameter-contract", required=True)
    parser.add_argument("--observable-spec", required=True)
    parser.add_argument("--library", default="A4_library")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Explicitly spend a new model call instead of reusing saved raw output.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    report_directory = categorized_output_directory(args.output_root, "reports")
    contract_directory = categorized_output_directory(args.output_root, "contracts")
    audit_directory = categorized_output_directory(args.output_root, "audit")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before calling Agent 4.2.")
    agent3_report, parameter_contract = load_agent3_inputs(
        args.parameter_contract, args.mapping_contract
    )
    observable_specification = load_observable_specification(args.observable_spec)
    references = extract_relevant_sections_from_library(
        library_path=args.library,
        target_keywords=OBSERVABLE_KEYWORDS,
        priority_keywords=model_priority_keywords(
            parameter_contract, task_context=observable_specification
        ),
        focus_terms=("correlation", "observable", "hadamard test", "trotter", "wilson line"),
    )
    agent = ObservableDesignAgent(
        api_key=api_key, base_url=args.base_url, model_name=args.model
    )
    raw_response_path = generator_raw_response_path(
        audit_directory, "Agent4_Observable", output_tag,
        regenerate=args.regenerate,
    )
    report, contract = agent.design(
        agent3_report, parameter_contract, observable_specification, references,
        raw_response_path=raw_response_path,
    )
    design_path = os.path.join(
        contract_directory,
        "Agent4_ObservableContract_{}.json".format(output_tag),
    )
    save_contract(contract, design_path, validate_observable_contract, parameter_contract)
    audit_record = publish_audit_report(
        report,
        stage_name="Agent4_Observable",
        output_tag=output_tag,
        report_directory=report_directory,
        audit_directory=audit_directory,
        compiled_filename="Agent4_Observable_{}.tex".format(output_tag),
        rejected_filename="Agent4_Observable_Rejected_{}.tex".format(output_tag),
        status_filename="Agent4_Observable_AuditStatus_{}.json".format(output_tag),
    )
    report_path = audit_record["report_path"]
    agent.logger.info(
        "Observable audit report saved to %s (%s; nonblocking)",
        report_path, audit_record["status"],
    )
    agent.logger.info("Observable contract saved to %s", design_path)
    write_stage_output_manifest(
        args.output_manifest,
        "agent4_observable",
        {
            "report": report_path,
            "audit_status": audit_record["status_path"],
            "design_contract": design_path,
            "generator_raw_response": raw_response_path,
        },
    )


if __name__ == "__main__":
    main()
