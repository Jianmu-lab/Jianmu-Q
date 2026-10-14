# -*- coding: utf-8 -*-
"""Agents 6-8: independent reference-code generation and result review.

No stage searches for a "latest" file and no stage automatically asks a model
to repair generated output. A failed reference program remains visible for
human diagnosis. Numerical acceptance and the hardware manifest are produced
by ``crosscheck_runtime.py``, not by an LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict

from openai import OpenAI

from parameter_contract import (
    format_parameter_contract,
    load_parameter_contract,
)
from machine_contracts import (
    format_machine_contract,
    load_machine_contract,
    validate_discretization_contract,
)
from observable_specification import (
    expand_observable_samples,
    load_observable_specification,
    resolve_observable_sampling,
)
from run_spec import load_run_spec


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def extract_single_code_block(text: str, language: str) -> str:
    pattern = r"```{}\s*(.*?)\s*```".format(re.escape(language))
    blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if len(blocks) != 1:
        raise ValueError(
            "Expected exactly one {} code block; found {}.".format(language, len(blocks))
        )
    return blocks[0].strip()


def _julia_call_arguments(source: str, function_name: str):
    """Return top-level argument strings for simple Julia function calls."""
    calls = []
    pattern = re.compile(r"\b{}\s*\(".format(re.escape(function_name)))
    for match in pattern.finditer(source):
        start = source.find("(", match.start())
        depth = 1
        quote = None
        escaped = False
        arguments = []
        argument_start = start + 1
        index = start + 1
        while index < len(source) and depth:
            char = source[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char == '"':
                quote = char
            elif char == "#":
                newline = source.find("\n", index)
                index = len(source) if newline < 0 else newline
                continue
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth == 0:
                    arguments.append(source[argument_start:index].strip())
                    break
            elif char == "," and depth == 1:
                arguments.append(source[argument_start:index].strip())
                argument_start = index + 1
            index += 1
        if depth != 0:
            raise ValueError("Unbalanced {} call in generated Julia source.".format(function_name))
        calls.append((source.count("\n", 0, match.start()) + 1, arguments))
    return calls


def validate_itensor_reference_source(source: str) -> str:
    """Reject unsupported ITensor contractions before publishing Agent7 output."""
    if re.search(r"\beigen\s*\(", source) and re.search(
        r"\b(?:vector_to_mps|dense_to_mps|mps_from_vector)\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            "The ITensor reference diagonalizes a dense Hamiltonian and converts "
            "its eigenvectors to MPS. Requested states must be solved independently "
            "with an MPS/DMRG algorithm."
        )
    calls = _julia_call_arguments(source, "inner")
    if not calls:
        raise ValueError("Generated ITensor reference contains no inner contractions.")
    for line, arguments in calls:
        if len(arguments) > 4:
            raise ValueError(
                "Unsupported {}-argument ITensor inner call at line {}; compose "
                "multi-MPO products first and use a four-argument contraction.".format(
                    len(arguments), line
                )
            )
        if (
            len(arguments) == 3
            and arguments[0] == arguments[2]
            and not arguments[0].endswith("'")
        ):
            raise ValueError(
                "Deprecated inner(psi, A, psi) form at line {}; use "
                "inner(psi', A, psi) or Apply.".format(line)
            )
        if len(arguments) == 4 and arguments[0].endswith("'"):
            raise ValueError(
                "Unsupported bra-first four-argument inner call at line {}; "
                "ITensor requires inner(A, psi, B, phi), with MPO arguments "
                "first and third. For <psi|H^2|psi>, use "
                "inner(H, psi, H, psi).".format(line)
            )
    return source


class ModelAgent:
    def __init__(self, name: str, model: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before invoking an Agent.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        self.model = model
        self.logger = setup_logger(name)

    def call_once(self, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return content


def _reference_common_prompt(
    agent2_report: str,
    parameter_contract: Dict[str, Any],
    observable_specification: Dict[str, Any],
    continuum_conventions: Dict[str, Any],
) -> str:
    execution_specification = resolve_observable_sampling(
        observable_specification, parameter_contract
    )
    samples = expand_observable_samples(execution_specification)
    return f"""
[AUTHORITATIVE AGENT 2 LATTICE-DISCRETIZATION CONTRACT]
```json
{agent2_report}
```

[AUTHORITATIVE PARAMETER CONTRACT]
```json
{format_parameter_contract(parameter_contract)}
```

[AUTHORITATIVE USER OBSERVABLE SPECIFICATION]
```json
{json.dumps(observable_specification, ensure_ascii=False, indent=2, sort_keys=True)}
```

The Observable Specification is the sole authority for requested components,
operator insertions, matrix elements, ordering and sampling. Do not reproduce
or validate these as Parameter Contract fields. For backward compatibility,
ignore a legacy Parameter Contract parameter named `requested_components`;
never compare its spelling (for example `imag` versus `imaginary`) with the
canonical Observable Specification.

[USER-RESOLVED CONTINUUM CONVENTIONS]
```json
{json.dumps(continuum_conventions, ensure_ascii=False, indent=2, sort_keys=True)}
```

These conventions are authoritative for the numerical observable. They resolve
corresponding convention-only placeholders that were previously left null; use
them without changing Hamiltonian, lattice, state, or algorithmic parameters.
Never infer the plus/minus sign in a gamma or light-cone definition from its
normalization factor. Read the explicit gamma definition from the authoritative
Observable Specification and Agent-2 lattice definitions and preserve it
literally. The normalization controls only the overall factor. Record the
applied convention in the output as optional metadata.

The generated program must accept required CLI options
`--parameter-contract PATH`, `--observable-spec PATH`, and `--output PATH`.
At runtime it must load the current two JSON files, derive every numerical
coupling, Hamiltonian coefficient, state target, sampling assignment and
observable weight from them, and atomically write exactly one JSON object with
this stable interface:
{{
  "reference_schema_version": "1.0",
  "method": "METHOD_ID",
  "ground_state_energy": finite number,
  "energy_variance": finite nonnegative number,
  "state_results": {{
    "each requested state label": {{"energy": finite number, "energy_variance": finite nonnegative number}}
  }},
  "observable_samples": [
    {{
      "sample_id": "one exact ID from the supplied sampling list",
      "sample_values": object copied exactly from that assignment,
      "observable_real": finite number,
      "observable_imag": finite number,
      "uncertainty_real": finite nonnegative number,
      "uncertainty_imag": finite nonnegative number
    }}
  ],
  "convergence": {{"passed": boolean, "details": object}}
}}

The expanded sampling assignments are:
```json
{json.dumps(samples, ensure_ascii=False, indent=2, sort_keys=True)}
```

The reference observable is the arbitrary n-point continuum/lattice observable
defined by Agent 2 and the immutable user specification. Produce exactly one
observable_samples entry for every supplied sampling assignment. Derive it
independently without reading any VQE,
Agent-5 adapter, QPY, noiseless-measurement, hardware, or cross-check result.
It is one reference value; do not manufacture an ansatz-depth or Trotter-step
dependence. Evaluate every term in observable.matrix_elements and apply its
coefficient exactly; when it is absent, evaluate the legacy phi/psi matrix
element. Preserve all named states independently unless the Parameter Contract
explicitly places them in one equivalent-state group. On any
contradiction or unsupported operation, raise an error rather than silently
changing the Hamiltonian, boundary, physical sector, state, or observable.
The program is a reusable model-specific numerical implementation. It may
embed the unchanged discretization and operator construction, but must not
freeze masses, couplings, lattice spacing, theta, lattice size `L`, sampling
points, or other run-varying numerical values from the generation request.
Construct site/link sets, Hilbert-space dimensions and observable samples from
the current runtime files, so the same source program supports every valid `L`
and grid for this physical target. The program retains its approved numerical
method (including boundary, discretization, field encoding and gauge treatment)
as its implementation choice; those internal choices are not a reason to reject
new `m`, `g`, `a`, `L` or grid values. Abort only when the model, external-state
quantum numbers or observable/operator form itself is different, or the current
inputs violate the declared numerical method's validity domain or exceed an
explicitly supported size. State such limitations and fail clearly; never use
old values or silently change the numerical method.
"""


class Agent6SciPy(ModelAgent):
    def __init__(self, model: str) -> None:
        super().__init__("Agent6_SciPy", model)

    def generate(
        self,
        agent2_report: str,
        parameter_contract: Dict[str, Any],
        observable_specification: Dict[str, Any],
        continuum_conventions: Dict[str, Any],
    ) -> str:
        prompt = _reference_common_prompt(
            agent2_report, parameter_contract, observable_specification,
            continuum_conventions,
        ) + r"""
Write one self-contained Python/SciPy exact-reference program.

Requirements:
- Load the selected Parameter Contract from `--parameter-contract` and validate
  all values used.
- Reconstruct Agent 2's Hamiltonian and observable independently; do not import
  an Agent-5 adapter or load its QPY circuits.
- Preserve Agent 2's exact boundary conditions and boundary fluxes.
- If this is a gauge theory with an explicit gauge Hilbert space, construct the
  exact simultaneous physical basis and restrict H and all states/operators to
  it before diagonalization. If Agent 2 eliminated Gauss law exactly, use that
  reduced Hilbert space and do not add a second gauge space or penalty. For a
  non-gauge theory, use the declared full Hilbert space.
- Abort if the required physical space is empty or H does not preserve it to
  numerical tolerance.
- Use SciPy sparse construction and exact diagonalization/exponentiation at the
  declared finite size. Calculate every required named state, its energy and
  variance, every requested matrix element, and their complex linear combination directly.
- Set method to "scipy_exact". Numerical roundoff estimates may be used for the
  uncertainty fields, but they must not hide discretization/model error.
- Output exactly one fenced Python block and no other text.
"""
        response = self.call_once(
            "You write auditable independent SciPy exact-reference programs.", prompt
        )
        return extract_single_code_block(response, "python")


class Agent7ITensor(ModelAgent):
    def __init__(self, model: str, julia_executable: str = "julia") -> None:
        super().__init__("Agent7_ITensor", model)
        self.julia_executable = julia_executable

    def generate(
        self,
        agent2_report: str,
        parameter_contract: Dict[str, Any],
        observable_specification: Dict[str, Any],
        continuum_conventions: Dict[str, Any],
    ) -> str:
        prompt = _reference_common_prompt(
            agent2_report, parameter_contract, observable_specification,
            continuum_conventions,
        ) + r"""
Write one self-contained Julia/ITensor tensor-network reference program.

Requirements:
- Start with explicit ITensors, ITensorMPS and JSON3 imports and put execution
  in main(). Apart from Julia standard libraries, do not import any other
  package. Accept the three common runtime CLI options without relying on
  hard-coded input or output filenames.
- Load the selected Parameter Contract from `--parameter-contract` and
  reconstruct Agent 2's Hamiltonian
  and observable independently; never import Agent-5 Qiskit code or QPY data.
- Keep runtime input validation minimal. Read only fields needed to construct
  the Hamiltonian, requested states and observable, plus the few structural
  choices that would change this implementation (lattice shape/boundary and
  gauge representation). Do not revalidate descriptive metadata, provenance,
  model-name spelling, schema prose, duplicated aliases, or fields unused by
  the calculation. Validate each required field once. Follow its actual JSON
  type: for a nested object such as index ranges, read its named child arrays;
  never also pass the containing object to an array validator.
- Preserve the exact boundary conditions, all named-state definitions, and the
  complete matrix-element linear combination.
- For a gauge theory, use the exact reduced Hilbert space if Agent 2 eliminated
  Gauss law. Otherwise enforce the declared physical sector by exact symmetry
  when available and explicitly test every constraint residual. Never accept a
  finite-penalty ground state without the residual test. Abort if no physical
  state is found. Do not invent constraints for a non-gauge theory.
- Distinguish those two gauge formulations explicitly. If Agent 2 eliminated
  gauge links exactly, Gauss law is a defining identity: verify the remaining
  matter-sector quantum numbers, record the analytic Gauss residual as zero,
  and never construct or truncate a zero Gauss-law or zero commutator QN MPO.
  If gauge links remain explicit, construct matter and link sites, build every
  nontrivial local Gauss operator, and evaluate each residual by scalar
  contraction; never assume those residuals vanish.
- With QN-conserving MPS, never evaluate an exactly annihilating conserved
  constraint by materializing `apply(Q, psi)`: an exact zero QN MPS can have no
  blocks and fail during compression. When QN flux exactly fixes the requested
  charge sector, use that exact sector statement and record its analytic
  residual as zero. Otherwise compute a nontrivial residual by scalar contraction
  such as `sqrt(max(0, real(inner(Q, psi, Q, psi))))`, without constructing
  `Q*psi`.
- When H is manifestly particle-number conserving term by term, verify that
  every term has equal creation and annihilation count and record [H,N]=0
  analytically. Never approximate HN and NH with separate `apply` calls and
  subtract their norms; independent MPO compression creates a false residual.
- Do not convert a parity-odd single-fermion OpSum to an MPO. Apply its local
  ITensor directly to the MPS with `product`, including the exact preceding
  fermion-parity/Jordan-Wigner string required by the chosen site ordering.
  Keep the full correlator parity even and verify the resulting QN flux changes.
- A state produced by a nonunitary insertion such as C or Cdag is generally
  unnormalized. For every particle-number or conserved-charge check on such an
  intermediate MPS, compute its norm squared first and divide both first and
  second moments by that norm before comparing the sector or variance. An
  identically zero branch contributes zero and has no normalized sector to
  test. Never compare a raw unnormalized expectation directly with an integer
  sector label.
- Treat `OpSum` accumulation as value-returning Julia code. If a helper uses
  `os += term`, return the updated `os` and assign it at every call
  (`os = helper(os, ...)`), or add terms inline. Never assume rebinding a helper's
  local argument mutates the caller's `OpSum`; verify every Hamiltonian sector
  (including diagonal Coulomb terms) is actually present.
- Use only these ITensor `inner` signatures for operator expectations:
  `inner(psi', A, phi)` for one MPO, or `inner(A, psi, B, phi)` for two MPOs.
  The four-argument signature always has MPOs in positions one and three and
  MPSs in positions two and four. In particular calculate
  `<psi|H^2|psi>` as `inner(H,psi,H,psi)`; never write the unsupported
  bra-first form `inner(psi',H,H,psi)`.
- Never invent a six-argument contraction such as
  `inner(H,Q,psi,H,Q,psi)`. Only when the product is not analytically the zero
  QN operator, first construct one composite MPO, for example `HQ = apply(H,Q)`,
  then use `inner(HQ,psi,HQ,psi)` or `inner(HQ,psi,QH,psi)`.
- Use `inner(psi', A, psi)` rather than the deprecated `inner(psi, A, psi)`.
- Choose a tensor-network solver appropriate to the Hamiltonian, boundary
  condition and requested states. The requested state energies and MPS must be
  obtained by an actual tensor-network variational solver (normally ground- and
  excited-state DMRG with explicit orthogonality penalties). Never diagonalize
  the dense/full-Hilbert Hamiltonian and then factor its eigenvectors into MPS;
  that is not an independent tensor-network reference. A dense calculation is
  allowed only as a separately labelled small-system diagnostic after the MPS
  result has already been obtained, and must not supply or initialize that MPS.
  When DMRG is used, use deterministic restarts
  and a declared convergence scan in sweep count, maximum bond dimension and
  truncation cutoff; include a time-step scan only when time evolution is
  required. Keep states as MPS and operators as MPO. Use the API provided by the
  installed ITensors/ITensorMPS version; for the currently tested 0.6 API,
  configure `Sweeps` with `maxdim!`, `mindim!`, `cutoff!`, and `noise!`, and pass
  Krylov controls such as `eigsolve_maxiter` directly to `dmrg` when needed.
- For real-time TDVP evolution `exp(-im * H * time)`, pass the complex total
  evolution parameter `-im * time` together with the positive integer keyword
  `nsteps=nsteps`, and let ITensorTDVP derive the complex time step. Never pair
  a complex total evolution parameter with a real-valued `time_step`.
- Calculate an uncertainty bound from final changes/truncation diagnostics.
  Set convergence.passed false when the requested tolerances are not achieved;
  do not merely print the last value as converged.
- Set method to "itensor_mps" and emit the stable reference JSON interface.
- Output exactly one fenced Julia block and no other text.
"""
        response = self.call_once(
            "You write auditable independent Julia/ITensor reference programs.", prompt
        )
        source = validate_itensor_reference_source(
            extract_single_code_block(response, "julia")
        )
        return source


class Agent8Reviewer(ModelAgent):
    def __init__(self, model: str) -> None:
        super().__init__("Agent8_CrossCheckReviewer", model)

    def review(self, crosscheck_result: Dict[str, Any]) -> str:
        prompt = f"""
Review the following deterministic cross-check result.
```json
{json.dumps(crosscheck_result, ensure_ascii=False, indent=2, sort_keys=True)}
```

Write a concise Markdown report that:
1. states whether the two classical references agree;
2. describes D_ans and N_T convergence separately without demanding strict
   point-by-point monotonicity;
3. lists every failed check using its recorded numerical evidence;
4. assesses whether the quantum results agree with the validated classical
   references within the recorded tolerances and uncertainties.

Do not change numbers, thresholds, statuses, or the hardware candidate list.
Do not repair generated programs or relabel best_effort_state as converged.
Review numerical and physical conclusions only. Do not judge formatting,
provenance, workflow authorization, historical compliance, stage ownership, or
assign responsibility for a failure.
"""
        return self.call_once(
            "You are a scientific cross-check reviewer. Assess numerical and "
            "physical conclusions without altering the validated record or "
            "performing workflow-policy review.", prompt
        ).strip()


def _load_inputs(args: argparse.Namespace):
    parameter_contract = load_parameter_contract(args.parameter_contract)
    discretization_contract = load_machine_contract(
        args.discretization_contract,
        validate_discretization_contract,
        expected_parameter_contract=parameter_contract,
    )
    conventions: Dict[str, Any] = {}
    if args.run_spec:
        conventions = dict(
            load_run_spec(args.run_spec)["user_input"].get(
                "continuum_conventions", {}
            )
        )
    return (
        format_machine_contract(
            discretization_contract,
            validate_discretization_contract,
            expected_parameter_contract=parameter_contract,
        ),
        parameter_contract,
        load_observable_specification(args.observable_spec),
        conventions,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one explicit Agent6-8 stage.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("scipy", "itensor"):
        child = subparsers.add_parser(stage)
        child.add_argument("--discretization-contract", required=True)
        child.add_argument("--parameter-contract", required=True)
        child.add_argument("--observable-spec", required=True)
        child.add_argument("--run-spec")
        child.add_argument("--output-script", required=True)
        child.add_argument("--model", default="gpt-5.6-sol")
        if stage == "itensor":
            child.add_argument("--julia-executable", default="julia")
    review = subparsers.add_parser("review")
    review.add_argument("--crosscheck-result", required=True)
    review.add_argument("--output-report", required=True)
    review.add_argument("--model", default="gpt-5.6-sol")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.stage == "review":
        content = Agent8Reviewer(args.model).review(_load_json(args.crosscheck_result))
        output = args.output_report
    else:
        inputs = _load_inputs(args)
        if args.stage == "scipy":
            content = Agent6SciPy(args.model).generate(*inputs)
        else:
            content = Agent7ITensor(args.model, args.julia_executable).generate(*inputs)
        output = args.output_script
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.write("\n")
    setup_logger("Agent678").info("Saved %s output to %s", args.stage, output)


if __name__ == "__main__":
    main()
