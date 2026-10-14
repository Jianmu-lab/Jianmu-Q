# Deterministic Cross-Check Review

## Classical-reference agreement

**The two classical references agree.** Both solvers converged, and the recorded classical-agreement check passed:

- Energy difference: `2.3447910280083306e-13`
- Observable difference: `2.4158453015843406e-13`
- Combined observable uncertainty: `3.1629991657344744e-06`
- Classical tolerance: `0.0001`

The observable discrepancy is far below both the combined uncertainty and the classical tolerance.

## Convergence assessment

### \(D_{\mathrm{ans}}\) convergence

At fixed `N_T = 0`, the algorithm error decreases substantially as `D_ans` increases:

| `D_ans` | Algorithm error vs SciPy | Within `0.001` tolerance |
|---:|---:|:---:|
| 1 | `0.1946762050630566` | No |
| 2 | `0.14139291757820516` | No |
| 4 | `0.0506903269726422` | No |
| 8 | `0.0018187508264793095` | No |

This shows clear improvement with depth, but not validated convergence. The observable’s adjacent depth-scan deltas are:

`0.05328328748485145`, `0.09070259060556296`, `0.04887157614616289`.

Strict point-by-point monotonicity is not required, but the tail remains unstable: the final adjacent delta is well above the `0.001` tail tolerance.

### \(N_T\) convergence

`N_T` was sampled only at `0` for every `D_ans`. Consequently, Trotter convergence is recorded as **not applicable**. There are insufficient points to establish any \(N_T\)-convergence trend; this is not itself a failed applicable check.

## Failed checks

1. **Depth convergence failed**
   - `passed: false`
   - `tail_stable: false`
   - Adjacent deltas: `0.05328328748485145`, `0.09070259060556296`, `0.04887157614616289`
   - Tail tolerance: `0.001`

2. **Endpoint algorithm-tolerance check failed**
   - Candidate: `default_D8_NT0`
   - Error vs SciPy: `0.0018187508264793095`
   - Algorithm tolerance: `0.001`
   - `within_algorithm_tolerance: false`

3. **Every scanned quantum point failed the algorithm-tolerance check**
   - `default_D1_NT0`: `0.1946762050630566`
   - `default_D2_NT0`: `0.14139291757820516`
   - `default_D4_NT0`: `0.0506903269726422`
   - `default_D8_NT0`: `0.0018187508264793095`
   - All exceed `0.001`.

4. **State preparation was not validated at every scanned point**
   - `state_preparation_validated: false` for `D_ans = 1, 2, 4, 8`.
   - For the recorded hardware-demonstration candidate `default_D1_NT0`, both states remain `best_effort_state`:
     - Glueball: energy error `0.20722850031108564`, variance `0.3198021708853389`
     - Vacuum: energy error `0.01255229524799617`, variance `0.1452604070260577`

## Quantum-versus-classical conclusion

The validated classical references are mutually consistent, but the quantum results **do not agree within the recorded algorithm tolerance**. Even the best endpoint error, `0.0018187508264793095`, exceeds the `0.001` tolerance and is also much larger than the recorded classical-reference uncertainties (`2.4266567746021817e-12` per real/imaginary component) and the combined classical observable uncertainty (`3.1629991657344744e-06`).

Accordingly, the recorded `not_approved` status, empty `hardware_eligible_candidates` list, and unchanged hardware-demonstration candidate classification are numerically supported.
