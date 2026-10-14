# Deterministic Cross-Check Review

## Classical references

The SciPy and ITensor results **agree numerically at all five sampled observables**:

- `sample_0000`: difference `1.5031676954219874e-05`, combined uncertainty `0.0016023547973917625`
- `sample_0001`: difference `1.599363070178987e-05`, combined uncertainty `0.0011033625998283717`
- `sample_0002`: difference `1.776588851188343e-15`, combined uncertainty `4.019436694230466e-14`
- `sample_0003`: difference `1.3914542312734176e-05`, combined uncertainty `0.0011054193826403234`
- `sample_0004`: difference `1.6993856633468185e-05`, combined uncertainty `0.0015994754328557197`

All observable differences are also below the classical tolerance `0.0001`, and the energy difference is `1.0929923632829741e-11`. Nevertheless, the recorded aggregate `classical_agreement.passed` is `false` because SciPy converged but ITensor did not (`scipy_converged: true`, `itensor_converged: false`). Thus numerical consistency is present, but converged two-reference validation is not established.

## Convergence

### \(D_{\mathrm{ans}}\) convergence

The depth scan uses \(D_{\mathrm{ans}}=[1,2,4,6]\). The changes generally become much smaller at the high-depth end, but the aggregate check failed because 19 of 25 series have final adjacent deltas above the tail tolerance `0.001`:

- `sample_0000`, fixed values `1,2,4,8,16`: `0.0019362738161162107`, `0.0036115805151132403`, `0.00493703852994053`, `0.006012974998030661`, `0.006233024649810821`
- `sample_0001`, fixed values `1,2,4,8,16`: `0.009226679726054897`, `0.011813415620149625`, `0.011996480870438503`, `0.012040614956311686`, `0.012051478135532465`
- `sample_0003`, fixed values `1,2,4,8,16`: `0.0032139785181645045`, `0.0036276882733446247`, `0.003275551332677733`, `0.0031883894840110364`, `0.003166614066865666`
- `sample_0004`, fixed values `2,4,8,16`: `0.00312840495172938`, `0.002070895488819519`, `0.00195731380375802`, `0.0019410973514752885`

All five `sample_0002` series are tail-stable, as is `sample_0004` at fixed value `1` with final delta `0.0006484454978899299`. Strict point-by-point monotonicity is neither required nor observed; the failure is specifically insufficient high-depth tail stability.

### \(N_T\) convergence

The Trotter scan uses \(N_T=[1,2,4,8,16]\). Most series show decreasing high-\(N_T\) changes, but 8 of 20 series remain above the tail tolerance:

- `sample_0000`, fixed \(D_{\mathrm{ans}}=1,2,4,6\): final deltas `0.007200300161446899`, `0.0036718062337617883`, `0.003375242268757878`, `0.003336671121897132`
- `sample_0004`, fixed \(D_{\mathrm{ans}}=1,2,4,6\): final deltas `0.008547725287002515`, `0.004171226721865349`, `0.0033049914630752146`, `0.003327639738537863`

All series for `sample_0001`, `sample_0002`, and `sample_0003` are tail-stable. The aggregate `trotter_convergence.passed` is therefore `false`, without implying that every individual series fails or requiring monotonic behavior at every step.

## Failed checks

- **Overall status:** `not_approved`
- **Classical agreement:** `passed: false`; ITensor did not converge, despite the numerical agreement documented above.
- **Depth convergence:** `passed: false`; the 19 failed tails are listed in the \(D_{\mathrm{ans}}\) section.
- **Trotter convergence:** `passed: false`; the 8 failed tails are listed in the \(N_T\) section.
- **Final \(D_{\mathrm{ans}}=6,\ N_T=16\) endpoint checks:** four of five exceed `algorithm_tolerance = 0.001`:
  - `sample_0000`: `0.0012012930246359343`
  - `sample_0001`: `0.0073318889907673015`
  - `sample_0003`: `0.0020462226950329764`
  - `sample_0004`: `0.0015526336699820651`
- **Failed point diagnostics:** every tested point for `sample_0000`, `sample_0001`, `sample_0003`, and `sample_0004` is outside algorithm tolerance. For `sample_0002`, all \(D_{\mathrm{ans}}=4\) points fail with error `0.0011898512340685475`; its \(D_{\mathrm{ans}}=1,2,6\) points pass.
- **State preparation:** all \(D_{\mathrm{ans}}=1,2,4\) point diagnostics have `state_preparation_validated: false`. The hardware-demonstration candidates are \(D_{\mathrm{ans}}=1,\ N_T=1\), remain `best_effort_state`, and are not state-preparation validated.
- **Hardware eligibility:** `hardware_eligible_candidates` is empty.

## Quantum-to-classical assessment

At the validated-state endpoint \(D_{\mathrm{ans}}=6,\ N_T=16\), only `sample_0002` agrees with SciPy within the recorded algorithm tolerance:

- `sample_0002`: error `0.0003948250934424234` — within `0.001`

The other four endpoint errors exceed `0.001`, so the quantum results do **not** agree with the validated classical result across the full sampled observable. Classical-reference uncertainties do not alter that conclusion: they are much smaller than the four endpoint discrepancies, and the endpoint decision is explicitly recorded against the algorithm tolerance.

The hardware-demonstration candidates remain same-circuit noiseless comparisons using unvalidated `best_effort_state` preparation. Although `sample_0002_D1_NT1` has error `2.604091557612907e-10`, it is not state-preparation validated; the other four demonstration candidates have errors from `0.3576025777825374` to `0.497614274066932`. Consequently, none is recorded as hardware eligible.
