# Deterministic Cross-Check Review

## Overall conclusion

**Status: `not_approved`.** The two classical references do **not** establish agreement, neither convergence scan passes globally, and no quantum candidate agrees with the converged SciPy reference within the recorded algorithm tolerance.

## 1. Classical-reference agreement

The classical-agreement check **failed**:

- SciPy converged: `true`
- ITensor converged: `false`
- Energy difference: `4.703792910731863e-12`
- Maximum observable difference: `0.5357740104465359`
- Reported combined observable uncertainty: `0.0013846435308039317`
- Classical tolerance: `0.0001`

The close energy values do not establish observable agreement, particularly because ITensor is explicitly nonconverged. Every sampled observable comparison failed:

| Sample | \((z_1,z_2)\) | Observable difference | Combined uncertainty | Result |
|---|---:|---:|---:|---|
| `sample_0000` | \((-2,0)\) | `0.3127533736845817` | `1.962515288915199e-08` | Failed |
| `sample_0001` | \((-1,0)\) | `0.43538815452269264` | `2.395485147626362e-10` | Failed |
| `sample_0002` | \((0,0)\) | `0.5357740104465359` | `1.688163411576795e-11` | Failed |
| `sample_0003` | \((1,0)\) | `0.4495836717341445` | `0.000646080449315726` | Failed |
| `sample_0004` | \((2,0)\) | `0.30317051068911877` | `0.0013846435308039317` | Failed |

Thus, there is no mutually converged pair of agreeing classical references. SciPy is the validated/converged classical reference in this record.

## 2. \(D_{\rm ans}\) convergence

The depth-convergence check is applicable and **failed**. All 25 series had sufficient points, but every series was recorded as `tail_stable: false` against the tail tolerance `0.001`.

The scans use \(D_{\rm ans}=[1,2,3,4,8]\). Strict point-by-point monotonicity is not required; nevertheless, the final adjacent changes remain well above tolerance:

| Sample | Fixed \(N_T\) values | Final adjacent deltas |
|---|---|---|
| `sample_0000` | 1, 2, 4, 8, 16 | `0.0927895097268872`, `0.027425680472620218`, `0.017995408417724037`, `0.016734222284275693`, `0.01648925858654107` |
| `sample_0001` | 1, 2, 4, 8, 16 | `0.10572739575071215`, `0.0896987036785751`, `0.08680300080733042`, `0.08613619098865619`, `0.08597285879580306` |
| `sample_0002` | 1, 2, 4, 8, 16 | `0.09730103266393891`, `0.09730103266392595`, `0.09730103266390235`, `0.09730103266385468`, `0.0973010326637582` |
| `sample_0003` | 1, 2, 4, 8, 16 | `0.07040697006833681`, `0.06867486160805053`, `0.06841629162458457`, `0.06834558061289878`, `0.06832744076767967` |
| `sample_0004` | 1, 2, 4, 8, 16 | `0.07198189827644533`, `0.041538741186063544`, `0.03799043032676757`, `0.03715480578020225`, `0.03692758005934926` |

Some changes decrease with depth, but none of the depth-scan tails reaches `0.001`; therefore no \(D_{\rm ans}\) convergence is demonstrated.

## 3. \(N_T\) convergence

The Trotter-convergence check is applicable and **failed globally**. The scans use \(N_T=[1,2,4,8,16]\). Several series show decreasing tail changes and pass individually, but 12 recorded series remain unstable:

| Sample | Fixed \(D_{\rm ans}\) | Final adjacent delta | Result |
|---|---:|---:|---|
| `sample_0000` | 1 | `0.005522330957372686` | Failed |
| `sample_0000` | 2 | `0.003767476923385965` | Failed |
| `sample_0000` | 3 | `0.0028462552411121786` | Failed |
| `sample_0000` | 4 | `0.002166972836292234` | Failed |
| `sample_0000` | 8 | `0.0018403672400419387` | Failed |
| `sample_0001` | 1 | `0.001114594647230767` | Failed |
| `sample_0003` | 1 | `0.0011930824992198146` | Failed |
| `sample_0004` | 1 | `0.005837620042141872` | Failed |
| `sample_0004` | 2 | `0.0032827101432555753` | Failed |
| `sample_0004` | 3 | `0.001140506078943968` | Failed |
| `sample_0004` | 4 | `0.0010054927445501142` | Failed |
| `sample_0004` | 8 | `0.001128592585513143` | Failed |

The other 13 series are recorded as tail-stable. In particular, all `sample_0002` \(N_T\) deltas are at approximately \(10^{-13}\), but its result remains inaccurate relative to SciPy; stability in \(N_T\) therefore does not imply correctness with respect to the classical reference.

## 4. Quantum agreement and remaining failed checks

The algorithm tolerance is `0.001`. **Every one of the 125 point diagnostics** has `within_algorithm_tolerance: false`. Recorded errors span from `0.07650934671772537` to `0.9447030042458429`, all far above tolerance.

The designated \(D_{\rm ans}=8,\ N_T=16\) endpoints all failed:

| Sample | Algorithm error versus SciPy |
|---|---:|
| `sample_0000` | `0.33882480685883515` |
| `sample_0001` | `0.47657004136187764` |
| `sample_0002` | `0.5953795366240534` |
| `sample_0003` | `0.49766201145367406` |
| `sample_0004` | `0.3320790290962231` |

Although these endpoint states are marked `state_preparation_validated: true`, their observables do not agree with SciPy within `0.001`.

The five retained hardware-demonstration candidates remain exactly the recorded \(D_{\rm ans}=1,\ N_T=1\) candidates. Their errors are:

- `0.19376276566165163`
- `0.22148364573969404`
- `0.9447030042458429`
- `0.32656775667715277`
- `0.21415437822292985`

All five have `state_preparation_validated: false`, are explicitly interpreted as `same_circuit_noiseless_comparison_only`, and use `best_effort_state` preparations. The reported state-quality evidence is:

- \(\phi\): energy error `0.36768055390507115`, variance `1.070626155722465`
- \(\psi\): energy error `1.6101910421045655`, variance `1.5237439445249343`

Accordingly, there are **no hardware-eligible candidates**, while the five hardware-demonstration candidates remain unchanged.

## Assessment

The quantum results do **not** agree with the converged SciPy classical reference within the recorded `0.001` algorithm tolerance. The discrepancies are also much larger than the classical-reference uncertainties reported for the samples. Together with failed \(D_{\rm ans}\) convergence, incomplete \(N_T\) convergence, nonconverged ITensor results, and failed classical observable agreement, the recorded numerical evidence supports the existing `not_approved` status.
