# IBM Quantum Hardware-Feasibility Review

## Deterministic gate result

**Failed.** All five `D_ans=1`, `N_T=1` demonstration candidates failed the deterministic hardware resource gate. The plan status is `no_hardware_feasible_candidate`; both `selected_deployments` and `pareto_deployments` are empty.

| Candidate | Max depth / recipe | Total 2Q gates | SWAPs | Max recipe 2Q error sum | Gate rejection reasons |
|---|---:|---:|---:|---:|---|
| `sample_0002_D1_NT1` | 3,352 | 9,368 | 0 | 1.416972 | 2Q error sum exceeds limit |
| `sample_0003_D1_NT1` | 7,856 | 26,296 | 0 | 4.163536 | Depth, total 2Q count, and 2Q error sum exceed limits |
| `sample_0001_D1_NT1` | 12,654 | 46,920 | 0 | 7.033644 | Depth, total 2Q count, and 2Q error sum exceed limits |
| `sample_0004_D1_NT1` | 12,753 | 47,864 | 0 | 7.053338 | Depth, total 2Q count, and 2Q error sum exceed limits |
| `sample_0000_D1_NT1` | 17,389 | 66,184 | 0 | 9.955392 | Depth, total 2Q count, and 2Q error sum exceed limits |

Configured limits were:

- Maximum ISA depth per recipe: **5,000**
- Total ISA two-qubit gates: **20,000**
- Maximum recipe two-qubit calibration-error sum: **1.0**
- Complete two-qubit error data: **required**

Even the shallowest candidate exceeded the error-sum limit. The other four also exceeded depth and total two-qubit-count limits.

## Backend, calibration, and capacity

All candidates were evaluated on:

- **Backend:** `ibm_kawasaki` — 156 qubits
- **Instance:** `Riken internal-dedicated` (`on-prem`)
- **Queue at planning:** 0 pending jobs
- **Calibration timestamp:** `2026-09-28T14:12:05+09:00`
- **Remaining execution allocation:** 288,542 seconds  
  (77,458 of 366,000 seconds consumed)

No IBM instance was selected because no candidate passed. The evaluated instance was not shown as exhausted, and no exhausted instance was selected.

## Noise and resource risks

- Two-qubit calibration data were complete: every recipe reports `two_qubit_errors_missing: 0`.
- Therefore, there is **no identified missing-error-data risk**.
- There is substantial hardware-noise/resource risk from aggregate two-qubit error sums of **1.416972–9.955392 per recipe**, all above the limit of 1.0.
- Four candidates additionally have excessive transpiled depth and two-qubit gate counts.
- All reported SWAP counts are zero, but this does not offset the depth, gate-count, or calibration-error failures.
- The algorithm-error values are explicitly characterized as same-circuit noiseless comparisons and do not override the hardware resource gate.

## Execution boundary

**Conclusion: no IBM job should be sent.** This review submits nothing.

The controller may continue only when the deterministic plan status is exactly `ready_for_explicit_submission`. The present status is `no_hardware_feasible_candidate`, so the controller **must skip submission**.
