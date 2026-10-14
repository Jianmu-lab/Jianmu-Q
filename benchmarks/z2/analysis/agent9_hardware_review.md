# Hardware Feasibility Review

## Resource-gate result

**Passed.** The shallow candidate uses `D_ans=1`, `N_T=0`; no evidence in this plan indicates that time evolution—and therefore `N_T=1`—is required.

| Resource gate | Observed | Limit |
|---|---:|---:|
| Maximum ISA depth per recipe | 69 | 5,000 |
| Total two-qubit gates | 1,189 | 20,000 |
| Maximum two-qubit error sum per recipe | 0.0281579299 | 1.0 |
| Missing two-qubit errors | 0 | Complete data required |

The deterministic gate records `passed: true` with no rejection reasons.

## Selected deployment

- **Logical candidate:** `default_D1_NT0`
- **Backend:** `ibm_kawasaki` (156 qubits)
- **Reason:** It passed every configured hardware-resource constraint and is the sole reported deployment/Pareto candidate, with hardware score `1.5684815248587063`.

## Transpiled resources and noise evidence

There are 58 recipes:

- Glueball recipes: depth 69, 30 two-qubit gates, 0 SWAPs, two-qubit error sum `0.02815792992629426` per recipe.
- Vacuum recipes: depth 37, 11 two-qubit gates, 0 SWAPs, two-qubit error sum `0.02172163989641973` per recipe.
- Aggregate: 1,189 two-qubit gates, 0 SWAPs, additive two-qubit error sum `1.4465075248587058`.
- Missing two-qubit calibration/error entries: **0**.
- Backend queue at planning: **0 pending jobs**.
- Calibration timestamp: **2026-09-20T17:16:54+09:00**.

The aggregate error sum exceeds 1, but it is not the configured gate metric; the applicable limit is **1.0 per recipe**, and the maximum recipe value is about 0.0282. This additive sum is a noise proxy, not a predicted total failure probability.

## IBM instance capacity

- **Instance:** `Riken internal-dedicated`
- **Plan:** on-prem
- **Remaining execution time at planning:** **252,524 seconds** (about 70 h 8 m 44 s)
- **Minimum configured remainder:** 600 seconds

The selected instance was therefore not exhausted. No missing two-qubit error data or excessive configured resource usage is evident. Residual risks include calibration drift after the stated timestamp and noise sources not quantified here, such as readout, single-qubit, decoherence, and sampling error.

## Execution boundary

This review submits nothing and does not claim that any IBM job was sent. The deterministic plan status is `ready_for_explicit_submission`; therefore, the controller **may continue under the authorization already recorded in the run specification**. For any other status, it must skip submission.
