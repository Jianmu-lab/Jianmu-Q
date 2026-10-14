# Hardware Feasibility Report

## Deterministic resource gate

**Passed.** The approved shallow configuration uses `D_ans=1` and `N_T=1`. All five logical samples in the full observable demonstration passed the configured per-deployment limits:

- Maximum depth per recipe: **5,000**
- Maximum two-qubit error sum per recipe: **1.0**
- Total two-qubit gates per deployment: **20,000**
- Complete two-qubit error calibration required: **yes**

No rejection reasons were recorded.

## Selected logical configuration and backend

The plan selected the complete `D_ans=1, N_T=1` logical sample set (`sample_0000` through `sample_0004`) on **`ibm_kawasaki`**. This backend passed every required sample and was retained by the deterministic lowest-total-hardware-score selection rule. The especially shallow `sample_0002_D1_NT1` has depth **101** and only **78** total two-qubit gates.

| Candidate | Max recipe depth | Total 2Q gates | SWAPs | Max recipe 2Q error sum | Missing errors |
|---|---:|---:|---:|---:|---:|
| `sample_0000_D1_NT1` | 2,033 | 14,368 | 0 | 0.978352 | 0 |
| `sample_0001_D1_NT1` | 2,038 | 14,368 | 0 | 0.980503 | 0 |
| `sample_0002_D1_NT1` | 101 | 78 | 0 | 0.042394 | 0 |
| `sample_0003_D1_NT1` | 2,072 | 14,560 | 0 | 0.991612 | 0 |
| `sample_0004_D1_NT1` | 2,077 | 14,592 | 0 | 0.994106 | 0 |

The corresponding aggregate two-qubit error sums per deployment are **15.653629**, **15.688054**, **0.084788**, **15.865798**, and **15.905698**. These aggregate values are not configured gate limits; the applicable error limit is per recipe.

## Backend and instance snapshot

- Backend: **`ibm_kawasaki`**, 156 qubits
- Queue at planning: **0 pending jobs**
- Calibration timestamp: **2026-09-29T11:04:40+09:00**
- IBM instance: **Riken internal-dedicated**
- Plan: **on-prem**
- Remaining execution time at planning: **289,269 seconds**
- Minimum required remaining time: **600 seconds**

The selected instance was therefore **not exhausted** at planning.

## Noise and resource risks

Calibration coverage is complete: every recipe reports zero missing two-qubit errors. There is no risk identified from absent error data.

However, four samples are resource-heavy, with roughly **14.4k two-qubit gates per deployment**, recipe depths around **2,000**, and per-recipe error sums between **0.978 and 0.994**, very close to the limit of 1.0. They pass deterministically but carry substantial accumulated-noise risk. The recorded ideal comparisons are explicitly `same_circuit_noiseless_comparison_only`; they do not establish expected hardware accuracy. Calibration drift after the stated timestamp could further change realized performance.

## Execution boundary

This review **submits nothing** and does not claim that any IBM job was sent. The immutable plan status is **`ready_for_explicit_submission`**; therefore, the controller may continue under the authorization already recorded in the run specification. For any other plan status, submission must be skipped.
