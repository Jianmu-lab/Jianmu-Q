# Benchmark snapshots

Only the final NJL, Schwinger and Z2 benchmark runs are included. Selection follows
the final revised figure sources: NJL `20260929T024841Z_b286d0a8`, Schwinger
`20260928T035738Z_5c1e78fc`, and Z2 `20260920T073518Z_799c77bd`.

Each directory retains original scientific inputs and contracts, generated
Qiskit adapters, optimized parameter bundles, noiseless measurement results,
independent reference programs/results, cross-checks and available hardware
records. Each final workflow report is retained in both Markdown and PDF, with its
original result images under `reports/figures/`. The final revised manuscript
figures are saved separately under `figures/paper/` as PNG/PDF, with available
plot data. Z2 uses a numerical table in the manuscript; its displayed values
are retained as `figures/paper/z2_results.csv`.
Manuscript drafts, plotting scripts, compilation intermediates, raw model
transcripts and stage logs are excluded. Finished result figures are included;
their drawing scripts are excluded. Exported QPY circuits are retained
where they exist in the source run.

`provenance.json` records the source run, omitted files and SHA-256 checksums of
the published artifacts. Source-machine absolute paths in JSON/Markdown are
normalized to repository-relative paths; IBM account instance CRNs are redacted.
Numerical JSON values and generated
Python/Julia program bytes are preserved. Embedded historical hashes remain
historical provenance; normalized path-bearing files can have different file
hashes. Paths use the repository root as their base.

`pipeline_manifest.json` is a historical index, not a ready-to-resume run: some
audit and report paths point to deliberately omitted artifacts. Original
controller/interpreter settings describe the old run. Use explicit files below
to replay generated programs, or `config.json` to start a new workflow. Avoid
submitting archived hardware plans.

## Inspect or recompute

Inspect `analysis/crosscheck_result.json`, `runtime/measurement_result.json`,
and `references/{scipy,itensor}_reference.json` first. All original cross-check
statuses are `not_approved`; stored hardware data do not change that verdict.
For Schwinger, independent external-state phases affect raw transition
amplitudes. The manuscript's constant phase alignment per numerical family is
not applied to these archived JSON records.

From the repository root, replay the Z2 SciPy reference into a separate output:

```bash
python benchmarks/z2/references/reference_scipy.py \
  --parameter-contract benchmarks/z2/contracts/Agent2_ParameterContract_20260920T073518Z_799c77bd.json \
  --observable-spec benchmarks/z2/inputs/agent1_observable.json \
  --output /tmp/jianmu_z2_scipy.json
```

Reference programs accept the same three arguments. Julia programs require the
ITensor packages described in the main README. Historical quantum adapters can
be passed to `workflow/vqe_runtime.py` and `workflow/measurement_runtime.py`;
consult their `--help` and the stored run configs. Reoptimization can be costly
and is not needed to inspect the published results.

`config.json` disables hardware and starts new generation with current prompts.
For Z2 it explicitly spells out the charge-free `G_v=+1` sector and lattice
spacing, removing ambiguous wording from the old user input. It does not change
the archived contracts or results. Current weighted-SSVQE designs and the
historical Z2 deflation method are distinct; new generation is not an exact
replay of an old LLM response.
