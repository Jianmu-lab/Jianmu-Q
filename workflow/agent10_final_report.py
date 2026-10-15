# -*- coding: utf-8 -*-
"""Agent 10: synthesize one evidence-grounded academic report for a pipeline run."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from openai import OpenAI
from llm_support import api_key_for_model, llm_connection_options

from pipeline_support import atomic_json_write


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def _artifact_path(manifest: Mapping[str, Any], name: str) -> str | None:
    record = manifest.get("artifacts", {}).get(name)
    path = record.get("path") if isinstance(record, Mapping) else None
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _without_large_fields(value: Any) -> Any:
    """Remove optimizer vectors and circuit payloads while preserving science."""
    if isinstance(value, list):
        return [_without_large_fields(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    omitted = {
        "parameters", "optimization_trace", "restart_results", "qpy_path",
        "qpy_sha256", "isa_qpy_path", "isa_qpy_sha256", "isa_observable",
    }
    return {
        str(key): _without_large_fields(item)
        for key, item in value.items() if key not in omitted
    }


def _compact_measurement(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep numerical scan results while excluding circuit recipes."""
    result = {
        key: value[key]
        for key in (
            "status", "observable_id", "depth_schedule", "trotter_schedule",
            "sampling_points", "hardware_eligibility_status",
        )
        if key in value
    }
    result["validation_grid"] = [
        {
            key: point[key]
            for key in (
                "sample_id", "sample_values", "D_ans", "N_T", "candidate_id",
                "observable_real", "observable_imag",
                "state_preparation_validated",
            )
            if key in point
        }
        for point in value.get("validation_grid", [])
        if isinstance(point, Mapping)
    ]
    return result


def collect_report_evidence(
    manifest: Mapping[str, Any], hardware_result_path: str | None = None,
) -> Dict[str, Any]:
    """Collect existing authoritative artifacts without another contract layer."""
    required = {
        "run_spec": "run_spec",
        "continuum_theory": "agent1_theory_contract",
        "discretization": "agent2_discretization_contract",
        "parameters": "agent2_parameter_contract",
        "qubit_mapping": "agent3_mapping_contract",
        "state_preparation": "agent4_state_design_contract",
        "observable_algorithm": "agent4_observable_design_contract",
        "vqe_result": "vqe_result",
        "measurement": "measurement_result",
        "scipy_reference": "scipy_reference",
        "itensor_reference": "itensor_reference",
        "crosscheck": "crosscheck_result",
    }
    evidence: Dict[str, Any] = {"run_id": manifest.get("run_id")}
    for label, artifact_name in required.items():
        path = _artifact_path(manifest, artifact_name)
        if path is None:
            raise ValueError("Final report requires artifact: {}".format(artifact_name))
        loaded = _load_json(path)
        evidence[label] = (
            _compact_measurement(loaded)
            if label == "measurement" else _without_large_fields(loaded)
        )

    review_path = _artifact_path(manifest, "agent8_review")
    if review_path:
        with open(review_path, "r", encoding="utf-8") as handle:
            evidence["crosscheck_human_review"] = handle.read()
    final_analysis_path = _artifact_path(manifest, "final_analysis")
    if final_analysis_path:
        evidence["final_analysis"] = _without_large_fields(
            _load_json(final_analysis_path)
        )
    agent9_path = _artifact_path(manifest, "agent9_review")
    if agent9_path:
        with open(agent9_path, "r", encoding="utf-8") as handle:
            evidence["hardware_feasibility_review"] = handle.read()
    agent9_decision_path = _artifact_path(manifest, "agent9_decision")
    agent9_decision = (
        _load_json(agent9_decision_path) if agent9_decision_path else None
    )
    if agent9_decision is not None:
        evidence["hardware_submission_decision"] = agent9_decision
    hardware_path = hardware_result_path or _artifact_path(manifest, "hardware_result")
    if hardware_path:
        if not os.path.isfile(hardware_path):
            raise ValueError("Hardware result is missing: {}".format(hardware_path))
        evidence["hardware"] = {
            "execution_status": "retrieved",
            "result": _without_large_fields(_load_json(hardware_path)),
        }
    else:
        plan_path = (
            _artifact_path(manifest, "hardware_audit")
            or _artifact_path(manifest, "hardware_plan")
        )
        plan = _without_large_fields(_load_json(plan_path)) if plan_path else None
        rejected_by_agent9 = (
            isinstance(agent9_decision, Mapping)
            and agent9_decision.get("decision") == "do_not_submit"
        )
        legacy_infeasible = (
            agent9_decision is None
            and isinstance(plan, Mapping)
            and plan.get("status") == "no_hardware_feasible_candidate"
        )
        rejected = rejected_by_agent9 or legacy_infeasible
        evidence["hardware"] = {
            "execution_status": "not_retrieved",
            "submission_status": (
                "skipped_by_agent9" if rejected_by_agent9 else
                "skipped_infeasible" if legacy_infeasible else
                "not_submitted_or_not_retrieved"
            ),
            "plan": plan,
            "required_statement": (
                "Agent 9 rejected hardware submission because the transpiled circuit was "
                "not expected to preserve a scientifically meaningful signal. State its "
                "recorded reason and do not imply that an IBM job was submitted."
                if rejected else
                "No retrieved quantum-hardware result was supplied; do not imply "
                "that hardware execution or hardware agreement occurred."
            ),
        }
    return evidence


def _sample_label(record: Mapping[str, Any], fallback: str) -> str:
    values = record.get("sample_values")
    if isinstance(values, Mapping) and values:
        return ", ".join("{}={}".format(key, values[key]) for key in sorted(values))
    return str(record.get("sample_id", fallback))


def _save_report_figure(figure: Any, path: str) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _vqe_quality_figure(vqe: Mapping[str, Any], path: str) -> bool:
    tested = vqe.get("tested_depths", [])
    state_labels = sorted({
        label for item in tested if isinstance(item, Mapping)
        for label in item.get("states", {})
    })
    if not tested or not state_labels:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label in state_labels:
        points = [
            (item.get("depth"), item.get("states", {}).get(label, {}))
            for item in tested
        ]
        points = [
            (depth, state) for depth, state in points
            if isinstance(depth, (int, float)) and isinstance(state, Mapping)
        ]
        depths = [point[0] for point in points]
        energy_errors = [max(float(point[1].get("energy_error", np.nan)), 1e-16)
                         for point in points]
        variances = [max(float(point[1].get(
            "variance", point[1].get("optimized_variance", np.nan)
        )), 1e-16) for point in points]
        axes[0].plot(depths, energy_errors, marker="o", label=label)
        axes[1].plot(depths, variances, marker="o", label=label)
    config = vqe.get("config", {})
    for axis, threshold, threshold_label in (
        (axes[0], config.get("energy_tolerance"), "energy tolerance"),
        (axes[1], config.get("variance_tolerance"), "variance tolerance"),
    ):
        if isinstance(threshold, (int, float)) and threshold > 0:
            axis.axhline(threshold, color="black", linestyle="--",
                         linewidth=1, label=threshold_label)
        axis.set_yscale("log")
        axis.set_xlabel(r"Ansatz depth $D_{\rm ans}$")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Absolute energy error")
    axes[1].set_ylabel(r"Energy variance $\langle H^2\rangle-\langle H\rangle^2$")
    axes[0].set_title("VQE energy accuracy")
    axes[1].set_title("VQE eigenstate quality")
    _save_report_figure(figure, path)
    return True


def _classical_reference_figure(evidence: Mapping[str, Any], path: str) -> bool:
    scipy_samples = evidence.get("scipy_reference", {}).get("observable_samples", [])
    tensor_samples = evidence.get("itensor_reference", {}).get("observable_samples", [])
    if not scipy_samples or not tensor_samples:
        return False
    tensor_by_id = {item.get("sample_id"): item for item in tensor_samples}
    pairs = [(item, tensor_by_id.get(item.get("sample_id"))) for item in scipy_samples]
    pairs = [(left, right) for left, right in pairs if isinstance(right, Mapping)]
    if not pairs:
        return False
    x = np.arange(len(pairs))
    labels = [_sample_label(left, str(i)) for i, (left, _) in enumerate(pairs)]
    figure, axes = plt.subplots(
        2, 1, figsize=(max(7, 1.15 * len(pairs)), 7), sharex=True
    )
    for axis, component in zip(axes, ("real", "imag")):
        for name, index, marker in (("SciPy", 0, "o"), ("tensor network", 1, "s")):
            values = [float(pair[index]["observable_" + component]) for pair in pairs]
            errors = [float(pair[index].get("uncertainty_" + component, 0.0))
                      for pair in pairs]
            axis.errorbar(x, values, yerr=errors, fmt=marker + "-", capsize=3,
                          label=name)
        axis.set_ylabel(("Re" if component == "real" else "Im") + " observable")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title("Independent classical-reference comparison")
    axes[1].set_xticks(x, labels=labels, rotation=30, ha="right")
    axes[1].set_xlabel("Observable sampling point")
    _save_report_figure(figure, path)
    return True


def _combined_observable_figure(evidence: Mapping[str, Any], path: str) -> bool:
    """Compare classical, noiseless-circuit, and available hardware results."""
    scipy_samples = evidence.get("scipy_reference", {}).get("observable_samples", [])
    tensor_samples = evidence.get("itensor_reference", {}).get("observable_samples", [])
    grid = evidence.get("measurement", {}).get("validation_grid", [])
    hardware_samples = (
        evidence.get("hardware", {}).get("result", {}).get("results", [])
    )
    hardware_by_point = {
        json.dumps(item.get("sample_values", {}), sort_keys=True): item
        for item in hardware_samples if isinstance(item, Mapping)
    }
    if not scipy_samples or not tensor_samples or not grid:
        return False
    tensor_by_id = {item.get("sample_id"): item for item in tensor_samples}
    grid_by_id: Dict[Any, List[Mapping[str, Any]]] = {}
    for point in grid:
        grid_by_id.setdefault(point.get("sample_id"), []).append(point)
    records = []
    for sample in scipy_samples:
        sample_id = sample.get("sample_id")
        tensor = tensor_by_id.get(sample_id)
        points = grid_by_id.get(sample_id, [])
        if not isinstance(tensor, Mapping) or not points:
            continue
        shallow_coordinate = min(
            (int(point["D_ans"]), int(point["N_T"])) for point in points
        )
        deepest_coordinate = max(
            (int(point["D_ans"]), int(point["N_T"])) for point in points
        )
        shallow = next(
            point for point in points
            if (int(point["D_ans"]), int(point["N_T"])) == shallow_coordinate
        )
        deepest = next(
            point for point in points
            if (int(point["D_ans"]), int(point["N_T"])) == deepest_coordinate
        )
        hardware = hardware_by_point.get(
            json.dumps(sample.get("sample_values", {}), sort_keys=True)
        )
        records.append((sample, tensor, shallow, deepest, hardware))
    if not records:
        return False
    x = np.arange(len(records))
    labels = [_sample_label(item[0], str(i)) for i, item in enumerate(records)]
    shallow_pair = (int(records[0][2]["D_ans"]), int(records[0][2]["N_T"]))
    deep_pair = (int(records[0][3]["D_ans"]), int(records[0][3]["N_T"]))
    series = (
        ("SciPy", 0, "o-", 2.2),
        ("tensor network", 1, "s--", 1.8),
        (r"noiseless $D_{\rm ans}=%d, N_T=%d$" % shallow_pair, 2, "^-.", 1.6),
        (r"noiseless $D_{\rm ans}=%d, N_T=%d$" % deep_pair, 3, "D:", 1.8),
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(max(8, 1.2 * len(records)), 7.4), sharex=True
    )
    for axis, component in zip(axes, ("real", "imag")):
        for label, index, style, width in series:
            values = [float(record[index]["observable_" + component]) for record in records]
            axis.plot(x, values, style, linewidth=width, markersize=6, label=label)
        hardware_points = [
            (index, record[4]) for index, record in enumerate(records)
            if (
                isinstance(record[4], Mapping)
                and isinstance(
                    record[4].get("hardware_observable_" + component),
                    (int, float),
                )
                and isinstance(
                    record[4].get("hardware_standard_error_" + component),
                    (int, float),
                )
            )
        ]
        if hardware_points:
            first = hardware_points[0][1]
            used_fallback = any(
                str(status).startswith("noise_factor_1_fallback")
                for _, item in hardware_points
                for status in item.get("mitigation_status", {}).values()
            )
            axis.errorbar(
                [item[0] for item in hardware_points],
                [float(item[1]["hardware_observable_" + component])
                 for item in hardware_points],
                yerr=[float(item[1]["hardware_standard_error_" + component])
                      for item in hardware_points],
                fmt="P", linestyle="none", capsize=4, markersize=7,
                label=(r"IBM hardware $D_{\rm ans}=%d, N_T=%d$%s"
                       % (int(first["D_ans"]), int(first["N_T"]),
                          " (ZNE/scale-1 fallback)" if used_fallback else " (ZNE)")),
            )
        axis.set_ylabel(("Re" if component == "real" else "Im") + " observable")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_title("Classical, noiseless-circuit, and hardware results")
    axes[1].set_xticks(x, labels=labels, rotation=30, ha="right")
    axes[1].set_xlabel("Observable sampling point")
    _save_report_figure(figure, path)
    return True


def _algorithm_error_figure(evidence: Mapping[str, Any], path: str) -> bool:
    grid = evidence.get("measurement", {}).get("validation_grid", [])
    references = evidence.get("scipy_reference", {}).get("observable_samples", [])
    if not grid or not references:
        return False
    reference_by_id = {
        item.get("sample_id"): complex(float(item["observable_real"]),
                                        float(item["observable_imag"]))
        for item in references
    }
    sample_ids = [
        item.get("sample_id") for item in references
        if any(point.get("sample_id") == item.get("sample_id") for point in grid)
    ]
    if not sample_ids:
        return False
    columns = min(3, len(sample_ids))
    rows = int(math.ceil(len(sample_ids) / columns))
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(4.1 * columns, 3.8 * rows),
        layout="constrained",
    )
    positive_values: List[float] = []
    matrices = []
    for sample_id in sample_ids:
        points = [point for point in grid if point.get("sample_id") == sample_id]
        depths = sorted({int(point["D_ans"]) for point in points})
        steps = sorted({int(point["N_T"]) for point in points})
        matrix = np.full((len(depths), len(steps)), np.nan)
        for point in points:
            value = complex(float(point["observable_real"]), float(point["observable_imag"]))
            error = abs(value - reference_by_id[sample_id])
            matrix[depths.index(int(point["D_ans"])), steps.index(int(point["N_T"]))] = error
            if error > 0:
                positive_values.append(error)
        matrices.append((matrix, depths, steps, points[0]))
    floor = max(min(positive_values) / 2.0, 1e-16) if positive_values else 1e-16
    ceiling = max(positive_values) if positive_values else floor * 10
    norm = LogNorm(vmin=floor, vmax=max(ceiling, floor * 10))
    image = None
    for axis, sample_id, (matrix, depths, steps, example) in zip(
        axes.flat, sample_ids, matrices
    ):
        shown = np.where(np.isfinite(matrix), np.maximum(matrix, floor), np.nan)
        image = axis.imshow(shown, origin="lower", aspect="auto", cmap="viridis", norm=norm)
        axis.set_xticks(range(len(steps)), labels=steps)
        axis.set_yticks(range(len(depths)), labels=depths)
        axis.set_xlabel(r"$N_T$")
        axis.set_ylabel(r"$D_{\rm ans}$")
        axis.set_title(_sample_label(example, str(sample_id)), fontsize=10)
    for axis in list(axes.flat)[len(sample_ids):]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=list(axes.flat), shrink=0.82,
                        label=r"$|O_{\rm ideal}-O_{\rm SciPy}|$")
    figure.suptitle("Noiseless algorithm error across the depth-Trotter grid")
    # Do not print a number in every cell: dense annotations are unreadable and
    # were the source of overlapping labels in earlier final reports.
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def generate_report_figures(
    evidence: Mapping[str, Any], output_directory: str, markdown_directory: str,
) -> List[Dict[str, str]]:
    """Generate deterministic plots and return exact Markdown references."""
    os.makedirs(output_directory, exist_ok=True)
    specifications = (
        ("vqe_quality", "vqe_quality.png", _vqe_quality_figure,
         "VQE energy error and energy variance as functions of ansatz depth."),
        ("classical_references", "classical_references.png", _classical_reference_figure,
         "SciPy and tensor-network results for the requested observable."),
        ("combined_observable", "combined_observable.png", _combined_observable_figure,
         "SciPy, tensor-network, shallow and deepest noiseless-circuit results, with available IBM hardware data and error bars, on the same observable grid."),
        ("algorithm_error_grid", "algorithm_error_grid.png", _algorithm_error_figure,
         "Noiseless quantum-algorithm error over the ansatz-depth and Trotter-step grid."),
    )
    figures: List[Dict[str, str]] = []
    for figure_id, filename, builder, caption in specifications:
        absolute_path = os.path.join(output_directory, filename)
        source = evidence.get("vqe_result", {}) if figure_id == "vqe_quality" else evidence
        if builder(source, absolute_path):
            relative_path = os.path.relpath(
                absolute_path, markdown_directory
            ).replace(os.sep, "/")
            figures.append({
                "id": figure_id,
                "path": relative_path,
                "caption": caption,
                "markdown": "![{}]({})".format(caption, relative_path),
            })
    return figures


def ensure_report_figures(report: str, figures: Sequence[Mapping[str, str]]) -> str:
    """Append only figures the language model omitted; never duplicate them."""
    missing = [item for item in figures if item.get("path") not in report]
    if not missing:
        return report
    block = ["## Numerical figures", ""]
    for item in missing:
        block.extend([item["markdown"], ""])
    return report.rstrip() + "\n\n" + "\n".join(block).rstrip()


def _strip_outer_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:markdown|md)?\s*(.*?)\s*```\s*", text, re.S | re.I)
    return match.group(1).strip() if match else text.strip()


class FinalReportAgent:
    def __init__(self, model_name: str, base_url: str) -> None:
        api_key = api_key_for_model(model_name)
        self.client = OpenAI(**llm_connection_options(model_name, api_key=api_key, base_url=base_url))
        self.model_name = model_name

    def write(self, evidence: Mapping[str, Any]) -> str:
        prompt = """
Write a self-contained, publication-style academic report in English from the
authoritative evidence below. Return Markdown only; Pandoc will convert it to
LaTeX. Use `$...$` for inline math and `$$...$$` for display math.

Required structure:
1. Title and abstract.
2. Physical problem and continuum formulation, including the continuum
   Hamiltonian, conventions, external states, and requested observable.
   If `matrix_elements` is present, write the complete linear combination
   $\\sum_k c_k\\langle a_k|O|b_k\\rangle$, define every named state, and keep
   every coefficient and sign. Do not report one component matrix element as
   though it were the requested combined observable.
3. Lattice discretization and boundary conditions, explicitly explaining the
   fermion discretization and the gauge-field treatment/truncation or exact
   Gauss-law elimination actually used.
4. Qubit mapping and physical-sector restrictions.
5. Quantum algorithm: state-preparation ansatz, depth policy, time evolution,
   Trotterization, observable measurement, and reconstruction.
6. Numerical results, explicitly separated into two validation branches:
   (a) a noiseless algorithm-validation branch that scans increasing ansatz
   depth D_ans and, only when time evolution is required, Trotter step count N_T,
   and tests convergence toward the SciPy and tensor-network references; and
   (b) a minimal-hardware demonstration at D_ans=1 and N_T=1 when Trotterized
   evolution is present, otherwise at D_ans=1 with N_T not applicable (the
   runtime may encode this as 0). Report VQE quality, both classical references, convergence,
   and the cross-check verdict without conflating these two branches.
7. Quantum-hardware section. The hardware demonstration is deliberately the
   shallowest D_ans=1 circuit (and N_T=1 only when time evolution is required),
   even when that point has appreciable
   algorithmic error relative to the converged classical answer. Its scientific
   comparison is only against the identical noiseless circuit with the same
   applicable runtime controls.
   Discuss ZNE and statistical error bars when present. If mitigation_status
   records a scale-1 fallback, identify it as a fallback rather than ZNE and
   explain that the extrapolated uncertainty failed the stability rule. Never require the
   hardware demo itself to agree with SciPy/ITensor, never imply that deeper
   convergence circuits ran on hardware, and never use the hardware demo as
   evidence of continuum-limit accuracy. If no retrieved result is present,
   state clearly that hardware execution has not been demonstrated and discuss
   only available candidate/resource evidence.
8. Limitations and conclusion.
9. References/evidence provenance using only literature sources explicitly
   present in the evidence. Never invent authors, titles, arXiv identifiers,
   equations, numerical values, hardware runs, citations, or conclusions.

Academic requirements:
- distinguish continuum, finite-lattice, qubit, noiseless, classical-reference,
  and hardware quantities;
- report parameter values and numerical tolerances with units/conventions;
- distinguish observed convergence from extrapolation;
- distinguish the reconstructed matrix-element combination from its component
  terms whenever the evidence supplies both;
- do not phase-rotate complex observables or infer even/odd real/imaginary
  symmetry merely because the sampling grid contains zero. Such an alignment
  is meaningful only for distinct external states when authoritative evidence
  explicitly establishes the relevant C/P symmetry relation and relative
  state quantum numbers; an overall phase cancels for diagonal matrix elements;
- state separately what the noiseless convergence study validates and what the
  minimal hardware/ZNE comparison validates;
- discuss any disagreement or missing evidence rather than hiding it;
- keep notation consistent and define every symbol used;
- do not mention Agent numbers in the scientific narrative, except in a short
  reproducibility/provenance paragraph if useful;
- do not embed raw JSON, code listings, or file-system paths in the main text.

Figure and table requirements:
- embed every item in `report_figures` exactly once, using its supplied
  `markdown` string verbatim and placing it near the discussion it supports;
- treat supplied figures as authoritative visualizations of the numerical
  evidence, but interpret them only to the precision supported by the data;
- when retrieved hardware data are present, embed and discuss the combined
  comparison figure that overlays IBM results with statistical error bars on
  the SciPy, tensor-network, and noiseless-circuit results. Compare hardware
  directly only with the identical shallow noiseless circuit and the same
  applicable runtime controls; do not
  present the hardware points as part of the deeper noiseless convergence scan;
- never create TikZ, PGFPlots, ASCII plots, or hand-written coordinate lists;
- do not print a numeric value or candidate identifier beside every plotted
  point; use legends, axes, and captions so labels cannot overlap;
- use tables only for compact headline values, round displayed values to
  meaningful precision, and leave dense scan data to the supplied figures;
- do not invent or request any figure that is absent from `report_figures`.

[AUTHORITATIVE EVIDENCE]
```json
%s
```
""" % json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful scientific editor. Synthesize only "
                        "the supplied evidence and preserve all scientific caveats."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        if not content.strip():
            raise RuntimeError("Agent 10 returned an empty report.")
        return _strip_outer_fence(content)


def find_local_tool(name: str) -> str | None:
    """Find host tools even when the pipeline runs inside a conda environment."""
    candidates = [shutil.which(name)]
    conda_base = os.path.dirname(os.path.dirname(sys.prefix))
    candidates.extend([
        os.path.join(conda_base, "bin", name),
        os.path.join("/opt/homebrew/bin", name),
        os.path.join("/usr/local/bin", name),
        os.path.join("/Library/TeX/texbin", name),
    ])
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def markdown_to_latex(markdown_path: str, latex_path: str) -> None:
    pandoc = find_local_tool("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to convert the final report to LaTeX.")
    with open(markdown_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    # Pandoc 2.12 does not recognize \(...\) and \[...\] reliably even when
    # its corresponding reader extension is requested. Normalize only the
    # delimiters outside fenced code; the mathematical content is unchanged.
    normalized = []
    in_fence = False
    for line in source.splitlines(keepends=True):
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
        elif not in_fence:
            line = line.replace(r"\(", "$").replace(r"\)", "$")
            line = line.replace(r"\[", "$$").replace(r"\]", "$$")
        normalized.append(line)
    subprocess.run(
        [
            pandoc, "-",
            "--from=markdown+tex_math_dollars+raw_tex",
            "--to=latex", "--standalone", "--output", latex_path,
        ],
        input="".join(normalized),
        text=True,
        check=True,
    )


def compile_pdf(latex_path: str) -> Dict[str, Any]:
    engine = find_local_tool("xelatex") or find_local_tool("pdflatex")
    if engine is None:
        return {"status": "not_attempted", "reason": "No LaTeX engine found."}
    directory = os.path.dirname(os.path.abspath(latex_path))
    log_path = os.path.splitext(latex_path)[0] + "_compile.log"
    process = subprocess.run(
        [engine, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(latex_path)],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(process.stdout)
    pdf_path = os.path.splitext(latex_path)[0] + ".pdf"
    return {
        "status": "compiled" if process.returncode == 0 else "compile_failed",
        "engine": engine,
        "log_path": os.path.abspath(log_path),
        "pdf_path": os.path.abspath(pdf_path) if os.path.isfile(pdf_path) else None,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the final academic report.")
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--latex-output", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--hardware-result")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--base-url", default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest = _load_json(args.pipeline_manifest)
    evidence = collect_report_evidence(manifest, args.hardware_result)
    markdown_directory = os.path.dirname(os.path.abspath(args.markdown_output))
    figures_directory = os.path.join(markdown_directory, "figures")
    evidence["report_figures"] = generate_report_figures(
        evidence, figures_directory, markdown_directory
    )
    atomic_json_write(args.evidence_output, evidence)
    report = FinalReportAgent(args.model, args.base_url).write(evidence)
    report = ensure_report_figures(report, evidence["report_figures"])
    os.makedirs(markdown_directory, exist_ok=True)
    with open(args.markdown_output, "w", encoding="utf-8") as handle:
        handle.write(report.rstrip() + "\n")
    markdown_to_latex(args.markdown_output, args.latex_output)
    status = compile_pdf(args.latex_output)
    status.update({
        "report_status_schema_version": "1.0",
        "markdown_path": os.path.abspath(args.markdown_output),
        "latex_path": os.path.abspath(args.latex_output),
        "evidence_path": os.path.abspath(args.evidence_output),
        "hardware_included": evidence["hardware"]["execution_status"] == "retrieved",
    })
    atomic_json_write(args.status_output, status)
    print("Final academic report written to {} ({})".format(
        args.latex_output, status["status"]
    ))


if __name__ == "__main__":
    main()
