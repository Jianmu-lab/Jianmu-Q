# -*- coding: utf-8 -*-
"""Agent 9: decide whether an IBM Quantum deployment is scientifically useful."""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


def _load_plan(path):
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("hardware_plan_schema_version") != "1.0":
        raise ValueError("Agent 9 requires hardware-plan schema 1.0.")
    deployments = plan.get("all_deployments")
    if not isinstance(deployments, list) or not deployments:
        raise ValueError("Hardware plan contains no transpiled deployment evidence.")
    return plan


def _compact_evidence(plan):
    """Remove executable payloads while retaining the evidence Agent 9 needs."""
    deployments = []
    for deployment in plan["all_deployments"]:
        recipes = deployment.get("recipes", [])
        resources = [item.get("resources", {}) for item in recipes]

        def finite_values(key):
            return [
                float(item[key]) for item in resources
                if isinstance(item.get(key), (int, float))
            ]

        def maximum(key):
            values = finite_values(key)
            return max(values) if values else None

        ideal_recipe_values = [
            abs(float(item["ideal_expectation"])) for item in recipes
            if isinstance(item.get("ideal_expectation"), (int, float))
        ]
        worst = max(
            recipes,
            key=lambda item: float(
                item.get("resources", {}).get("two_qubit_error_sum", -1.0)
            ),
            default={},
        )
        deployments.append({
            key: deployment.get(key) for key in (
                "candidate_id", "sample_values", "D_ans", "N_T",
                "ideal_observable_real", "ideal_observable_imag",
                "algorithm_error_vs_scipy", "candidate_class", "interpretation",
                "backend", "backend_instance_name", "backend_instance_plan",
                "backend_num_qubits", "backend_pending_jobs",
                "backend_calibration_time", "instance_usage", "score",
                "error_metric_complete", "resource_gate",
            )
        } | {
            "recipe_summary": {
                "count": len(recipes),
                "maximum_isa_depth": maximum("isa_depth"),
                "maximum_isa_one_qubit_gates": maximum("isa_one_qubit_gates"),
                "maximum_isa_two_qubit_gates": maximum("isa_two_qubit_gates"),
                "maximum_isa_swaps": maximum("isa_swaps"),
                "maximum_one_qubit_error_sum": maximum("one_qubit_error_sum"),
                "maximum_two_qubit_error_sum": maximum("two_qubit_error_sum"),
                "maximum_estimated_duration_seconds": maximum(
                    "estimated_duration_seconds"
                ),
                "maximum_duration_over_minimum_t1": maximum(
                    "duration_over_minimum_t1"
                ),
                "maximum_duration_over_minimum_t2": maximum(
                    "duration_over_minimum_t2"
                ),
                "maximum_mean_readout_error": maximum("mean_readout_error"),
                "one_qubit_errors_missing": sum(
                    int(item.get("one_qubit_errors_missing", 0)) for item in resources
                ),
                "two_qubit_errors_missing": sum(
                    int(item.get("two_qubit_errors_missing", 0)) for item in resources
                ),
                "readout_errors_missing": sum(
                    int(item.get("readout_errors_missing", 0)) for item in resources
                ),
                "maximum_absolute_ideal_recipe_expectation": (
                    max(ideal_recipe_values) if ideal_recipe_values else None
                ),
                "worst_two_qubit_error_recipe": worst.get("recipe_id"),
            }
        })
    return {
        "candidate_class": plan.get("candidate_class"),
        "plan_kind": plan.get("plan_kind"),
        "planning_policy": plan.get("planning_policy"),
        "deployments": deployments,
    }


def _validate_decision(raw, plan):
    if not isinstance(raw, dict):
        raise ValueError("Agent 9 decision must be a JSON object.")
    decision = raw.get("decision")
    if decision not in ("submit", "do_not_submit"):
        raise ValueError("Agent 9 decision must be submit or do_not_submit.")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Agent 9 decision must explain its physical reasoning.")
    selected = raw.get("selected_deployments", [])
    if not isinstance(selected, list):
        raise ValueError("Agent 9 selected_deployments must be a list.")
    available = {
        (item.get("candidate_id"), item.get("backend"))
        for item in plan["all_deployments"]
    }
    normalized = []
    seen = set()
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("Each Agent 9 selection must be an object.")
        key = (item.get("candidate_id"), item.get("backend"))
        if key not in available or key in seen:
            raise ValueError("Agent 9 selected an unavailable or duplicate deployment.")
        seen.add(key)
        normalized.append({"candidate_id": key[0], "backend": key[1]})
    if decision == "do_not_submit" and normalized:
        raise ValueError("A do_not_submit decision cannot select deployments.")
    if decision == "submit" and not normalized:
        raise ValueError("A submit decision must select at least one deployment.")
    if decision == "submit" and plan.get("plan_kind") == "full_demo_observable":
        backends = {item["backend"] for item in normalized}
        required = {item.get("candidate_id") for item in plan["all_deployments"]}
        chosen = {item["candidate_id"] for item in normalized}
        if len(backends) != 1 or chosen != required:
            raise ValueError(
                "A full-observable demo must select every sample on one backend."
            )
    report = raw.get("report_markdown")
    if not isinstance(report, str) or not report.strip():
        report = "# Agent 9 hardware decision\n\n**{}**\n\n{}".format(
            decision, reason.strip()
        )
    return {
        "decision": decision,
        "reason": reason.strip(),
        "selected_deployments": normalized,
        "report_markdown": report.strip(),
    }


class HardwarePlanReviewer:
    def __init__(self, model_name):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before invoking Agent 9.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        self.model_name = model_name

    def review(self, plan):
        evidence = _compact_evidence(plan)
        prompt = """
You are Agent 9, the final scientific decision-maker for IBM Quantum submission.
Decide whether the transpiled circuit can plausibly retain enough physical signal
to produce a meaningful comparison with its same-circuit noiseless result.

Hardware evidence:
```json
{}
```

Use the actual backend evidence: ISA depth and duration proxies, one- and two-qubit
gate counts, accumulated calibration errors, missing error data, layout/SWAP cost,
readout risk, queue/calibration age, available execution time, ideal signal size,
shot uncertainty, and ZNE amplification cost. The recorded resource_gate limits
are historical reference values only; they are not an approval and must not replace
your scientific judgment. A circuit merely being executable does not mean that it
can produce a meaningful result.

For plan_kind=full_demo_observable, approve only if every requested sample can be
run meaningfully on one backend. Do not silently approve a subset. For a
representative pilot, you may select the meaningful candidate(s). If evidence is
missing, the expected signal is likely to be washed out, or the conclusion is
uncertain, choose do_not_submit. Do not judge user authorization; it was handled
upstream. Do not claim that submission already occurred.

Return one JSON object with exactly these semantic fields:
{{
  "decision": "submit" or "do_not_submit",
  "reason": "concise scientific reason",
  "selected_deployments": [
    {{"candidate_id": "exact candidate id", "backend": "exact backend name"}}
  ],
  "report_markdown": "concise human-readable audit"
}}
For do_not_submit, selected_deployments must be empty.
""".format(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantum-hardware feasibility expert with final "
                        "authority over scientific submission. Be conservative: "
                        "when meaningful signal survival is not supported, reject."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Agent 9 returned an empty decision.")
        return _validate_decision(json.loads(content), plan)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Decide one IBM hardware submission.")
    parser.add_argument("--hardware-plan", required=True)
    parser.add_argument("--output-decision", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    return parser.parse_args()


def main():
    args = parse_arguments()
    decision = HardwarePlanReviewer(args.model).review(_load_plan(args.hardware_plan))
    for path in (args.output_decision, args.output_report):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.output_decision, "w", encoding="utf-8") as handle:
        json.dump(
            {key: decision[key] for key in ("decision", "reason", "selected_deployments")},
            handle, ensure_ascii=False, indent=2, sort_keys=True,
        )
        handle.write("\n")
    with open(args.output_report, "w", encoding="utf-8") as handle:
        handle.write(decision["report_markdown"])
        handle.write("\n")
    print("Agent 9 decision saved: {}.".format(decision["decision"]))


if __name__ == "__main__":
    main()
