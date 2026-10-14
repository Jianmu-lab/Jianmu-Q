# -*- coding: utf-8 -*-
"""Small, read-only scientific-review interface.

Reviewers report scientific or domain-technical findings only. They never
judge formatting, provenance, workflow authorization, historical compliance,
or stage ownership, and never mutate authoritative machine contracts.
Formatting variation is normalized at the model boundary; only an explicit
scientific ``needs_revision`` verdict is blocking.
"""

from __future__ import annotations

import json
import glob
import os
from typing import Any, Dict, Mapping, Sequence

from model_output_ingestion import json_object_candidates
from parameter_contract import ParameterContractError


class ReviewProtocolError(ParameterContractError):
    """Compatibility name for an unreadable reviewer response."""


AUTOMATIC_REVISION_POLICY_VERSION = "2.0"


def next_review_paths(directory: str, stage: str, output_tag: str) -> tuple[str, str]:
    """Return non-overwriting paths for one reviewer call."""
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_Attempt{}".format(attempt)
        raw = os.path.join(
            directory, "{}_ReviewerRaw_{}{}.txt".format(stage, output_tag, suffix)
        )
        status = os.path.join(
            directory, "{}_ReviewStatus_{}{}.json".format(stage, output_tag, suffix)
        )
        if not os.path.exists(raw) and not os.path.exists(status):
            return raw, status
        attempt += 1


def automatic_revision_was_attempted(
    directory: str, stage: str, output_tag: str
) -> bool:
    """Return whether this stage/run already spent any correction attempt."""
    return automatic_revision_count(directory, stage, output_tag) > 0


def automatic_revision_count(
    directory: str, stage: str, output_tag: str
) -> int:
    """Count distinct persisted correction attempts for one stage/run."""
    pattern = os.path.join(
        directory, "{}_ReviewStatus_{}*.json".format(stage, output_tag)
    )
    revision_paths = set()
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        revision = record.get("automatic_revision")
        # A legacy Agent3 preflight repair was performed before its scientific
        # reviewer could inspect the candidate. It must not consume the single
        # consolidated scientific-correction pass.
        if record.get("status") == "preflight_corrected":
            continue
        if (
            isinstance(revision, Mapping)
            and revision.get("attempted") is True
            and revision.get("policy_version")
            == AUTOMATIC_REVISION_POLICY_VERSION
        ):
            raw_path = revision.get("revision_raw_response_path")
            revision_paths.add(str(raw_path or os.path.abspath(path)))
    return len(revision_paths)


def _finding_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("message", "finding", "description", "problem", "issue", "reason"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_finding(value: Any) -> Dict[str, str]:
    source = value if isinstance(value, Mapping) else {}
    severity = str(source.get("severity", "warning")).strip().lower()
    severity = {"critical": "fatal", "warn": "warning", "error": "fatal"}.get(
        severity, severity
    )
    if severity not in ("info", "warning", "fatal"):
        severity = "warning"
    finding = {"severity": severity, "message": _finding_text(value)}
    suggestion = next(
        (
            source.get(key)
            for key in (
                "suggested_correction", "correction", "suggested_fix",
                "recommendation", "fix",
            )
            if isinstance(source.get(key), str) and source[key].strip()
        ),
        None,
    )
    if suggestion:
        finding["suggested_correction"] = suggestion.strip()
    return finding


def _select_review_object(text: str, contract_type: str) -> Mapping[str, Any]:
    candidates = list(json_object_candidates(text))
    if not candidates:
        raise ReviewProtocolError("Reviewer returned no readable JSON object.")
    exact = [item for item in candidates if item.get("contract_type") == contract_type]
    if exact:
        return exact[-1]
    signal_keys = {
        "verdict", "decision", "status", "result", "findings", "issues",
        "problems", "comments", "notes",
    }
    signaled = [item for item in candidates if any(key in item for key in signal_keys)]
    return (signaled or candidates)[-1]


def _extract_review(text: str, contract_type: str) -> Dict[str, Any]:
    raw = _select_review_object(text, contract_type)
    raw_findings = next(
        (
            raw[key]
            for key in ("findings", "issues", "problems", "comments", "notes")
            if key in raw
        ),
        [],
    )
    if raw_findings is None:
        raw_findings = []
    if not isinstance(raw_findings, list):
        raw_findings = [raw_findings]
    findings = [_normalize_finding(item) for item in raw_findings]

    raw_verdict = next(
        (raw[key] for key in ("verdict", "decision", "status", "result") if key in raw),
        None,
    )
    verdict = str(raw_verdict).strip().lower() if raw_verdict is not None else ""
    verdict = {
        "accepted": "accept", "approve": "accept", "approved": "accept",
        "pass": "accept", "passed": "accept", "ok": "accept",
        "revise": "needs_revision", "revised": "needs_revision",
        "reject": "needs_revision", "rejected": "needs_revision",
        "changes_required": "needs_revision", "needs revision": "needs_revision",
    }.get(verdict, verdict)
    if verdict not in ("accept", "needs_revision"):
        verdict = "needs_revision" if any(
            item["severity"] == "fatal" for item in findings
        ) else "accept"
    if any(item["severity"] == "fatal" for item in findings):
        verdict = "needs_revision"
    return {
        "contract_type": contract_type,
        "verdict": verdict,
        "findings": findings,
    }


def extract_review_decision(text: str, contract_type: str) -> Dict[str, Any]:
    return _extract_review(text, contract_type)


def extract_single_artifact_review(text: str, contract_type: str) -> Dict[str, Any]:
    return _extract_review(text, contract_type)


def render_review_audit(decision: Mapping[str, Any]) -> str:
    """Render a deterministic short TeX report; compilation is optional."""
    def escape(value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        }
        return "".join(replacements.get(character, character) for character in value)

    findings: Sequence[Mapping[str, Any]] = decision.get("findings", [])
    items = []
    for finding in findings:
        text = "[{}] {}".format(
            finding.get("severity", "warning"), finding.get("message", "")
        )
        if finding.get("suggested_correction"):
            text += " Suggested correction: " + str(finding["suggested_correction"])
        items.append(r"\item " + escape(text))
    if not items:
        items.append(r"\item No scientific issue was reported.")
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{Machine-contract review}\nVerdict: \\texttt{" +
        escape(str(decision.get("verdict", "accept"))) + "}.\n\\begin{itemize}\n" +
        "\n".join(items) + "\n\\end{itemize}\n"
        "The reviewer did not modify the authoritative JSON contract.\n"
        "\\end{document}\n"
    )
