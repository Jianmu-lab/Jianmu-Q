# -*- coding: utf-8 -*-
"""Publish human audit reports without making typography a machine-chain gate.

Reports are deliberately stored as-is.  They are useful to people, but they are
not machine interfaces and therefore are neither compiled nor repaired by an
LLM during the scientific pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict
from pipeline_support import atomic_json_write


def publish_audit_report(
    source: str,
    *,
    stage_name: str,
    output_tag: str,
    report_directory: str,
    audit_directory: str,
    compiled_filename: str,
    rejected_filename: str,
    status_filename: str,
) -> Dict[str, Any]:
    """Publish one non-authoritative human report.

    Scientific validity comes from the JSON contract validators, not from TeX
    compilation or typography repair.
    """
    del output_tag, rejected_filename
    final_source = source
    filename = compiled_filename
    path = os.path.join(report_directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(final_source)
        if final_source and not final_source.endswith("\n"):
            handle.write("\n")
    record: Dict[str, Any] = {
        "audit_report_schema_version": "1.0",
        "stage": stage_name,
        "status": "published_unchecked",
        "report_path": os.path.abspath(path),
        "machine_chain_blocked": False,
    }
    status_path = os.path.join(audit_directory, status_filename)
    atomic_json_write(status_path, record)
    record["status_path"] = os.path.abspath(status_path)
    return record
