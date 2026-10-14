# -*- coding: utf-8 -*-
"""Small, deterministic helpers shared by agents and the pipeline runtime."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Dict, Mapping


STAGE_OUTPUT_SCHEMA_VERSION = "1.0"


def resolve_output_tag(value: str | None) -> str:
    tag = str(int(time.time())) if value is None else str(value)
    if not tag or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
        raise ValueError("output tag may contain only letters, digits, '_' and '-'")
    return tag


def categorized_output_directory(output_root: str | None, category: str) -> str:
    """Return a per-run category directory, or legacy ``outputs`` for manual calls."""
    if output_root is None:
        directory = "outputs"
    else:
        if not isinstance(category, str) or not category:
            raise ValueError("output category must be nonempty")
        directory = os.path.join(os.path.abspath(output_root), category)
    os.makedirs(directory, exist_ok=True)
    return directory


def atomic_json_write(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".pipeline_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def write_stage_output_manifest(
    path: str | None, stage: str, outputs: Mapping[str, str]
) -> Dict[str, Any] | None:
    if path is None:
        return None
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    records: Dict[str, Dict[str, str]] = {}
    for name, output_path in outputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("output names must be nonempty strings")
        if not isinstance(output_path, str) or not os.path.isfile(output_path):
            raise ValueError("stage output is missing: {}".format(output_path))
        absolute = os.path.abspath(output_path)
        records[name] = {"path": absolute}
    payload: Dict[str, Any] = {
        "stage_output_schema_version": STAGE_OUTPUT_SCHEMA_VERSION,
        "stage": stage,
        "status": "completed",
        "outputs": records,
    }
    atomic_json_write(path, payload)
    return payload


def load_stage_output_manifest(path: str, expected_stage: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("stage_output_schema_version") != STAGE_OUTPUT_SCHEMA_VERSION:
        raise ValueError("stage output manifest schema must be 1.0")
    if payload.get("stage") != expected_stage or payload.get("status") != "completed":
        raise ValueError("stage output manifest identity/status mismatch")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("stage output manifest contains no outputs")
    for name, record in outputs.items():
        if not isinstance(record, dict) or not os.path.isfile(record.get("path", "")):
            raise ValueError("missing output {}".format(name))
    return payload
