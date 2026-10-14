# -*- coding: utf-8 -*-
"""Register a manual output in the pipeline's flat manifest."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from pipeline_support import atomic_json_write


class ArtifactRegistrationError(RuntimeError):
    pass


def register_external_artifact(
    manifest_path: str,
    *,
    logical_name: str,
    payload_path: str,
) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.setdefault("artifacts", {})
    absolute = os.path.abspath(payload_path)
    if not os.path.isfile(absolute):
        raise ArtifactRegistrationError("Output is missing: {}".format(absolute))
    record = {"path": absolute}
    records[logical_name] = record
    atomic_json_write(manifest_path, manifest)
    return record
