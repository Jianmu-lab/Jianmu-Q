# -*- coding: utf-8 -*-
"""Tolerant JSON ingestion at the stochastic-model boundary.

This module never validates scientific meaning. It only recovers plausible
JSON objects from fenced or plain model output and selects one by semantic
shape. Strict schema and physics validation happens after controller
normalization.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Sequence


class ModelOutputIngestionError(ValueError):
    pass


def json_object_candidates(text: str) -> Sequence[Dict[str, Any]]:
    candidates = []
    seen = set()

    def remember(value: Any) -> None:
        if not isinstance(value, dict):
            return
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            candidates.append(value)

    for block in re.findall(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    ):
        try:
            remember(json.loads(block))
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        remember(value)
    return candidates


def select_semantic_object(
    text: str,
    *,
    label: str,
    fingerprint_keys: Sequence[str],
    discriminator_key: str | None = None,
    discriminator_value: str | None = None,
    minimum_fingerprint_matches: int = 2,
) -> Dict[str, Any]:
    """Select one object by controller-known identity or structural shape."""
    candidates = list(json_object_candidates(text))
    exact = []
    if discriminator_key and discriminator_value:
        exact = [
            item for item in candidates
            if item.get(discriminator_key) == discriminator_value
        ]
    if len(exact) == 1:
        return json.loads(json.dumps(exact[0], ensure_ascii=False))
    if len(exact) > 1:
        raise ModelOutputIngestionError(
            "Multiple explicit {} JSON objects were found.".format(label)
        )

    scored = []
    for item in candidates:
        score = sum(1 for key in fingerprint_keys if key in item)
        if score >= minimum_fingerprint_matches:
            scored.append((score, len(item), item))
    if not scored:
        raise ModelOutputIngestionError(
            "No plausible {} JSON object was found.".format(label)
        )
    best_score = max(score for score, _, _ in scored)
    best = [entry for entry in scored if entry[0] == best_score]
    # Prefer the complete outer object over a nested object with the same
    # semantic-key score.
    best_size = max(size for _, size, _ in best)
    best = [item for score, size, item in best if size == best_size]
    if len(best) != 1:
        raise ModelOutputIngestionError(
            "Multiple equally plausible {} JSON objects were found.".format(label)
        )
    return json.loads(json.dumps(best[0], ensure_ascii=False))
