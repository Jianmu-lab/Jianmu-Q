# -*- coding: utf-8 -*-
"""Small shared policy for machine-interface validation.

Only semantic conditions that make a calculation impossible are fatal.
Provenance/hash differences are warnings: hashes remain useful for audit and
corruption diagnosis, but they do not define scientific compatibility.
"""

import json
import logging
from typing import Any, Dict, Optional

from parameter_contract import ParameterContractError


LOGGER = logging.getLogger("PipelineValidation")


def diagnostic(
    level: str,
    code: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "level": level,
        "code": code,
        "message": message,
    }
    if details:
        record["details"] = details
    rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if level == "fatal":
        raise ParameterContractError(rendered)
    if level == "warning":
        LOGGER.warning(rendered)
    else:
        LOGGER.info(rendered)
    return record


def provenance_warning(code: str, message: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        diagnostic(
            "warning",
            code,
            message,
            details={"expected": expected, "actual": actual},
        )
