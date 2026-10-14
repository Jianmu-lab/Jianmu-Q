"""Verify the three published snapshots offline; no LLM or hardware calls."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    expected = {"njl", "schwinger", "z2"}
    actual = {p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"}
    if actual != expected:
        raise SystemExit(f"Expected exactly {sorted(expected)}, found {sorted(actual)}")
    for name in sorted(expected):
        directory = root / name
        provenance = json.loads((directory / "provenance.json").read_text())
        for relative, checksum in provenance["sha256"].items():
            path = directory / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise SystemExit(f"Missing or changed snapshot artifact: {name}/{relative}")
            if path.suffix == ".json":
                json.loads(path.read_text())
        scipy = json.loads((directory / "references/scipy_reference.json").read_text())
        itensor = json.loads((directory / "references/itensor_reference.json").read_text())
        measurement = json.loads((directory / "runtime/measurement_result.json").read_text())
        sample_sets = []
        for label, record in (("SciPy", scipy), ("ITensor", itensor)):
            samples = record["observable_samples"]
            ids = [sample["sample_id"] for sample in samples]
            if len(ids) != len(set(ids)):
                raise SystemExit(f"Duplicate {label} sample ID in {name}")
            sample_sets.append(set(ids))
        if sample_sets[0] != sample_sets[1]:
            raise SystemExit(f"Classical reference sample coverage differs in {name}")
        verdict = json.loads((directory / "analysis/crosscheck_result.json").read_text())["status"]
        print(f"{name}: {len(provenance['sha256'])} artifacts verified; "
              f"{len(sample_sets[0])} reference samples; cross-check={verdict}")


if __name__ == "__main__":
    main()
