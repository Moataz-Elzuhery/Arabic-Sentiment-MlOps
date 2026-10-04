"""CI quality gate: fail if the candidate model is worse than the baseline.

Uses only the standard library so it runs in CI without installing ML packages.

python -m scripts.quality_gate --candidate metrics/production.json --baseline metrics/baseline.json
"""
import argparse
import json
import sys
from pathlib import Path


def check(candidate: dict, baseline: dict, tolerance: float, min_class_f1: float) -> list[str]:
    """Return a list of failure messages (empty list = gate passed)."""
    failures = []
    cand_f1, base_f1 = candidate["f1_macro"], baseline["f1_macro"]
    if cand_f1 < base_f1 - tolerance:
        failures.append(
            f"F1 macro {cand_f1:.4f} is below baseline {base_f1:.4f} (tolerance {tolerance})"
        )
    for label, score in candidate.get("per_class_f1", {}).items():
        if score < min_class_f1:
            failures.append(f"F1 for class '{label}' is {score:.4f}, below the floor {min_class_f1}")
    return failures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="metrics/production.json")
    p.add_argument("--baseline", default="metrics/baseline.json")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="allowed drop vs baseline (absorbs rounding/noise)")
    p.add_argument("--min-class-f1", type=float, default=0.70,
                   help="no single class may fall below this F1")
    a = p.parse_args()

    candidate = json.loads(Path(a.candidate).read_text())
    baseline = json.loads(Path(a.baseline).read_text())
    print(f"candidate: {candidate.get('model_version', '?')}  F1 macro = {candidate['f1_macro']:.4f}")
    print(f"baseline : {baseline.get('model_version', '?')}  F1 macro = {baseline['f1_macro']:.4f}")

    failures = check(candidate, baseline, a.tolerance, a.min_class_f1)
    if failures:
        print("\nQUALITY GATE FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
