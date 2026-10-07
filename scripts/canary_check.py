"""Send N requests through the canary router and report which model version answered.

python -m scripts.canary_check --url http://localhost:8080 --n 400 --expected 5
"""
import argparse
import json
import math
import urllib.request
from collections import Counter

SAMPLE = "المنتج ممتاز جدًا"


def share_within_range(count: int, n: int, expected_pct: float, sigmas: float = 3.0) -> bool:
    """True if `count` of `n` is within `sigmas` binomial std-devs of the expected share."""
    if expected_pct <= 0:       # rollback: even a single canary answer is a failure
        return count == 0
    if expected_pct >= 100:     # full rollout: every answer must come from the canary
        return count == n
    p = expected_pct / 100
    sd = math.sqrt(n * p * (1 - p))
    return abs(count - n * p) <= sigmas * max(sd, 1e-9) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--expected", type=float, required=True,
                    help="expected % of traffic served by the canary version")
    ap.add_argument("--canary-version", default="v2")
    a = ap.parse_args()

    seen, errors = Counter(), 0
    body = json.dumps({"text": SAMPLE}).encode()
    for _ in range(a.n):
        req = urllib.request.Request(a.url + "/predict", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                seen[json.loads(r.read())["model_version"]] += 1
        except Exception:  # noqa: BLE001
            errors += 1

    ok_total = sum(seen.values())
    print(f"requests: {a.n}  ok: {ok_total}  errors: {errors}")
    for version, c in sorted(seen.items()):
        print(f"  {version}: {c} ({100 * c / max(ok_total, 1):.1f}%)")
    canary = seen.get(a.canary_version, 0)
    in_range = share_within_range(canary, ok_total, a.expected)
    print(f"expected canary share {a.expected}% -> {'within' if in_range else 'OUTSIDE'} "
          f"the 3-sigma range")
    return 0 if (in_range and errors == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
