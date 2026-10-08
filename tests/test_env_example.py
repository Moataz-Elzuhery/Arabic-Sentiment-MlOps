import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _compose_vars() -> set[str]:
    text = (ROOT / "docker-compose.yml").read_text()
    return set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", text))


def _env_keys(path: Path) -> set[str]:
    keys = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0])
    return keys


def test_env_example_covers_every_compose_variable():
    missing = _compose_vars() - _env_keys(ROOT / ".env.example")
    assert not missing, f"add these to .env.example: {sorted(missing)}"
