"""Quick check: which Python packages can be imported on this machine."""
import importlib

mods = ["numpy", "pandas", "scipy", "sklearn", "sklearn.utils.murmurhash",
        "torch", "tokenizers", "transformers", "mlflow", "dvc"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK    {m:28} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {m:28} {type(e).__name__}: {str(e)[:90]}")
