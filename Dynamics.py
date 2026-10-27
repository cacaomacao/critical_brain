"""Explicit entry point for the preserved historical grid demonstration.

Importing this module does not build a random network, simulate, or open plots.
The legacy model has unresolved semantics documented in docs/dynamics-contract.md.
"""
from pathlib import Path
from runpy import run_path


def run_demo():
    """Run the original experiment explicitly and return its script namespace."""
    script = Path(__file__).resolve().parent / "legacy" / "dynamics_demo.py"
    return run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    run_demo()
