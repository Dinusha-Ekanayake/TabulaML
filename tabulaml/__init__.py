"""TabulaML: interactive ML workbench for tabular data (the app entry point is ``app.py``)."""

if __name__ == "__main__":  # pragma: no cover - only when a host runs this file as the app
    # Some hosts (e.g. a Streamlit Cloud app created with this file as its main file) run
    # the package __init__ as a script. Start the real app instead of showing a blank page.
    # A normal ``import tabulaml`` never enters this branch.
    import runpy
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    runpy.run_path(str(_root / "app.py"), run_name="__main__")
