"""Documentation that executes: doctests, every Python code block in the docs, and the example scripts."""

import doctest
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


@pytest.mark.parametrize("module", ["neuralnaca.api", "neuralnaca.xfoil", "neuralnaca"])
def test_doctests(module):
    mod = importlib.import_module(module)
    result = doctest.testmod(mod, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0


def python_blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), flags=re.S) if page.exists() else []


PAGES_WITH_CODE = [p for p in DOC_PAGES if python_blocks(p)]


@pytest.mark.parametrize("page", PAGES_WITH_CODE, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_documentation_code_blocks_run(page, tmp_path, monkeypatch):
    """The Python blocks of a page run in order in one namespace, like a notebook. Blocks guarded by
    ``if __name__ == "__main__":`` (they need XFOIL or train models) are only compiled."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "docs_example"}
    for i, code in enumerate(python_blocks(page)):
        compiled = compile(code, f"{page.name}#{i}", "exec")
        if "__main__" not in code:
            exec(compiled, ns)


def test_cli_reference_lists_every_command():
    from neuralnaca.cli import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    text = (ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    for name in sub.choices:
        assert f"| `{name}` |" in text, name


@pytest.mark.parametrize("script", ["quickstart.py", "compare_neuralfoil.py"])
def test_examples_run(script, tmp_path):
    if script == "compare_neuralfoil.py":
        pytest.importorskip("neuralfoil")
    out = subprocess.run([sys.executable, str(ROOT / "examples" / script)], cwd=tmp_path, capture_output=True,
                         text=True, encoding="utf-8", timeout=600, env={**os.environ, "PYTHONUTF8": "1"})
    assert out.returncode == 0, out.stderr


def test_plot_example_runs(tmp_path):
    pytest.importorskip("matplotlib")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "polar_plot.py"), "1e6"], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8", timeout=600, env=env)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "polars.png").exists()


def test_training_example_compiles():
    compile((ROOT / "examples" / "train_custom_model.py").read_text(encoding="utf-8"), "train_custom_model.py", "exec")
