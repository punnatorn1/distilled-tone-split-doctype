"""The result tables in README.md, docs/7_results.md and docs/HYPERPARAMETERS.md are generated from results/; this fails
when one changes without the other (fix: python scripts/make_results_tables.py)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "make_results_tables.py"


def _script():
    spec = importlib.util.spec_from_file_location("make_results_tables", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results_tables_in_docs_match_results():
    out = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_readme_headline_tables_are_generated():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "<!-- headline:start" in text and "<!-- headline:end -->" in text


def test_verdict_needs_both_checks():
    verdict = _script().verdict
    assert verdict(p_main=0.01, p_fake=0.30, p_real_fake=0.02) == "**passed**"
    assert verdict(p_main=0.005, p_fake=0.24, p_real_fake=0.08).startswith("not passed")
    assert verdict(p_main=0.22, p_fake=0.37, p_real_fake=0.03).startswith("not passed")
    assert verdict(p_main=0.01, p_fake=0.01, p_real_fake=0.01).startswith("not passed")
    assert verdict(p_main=0.22, p_fake=0.10, p_real_fake=None).startswith("not passed")
