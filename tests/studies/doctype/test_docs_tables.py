"""The result tables in docs/7_results.md and docs/HYPERPARAMETERS.md are generated from results/; this fails when one
changes without the other (fix: python scripts/make_results_tables.py)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_results_tables_in_docs_match_results():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "make_results_tables.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
