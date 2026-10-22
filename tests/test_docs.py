import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def test_explanation_is_at_most_300_words():
    assert _words((ROOT / "docs" / "explanation-300w.md").read_text(encoding="utf-8")) <= 300


def test_readme_covers_run_steps_privacy_limitations_and_signals():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("bootstrap.ps1", "run.ps1", "funnel.ps1", "## Privacy", "## Limitations", "Signals we trusted"):
        assert needle in readme, needle
    assert "submission form" in readme  # the access code is handed over there, never kept in the repository


def test_readme_names_every_rubric_dimension():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for dim in (f"R{i} " for i in range(1, 13)):
        assert dim in readme, dim
