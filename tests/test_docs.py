import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def test_explanation_is_at_most_300_words():
    assert _words((ROOT / "docs" / "explanation-300w.md").read_text(encoding="utf-8")) <= 300


def test_readme_covers_setup_use_privacy_limitations_signals_and_what_was_tested():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in (
        "scripts/install.py",
        "scripts/run.py",
        "scripts/run_pipeline.py",
        "SELFTEST OK",
        "## Privacy",
        "## Limitations",
        "Signals we trusted",
        "## What has been tested",
    ):
        assert needle in readme, needle


def test_the_readme_has_no_unfilled_placeholders_and_no_stale_script_names():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in readme
    for stale in ("bootstrap.ps1", "run.ps1", "funnel.ps1", "Funnel"):
        assert stale not in readme, stale


def test_every_script_the_readme_mentions_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in sorted(set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", readme))):
        assert (ROOT / "scripts" / name).exists(), name


def test_readme_names_every_rubric_dimension():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for dim in (f"R{i} " for i in range(1, 13)):
        assert dim in readme, dim
