import subprocess
from pathlib import Path

import pytest

from scripts.release_scan import main


def git_cmd(repo: Path, *args: str) -> None:
    cmd = [
        "git",
        "-c",
        "user.name=TestUser",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
    ] + list(args)
    subprocess.run(cmd, cwd=repo, check=True, capture_output=True)


def test_clean_temp_repo_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    f = tmp_path / "README.md"
    f.write_text("# Test Repo\nClean documentation.", encoding="utf-8")
    git_cmd(tmp_path, "add", "README.md")
    git_cmd(tmp_path, "commit", "-m", "Initial commit")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "Total violations: 0" in captured.out


def test_release_scan_flags_planted_secret_in_temp_repo(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    planted = "ghp_" + "A" * 36
    f = tmp_path / "token.txt"
    f.write_text(f"api_key = '{planted}'\n", encoding="utf-8")
    git_cmd(tmp_path, "add", "token.txt")
    git_cmd(tmp_path, "commit", "-m", "Add token")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "secret_key: 1" in captured.out or "secret_key: 2" in captured.out
    assert planted not in captured.out
    assert planted not in captured.err


def test_flags_secret_that_exists_only_in_history(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    planted = "sk-" + "Z" * 30
    f = tmp_path / "old_key.txt"
    f.write_text(f"KEY={planted}\n", encoding="utf-8")
    git_cmd(tmp_path, "add", "old_key.txt")
    git_cmd(tmp_path, "commit", "-m", "Commit key")

    # Delete the file in a subsequent commit
    f.unlink()
    git_cmd(tmp_path, "rm", "old_key.txt")
    git_cmd(tmp_path, "commit", "-m", "Remove key")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "Commits affected: 1" in captured.out
    assert planted not in captured.out


def test_flags_media_extension(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    media = tmp_path / "recording.mp4"
    media.write_bytes(b"dummy mp4 bytes")
    git_cmd(tmp_path, "add", "recording.mp4")
    git_cmd(tmp_path, "commit", "-m", "Add recording")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "forbidden_extension: 1" in captured.out


def test_env_values_are_detected_but_not_printed(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(".env\n", encoding="utf-8")
    git_cmd(tmp_path, "add", ".gitignore")
    git_cmd(tmp_path, "commit", "-m", "Add gitignore")

    secret_val = "super_secret_env_value_xyz"
    env_file = tmp_path / ".env"
    env_file.write_text(f"API_SECRET={secret_val}\n", encoding="utf-8")

    leaked_file = tmp_path / "config.txt"
    leaked_file.write_text(f"token={secret_val}\n", encoding="utf-8")
    git_cmd(tmp_path, "add", "config.txt")
    git_cmd(tmp_path, "commit", "-m", "Leak secret")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "env_secret: 1" in captured.out or "env_secret: 2" in captured.out
    assert secret_val not in captured.out
    assert secret_val not in captured.err


def test_names_file_words_flagged(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    names_file = data_dir / "names.txt"
    names_file.write_text("Siddharth\nAnanya\n", encoding="utf-8")
    git_cmd(tmp_path, "add", "data/names.txt")
    git_cmd(tmp_path, "commit", "-m", "Add names list")

    transcript_file = tmp_path / "notes.txt"
    transcript_file.write_text("Student Siddharth made a move.", encoding="utf-8")
    git_cmd(tmp_path, "add", "notes.txt")
    git_cmd(tmp_path, "commit", "-m", "Add notes")

    code = main(["--root", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "name:" in captured.out


def test_flags_media_that_was_committed_and_later_deleted(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\0 not really a video")
    git_cmd(tmp_path, "add", "clip.mp4")
    git_cmd(tmp_path, "commit", "-m", "Add clip")
    git_cmd(tmp_path, "rm", "clip.mp4")
    git_cmd(tmp_path, "commit", "-m", "Remove clip")

    code = main(["--root", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 1
    assert "forbidden_extension_in_history: 1" in out and "forbidden_extension: 0" in out


@pytest.mark.parametrize("name", ["frame_000001.jpg", "docs/shot.png", "voices.wav", "embeddings.npz", "face.JPEG"])
def test_flags_pictures_and_audio_derived_from_recordings(tmp_path: Path, capsys: pytest.CaptureFixture, name: str):
    git_cmd(tmp_path, "init")
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0 pretend media")
    git_cmd(tmp_path, "add", "-f", name)
    git_cmd(tmp_path, "commit", "-m", "Add media")

    code = main(["--root", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 1 and "forbidden_extension: 1" in out


def test_the_logo_asset_folder_is_allowed(tmp_path: Path, capsys: pytest.CaptureFixture):
    git_cmd(tmp_path, "init")
    logo = tmp_path / "assets" / "logo.png"
    logo.parent.mkdir()
    logo.write_bytes(b"\0 pretend logo")
    git_cmd(tmp_path, "add", "assets/logo.png")
    git_cmd(tmp_path, "commit", "-m", "Add logo")

    code = main(["--root", str(tmp_path)])
    assert code == 0 and "forbidden_extension: 0" in capsys.readouterr().out
