import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

FORBIDDEN_EXTS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".vtt",
    ".srt",
    ".pdf",
    ".zip",
}

KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def is_allowed_email(email: str) -> bool:
    email_lower = email.lower()
    return email_lower.endswith("@example.com") or email_lower == "noreply@anthropic.com"


def load_env_values(root: Path) -> list[str]:
    env_path = root / ".env"
    if not env_path.is_file():
        return []
    values: list[str] = []
    try:
        lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, val = line.split("=", 1)
            val = val.strip().strip("'\"")
            if len(val) >= 8:
                values.append(val)
    return values


def load_names_patterns(root: Path) -> list[re.Pattern]:
    names_path = root / "data" / "names.txt"
    if not names_path.is_file():
        return []
    patterns: list[re.Pattern] = []
    try:
        lines = names_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        name = line.strip()
        if name:
            patterns.append(re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE))
    return patterns


def is_binary(chunk: bytes) -> bool:
    return b"\0" in chunk[:1024]


def scan_text(
    text: str,
    env_values: list[str],
    name_patterns: list[re.Pattern],
) -> list[str]:
    """Return categories of violations found in text (without returning matched content)."""
    found: list[str] = []
    for kp in KEY_PATTERNS:
        if kp.search(text):
            found.append("secret_key")
            break

    for env_val in env_values:
        if env_val in text:
            found.append("env_secret")
            break

    for m in EMAIL_RE.finditer(text):
        if not is_allowed_email(m.group(0)):
            found.append("email")
            break

    for np in name_patterns:
        if np.search(text):
            found.append("name")
            break

    return found


def run_scan(root: Path) -> tuple[int, str]:
    root = root.resolve()
    category_counts: Counter[str] = Counter()
    tree_matches: list[tuple[str, str]] = []
    files_affected: set[str] = set()
    commits_affected: set[str] = set()

    env_values = load_env_values(root)
    name_patterns = load_names_patterns(root)

    # 1. Scan working tree files tracked or untracked-but-not-ignored
    try:
        res = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(root),
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )
        tree_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        tree_files = []

    for rel_path in tree_files:
        path = root / rel_path
        if os.path.islink(path):
            continue

        # Check extension
        ext = path.suffix.lower()
        if ext in FORBIDDEN_EXTS:
            category_counts["forbidden_extension"] += 1
            tree_matches.append((rel_path, "forbidden_extension"))
            files_affected.add(rel_path)

        if not path.is_file():
            continue

        try:
            with open(path, "rb") as f:
                data = f.read(5 * 1024 * 1024)
        except OSError:
            continue

        if is_binary(data):
            continue

        # Skip checking .env against itself
        curr_env_values = [] if rel_path == ".env" else env_values

        text = data.decode("utf-8", errors="replace")
        violations = scan_text(text, curr_env_values, name_patterns)
        for cat in violations:
            category_counts[cat] += 1
            tree_matches.append((rel_path, cat))
            files_affected.add(rel_path)

    # 2. Scan git history added lines
    try:
        log_res = subprocess.run(
            ["git", "log", "-p", "--all"],
            cwd=str(root),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        log_output = log_res.stdout
    except FileNotFoundError:
        log_output = ""

    current_commit: str | None = None
    commit_re = re.compile(r"^commit\s+([0-9a-fA-F]{7,40})")

    for line in log_output.splitlines():
        c_m = commit_re.match(line)
        if c_m:
            current_commit = c_m.group(1)
            continue

        if line.startswith("+") and not line.startswith("+++"):
            added_content = line[1:]
            violations = scan_text(added_content, env_values, name_patterns)
            for cat in violations:
                category_counts[cat] += 1
                if current_commit:
                    commits_affected.add(current_commit)

    # 3. Files that were ever added with a forbidden extension, even if a later commit removed them
    try:
        names_res = subprocess.run(
            ["git", "log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:%H"],
            cwd=str(root), capture_output=True, text=True, errors="replace", check=False,
        )
        names_output = names_res.stdout
    except FileNotFoundError:
        names_output = ""
    hash_re = re.compile(r"^[0-9a-f]{40}$")
    for line in names_output.splitlines():
        if hash_re.match(line):
            current_commit = line
        elif line.strip() and Path(line.strip()).suffix.lower() in FORBIDDEN_EXTS:
            category_counts["forbidden_extension_in_history"] += 1
            if current_commit:
                commits_affected.add(current_commit)

    total_violations = sum(category_counts.values())

    lines: list[str] = ["Release Scan Summary:"]
    if tree_matches:
        lines.append("Working Tree Findings (relative paths):")
        for p, cat in tree_matches:
            lines.append(f"  {p} [{cat}]")
    else:
        lines.append("No working tree findings.")

    lines.append("Categories:")
    for cat in sorted(["forbidden_extension", "forbidden_extension_in_history", "secret_key", "env_secret", "email", "name"]):
        lines.append(f"  {cat}: {category_counts.get(cat, 0)}")

    lines.append(f"Files affected: {len(files_affected)}")
    lines.append(f"Commits affected: {len(commits_affected)}")
    lines.append(f"Total violations: {total_violations}")

    exit_code = 1 if total_violations > 0 else 0
    return exit_code, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release hygiene and secret scanner.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Path to repository root")
    args = parser.parse_args(argv)

    exit_code, report = run_scan(args.root)
    print(report)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
