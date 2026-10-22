import re
from pathlib import Path


def _parse_timestamp(ts: str) -> float:
    """Parse WebVTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) into seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    return float(ts)


def parse_vtt(path: Path) -> list[tuple[float, float, str | None, str]]:
    """Parse a WebVTT file into a list of (start, end, speaker, text) tuples."""
    path = Path(path).resolve()
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    cues: list[tuple[float, float, str | None, str]] = []
    i = 0
    time_pattern = re.compile(
        r"((?:\d{1,2}:)?\d{2}:\d{2}\.\d{3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}\.\d{3})"
    )
    speaker_pattern = re.compile(r"<v\s+([^>]+)>(.*)")
    colon_pattern = re.compile(r"^([^:<>]{1,40}):\s+(.*)$")

    while i < len(lines):
        line = lines[i].strip()
        match = time_pattern.search(line)
        if match:
            start_s = _parse_timestamp(match.group(1))
            end_s = _parse_timestamp(match.group(2))
            i += 1
            text_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            full_text = " ".join(text_lines)

            speaker: str | None = None
            sp_match = speaker_pattern.match(full_text)
            if sp_match:
                speaker = sp_match.group(1).strip()
                full_text = sp_match.group(2).strip()
            else:
                # The platform's own transcripts write "Speaker Name: text"
                colon_match = colon_pattern.match(full_text)
                if colon_match:
                    speaker = colon_match.group(1).strip()
                    full_text = colon_match.group(2).strip()

            # Strip any remaining HTML-like tags
            clean_text = re.sub(r"<[^>]+>", "", full_text).strip()
            cues.append((start_s, end_s, speaker, clean_text))
        else:
            i += 1

    cues.sort(key=lambda c: c[0])
    return cues


def wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis text."""
    r_words = ref.strip().split()
    h_words = hyp.strip().split()

    if not r_words:
        return 0.0 if not h_words else 1.0

    r_len = len(r_words)
    h_len = len(h_words)

    # Dynamic programming matrix for Levenshtein distance on words
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if r_words[i - 1] == h_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return float(dp[r_len][h_len]) / float(r_len)
