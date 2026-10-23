"""Turn AI session records into one readable, redacted log of the prompts and replies.

    uv run python scripts/export_ai_history.py <out.md> <record> [<record> ...] [--pdf out.pdf]

A record is a Claude Code transcript (.jsonl) or an Antigravity / Gemini conversation database (.db, SQLite with
protocol-buffer steps). Both are read-only.

Kept: every message of the person (pasted reports included), every reply of the assistant, and a one-line note for each tool
call. Dropped: hidden reasoning, system reminders, and the output of tools (it only ever held numbers and file listings, but
nothing is shipped unseen). Redacted: e-mail addresses, access tokens, the home folder name, the machine's network name, and
any value found in a local .env file. Nothing here ever contained a recording, a frame or a transcript: the assistants were
only shown numbers (see GEMINI.md).
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_TOKENS = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|ya29\.[0-9A-Za-z_-]{20,})\b"
    r"|\b1//[0-9A-Za-z_-]{30,}"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)
_HOST = re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.ts\.net\b|\bdesktop-[a-z0-9]{5,}\b", re.IGNORECASE)
_HOME = re.compile(r"(?i)(?:[A-Z]:[\\/]+|/[a-z]/)Users[\\/]+[^\\/\s\"']+")
_ASSISTANT_LIMIT = 6000  # characters per reply; the rest is noted, not shipped
_TOOL_LIMIT = 220


def env_values(env_path: Path) -> list[str]:
    """Values of a local .env (read in memory only, never printed) so that none of them can leak into the log."""
    if not env_path.is_file():
        return []
    values = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if len(value) >= 8:
                values.append(value)
    return values


def redact(text: str, secrets: list[str] | None = None, names: list[str] | None = None) -> str:
    for value in secrets or []:
        text = text.replace(value, "[secret]")
    for name in names or []:  # the account name as a whole word, so "fastapi" survives a user called "asta"
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "[user]", text, flags=re.IGNORECASE)
    text = _PRIVATE_KEY.sub("[private key]", text)
    text = _TOKENS.sub("[token]", text)
    text = _EMAIL.sub("[email]", text)
    text = _HOST.sub("[host]", text)
    text = _HOME.sub(lambda m: re.sub(r"Users([\\/]+)[^\\/\s\"']+", r"Users\1[user]", m.group(0), flags=re.IGNORECASE), text)
    return text


def _blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _tool_note(block: dict) -> str:
    data = block.get("input") or {}
    detail = data.get("description") or data.get("command") or data.get("file_path") or data.get("pattern") or data.get("skill") or ""
    return f"[tool: {block.get('name', '?')}] {' '.join(str(detail).split())[:_TOOL_LIMIT]}"


def entries(path: Path):
    """(time, role, text) for every visible message of one transcript."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        kind = obj.get("type")
        if kind == "attachment":  # a message the person typed while the assistant was still working
            att = obj.get("attachment") or {}
            prompt = _REMINDER.sub("", str(att.get("prompt", ""))).strip()
            if att.get("type") == "queued_command" and "human" in str(att.get("origin", "")) and prompt:
                yield att.get("timestamp") or obj.get("timestamp", ""), "PERSON", prompt
            continue
        if kind not in ("user", "assistant") or obj.get("isMeta"):
            continue
        message = obj.get("message") or {}
        parts = []
        for block in _blocks(message.get("content")):
            if block.get("type") == "text":
                text = _REMINDER.sub("", block.get("text", "")).strip()
                if text and not text.startswith(("<command-name>", "<local-command", "Caveat:")):
                    parts.append(text)
            elif block.get("type") == "tool_use" and kind == "assistant":
                parts.append(_tool_note(block))
        if parts:
            yield obj.get("timestamp", ""), "ASSISTANT" if kind == "assistant" else "PERSON", "\n".join(parts)


def _varint(data: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """(field number, wire type, value) of one protocol-buffer message; ValueError when the bytes are not one."""
    i, out = 0, []
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(data, i)
            out.append((number, wire, value))
        elif wire == 2:
            size, i = _varint(data, i)
            if i + size > len(data):
                raise ValueError("truncated")
            out.append((number, wire, data[i : i + size]))
            i += size
        elif wire in (1, 5):
            i += 8 if wire == 1 else 4
        else:
            raise ValueError("unsupported wire type")
    return out


def _text_fields(data: bytes, path: tuple[int, ...] = ()):
    """Every readable text field of a message, nested messages included, with its path of field numbers."""
    try:
        fields = _fields(data)
    except (ValueError, IndexError):
        return
    for number, wire, value in fields:
        if wire != 2 or not isinstance(value, bytes) or not value:
            continue
        try:
            _fields(value)
            nested = True
        except (ValueError, IndexError):
            nested = False
        if nested:
            yield from _text_fields(value, (*path, number))
            continue
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if sum(ch.isprintable() or ch in "\n\r\t" for ch in text) / len(text) > 0.95:
            yield (*path, number), text


def _step_time(metadata: bytes | None) -> str:
    """ISO time from the first (seconds) field of a step's metadata, or an empty string."""
    try:
        for number, wire, value in _fields(metadata or b""):
            if number == 1 and wire == 2 and isinstance(value, bytes):
                for n2, w2, v2 in _fields(value):
                    if n2 == 1 and w2 == 0 and isinstance(v2, int):
                        return datetime.fromtimestamp(v2, tz=UTC).isoformat()
    except (ValueError, IndexError, OverflowError, OSError):
        pass
    return ""


# Antigravity step types seen in its conversation databases: 14 = the person's message, 15 = a model turn.
_AG_USER, _AG_MODEL = 14, 15


def antigravity_entries(path: Path):
    """(time, role, text) for the person's messages and the model's visible replies of one Antigravity conversation.

    Model turns also carry a hidden reasoning summary and the arguments of tool calls; the reasoning is dropped and each
    tool call becomes a one-line note, as in the Claude transcripts."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select step_type, metadata, step_payload from steps where step_type in (?, ?) order by idx", (_AG_USER, _AG_MODEL)
        ).fetchall()
    finally:
        con.close()
    for step_type, metadata, payload in rows:
        texts = {field: text for field, text in _text_fields(payload)}
        when = _step_time(metadata)
        if step_type == _AG_USER:
            text = (texts.get((19, 2)) or texts.get((19, 3, 1)) or "").strip()
            if text:
                yield when, "PERSON", text
            continue
        parts = []
        reply = (texts.get((20, 1)) or "").strip()
        if reply:
            parts.append(reply)
        call = texts.get((20, 7, 3))
        if call:
            parts.append(f"[tool call] {' '.join(call.split())[:_TOOL_LIMIT]}")
        if parts:
            yield when, "ASSISTANT", "\n".join(parts)


def stamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def build(paths: list[Path], secrets: list[str], names: list[str] | None = None) -> str:
    out = [
        "# AI interaction history (redacted)",
        "",
        (
            "Planning, review and coding sessions for the Class Quality Assessor. PERSON is the human; in the Claude Code sessions "
            "ASSISTANT is Claude (planner and reviewer), in the Antigravity conversation it is Gemini (implementer, working from the "
            "prompts that are also in docs/ai-history/prompts). Tool calls appear as one-line notes; tool output and hidden reasoning "
            "are omitted. Redacted: e-mail addresses, tokens, home-folder name, machine name. No recording, frame or transcript was "
            "ever shown to any assistant."
        ),
        "",
    ]
    for path in paths:
        if path.suffix == ".db":
            out += [f"## Antigravity (Gemini) conversation {path.stem[:8]}", ""]
            source = antigravity_entries(path)
        else:
            out += [f"## Claude Code session {path.stem[:8]}", ""]
            source = entries(path)
        for when, role, text in source:
            text = redact(text, secrets, names)
            if role == "ASSISTANT" and len(text) > _ASSISTANT_LIMIT:
                text = text[:_ASSISTANT_LIMIT].rstrip() + f"\n[... {len(text) - _ASSISTANT_LIMIT} more characters not included]"
            out += [f"**{role}** {stamp(when)}", "", text, ""]
    return "\n".join(out)


def write_pdf(markdown: str, pdf_path: Path) -> None:
    from fpdf import FPDF

    font = next((p for p in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists()), None)
    pdf = FPDF()
    pdf.set_auto_page_break(True, 12)
    pdf.add_page()
    if font:
        pdf.add_font("Body", fname=font)
        pdf.set_font("Body", size=8)
    else:
        pdf.set_font("Helvetica", size=8)
    for raw in markdown.splitlines():
        line = raw if font else raw.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 3.6, line or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a redacted AI interaction history.")
    parser.add_argument("out", type=Path)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--pdf", type=Path, default=None, help="also write a PDF (needs the fpdf2 package)")
    args = parser.parse_args(argv)

    secrets = env_values(ROOT / ".env")
    text = build(args.sessions, secrets, [Path.home().name])
    leaked = [v for v in secrets if v in text]
    if leaked:
        print("refusing to write: a secret survived redaction", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text) / 1e6:.1f} MB, {text.count('**PERSON**')} messages from the person, "
          f"{text.count('**ASSISTANT**')} replies)")
    if args.pdf:
        write_pdf(text, args.pdf)
        print(f"wrote {args.pdf} ({args.pdf.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
