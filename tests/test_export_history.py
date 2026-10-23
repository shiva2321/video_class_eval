import json

import pytest

from scripts import export_ai_history as export


def test_emails_tokens_hosts_and_home_folders_are_redacted():
    text = (
        "mail me at someone.name+tag@example.com or a@example.com; token hf_" + "A" * 30 + " and ghp_" + "b" * 36 + "; "
        r"files in C:\Users\Asta\Documents and /c/Users/Asta/.claude; host desktop-abc123xyz.tail1234.ts.net"
    )
    out = export.redact(text)
    for leaked in ("someone.name", "a@example.com", "A" * 30, "b" * 36, "Asta", "desktop-abc123xyz", "tail1234"):
        assert leaked not in out
    assert "[email]" in out and "[token]" in out and "[host]" in out and r"C:\Users\[user]" in out and "/c/Users/[user]" in out


def test_values_from_a_local_env_file_are_removed_wherever_they_appear():
    assert export.redact("code is s3cr3t-value-123 ok", ["s3cr3t-value-123"]) == "code is [secret] ok"


def test_env_values_are_read_without_comments_or_short_values(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# note=ignored_comment_value\nACCESS_CODE=abcdefghijkl\nSHORT=ab\nHF_TOKEN=\n", encoding="utf-8")
    assert export.env_values(env) == ["abcdefghijkl"]
    assert export.env_values(tmp_path / "missing") == []


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_the_log_keeps_prompts_replies_and_shell_output_and_drops_reasoning_reminders_and_other_tool_output(tmp_path):
    session = tmp_path / "abcd1234-session.jsonl"
    _write(
        session,
        [
            {"type": "user", "timestamp": "2026-10-22T10:00:00Z", "message": {"content": "Please build the app. <system-reminder>hidden rules</system-reminder>"}},
            {
                "type": "assistant",
                "timestamp": "2026-10-22T10:01:00Z",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "PRIVATE-THOUGHTS-MARKER"},
                        {"type": "text", "text": "I will start with the plan."},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest -q", "description": "Run the tests"}},
                        {"type": "tool_use", "id": "t2", "name": "mcp__PDF_Tools__convert_pdf_to_markdown", "input": {"file_path": "notes.pdf"}},
                        {"type": "tool_use", "id": "t3", "name": "Read", "input": {"file_path": "code.py"}},
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2026-10-22T10:02:00Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "OUTPUT 123 numbers"},
                        {"type": "tool_result", "tool_use_id": "t2", "content": "TUTOR NAME IN A DOCUMENT"},
                        {"type": "tool_result", "tool_use_id": "t3", "content": "FILE CONTENTS"},
                    ]
                },
            },
            {"type": "user", "isMeta": True, "message": {"content": "META-ONLY-MARKER"}},
        ],
    )
    log = export.build([session], [])
    assert "Please build the app." in log and "I will start with the plan." in log and "[tool: Bash] Run the tests" in log
    assert "OUTPUT 123 numbers" in log and "**RESULT**" in log  # what the shell printed is kept
    for dropped in ("hidden rules", "PRIVATE-THOUGHTS-MARKER", "TUTOR NAME IN A DOCUMENT", "FILE CONTENTS", "META-ONLY-MARKER"):
        assert dropped not in log
    assert log.count("**PERSON**") == 1 and log.count("**CLAUDE**") == 1 and log.count("**RESULT**") == 1


def test_shell_output_is_cut_after_a_few_hundred_characters(tmp_path):
    session = tmp_path / "out.jsonl"
    _write(
        session,
        [
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "a", "name": "PowerShell", "input": {"command": "dir"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "a", "content": "y" * 3000}]}},
        ],
    )
    log = export.build([session], [])
    assert "y" * 500 in log and "y" * 501 not in log


def test_very_long_replies_are_cut_with_a_note_and_normal_long_ones_are_whole(tmp_path):
    session = tmp_path / "long.jsonl"
    _write(session, [{"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 45000}]}}])
    log = export.build([session], [])
    assert "more characters not included" in log and len(log) < export._REPLY_LIMIT + 3000  # cut to the limit, plus the headings and the note
    session2 = tmp_path / "ok.jsonl"
    _write(session2, [{"type": "assistant", "message": {"content": [{"type": "text", "text": "z" * 12000}]}}])
    whole = export.build([session2], [])
    assert "z" * 12000 in whole and "not included" not in whole


def test_nothing_is_written_when_a_secret_survives(tmp_path, monkeypatch):
    session = tmp_path / "s.jsonl"
    _write(session, [{"type": "user", "message": {"content": "my value is abc"}}])
    monkeypatch.setattr(export, "env_values", lambda _path: ["abc"])
    monkeypatch.setattr(export, "redact", lambda text, secrets=None, names=None: text)  # a broken redaction must be caught by the final check
    out = tmp_path / "out.md"
    assert export.main([str(out), str(session)]) == 1 and not out.exists()


def test_a_message_typed_while_the_assistant_was_working_is_kept(tmp_path):
    session = tmp_path / "queued.jsonl"
    _write(
        session,
        [
            {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "the run is taking long", "origin": {"kind": "human"}, "timestamp": "2026-10-22T10:00:00Z"}},
            {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "automated notice", "origin": {"kind": "task-notification"}}},
        ],
    )
    log = export.build([session], [])
    assert "the run is taking long" in log and "automated notice" not in log


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _antigravity_db(path, steps):
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("create table steps (idx integer primary key, step_type integer, metadata blob, step_payload blob)")
    for idx, (step_type, seconds, payload) in enumerate(steps):
        metadata = _field(1, _varint(1 << 3) + _varint(seconds))
        con.execute("insert into steps values (?, ?, ?, ?)", (idx, step_type, metadata, payload))
    con.commit()
    con.close()


def test_an_antigravity_conversation_is_read_for_messages_replies_thinking_tool_calls_and_command_output(tmp_path):
    user = _field(19, _field(2, b"Workspace is the repo. Build task one.") + _field(3, _field(1, b"Workspace is the repo. Build task one.")))
    model = _field(20, _field(1, b"Done, all tests pass.") + _field(3, b"I should run the tests first") + _field(7, _field(3, b'{"Path":"x.py"}')))
    command = _field(140, _field(2, _field(1, b"Command exited with code 0. all good") + _field(6, _field(1, b"type.googleapis.com/exa.cortex.core.tools.RunCommand"))))
    file_read = _field(140, _field(2, _field(1, b"CONTENTS OF A FILE THAT WAS READ") + _field(6, _field(1, b"type.googleapis.com/exa.cortex.core.tools.ReadFile"))))
    other = _field(30, _field(5, b"internal checkpoint text"))
    db = tmp_path / "e7b3e06f-test.db"
    _antigravity_db(
        db,
        [(14, 1_792_000_000, user), (15, 1_792_000_060, model), (132, 1_792_000_065, command), (132, 1_792_000_066, file_read), (23, 1_792_000_070, other)],
    )

    rows = list(export.antigravity_entries(db))
    assert [role for _, role, _ in rows] == ["PERSON", "GEMINI", "RESULT"]
    assert rows[0][2] == "Workspace is the repo. Build task one."
    assert "Done, all tests pass." in rows[1][2] and '[tool call] {"Path":"x.py"}' in rows[1][2]
    assert "[thinking] I should run the tests first" in rows[1][2]  # the reasoning summary the interface showed
    assert rows[2][2].startswith("Command exited with code 0")
    assert "CONTENTS OF A FILE" not in str(rows) and "internal checkpoint" not in str(rows)
    assert rows[0][0].startswith("2026-")  # the step time was decoded from the metadata


def test_the_log_labels_parts_lists_them_up_front_and_redacts_like_the_others(tmp_path):
    folder = "C:" + chr(92) + "Users" + chr(92) + "Asta" + chr(92) + "work"
    user = _field(19, _field(2, f"mail me at someone@example.com from {folder}".encode()))
    db = tmp_path / "abcd1234-conv.db"
    _antigravity_db(db, [(14, 1_792_000_000, user)])
    session = tmp_path / "ffff0000-claude.jsonl"
    _write(session, [{"type": "user", "message": {"content": "hello"}}, {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi there"}]}}])
    log = export.build([session, db], [])
    assert "# PART 1: Claude Code" in log and "# PART 2: Gemini in Antigravity" in log
    contents = log.split("## Contents")[1].split("# PART 1")[0]
    assert "PART 1: Claude Code" in contents and "1 messages from the person, 1 replies" in contents and "PART 2: Gemini" in contents
    assert "someone@example.com" not in log and "Asta" not in log and "[email]" in log


def test_the_pdf_lists_the_parts_with_their_pages_and_has_a_bookmark_for_each(tmp_path):
    pytest.importorskip("fpdf")
    session = tmp_path / "ffff0000-claude.jsonl"
    _write(session, [{"type": "user", "message": {"content": "hello"}}, {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}])
    db = tmp_path / "abcd1234-conv.db"
    _antigravity_db(db, [(14, 1_792_000_000, _field(19, _field(2, b"a prompt")))])
    markdown = export.build([session, db], [])
    pdf = tmp_path / "out.pdf"
    export.write_pdf(markdown, pdf)
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page") >= 3  # the cover, part 1, part 2
    assert b"/Outlines" in data  # the parts are bookmarks
    _, starts = export._render_pdf(markdown, None)
    assert sorted(starts.values()) == [2, 3]  # each part starts on its own page, right after the cover
    _, with_pages = export._render_pdf(markdown, starts)
    assert with_pages == starts  # adding the page numbers to the cover does not move anything


def test_google_access_and_refresh_tokens_are_redacted():
    text = "ya29." + "A" * 40 + " and 1//" + "b" * 45
    out = export.redact(text)
    assert "A" * 40 not in out and "b" * 45 not in out and out.count("[token]") == 2


def test_the_database_is_opened_read_only(tmp_path):
    user = _field(19, _field(2, b"hello there"))
    db = tmp_path / "ro.db"
    _antigravity_db(db, [(14, 1_792_000_000, user)])
    before = db.read_bytes()
    list(export.antigravity_entries(db))
    assert db.read_bytes() == before


def test_the_account_name_is_removed_as_a_whole_word_only():
    out = export.redact("Get-ChildItem pytest-of-Asta and /home/asta/x but fastapi and Fastapi stay", names=["Asta"])
    assert "Asta" not in out and "asta/" not in out
    assert "fastapi" in out and "Fastapi" in out and out.count("[user]") == 2
