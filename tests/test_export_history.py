import json

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


def test_the_log_keeps_prompts_and_replies_and_drops_reasoning_reminders_and_tool_output(tmp_path):
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
                        {"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "I will start with the plan."},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q", "description": "Run the tests"}},
                    ]
                },
            },
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": "OUTPUT 123 numbers"}]}},
            {"type": "user", "isMeta": True, "message": {"content": "meta"}},
        ],
    )
    log = export.build([session], [])
    assert "Please build the app." in log and "I will start with the plan." in log and "[tool: Bash] Run the tests" in log
    for dropped in ("hidden rules", "private reasoning", "OUTPUT 123", "meta"):
        assert dropped not in log
    assert log.count("**PERSON**") == 1 and log.count("**ASSISTANT**") == 1


def test_very_long_replies_are_cut_with_a_note(tmp_path):
    session = tmp_path / "long.jsonl"
    _write(session, [{"type": "assistant", "message": {"content": [{"type": "text", "text": "x" * 9000}]}}])
    log = export.build([session], [])
    assert "more characters not included" in log and len(log) < 8000


def test_nothing_is_written_when_a_secret_survives(tmp_path, monkeypatch):
    session = tmp_path / "s.jsonl"
    _write(session, [{"type": "user", "message": {"content": "my value is abc"}}])
    monkeypatch.setattr(export, "env_values", lambda _path: ["abc"])
    monkeypatch.setattr(export, "redact", lambda text, secrets=None: text)  # a broken redaction must be caught by the final check
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
