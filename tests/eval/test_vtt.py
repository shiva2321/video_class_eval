import pytest

from classeval.eval.vtt import parse_vtt, wer


def test_vtt_parse_and_wer(tmp_path):
    # Test wer
    assert wer("a b c", "a x c") == pytest.approx(1.0 / 3.0)
    assert wer("hello world", "hello world") == 0.0
    assert wer("hello world", "") == 1.0
    assert wer("", "") == 0.0

    # Test parse_vtt
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.500
<v Teacher>Hello class, welcome!

00:00:04.000 --> 00:00:06.200
<v Student>Hi sir.
"""
    vtt_file = tmp_path / "sample.vtt"
    vtt_file.write_text(vtt_content, encoding="utf-8")

    cues = parse_vtt(vtt_file)
    assert len(cues) == 2
    assert cues[0][0] == 1.0
    assert cues[0][1] == 3.5
    assert cues[0][2] == "Teacher"
    assert cues[0][3] == "Hello class, welcome!"

    assert cues[1][0] == 4.0
    assert cues[1][1] == 6.2
    assert cues[1][2] == "Student"
    assert cues[1][3] == "Hi sir."
