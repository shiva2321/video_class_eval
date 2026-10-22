import pytest

from classeval.pipeline import class_stats
from classeval.schema import (
    AudioFacts,
    Diarization,
    FrameState,
    MediaInfo,
    Segment,
    SpeakerTurn,
    Transcript,
    TranscriptFacts,
    VideoFacts,
)


def _inputs():
    segs = [
        Segment(start=0, end=5, text="Hello everyone. Why does the knight move like that?", speaker="T"),
        Segment(start=6, end=8, text="Because it jumps over pieces", speaker="S1"),
        Segment(start=9, end=12, text="Great answer Gautham, what is the next move?", speaker="T"),
        Segment(start=13, end=15, text="Is it a fork? I think so", speaker="S2"),
        Segment(start=16, end=30, text="Let us practice. Correct, that is a fork.", speaker="T"),
    ]
    tr = Transcript(language="en", language_confidence=0.99, asr_model="t", segments=segs)
    d = Diarization(
        turns=[
            SpeakerTurn(start=0, end=5, speaker="T"),
            SpeakerTurn(start=6, end=8, speaker="S1"),
            SpeakerTurn(start=9, end=12, speaker="T"),
            SpeakerTurn(start=13, end=15, speaker="S2"),
            SpeakerTurn(start=16, end=30, speaker="T"),
        ],
        tutor="T",
        tutor_confidence=1.0,
        method="ecapa",
        n_speakers=3,
    )
    audio = AudioFacts(
        duration_s=60,
        speech_ratio=0.4,
        tutor_talk_share=0.85,
        student_words_per_min=8.0,
        dead_air_per_min=0.0,
        median_response_latency_s=2.0,
        tutor_speech_rate_wpm=140.0,
        loudness_lufs=-20,
        clipping_ratio=0,
        snr_db_est=30,
        pitch_std_semitones_tutor=3.0,
        energy_cv_tutor=0.6,
    )
    video = VideoFacts(
        states=[
            FrameState(t=10, state="BOARD_SHARE", confidence=0.9),
            FrameState(t=20, state="BOARD_SHARE", confidence=0.9),
            FrameState(t=30, state="BRANDED_TALKING_HEAD", confidence=0.9),
            FrameState(t=70, state="LESSON_VIDEO_SHARE", confidence=0.9),
        ],
        tutor_name=None,
        branded_ratio_tutor_speaking=0.5,
        board_share_ratio=0.5,
        lesson_video_ratio=0.25,
        camera_off_total_ratio=0.0,
        camera_off_while_tutor_speaks_s=0.0,
        first5_branded=None,
        last5_branded=None,
        talking_head_luma_mean=100.0,
        n_frames=4,
    )
    tf = TranscriptFacts(detections={}, distinct_responders=2, question_reply_rate=0.5, question_count=2)
    media = MediaInfo(duration_s=60.0, has_audio=True, has_video=True, container="mp4", size_bytes=1)
    return tr, d, audio, video, tf, media


def test_counts_are_exact():
    stats = class_stats.compute(*_inputs())
    assert stats.duration_min == 1.0
    assert (stats.tutor_talk_pct, stats.student_talk_pct, stats.silence_pct) == (36.7, 6.7, 56.7)
    assert stats.tutor_questions == 2 and stats.open_questions == 1
    assert stats.student_questions == 1 and stats.student_utterances == 2
    assert stats.praise_count == 2  # "Great" and "Correct"
    assert stats.students_addressed_by_name == 1 and stats.name_address_count == 1
    assert stats.question_reply_pct == 50.0 and stats.median_wait_s == 2.0
    assert stats.longest_tutor_monologue_s == 14.0
    assert (stats.first_tutor_speech_s, stats.last_tutor_speech_s) == (0.0, 30.0)
    assert stats.tutor_wpm == 140.0


def test_timeline_splits_speech_and_screen_by_minute():
    stats = class_stats.compute(*_inputs())
    assert len(stats.timeline) == 1
    assert (stats.timeline[0].tutor_s, stats.timeline[0].student_s) == (22.0, 4.0)
    assert stats.timeline[0].screen == "board"
    assert stats.screen_pct == {"board": 50.0, "face_on_screen": 25.0, "lesson_video": 25.0}


def test_timeline_has_one_bin_per_started_minute_and_attributes_speech_across_minutes():
    tr, d, audio, video, tf, media = _inputs()
    d = d.model_copy(update={"turns": [SpeakerTurn(start=50, end=130, speaker="T")]})
    stats = class_stats.compute(tr, d, audio, video, tf, media.model_copy(update={"duration_s": 150.0}))
    assert [round(b.tutor_s) for b in stats.timeline] == [10, 60, 10]


def test_a_voice_counts_as_a_student_once_it_has_said_enough_words():
    tr, d, audio, video, tf, media = _inputs()
    chatty = Segment(start=40, end=50, text=" ".join(["word"] * class_stats.MIN_STUDENT_VOICE_WORDS), speaker="S3")
    tr = tr.model_copy(update={"segments": [*tr.segments, chatty]})
    assert class_stats.compute(tr, d, audio, video, tf, media).student_voices == 1


def test_no_tutor_and_no_frames_do_not_crash():
    tr, d, audio, video, tf, media = _inputs()
    d = d.model_copy(update={"tutor": None, "turns": []})
    video = video.model_copy(update={"states": [], "n_frames": 0})
    stats = class_stats.compute(tr, d, audio, video, tf, media)
    assert stats.tutor_questions == 0 and stats.screen_pct == {}
    assert stats.first_tutor_speech_s is None and stats.timeline[0].screen == "NONE"


def test_names_are_counted_but_never_stored():
    stats = class_stats.compute(*_inputs())
    assert "gautham" not in stats.model_dump_json().lower()


@pytest.mark.parametrize(
    "state,group",
    [("BOARD_SHARE", "board"), ("OTHER_TALKING_HEAD", "face_on_screen"), ("CAMERA_OFF_CARD", "camera_off"), ("NONE", "NONE")],
)
def test_screen_groups(state, group):
    assert class_stats.screen_group(state) == group
