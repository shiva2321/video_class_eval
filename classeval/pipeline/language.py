"""Which transcripts the text-based checks can be trusted on."""

from classeval.config import Settings
from classeval.schema import Transcript


def text_checks_supported(tr: Transcript, settings: Settings) -> bool:
    """Word lists, model judgements and quotes are validated for the languages in settings.languages.full_support
    (English). For anything else those checks are reported as not assessed instead of being guessed; everything that
    does not depend on language (talk time, participation, pace, screen use, audio) is still measured."""
    return tr.language_confidence >= 0.5 and tr.language in settings.languages.full_support
