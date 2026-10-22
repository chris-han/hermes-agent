"""Optional transcription provider; registration never selects the provider."""
from .provider import SenseVoiceProvider


def register(ctx):
    ctx.register_transcription_provider(SenseVoiceProvider())
