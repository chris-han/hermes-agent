"""Pinned, opt-in SenseVoice adapter for Hermes' existing transcription ABC."""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from agent.transcription_provider import TranscriptionProvider
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
ASSETS = {
    'model.int8.onnx': (239233841, 'c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51'),
    'tokens.txt': (315894, 'f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc'),
}


def _is_cpu_distribution():
    from pm.features import runtime_distribution_extras
    return runtime_distribution_extras() is not None


class SenseVoiceProvider(TranscriptionProvider):
    name = 'sensevoice'
    display_name = 'SenseVoice Small INT8 (CPU)'

    def __init__(self):
        self._lock = threading.RLock()
        self._key = None
        self._model = None

    def list_models(self):
        return [{'id': 'sensevoice-small-int8', 'display': self.display_name}]

    def _asset_dir(self):
        home = get_hermes_home().resolve()
        path = (home / 'models' / 'sensevoice').resolve()
        if not path.is_relative_to(home):
            raise ValueError('SenseVoice assets must remain under the active Hermes home')
        # A present user override is authoritative, even if incomplete/corrupt.
        # Only this product's launcher-provisioned shared platform root is a fallback.
        if path.exists() or not _is_cpu_distribution():
            return path
        shared = os.environ.get('SEMANTIER_LOCAL_STATE_DIR')
        if not shared:
            raise ValueError('SenseVoice packaged platform assets are not bootstrapped')
        platform = Path(shared).resolve()
        path = (platform / 'models' / 'sensevoice').resolve()
        if not path.is_relative_to(platform):
            raise ValueError('SenseVoice shared assets escape the governed platform root')
        return path

    def is_available(self):
        try:
            return (all(importlib.util.find_spec(m) is not None for m in ['sherpa_onnx','numpy','soundfile'])
                    and all((self._asset_dir()/name).is_file() for name in ASSETS))
        except (ImportError, ValueError, OSError):
            return False

    def _recognizer(self):
        path = self._asset_dir()
        threads = 4 if _is_cpu_distribution() else 1
        key = (str(path), threads, 'auto', False, tuple((name, (path/name).stat().st_size,
                                (path/name).stat().st_mtime_ns) for name in ASSETS))
        if self._key == key and self._model is not None:
            return self._model
        # Verify before importing/loading the optional runtime. No live download fallback.
        for name, (size, expected) in ASSETS.items():
            p = path/name
            if not p.resolve().is_relative_to(path):
                raise ValueError('SenseVoice asset escapes governed model directory')
            with p.open('rb') as source:
                matches = p.stat().st_size == size and hashlib.file_digest(source, 'sha256').hexdigest() == expected
            if not matches:
                raise ValueError(f'SenseVoice pinned asset integrity mismatch: {name}')
        import sherpa_onnx
        if sherpa_onnx.__version__ != '1.13.8':
            raise ValueError('SenseVoice requires sherpa-onnx 1.13.8')
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(path/'model.int8.onnx'), tokens=str(path/'tokens.txt'),
            num_threads=threads, provider='cpu', language='auto', use_itn=False)
        self._model, self._key = rec, key
        return rec

    @staticmethod
    def _read_audio(file_path):
        import soundfile as sf
        try:
            samples, rate = sf.read(file_path, dtype='float32')
            if rate == 16000 and samples.ndim == 1:
                return samples
        except (RuntimeError, ValueError):
            pass
        from gateway.execution_boundary import execution_scratch_dir
        from tools.transcription_audio import _find_ffmpeg_binary
        from hermes_cli._subprocess_compat import windows_hide_flags
        ffmpeg = _find_ffmpeg_binary()
        if not ffmpeg:
            raise ValueError('SenseVoice needs ffmpeg for non-16kHz-mono audio')
        with tempfile.TemporaryDirectory(prefix='sensevoice-', dir=execution_scratch_dir()) as scratch:
            target = str(Path(scratch)/'audio.wav')
            subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-i',str(file_path),
                            '-ar','16000','-ac','1','-c:a','pcm_f32le',target],
                           check=True, capture_output=True, timeout=60, creationflags=windows_hide_flags())
            samples, rate = sf.read(target, dtype='float32')
            if rate != 16000 or samples.ndim != 1:
                raise ValueError('SenseVoice audio conversion did not produce mono16kHz')
            return samples

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        started = time.perf_counter()
        try:
            if model not in (None, '', 'sensevoice-small-int8'):
                raise ValueError('SenseVoice provider supports only the pinned small INT8 export')
            # The frozen provider always uses auto language and no text correction/hotwords.
            with self._lock:
                t = time.perf_counter()
                rec = self._recognizer()
                loaded = time.perf_counter()
                samples = self._read_audio(file_path)
                audio_ready = time.perf_counter()
                stream = rec.create_stream()
                stream.accept_waveform(16000, samples)
                rec.decode_stream(stream)
                text = stream.result.text
                completed = time.perf_counter()
                logger.debug('SenseVoice recognition completed', extra={'sensevoice_timing': {
                    'load_seconds': loaded-t, 'audio_prepare_seconds': audio_ready-loaded,
                    'inference_seconds': completed-audio_ready, 'total_seconds': completed-started,
                    'samples': len(samples), 'success': True}})
                return {'success': True, 'transcript': text, 'provider': self.name}
        except Exception as exc:
            logger.debug('SenseVoice recognition failed', extra={'sensevoice_timing': {
                'total_seconds': time.perf_counter()-started, 'success': False}})
            return {'success': False, 'transcript': '', 'provider': self.name, 'error': str(exc)}
