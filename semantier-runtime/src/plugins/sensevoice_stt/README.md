# Optional SenseVoice Small INT8

This package registers `stt.provider: sensevoice`; it never changes the default
provider. Enable `sensevoice_stt` through the existing plugin/config surface and
install the pinned optional requirements into the runtime used by Hermes.

Provision the exact model and tokens under the active profile's
`models/sensevoice/`. Source: csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17,
revision `2365baeacb507f821a0c8120fcee3d484dba7a07`. Sizes and SHA-256 pins are
executable in provider.py; missing or altered assets fail clearly without downloads
or provider fallback. Native qualification and limitations are in
`docs/research/logs/sensevoice-windows-qualification-v1/` in semantier-runtime.

CPU, one inference thread, automatic language, ITN disabled. Generic language and
prompt hints do not modify this fixed provider. No hotword/text correction arm.
Existing UI dictation stop/cancel behavior remains unchanged; running native decode
is not forcibly interrupted. The per-provider cache retains one profile's model at
a time, replaces it on home/asset identity changes and serializes decoding.
Non-16kHz-mono input requires the existing ffmpeg executable and governed scratch
boundary; direct 16kHz mono float WAV input bypasses conversion. Standard Hermes
upload validation remains in force. Default/local Whisper remains selectable.
