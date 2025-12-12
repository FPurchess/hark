# hark 😇

[![PyPI version](https://img.shields.io/pypi/v/hark-cli)](https://pypi.org/project/hark-cli/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 100% offline, Whisper-powered voice notes from your terminal

### Use Cases

- **Voice-to-LLM pipelines** — `hark | llm` turns speech into AI prompts instantly
- **Meeting minutes** — Transcribe calls with speaker identification (`--diarize`)
- **System audio capture** — Record what you hear, not just what you say (`--input speaker`)
- **Private by design** — No cloud, no API keys, no data leaves your machine

## Features

- 🎙️ **Record** - Press space to start, Ctrl+C to stop
- 🔊 **Multi-source** - Capture microphone, system audio, or both
- ✨ **Transcribe** - Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- 🗣️ **Diarize** - Identify who said what with [WhisperX](https://github.com/m-bain/whisperX)
- 🔒 **Local** - 100% offline, no cloud required
- 📄 **Flexible** - Output as plain text, markdown, or SRT subtitles

## Installation

```bash
pipx install hark-cli
```

### System Dependencies

**Ubuntu/Debian:**

```bash
sudo apt install portaudio19-dev
```

**macOS:**

```bash
brew install portaudio
```

**Windows:**

No system dependencies required. Audio libraries are bundled with the Python packages.

## Quick Start

```bash
# Record and print to stdout
hark

# Save to file
hark notes.txt

# Use larger model for better accuracy
hark --model large-v3 meeting.md

# Transcribe in German
hark --lang de notes.txt

# Output as SRT subtitles
hark --format srt captions.srt

# Capture system audio (e.g., online meetings)
hark --input speaker meeting.txt

# Capture both microphone and system audio (stereo: L=mic, R=speaker)
hark --input both conversation.txt
```

## Configuration

Hark uses a YAML config file at `~/.config/hark/config.yaml`. CLI flags override config file settings.

```yaml
# ~/.config/hark/config.yaml
recording:
  sample_rate: 16000
  channels: 1 # Use 2 for --input both
  max_duration: 600
  input_source: mic # mic, speaker, or both

whisper:
  model: base # tiny, base, small, medium, large, large-v2, large-v3
  language: auto # auto, en, de, fr, es, ...
  device: auto # auto, cpu, cuda

preprocessing:
  noise_reduction:
    enabled: true
    strength: 0.5 # 0.0-1.0
  normalization:
    enabled: true
  silence_trimming:
    enabled: true

output:
  format: plain # plain, markdown, srt
  timestamps: false

diarization:
  hf_token: null # HuggingFace token (required for --diarize)
  local_speaker_name: null # Your name in stereo mode, or null for SPEAKER_00
```

## Audio Input Sources

Hark supports three input modes via `--input` or `recording.input_source`:

| Mode      | Description                                            |
| --------- | ------------------------------------------------------ |
| `mic`     | Microphone only (default)                              |
| `speaker` | System audio only (loopback capture)                   |
| `both`    | Microphone + system audio as stereo (L=mic, R=speaker) |

### System Audio Capture

System audio capture (`--input speaker` or `--input both`) works differently on each platform:

**Linux (PulseAudio/PipeWire):**

Uses monitor sources automatically. To verify your system supports it:

```bash
pactl list sources | grep -i monitor
```

You should see output like:

```
Name: alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
Description: Monitor of Built-in Audio
```

**macOS:**

Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual audio driver:

1. Install BlackHole:
   ```bash
   brew install blackhole-2ch
   ```

2. Open **Audio MIDI Setup** (in Applications → Utilities)

3. Click **+** → **Create Multi-Output Device**

4. Check both your speakers/headphones AND BlackHole 2ch

5. Set the Multi-Output Device as your default output in System Preferences → Sound

Now hark can capture system audio through BlackHole.

**Windows 10/11:**

Uses WASAPI loopback automatically. No setup required—just ensure your audio output device is working.

## Speaker Diarization

Identify who said what in multi-speaker recordings using [WhisperX](https://github.com/m-bain/whisperX).

### Setup

1. Install diarization dependencies:

   ```bash
   pipx inject hark-cli whisperx
   # Or with pip:
   pip install hark-cli[diarization]
   ```

2. Get a HuggingFace token (required for pyannote models):

   - Create account at https://huggingface.co
   - Accept model licenses:
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/speaker-diarization-3.1
   - Create token at https://huggingface.co/settings/tokens

3. Add token to config:
   ```yaml
   # ~/.config/hark/config.yaml
   diarization:
     hf_token: "hf_xxxxxxxxxxxxx"
   ```

### Usage

The `--diarize` flag enables speaker identification. It requires `--input speaker` or `--input both`.

```bash
# Transcribe a meeting with speaker identification
hark --diarize --input speaker meeting.txt

# Specify expected number of speakers (improves accuracy)
hark --diarize --speakers 3 --input speaker meeting.md

# Skip interactive speaker naming for batch processing
hark --diarize --no-interactive --input speaker meeting.txt

# Stereo mode: separate local user from remote speakers
hark --diarize --input both conversation.md

# Combine with other options
hark --diarize --input speaker --format markdown --model large-v3 meeting.md
```

| Flag               | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `--diarize`        | Enable speaker identification                         |
| `--speakers N`     | Hint for expected speaker count (improves clustering) |
| `--no-interactive` | Skip post-transcription speaker naming prompt         |

**Note:** Diarization adds processing time. For a 5-minute recording, expect ~1-2 minutes on GPU or ~5-10 minutes on CPU.

### Output Format

With diarization enabled, output includes speaker labels and timestamps:

**Plain text:**

```
[00:02] [SPEAKER_01] Hello everyone, let's get started.
[00:05] [SPEAKER_02] Thanks for joining. Let me share my screen.
```

**Markdown:**

```markdown
# Meeting Transcript

**SPEAKER_01** (00:02)
Hello everyone, let's get started.

**SPEAKER_02** (00:05)
Thanks for joining. Let me share my screen.

---

_2 speakers detected • Duration: 5:23 • Language: en (98% confidence)_
```

### Interactive Naming

After transcription, hark will prompt you to identify speakers:

```
Detected 2 speaker(s) to identify.

SPEAKER_01 said: "Hello everyone, let's get started."
Who is this? [name/skip/done]: Alice

SPEAKER_02 said: "Thanks for joining. Let me share my screen."
Who is this? [name/skip/done]: Bob
```

Use `--no-interactive` to skip this prompt.

### Known Issues

**Slow diarization?** The pyannote models may default to CPU inference. For GPU acceleration:

```bash
pip install --force-reinstall onnxruntime-gpu
```

See [WhisperX #499](https://github.com/m-bain/whisperX/issues/499) for details.

## Development

```bash
git clone https://github.com/FPurchess/hark.git
cd hark
uv sync --extra test
uv run pre-commit install
uv run pytest
```

## License

[AGPLv3](LICENSE)
