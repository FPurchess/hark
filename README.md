# mrec-cli

> Whisper-powered voice notes from your terminal

<!-- TODO: Add demo GIF -->

## Features

- **Record** - Press space to start, space to stop
- **Transcribe** - Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Local** - 100% offline, no cloud required
- **Flexible** - Output as plain text, markdown, or SRT subtitles

## Installation

```bash
pipx install mrec-cli
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

## Quick Start

```bash
# Record and print to stdout
mrec-cli

# Save to file
mrec-cli notes.txt

# Use larger model for better accuracy
mrec-cli --model large-v3 meeting.md

# Output as SRT subtitles
mrec-cli --format srt captions.srt
```

## Development

```bash
git clone https://github.com/your-username/mrec-cli.git
cd mrec-cli
uv sync --extra test
uv run pre-commit install
uv run pytest
```

## License

[AGPLv3](LICENSE)
