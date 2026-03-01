# stt - fast speech-to-text CLI

> [!NOTE]
> Also: Optimised for agentic usage with Claude Code, Codex, Open Code and more!

# Supported languages
English, and:
Bulgarian, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian

# Speed

| Benchmark | Audio Total Duration | Processing Time | Speed Ratio |
|-----------|----------------|-----------------|-------------|
| First run (model load) | any | longer, ~200MB to download | - |
| Single file (long) | 10m 39s | hardware-dependent | - |
| Single file (short) | 25s | hardware-dependent | - |
| Batch 100x (short) | 41m 40s | hardware-dependent | - |

**Notes:**
- Performance varies by hardware. GPU acceleration available with CUDA.

## Requirements

- Windows or Linux
- Python 3.10+
- ffmpeg (`winget install ffmpeg` on Windows, `apt install ffmpeg` on Linux)
- uv ([see official docs](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

**CPU only:**
```bash
uv pip install --system git+https://github.com/plipowczan/stt
```

**With CUDA GPU acceleration:**
```bash
uv pip install --system "stt-cli[cuda] @ git+https://github.com/plipowczan/stt"
```

## Output Formats

`txt` (default), `srt`, `vtt`, `json` (with timestamps)

## CLAUDE.md / AGENTS.md

Example plug-and-play .md paragraph so your favorite agent can use this amazing tool:

```markdown
## stt - Speech to Text

Transcribe audio/video files to text. Supports wav, mp3, m4a, mp4, etc.

stt audio.mp3                    # Creates audio.txt
stt video.mp4 -f srt             # Creates video.srt (subtitles)
stt recording.m4a -f json        # Creates recording.json (with timestamps)
stt *.wav -o transcripts/        # Batch process with wildcard to directory
stt file1.mp3 file2.mp3          # Batch process with multiple files
stt podcast.mp3 -o out.txt       # Custom output path
stt interview.m4a -f vtt -q      # Quiet mode, only prints output path (CLI usage)
stt -h                           # for help / all options
```