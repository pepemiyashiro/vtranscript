# Quick Start Guide

## Installation

```bash
# Run the automated installer
./install.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Activate Virtual Environment (every time you open a new terminal)
```bash
source venv/bin/activate
```

### Basic Commands

```bash
# Transcribe a single video
vtranscribe transcribe video.mp4

# Transcribe with SRT subtitles
vtranscribe transcribe video.mp4 --format srt

# Batch process multiple videos
vtranscribe batch *.mp4 --format all

# Translate to English
vtranscribe transcribe foreign_video.mp4 --task translate

# Use better model for higher accuracy
vtranscribe transcribe video.mp4 --model medium

# List available models
vtranscribe models

# Check system info (GPU, CUDA, etc.)
vtranscribe info
```

### Common Options

- `--format`: Output format (`txt`, `srt`, `json`, `all`)
- `--model`: Whisper model (`tiny`, `base`, `small`, `medium`, `large`)
- `--language`: Language code (e.g., `en`, `es`, `fr`) or auto-detect
- `--output-dir`: Where to save transcriptions (default: `./transcriptions`)
- `--task`: `transcribe` or `translate` (translate to English)
- `--no-gpu`: Disable GPU acceleration
- `--no-timestamps`: Exclude timestamps from text output

### Test Installation

```bash
python3 test_installation.py
```

## Examples

```bash
# Spanish video with subtitles
vtranscribe transcribe video_es.mp4 --language es --format srt

# High-quality transcription
vtranscribe transcribe interview.mp4 --model large --format all

# Quick transcription of multiple files
vtranscribe batch lecture1.mp4 lecture2.mp4 lecture3.mp4 --model tiny

# Translate foreign language to English
vtranscribe transcribe japanese_video.mp4 --task translate --format srt
```

## Troubleshooting

```bash
# If command not found, make sure you:
# 1. Activated the virtual environment
source venv/bin/activate

# 2. Installed the package
pip install -e .

# If still using old method, you can always use:
python -m src.cli transcribe video.mp4
```
