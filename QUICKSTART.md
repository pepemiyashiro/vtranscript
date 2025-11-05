# Quick Start Guide

Get started with Video Transcriptor in minutes!

## Installation

### New Installation

```bash
# Automated installer (recommended)
./install.sh
```

### Upgrading from v0.1.x

```bash
# Upgrade to optimized version (10-50x faster!)
./UPGRADE.sh
```

### Manual Installation

**1. Create Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Install Package:**
```bash
pip install -e .
```

## Activate Virtual Environment

**Important**: Run this every time you open a new terminal:

```bash
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

## Basic Usage

### Simple Transcription

```bash
# Basic transcription
vtranscribe transcribe video.mp4

# With language (1.5-2x faster!)
vtranscribe transcribe video.mp4 --language en

# With SRT subtitles
vtranscribe transcribe video.mp4 --format srt --language en

# All formats
vtranscribe transcribe video.mp4 --format all --language en
```

### Batch Processing

```bash
# Multiple videos
vtranscribe batch video1.mp4 video2.mp4 video3.mp4 --format srt

# All MP4 files in directory
vtranscribe batch *.mp4 --format all --language en
```

### Translation

```bash
# Translate any language to English
vtranscribe transcribe spanish_video.mp4 --task translate --format srt
```

## ⚡ Optimization Options (v0.3.0)

### Maximum Speed (NEW!)

```bash
# Fastest: parallel + language + fast mode
vtranscribe transcribe video.mp4 --parallel --language en --fast

# Alternative: parallel + language + int8 precision
vtranscribe transcribe video.mp4 --parallel --language en --compute-type int8
```

### Maximum Accuracy

```bash
# Best quality: disable VAD + float32 precision
vtranscribe transcribe video.mp4 --no-vad --compute-type float32
```

### Custom Configuration

```bash
# Disable silence skipping (for music videos)
vtranscribe transcribe music_video.mp4 --no-vad

# Force CPU mode
vtranscribe transcribe video.mp4 --no-gpu

# Different model size
vtranscribe transcribe video.mp4 --model medium --language en
```

## Common Options

| Option | Values | Description |
|--------|--------|-------------|
| `--format` | txt, srt, json, all | Output format (default: txt) |
| `--model` | tiny, base, small, medium, large | Whisper model size (default: base) |
| `--language` | en, es, fr, etc. | **Specify for 1.5-2x speedup** |
| `--parallel` | flag | **NEW! 3-4x speedup for long videos** |
| `--workers` | number | Parallel workers (default: CPU cores - 1) |
| `--fast` | flag | Fast mode (beam_size=1, default) |
| `--accurate` | flag | Accurate mode (beam_size=5) |
| `--compute-type` | int8, float16, float32, auto | Speed vs accuracy (default: auto) |
| `--output-dir` | path | Output directory (default: ./transcriptions) |
| `--task` | transcribe, translate | Transcribe or translate to English |
| `--no-vad` | flag | Disable silence skipping |
| `--no-gpu` | flag | Force CPU mode |
| `--no-timestamps` | flag | Exclude timestamps from text output |

## Useful Commands

```bash
# List available models and their specs
vtranscribe models

# Check system capabilities and optimization status
vtranscribe info

# Test installation
python test_installation.py
```

## Performance Comparison

### Before (v0.1.x)
```bash
vtranscribe transcribe 220min_video.mp4
# Time: ~220 minutes
```

### v0.2.0 (Optimized)
```bash
vtranscribe transcribe 220min_video.mp4 --language en
# Time: ~16 minutes ⚡ (13.8x faster!)
```

### v0.3.0 (Parallel Processing)
```bash
vtranscribe transcribe 220min_video.mp4 --parallel --language en
# Time: ~3-4 minutes 🚀 (55-73x faster!)
```

## Real-World Examples

### YouTube Tutorial Video
```bash
vtranscribe transcribe tutorial.mp4 --language en --format srt --model base
# Output: ./transcriptions/tutorial.srt
```

### Podcast Interview
```bash
vtranscribe transcribe podcast.mp4 --language en --format all --model medium
# Output: .txt, .srt, .json files
```

### Foreign Language Film
```bash
vtranscribe transcribe spanish_film.mp4 --language es --format srt
# Output: Spanish subtitles
```

### Translate to English
```bash
vtranscribe transcribe japanese_video.mp4 --task translate --format srt
# Output: English subtitles
```

### Batch Conference Recordings (with Parallel Processing)
```bash
vtranscribe batch session*.mp4 --parallel --language en --format txt --model base
# Process all session videos with parallel speedup
```

### Long Video (220 minutes)
```bash
vtranscribe transcribe long_video.mp4 --parallel --fast --language en
# ~3-4 minutes on 8-core CPU (vs ~16 min without parallel)
```

### Music Video (Disable VAD)
```bash
vtranscribe transcribe music_video.mp4 --no-vad --language en --format srt
# VAD disabled to avoid skipping song parts
```

## Troubleshooting

### Command Not Found

```bash
# Make sure you activated the virtual environment
source venv/bin/activate

# Reinstall package
pip install -e .
```

### Slow Performance

```bash
# 1. Enable parallel processing (3-4x speedup for long videos)
vtranscribe transcribe video.mp4 --parallel --language en

# 2. Check optimization status
vtranscribe info

# Expected output should show:
#   - faster-whisper: X.X.X (optimized)
#   - CUDA/MPS: Available
#   - Silero VAD: Available

# 3. Use int8 for maximum speed
vtranscribe transcribe video.mp4 --parallel --language en --compute-type int8

# 4. Adjust workers if needed
vtranscribe transcribe video.mp4 --parallel --workers 4
```

### GPU Not Detected

```bash
# Check info
vtranscribe info

# Force CPU mode if GPU issues
vtranscribe transcribe video.mp4 --no-gpu
```

### VAD Issues (Skipping Speech)

```bash
# Disable VAD for music or continuous audio
vtranscribe transcribe video.mp4 --no-vad
```

### Out of Memory

```bash
# Use smaller model
vtranscribe transcribe video.mp4 --model tiny

# Or force CPU mode
vtranscribe transcribe video.mp4 --no-gpu --model base
```

## Configuration File

Create `config.yaml` for default settings:

```yaml
whisper:
  model_size: base
  language: en  # Specify for faster processing
  task: transcribe

output:
  default_format: srt
  output_dir: ./transcriptions
  include_timestamps: true
```

Use with:
```bash
vtranscribe transcribe video.mp4 --config config.yaml
```

## Next Steps

- **Full documentation**: See [README.md](README.md)
- **Optimization guide**: See [OPTIMIZATION_UPGRADE.md](OPTIMIZATION_UPGRADE.md)
- **Changelog**: See [CHANGES.md](CHANGES.md)
- **Python API**: See `examples/example_usage.py`

---

**Tip**: Always specify `--language` when you know it for 1.5-2x speedup! 🚀
