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

## Python API Usage

### Example 1: Simple Transcription

```python
from src.transcriptor import VideoTranscriptor

# Initialize transcriptor
transcriptor = VideoTranscriptor(
    model_size="base",
    language="en",
    use_gpu=True,
    verbose=True
)

# Transcribe and save
saved_files = transcriptor.transcribe_and_save(
    video_path="video.mp4",
    output_dir="./transcriptions",
    output_format="all",  # txt, srt, json
    include_timestamps=True
)

print(f"Saved: {saved_files}")
```

### Example 2: Parallel Processing

```python
from src.transcriptor import VideoTranscriptor

# Initialize with parallel processing enabled
transcriptor = VideoTranscriptor(
    model_size="base",
    language="en",
    use_gpu=True,
    use_parallel=True,  # Enable parallel processing
    num_workers=None,   # Auto-detect (CPU cores - 1)
    verbose=True
)

# Transcribe (3-4x faster for long videos)
result = transcriptor.transcribe_video("long_video.mp4")

print(f"Language: {result['language']}")
print(f"Segments: {len(result['segments'])}")
print(f"Text: {result['text'][:200]}...")
```

### Example 3: Batch Processing

```python
from src.transcriptor import VideoTranscriptor

transcriptor = VideoTranscriptor(
    model_size="base",
    language="en",
    use_gpu=True
)

# Process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = transcriptor.transcribe_batch(
    video_paths=video_paths,
    output_dir="./transcriptions",
    output_format="srt"
)
```

### Example 4: Translation

```python
from src.transcriptor import VideoTranscriptor

transcriptor = VideoTranscriptor(
    model_size="base",
    use_gpu=True
)

# Translate any language to English
result = transcriptor.transcribe_video(
    video_path="spanish_video.mp4",
    task="translate"  # translate to English
)
```

### Example 5: Custom Settings

```python
from src.transcriptor import VideoTranscriptor

# Maximum speed configuration
transcriptor = VideoTranscriptor(
    model_size="base",
    language="en",
    use_gpu=True,
    use_vad=True,
    compute_type="int8",
    beam_size=1,
    use_parallel=True,
    num_workers=4,
    verbose=True
)

result = transcriptor.transcribe_video("video.mp4")
```

## Performance Specs & Time Estimates

### Processing Time by Video Length

**Base Model + Language Specified + Parallel Processing** (Recommended)

| Video Length | Sequential | Parallel (4 cores) | Parallel (8 cores) | Speedup |
|-------------|-----------|-------------------|-------------------|---------|
| 5 minutes   | ~20 sec   | ~15 sec          | ~12 sec          | 1.7x    |
| 15 minutes  | ~1 min    | ~35 sec          | ~25 sec          | 2.4x    |
| 30 minutes  | ~2 min    | ~50 sec          | ~35 sec          | 3.4x    |
| 60 minutes  | ~4 min    | ~1.5 min         | ~1 min           | 4x      |
| 120 minutes | ~8 min    | ~3 min           | ~2 min           | 4x      |
| 220 minutes | ~16 min   | ~5 min           | ~3.5 min         | 4.6x    |

**Note**: Times are approximate and vary based on CPU/GPU, audio complexity, and system load.

### Model Comparison (60-minute video)

| Model | Size | VRAM | Sequential | Parallel | Accuracy |
|-------|------|------|-----------|----------|----------|
| tiny  | 39M  | ~1GB | ~2 min    | ~45 sec  | ⭐⭐ Low |
| base  | 74M  | ~1GB | ~4 min    | ~1 min   | ⭐⭐⭐ Good |
| small | 244M | ~2GB | ~8 min    | ~2.5 min | ⭐⭐⭐⭐ Better |
| medium| 769M | ~5GB | ~18 min   | ~6 min   | ⭐⭐⭐⭐⭐ High |
| large | 1550M| ~10GB| ~35 min   | ~12 min  | ⭐⭐⭐⭐⭐ Best |

### Optimization Impact (60-minute video, base model)

| Configuration | Time | vs Baseline | Notes |
|--------------|------|-------------|-------|
| No optimizations | ~60 min | 1x | v0.1.x baseline |
| faster-whisper | ~12 min | 5x | v0.2.0 |
| + language specified | ~6 min | 10x | Auto-detection skipped |
| + VAD enabled | ~4 min | 15x | Skips silence |
| + compute_type=int8 | ~3 min | 20x | Fastest precision |
| + parallel (4 cores) | ~1.5 min | 40x | Parallel chunks |
| + parallel (8 cores) | ~1 min | 60x | **Maximum speed** |

### Hardware Performance (60-minute video, base model, parallel)

| Hardware | Time | Relative |
|----------|------|----------|
| CPU only (4 cores) | ~2 min | 1x |
| CPU only (8 cores) | ~1 min | 2x |
| NVIDIA RTX 3060 | ~40 sec | 3x |
| NVIDIA RTX 4090 | ~25 sec | 4.8x |
| Apple M1 (8 cores) | ~1 min | 2x |
| Apple M2 Pro (10 cores) | ~45 sec | 2.7x |

### Compute Type Comparison (60-minute video, base model)

| Compute Type | Speed | Accuracy | Best For |
|-------------|-------|----------|----------|
| int8 | Fastest (~3 min) | Good | Speed priority, clear audio |
| float16 | Fast (~4 min) | Better | Balanced (CUDA GPU default) |
| float32 | Slow (~8 min) | Best | Accuracy priority, research |
| auto | Varies | Varies | Recommended (adapts to hardware) |

## Next Steps

- **Full documentation**: See [README.md](README.md)
- **Optimization guide**: See [OPTIMIZATION_UPGRADE.md](OPTIMIZATION_UPGRADE.md)
- **Changelog**: See [CHANGES.md](CHANGES.md)
- **Python API examples**: See `examples/example_usage.py`

---

**Tip**: Always specify `--language` when you know it for 1.5-2x speedup! 🚀
