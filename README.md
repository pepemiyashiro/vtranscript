# Video Transcriptor

A Python tool for extracting text transcriptions from video files using OpenAI's Whisper speech recognition model.

## ⚡ Performance (v0.2.0)

**10-50x faster transcription** with automatic optimizations:

- **faster-whisper**: 4-10x speedup using CTranslate2 optimization
- **Apple Silicon MPS**: Native GPU acceleration on M1/M2/M3 Macs
- **Voice Activity Detection**: Automatically skips silent sections (20-50% faster)
- **Smart precision**: Auto-selects best compute type (int8/float16/float32)

```bash
# Before: 10 min video = 10 min processing
# After:  10 min video = 1-2 min processing ⚡

vtranscribe transcribe video.mp4 --language en  # Fastest!
```

📖 **See [OPTIMIZATION_UPGRADE.md](OPTIMIZATION_UPGRADE.md) for full optimization guide**

## Features

- **Multiple Video Formats**: Supports MP4, AVI, MOV, MKV, FLV, WMV, WebM
- **High-Quality Transcription**: Uses Whisper models (tiny to large) with faster-whisper optimization
- **Multiple Output Formats**: TXT, SRT (subtitles), JSON
- **Batch Processing**: Transcribe multiple videos at once
- **GPU Acceleration**: Automatic CUDA and Apple Silicon MPS detection
- **Language Detection**: Auto-detect language or specify explicitly for 1.5-2x speedup
- **Translation**: Translate non-English audio to English
- **Voice Activity Detection**: Skip silent sections automatically
- **CLI Interface**: Easy-to-use command-line interface
- **Python API**: Use as a library in your own projects

## Requirements

- Python 3.11+ (tested with 3.11, 3.12, 3.13)
- CUDA-capable GPU (optional, but highly recommended for faster processing)

**Note**: FFmpeg is automatically included via `imageio-ffmpeg` - no system installation required!

## Installation

### New Installation

Run the automated installation script:

```bash
./install.sh
```

This will:
1. Create a virtual environment
2. Install all optimized dependencies (faster-whisper, VAD, etc.)
3. Set up the `vtranscribe` command

### Upgrading from v0.1.x

```bash
./UPGRADE.sh
```

This will upgrade to the optimized version with 10-50x speedup!

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

**3. Install Package (enables simple CLI commands):**
```bash
pip install -e .
```

**Note**: The first time you run the transcriptor, it will download:
- The Whisper model (39MB - 1.5GB depending on model size)
- FFmpeg binaries automatically via `imageio-ffmpeg` (if not already cached)

## Quick Start

### Command Line Usage

After installation, you can use the simple `vtranscribe` command:

**Transcribe a single video (with optimizations):**
```bash
# Fastest (specify language for 1.5-2x speedup)
vtranscribe transcribe video.mp4 --language en

# With subtitles
vtranscribe transcribe video.mp4 --format srt --language en
```

**Maximum speed (int8 precision):**
```bash
vtranscribe transcribe video.mp4 --language en --compute-type int8
```

**Maximum accuracy (disable optimizations):**
```bash
vtranscribe transcribe video.mp4 --no-vad --compute-type float32
```

**Full options:**
```bash
vtranscribe transcribe video.mp4 \
  --output-dir ./my_transcriptions \
  --format all \
  --model base \
  --language en \
  --compute-type auto \
  --no-timestamps
```

**Batch transcription:**
```bash
vtranscribe batch video1.mp4 video2.mp4 video3.mp4 --format srt
```

**Translate to English:**
```bash
vtranscribe transcribe foreign_video.mp4 --task translate
```

**List available models:**
```bash
vtranscribe models
```

**Check system info and optimization status:**
```bash
vtranscribe info
```

Output shows:
- PyTorch version
- CUDA/MPS GPU availability
- faster-whisper status (optimized or fallback)
- VAD availability

### Python API Usage

```python
from src.transcriptor import VideoTranscriptor

# Initialize with optimizations (v0.2.0)
transcriptor = VideoTranscriptor(
    model_size="base",
    language="en",  # Specify for 1.5-2x speedup
    use_gpu=True,
    use_vad=True,  # Skip silent sections
    compute_type="auto"  # Auto-select precision
)

# Transcribe and save
saved_files = transcriptor.transcribe_and_save(
    video_path="video.mp4",
    output_dir="./transcriptions",
    output_format="all"
)

print(f"Transcription saved: {saved_files}")
```

More examples in `examples/example_usage.py`

## Configuration

Edit `config.yaml` to set default options:

```yaml
whisper:
  model_size: base  # tiny, base, small, medium, large
  language: null    # Auto-detect or specify: en, es, fr, etc.
  task: transcribe  # transcribe or translate

output:
  default_format: txt
  output_dir: ./transcriptions
  include_timestamps: true
```

Use config file:
```bash
python -m src.cli transcribe video.mp4 --config config.yaml
```

## Whisper Models

| Model  | Parameters | VRAM Required | Speed      | Accuracy |
|--------|-----------|---------------|------------|----------|
| tiny   | 39M       | ~1GB          | Fastest    | Basic    |
| base   | 74M       | ~1GB          | Fast       | Good     |
| small  | 244M      | ~2GB          | Moderate   | Better   |
| medium | 769M      | ~5GB          | Slow       | Great    |
| large  | 1550M     | ~10GB         | Slowest    | Best     |

**Recommendation**: Start with `base` for a good balance of speed and accuracy.

## Output Formats

### TXT (Plain Text)
```
[00:00:00,000 -> 00:00:03,500]
Welcome to this video tutorial.

[00:00:03,500 -> 00:00:07,200]
Today we'll learn about Python programming.
```

### SRT (Subtitles)
```
1
00:00:00,000 --> 00:00:03,500
Welcome to this video tutorial.

2
00:00:03,500 --> 00:00:07,200
Today we'll learn about Python programming.
```

### JSON
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Welcome to this video tutorial."
    }
  ],
  "metadata": {
    "video_path": "video.mp4",
    "model_size": "base",
    "language": "en"
  }
}
```

## CLI Reference

### `transcribe` - Transcribe a single video

```bash
vtranscribe transcribe VIDEO_PATH [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH`: Output directory (default: ./transcriptions)
- `-f, --format`: Output format: txt, srt, json, all (default: txt)
- `-m, --model`: Model size: tiny, base, small, medium, large (default: base)
- `-l, --language`: Language code (e.g., en, es, fr) - **specify for 1.5-2x speedup**
- `-t, --task`: Task type: transcribe or translate (default: transcribe)
- `--compute-type`: Precision: int8, float16, float32, auto (default: auto)
- `--no-vad`: Disable Voice Activity Detection (silence skipping)
- `--no-timestamps`: Exclude timestamps from text output
- `--no-gpu`: Disable GPU acceleration
- `-c, --config PATH`: Path to configuration file

### `batch` - Transcribe multiple videos

```bash
vtranscribe batch VIDEO1 VIDEO2 VIDEO3 [OPTIONS]
```

Same options as `transcribe` command.

### `models` - List available models

```bash
vtranscribe models
```

### `info` - Display system information

```bash
vtranscribe info
```

## Performance Tips

### ⚡ Optimization Tips (v0.2.0)

1. **Specify language**: Add `--language en` for 1.5-2x speedup
2. **Use faster-whisper**: Installed by default, gives 4-10x speedup automatically
3. **Enable VAD**: On by default, skips silent sections (20-50% faster)
4. **Use GPU**: CUDA or Apple Silicon MPS gives 2-3x additional speedup
5. **Choose compute type**: `--compute-type int8` for max speed, `float32` for max accuracy
6. **Batch processing**: Process multiple videos to reuse the loaded model

### Expected Performance

| Configuration | 10min Video | Speedup vs v0.1.0 |
|--------------|-------------|-------------------|
| CPU + faster-whisper + VAD + language | ~1-2 min | 5-10x |
| GPU + faster-whisper + VAD + language | ~30-60 sec | 10-20x |
| Apple M1/M2/M3 + all optimizations | ~40-80 sec | 8-15x |

**See [OPTIMIZATION_UPGRADE.md](OPTIMIZATION_UPGRADE.md) for detailed benchmarks and tuning guide.**

## Troubleshooting

**Import errors during installation:**
- Make sure you're using Python 3.11 or later: `python --version`
- Activate your virtual environment
- Update pip: `pip install --upgrade pip`

**FFmpeg/video processing errors:**
- The `imageio-ffmpeg` package provides FFmpeg automatically
- If issues persist, run: `pip install --upgrade imageio-ffmpeg moviepy`
- No system FFmpeg installation needed!

**CUDA/GPU not detected:**
- Check CUDA installation: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
- Install PyTorch with CUDA support: Visit [pytorch.org](https://pytorch.org)

**Out of memory error:**
- Use a smaller model (try `tiny` or `base`)
- Process shorter videos
- Use CPU instead of GPU with `--no-gpu`

**Transcription quality issues:**
- Use a larger model (`medium` or `large`)
- Specify the language with `--language` if known
- Ensure video has clear audio

## Project Structure

```
video-transcriptor/
├── .python-version          # Python version specification
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── config.yaml            # Configuration file
├── README.md              # This file
├── src/
│   ├── __init__.py        # Package initialization
│   ├── transcriptor.py    # Main transcription logic
│   ├── audio_extractor.py # Audio extraction from video
│   ├── utils.py           # Utility functions
│   └── cli.py             # Command-line interface
└── examples/
    └── example_usage.py   # Example usage scripts
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project uses OpenAI Whisper, which is released under the MIT License.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [MoviePy](https://github.com/Zulko/moviepy) - Video editing library
- [FFmpeg](https://ffmpeg.org/) - Multimedia framework

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [examples](examples/example_usage.py)
3. Open an issue on GitHub

---

**Happy Transcribing!** 🎥 ➡️ 📝
