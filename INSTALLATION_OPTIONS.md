# Installation Options - No System FFmpeg Required!

## Current Setup (Recommended)

Your project now uses `imageio-ffmpeg` which **bundles FFmpeg as a Python package**. No system installation needed!

```bash
pip install -r requirements.txt
```

That's it! MoviePy will automatically use the bundled FFmpeg from `imageio-ffmpeg`.

## How It Works

1. When you `pip install imageio-ffmpeg`, it downloads platform-specific FFmpeg binaries
2. MoviePy automatically detects and uses these binaries
3. No `brew install ffmpeg` or system PATH configuration needed!

## Alternative: Pure Python Audio Processing (Advanced)

If you want to avoid FFmpeg entirely, you can use **OpenCV + NumPy** for basic video processing:

### Option A: OpenCV with PyAV

```bash
# Install these instead of moviepy
pip install opencv-python-headless
pip install av  # PyAV - pure Python video processing
pip install numpy
```

Then modify `src/audio_extractor.py` to use PyAV instead of MoviePy.

### Option B: Direct Whisper Audio Input

Whisper can directly accept audio from various sources:

```python
import whisper
import numpy as np

# Load audio directly (Whisper handles many formats internally)
model = whisper.load_model("base")
result = model.transcribe("video.mp4")  # Whisper extracts audio internally!
```

## Why imageio-ffmpeg is Best

✅ **No system dependencies** - Works on any Python environment
✅ **Cross-platform** - Automatic binary selection (Windows/Mac/Linux)
✅ **Easy deployment** - Just `pip install`, no IT admin needed
✅ **Always available** - No "FFmpeg not found" errors
✅ **Version controlled** - Same FFmpeg version across all environments

## Verification

After installing, verify it works:

```python
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())  # Shows path to bundled FFmpeg
```

Or test the full pipeline:

```bash
python -m src.cli info  # Check system setup
python -m src.cli transcribe your_video.mp4  # Test transcription
```

## Docker/Cloud Deployment

With `imageio-ffmpeg`, your Docker images become simpler:

```dockerfile
FROM python:3.11.9-slim

# No need for: apt-get install ffmpeg
# Just install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# That's it! FFmpeg is included via imageio-ffmpeg
```

---

**Summary**: Your project is now **100% Python pip-installable** with no system dependencies! 🎉
