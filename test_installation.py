#!/usr/bin/env python3
"""
Test script to verify FFmpeg-free installation works.
Run this after: pip install -r requirements.txt
"""

import sys


def check_imageio_ffmpeg():
    """Check if imageio-ffmpeg is installed and working."""
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"✓ imageio-ffmpeg installed")
        print(f"  FFmpeg binary: {ffmpeg_path}")
        return True
    except ImportError:
        print("✗ imageio-ffmpeg not installed")
        print("  Run: pip install imageio-ffmpeg")
        return False
    except Exception as e:
        print(f"✗ Error with imageio-ffmpeg: {e}")
        return False


def check_moviepy():
    """Check if MoviePy can use the bundled FFmpeg."""
    try:
        import moviepy
        from moviepy import VideoFileClip
        print(f"✓ MoviePy installed (version {moviepy.__version__})")
        print("  MoviePy will auto-detect FFmpeg from imageio-ffmpeg")
        
        return True
    except ImportError:
        print("✗ MoviePy not installed")
        print("  Run: pip install moviepy")
        return False
    except Exception as e:
        print(f"✗ Error with MoviePy: {e}")
        return False


def check_whisper():
    """Check if Whisper is installed."""
    try:
        import whisper
        print(f"✓ OpenAI Whisper installed")
        return True
    except ImportError:
        print("✗ Whisper not installed")
        print("  Run: pip install openai-whisper")
        return False


def check_torch():
    """Check if PyTorch is installed with CUDA info."""
    try:
        import torch
        print(f"✓ PyTorch installed (version {torch.__version__})")
        
        if torch.cuda.is_available():
            print(f"  CUDA available: GPU ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  CUDA available: No (CPU only)")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Run: pip install torch torchaudio")
        return False


def main():
    print("=" * 60)
    print("Video Transcriptor - Installation Check")
    print("=" * 60)
    print("\nChecking dependencies...\n")
    
    checks = [
        check_imageio_ffmpeg(),
        check_moviepy(),
        check_whisper(),
        check_torch(),
    ]
    
    print("\n" + "=" * 60)
    
    if all(checks):
        print("✓ All dependencies installed correctly!")
        print("\nYou can now run:")
        print("  vtranscribe transcribe video.mp4")
        print("\nNo system FFmpeg installation required! 🎉")
        return 0
    else:
        print("✗ Some dependencies are missing")
        print("\nInstall all requirements:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
