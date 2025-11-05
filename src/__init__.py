"""
Video Transcriptor - A Python tool for extracting text transcriptions from video files.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .transcriptor import VideoTranscriptor
from .audio_extractor import AudioExtractor

__all__ = ["VideoTranscriptor", "AudioExtractor"]
