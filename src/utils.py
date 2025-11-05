"""
Utility functions for the video transcriptor project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import timedelta


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    millis = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_output_path(video_path: str, output_dir: str, extension: str) -> Path:
    """
    Generate output path for transcription file.
    
    Args:
        video_path: Input video file path
        output_dir: Output directory
        extension: Output file extension (without dot)
        
    Returns:
        Output file path
    """
    video_name = Path(video_path).stem
    output_path = Path(output_dir) / f"{video_name}.{extension}"
    return output_path


def save_as_txt(segments: List[Dict[str, Any]], output_path: str, 
                include_timestamps: bool = True) -> None:
    """
    Save transcription as plain text file.
    
    Args:
        segments: List of transcription segments
        output_path: Output file path
        include_timestamps: Whether to include timestamps
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            if include_timestamps:
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                f.write(f"[{start_time} -> {end_time}]\n")
            f.write(f"{segment['text'].strip()}\n\n")


def save_as_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save transcription as SRT subtitle file.
    
    Args:
        segments: List of transcription segments
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def save_as_json(segments: List[Dict[str, Any]], output_path: str, 
                 metadata: Dict[str, Any] = None) -> None:
    """
    Save transcription as JSON file.
    
    Args:
        segments: List of transcription segments
        output_path: Output file path
        metadata: Optional metadata to include
    """
    data = {
        'segments': segments,
        'metadata': metadata or {}
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    try:
        from moviepy import VideoFileClip
        with VideoFileClip(video_path) as video:
            return video.duration
    except Exception as e:
        raise RuntimeError(f"Failed to get video duration: {e}")


def validate_video_file(video_path: str) -> bool:
    """
    Check if the video file exists and has a valid extension.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, raises exception otherwise
    """
    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.isfile(video_path):
        raise ValueError(f"Path is not a file: {video_path}")
    
    extension = Path(video_path).suffix.lower()
    if extension not in supported_formats:
        raise ValueError(
            f"Unsupported video format: {extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    return True
