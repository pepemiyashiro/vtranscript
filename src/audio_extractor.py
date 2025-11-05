"""
Audio extraction from video files.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from moviepy import VideoFileClip
from rich.console import Console

console = Console()


class AudioExtractor:
    """Extract audio from video files."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the AudioExtractor.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000 for Whisper)
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
    
    def extract(self, video_path: str, output_path: Optional[str] = None, 
                verbose: bool = True) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save audio file (optional, creates temp file if not provided)
            verbose: Whether to print progress messages
            
        Returns:
            Path to extracted audio file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if verbose:
            console.print(f"[cyan]Extracting audio from:[/cyan] {video_path}")
        
        try:
            # Create temporary file if output path not provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                video_name = Path(video_path).stem
                output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Extract audio using moviepy
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is None:
                    raise RuntimeError("Video file has no audio track")
                
                # Write audio file
                audio.write_audiofile(
                    output_path,
                    fps=self.sample_rate,
                    nbytes=2,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None
                )
            
            if verbose:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                console.print(f"[green]✓[/green] Audio extracted: {output_path} ({file_size:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio: {str(e)}")
    
    def extract_segment(self, video_path: str, start_time: float, 
                       end_time: float, output_path: Optional[str] = None,
                       verbose: bool = True) -> str:
        """
        Extract audio from a specific segment of the video.
        
        Args:
            video_path: Path to input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save audio file (optional)
            verbose: Whether to print progress messages
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if verbose:
            console.print(f"[cyan]Extracting audio segment:[/cyan] {start_time}s to {end_time}s")
        
        try:
            # Create temporary file if output path not provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                video_name = Path(video_path).stem
                output_path = os.path.join(temp_dir, f"{video_name}_segment_{start_time}_{end_time}.wav")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Extract audio segment
            with VideoFileClip(video_path) as video:
                audio_segment = video.subclip(start_time, end_time).audio
                if audio_segment is None:
                    raise RuntimeError("Video segment has no audio track")
                
                audio_segment.write_audiofile(
                    output_path,
                    fps=self.sample_rate,
                    nbytes=2,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None
                )
            
            if verbose:
                console.print(f"[green]✓[/green] Audio segment extracted: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio segment: {str(e)}")
