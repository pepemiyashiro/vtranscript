"""
Audio extraction from video files using PyAV (pure Python, no system dependencies).
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional
import av
import wave
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
            
            # Extract audio using PyAV
            container = av.open(video_path)
            
            # Find audio stream
            audio_stream = None
            for stream in container.streams.audio:
                audio_stream = stream
                break
            
            if audio_stream is None:
                container.close()
                raise RuntimeError("Video file has no audio track")
            
            # Collect audio samples
            audio_frames = []
            for frame in container.decode(audio_stream):
                # Convert frame to numpy array
                array = frame.to_ndarray()
                audio_frames.append(array)
            
            container.close()
            
            if not audio_frames:
                raise RuntimeError("No audio data could be extracted")
            
            # Concatenate all frames
            audio_data = np.concatenate(audio_frames, axis=1)
            
            # Convert to mono if needed
            if audio_data.shape[0] > 1 and self.channels == 1:
                audio_data = np.mean(audio_data, axis=0, keepdims=True)
            
            # Resample if needed
            original_sample_rate = audio_stream.rate
            if original_sample_rate != self.sample_rate:
                import scipy.signal
                num_samples = int(audio_data.shape[1] * self.sample_rate / original_sample_rate)
                audio_data = scipy.signal.resample(audio_data, num_samples, axis=1)
            
            # Convert to int16 format
            audio_data = audio_data.T  # Transpose to (samples, channels)
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
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
            
            # Extract audio segment using PyAV
            container = av.open(video_path)
            
            # Find audio stream
            audio_stream = None
            for stream in container.streams.audio:
                audio_stream = stream
                break
            
            if audio_stream is None:
                container.close()
                raise RuntimeError("Video segment has no audio track")
            
            # Seek to start time
            container.seek(int(start_time * av.time_base))
            
            # Collect audio samples within the time range
            audio_frames = []
            for frame in container.decode(audio_stream):
                frame_time = float(frame.pts * audio_stream.time_base)
                
                if frame_time < start_time:
                    continue
                if frame_time > end_time:
                    break
                    
                # Convert frame to numpy array
                array = frame.to_ndarray()
                audio_frames.append(array)
            
            container.close()
            
            if not audio_frames:
                raise RuntimeError("No audio data could be extracted from segment")
            
            # Concatenate all frames
            audio_data = np.concatenate(audio_frames, axis=1)
            
            # Convert to mono if needed
            if audio_data.shape[0] > 1 and self.channels == 1:
                audio_data = np.mean(audio_data, axis=0, keepdims=True)
            
            # Resample if needed
            original_sample_rate = audio_stream.rate
            if original_sample_rate != self.sample_rate:
                import scipy.signal
                num_samples = int(audio_data.shape[1] * self.sample_rate / original_sample_rate)
                audio_data = scipy.signal.resample(audio_data, num_samples, axis=1)
            
            # Convert to int16 format
            audio_data = audio_data.T  # Transpose to (samples, channels)
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            if verbose:
                console.print(f"[green]✓[/green] Audio segment extracted: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio segment: {str(e)}")
