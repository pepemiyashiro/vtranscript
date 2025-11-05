"""
Main transcription module using OpenAI Whisper.
"""

import os
import torch
import whisper
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .audio_extractor import AudioExtractor
from .utils import (
    ensure_dir,
    get_output_path,
    save_as_txt,
    save_as_srt,
    save_as_json,
    validate_video_file
)

console = Console()


class VideoTranscriptor:
    """Transcribe video files to text using Whisper."""
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the VideoTranscriptor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            use_gpu: Whether to use GPU if available
            verbose: Whether to print progress messages
        """
        self.model_size = model_size
        self.language = language
        self.verbose = verbose
        
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            if verbose:
                console.print("[green]✓[/green] Using GPU for transcription")
        else:
            self.device = "cpu"
            if verbose:
                console.print("[yellow]ℹ[/yellow] Using CPU for transcription (this will be slower)")
        
        # Load Whisper model
        if verbose:
            console.print(f"[cyan]Loading Whisper model:[/cyan] {model_size}")
        
        self.model = whisper.load_model(model_size, device=self.device)
        
        if verbose:
            console.print("[green]✓[/green] Model loaded successfully")
        
        # Initialize audio extractor
        self.audio_extractor = AudioExtractor(sample_rate=16000, channels=1)
    
    def transcribe_video(
        self,
        video_path: str,
        task: str = "transcribe",
        temperature: float = 0.0,
        best_of: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribe a video file.
        
        Args:
            video_path: Path to video file
            task: 'transcribe' or 'translate' (translate to English)
            temperature: Sampling temperature (0 = deterministic)
            best_of: Number of candidates when temperature > 0
            
        Returns:
            Dictionary containing transcription results with segments
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        # Validate video file
        validate_video_file(video_path)
        
        if self.verbose:
            console.print(f"\n[bold cyan]Transcribing:[/bold cyan] {video_path}\n")
        
        # Extract audio from video
        audio_path = self.audio_extractor.extract(video_path, verbose=self.verbose)
        
        try:
            # Transcribe audio
            if self.verbose:
                console.print("[cyan]Running transcription...[/cyan]")
            
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                task=task,
                temperature=temperature,
                best_of=best_of,
                verbose=False
            )
            
            if self.verbose:
                console.print("[green]✓[/green] Transcription complete")
            
            return result
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path) and 'temp' in audio_path:
                os.remove(audio_path)
    
    def transcribe_and_save(
        self,
        video_path: str,
        output_dir: str = "./transcriptions",
        output_format: str = "txt",
        include_timestamps: bool = True,
        task: str = "transcribe"
    ) -> Dict[str, str]:
        """
        Transcribe video and save to file(s).
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save transcription
            output_format: Output format ('txt', 'srt', 'json', or 'all')
            include_timestamps: Whether to include timestamps in txt output
            task: 'transcribe' or 'translate'
            
        Returns:
            Dictionary with paths to saved files
        """
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Transcribe video
        result = self.transcribe_video(video_path, task=task)
        
        # Prepare segments
        segments = result.get('segments', [])
        
        # Save in requested format(s)
        saved_files = {}
        
        if output_format in ['txt', 'all']:
            txt_path = get_output_path(video_path, output_dir, 'txt')
            save_as_txt(segments, str(txt_path), include_timestamps)
            saved_files['txt'] = str(txt_path)
            if self.verbose:
                console.print(f"[green]✓[/green] Saved TXT: {txt_path}")
        
        if output_format in ['srt', 'all']:
            srt_path = get_output_path(video_path, output_dir, 'srt')
            save_as_srt(segments, str(srt_path))
            saved_files['srt'] = str(srt_path)
            if self.verbose:
                console.print(f"[green]✓[/green] Saved SRT: {srt_path}")
        
        if output_format in ['json', 'all']:
            json_path = get_output_path(video_path, output_dir, 'json')
            metadata = {
                'video_path': video_path,
                'model_size': self.model_size,
                'language': result.get('language', 'unknown'),
                'task': task
            }
            save_as_json(segments, str(json_path), metadata)
            saved_files['json'] = str(json_path)
            if self.verbose:
                console.print(f"[green]✓[/green] Saved JSON: {json_path}")
        
        if self.verbose:
            console.print(f"\n[bold green]✓ Transcription complete![/bold green]\n")
        
        return saved_files
    
    def transcribe_batch(
        self,
        video_paths: List[str],
        output_dir: str = "./transcriptions",
        output_format: str = "txt",
        include_timestamps: bool = True,
        task: str = "transcribe"
    ) -> Dict[str, Dict[str, str]]:
        """
        Transcribe multiple video files.
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory to save transcriptions
            output_format: Output format ('txt', 'srt', 'json', or 'all')
            include_timestamps: Whether to include timestamps in txt output
            task: 'transcribe' or 'translate'
            
        Returns:
            Dictionary mapping video paths to their saved file paths
        """
        results = {}
        total = len(video_paths)
        
        console.print(f"\n[bold]Processing {total} video file(s)...[/bold]\n")
        
        for i, video_path in enumerate(video_paths, 1):
            console.print(f"[bold cyan]File {i}/{total}[/bold cyan]")
            
            try:
                saved_files = self.transcribe_and_save(
                    video_path,
                    output_dir=output_dir,
                    output_format=output_format,
                    include_timestamps=include_timestamps,
                    task=task
                )
                results[video_path] = saved_files
                
            except Exception as e:
                console.print(f"[bold red]✗ Error processing {video_path}:[/bold red] {str(e)}")
                results[video_path] = {'error': str(e)}
            
            console.print()  # Empty line between files
        
        console.print(f"[bold green]Batch processing complete![/bold green]")
        console.print(f"Successfully processed: {sum(1 for r in results.values() if 'error' not in r)}/{total}")
        
        return results
