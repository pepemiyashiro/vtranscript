"""
Main transcription module using Whisper (faster-whisper or openai-whisper).
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Try importing faster-whisper first, fall back to openai-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    import whisper
    FASTER_WHISPER_AVAILABLE = False

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
        verbose: bool = True,
        use_vad: bool = True,
        compute_type: str = "auto"
    ):
        """
        Initialize the VideoTranscriptor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            use_gpu: Whether to use GPU if available
            verbose: Whether to print progress messages
            use_vad: Whether to use Voice Activity Detection to skip silent sections
            compute_type: Computation precision for faster-whisper ("int8", "float16", "float32", "auto")
        """
        self.model_size = model_size
        self.language = language
        self.verbose = verbose
        self.use_vad = use_vad
        self.use_faster_whisper = FASTER_WHISPER_AVAILABLE
        
        # Determine device and compute type
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            if compute_type == "auto":
                self.compute_type = "float16"
            else:
                self.compute_type = compute_type
            if verbose:
                console.print("[green]✓[/green] Using CUDA GPU for transcription")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps" if not FASTER_WHISPER_AVAILABLE else "cpu"
            if compute_type == "auto":
                self.compute_type = "float32"
            else:
                self.compute_type = compute_type
            if verbose:
                if self.device == "mps":
                    console.print("[green]✓[/green] Using Apple Silicon MPS GPU for transcription")
                else:
                    console.print("[yellow]ℹ[/yellow] MPS available but using CPU (faster-whisper doesn't support MPS)")
        else:
            self.device = "cpu"
            if compute_type == "auto":
                self.compute_type = "int8"
            else:
                self.compute_type = compute_type
            if verbose:
                console.print("[yellow]ℹ[/yellow] Using CPU for transcription (slower than GPU)")
        
        # Load Whisper model
        if verbose:
            console.print(f"[cyan]Loading Whisper model:[/cyan] {model_size}")
            if FASTER_WHISPER_AVAILABLE:
                console.print(f"[green]✓[/green] Using faster-whisper (optimized)")
                console.print(f"[cyan]Compute type:[/cyan] {self.compute_type}")
            else:
                console.print("[yellow]ℹ[/yellow] Using openai-whisper (install faster-whisper for 4-10x speedup)")
        
        if FASTER_WHISPER_AVAILABLE:
            # Use faster-whisper for major speedup
            device_str = self.device if self.device != "mps" else "cpu"
            self.model = WhisperModel(
                model_size,
                device=device_str,
                compute_type=self.compute_type
            )
        else:
            # Fall back to openai-whisper
            device_str = self.device
            self.model = whisper.load_model(model_size, device=device_str)
        
        if verbose:
            console.print("[green]✓[/green] Model loaded successfully")
        
        # Initialize audio extractor
        self.audio_extractor = AudioExtractor(sample_rate=16000, channels=1)
        
        # Initialize VAD if requested
        if use_vad:
            if verbose:
                console.print("[cyan]Initializing Voice Activity Detection...[/cyan]")
            try:
                import torch
                self.vad_model, self.vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    verbose=False
                )
                if verbose:
                    console.print("[green]✓[/green] VAD initialized (will skip silent sections)")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠[/yellow] Could not load VAD model: {e}")
                self.vad_model = None
        else:
            self.vad_model = None
    
    def _apply_vad(self, audio_path: str) -> Optional[List[Tuple[float, float]]]:
        """
        Apply Voice Activity Detection to find speech segments.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start, end) tuples for speech segments in seconds, or None if VAD fails
        """
        if not self.vad_model:
            return None
        
        try:
            import torch
            import torchaudio
            
            # Load audio
            wav, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed (VAD expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                wav = resampler(wav)
                sample_rate = 16000
            
            # Get speech timestamps
            (get_speech_timestamps, _, read_audio, *_) = self.vad_utils
            speech_timestamps = get_speech_timestamps(
                wav[0],
                self.vad_model,
                sampling_rate=sample_rate,
                return_seconds=True
            )
            
            if self.verbose and len(speech_timestamps) > 0:
                total_duration = len(wav[0]) / sample_rate
                speech_duration = sum(seg['end'] - seg['start'] for seg in speech_timestamps)
                console.print(f"[green]✓[/green] VAD found {len(speech_timestamps)} speech segments "
                            f"({speech_duration:.1f}s / {total_duration:.1f}s)")
            
            return [(seg['start'], seg['end']) for seg in speech_timestamps]
            
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]⚠[/yellow] VAD failed: {e}")
            return None
    
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
            # Apply VAD if enabled
            vad_segments = self._apply_vad(audio_path) if self.use_vad else None
            
            # Transcribe audio
            if self.verbose:
                console.print("[cyan]Running transcription...[/cyan]")
            
            if FASTER_WHISPER_AVAILABLE and self.use_faster_whisper:
                # Use faster-whisper API
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.language,
                    task=task,
                    temperature=temperature,
                    best_of=best_of,
                    vad_filter=True if vad_segments else False,
                    vad_parameters=dict(min_silence_duration_ms=500) if vad_segments else None
                )
                
                # Convert faster-whisper segments to openai-whisper format
                result = {
                    'text': '',
                    'segments': [],
                    'language': info.language
                }
                
                for segment in segments:
                    result['segments'].append({
                        'id': segment.id,
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    })
                    result['text'] += segment.text.strip() + ' '
                
                result['text'] = result['text'].strip()
                
            else:
                # Use openai-whisper API
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
