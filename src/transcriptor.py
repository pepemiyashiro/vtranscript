"""
Main transcription module using Whisper (faster-whisper or openai-whisper).
"""

import os
import sys
import signal
import atexit
import torch
import numpy as np
import math
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
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


def _transcribe_chunk_worker(chunk_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel chunk transcription.
    This is a module-level function so it can be pickled by multiprocessing.
    
    Args:
        chunk_info: Dictionary containing all necessary parameters
        
    Returns:
        Dictionary with chunk results and metadata
    """
    video_path = chunk_info['video_path']
    start_time = chunk_info['start_time']
    end_time = chunk_info['end_time']
    chunk_id = chunk_info['chunk_id']
    task = chunk_info.get('task', 'transcribe')
    temperature = chunk_info.get('temperature', 0.0)
    best_of = chunk_info.get('best_of', 5)
    model_size = chunk_info['model_size']
    language = chunk_info.get('language')
    device = chunk_info['device']
    compute_type = chunk_info['compute_type']
    beam_size = chunk_info.get('beam_size', 1)
    
    # Initialize model in this worker process
    if FASTER_WHISPER_AVAILABLE:
        device_str = device if device != "mps" else "cpu"
        model = WhisperModel(
            model_size,
            device=device_str,
            compute_type=compute_type
        )
    else:
        import whisper
        model = whisper.load_model(model_size, device=device)
    
    # Initialize audio extractor
    audio_extractor = AudioExtractor(sample_rate=16000, channels=1)
    
    # Extract audio segment
    audio_path = None
    try:
        audio_path = audio_extractor.extract_segment(
            video_path, 
            start_time, 
            end_time,
            verbose=False
        )
        
        # Transcribe the chunk
        if FASTER_WHISPER_AVAILABLE:
            segments, info = model.transcribe(
                audio_path,
                language=language,
                task=task,
                temperature=temperature,
                best_of=best_of,
                beam_size=beam_size,
                vad_filter=False  # Already chunked, no need for VAD
            )
            
            result = {
                'chunk_id': chunk_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': '',
                'segments': [],
                'language': info.language
            }
            
            for segment in segments:
                # Adjust timestamps to absolute time
                result['segments'].append({
                    'id': segment.id,
                    'start': segment.start + start_time,
                    'end': segment.end + start_time,
                    'text': segment.text.strip()
                })
                result['text'] += segment.text.strip() + ' '
            
            result['text'] = result['text'].strip()
            
        else:
            # Use openai-whisper
            transcription = model.transcribe(
                audio_path,
                language=language,
                task=task,
                temperature=temperature,
                best_of=best_of,
                verbose=False
            )
            
            result = {
                'chunk_id': chunk_id,
                'start_time': start_time,
                'end_time': end_time,
                'text': transcription['text'],
                'segments': [],
                'language': transcription.get('language', language)
            }
            
            # Adjust timestamps
            for segment in transcription.get('segments', []):
                result['segments'].append({
                    'id': segment['id'],
                    'start': segment['start'] + start_time,
                    'end': segment['end'] + start_time,
                    'text': segment['text'].strip()
                })
        
        return result
        
    except Exception as e:
        # Log the error and re-raise
        console.print(f"[red]✗[/red] Error processing chunk {chunk_id}: {e}")
        raise
    finally:
        # Clean up temporary audio file - ensure it's always removed
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as cleanup_error:
                # Log cleanup errors but don't fail the whole process
                console.print(f"[yellow]⚠[/yellow] Warning: Failed to clean up {audio_path}: {cleanup_error}")


class VideoTranscriptor:
    """Transcribe video files to text using Whisper."""
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        use_gpu: bool = True,
        verbose: bool = True,
        use_vad: bool = True,
        compute_type: str = "auto",
        beam_size: int = 1,
        use_parallel: bool = False,
        num_workers: Optional[int] = None
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
            beam_size: Beam search size (1=fastest, 5=default/more accurate). Lower is faster but slightly less accurate.
            use_parallel: Whether to use parallel processing for long videos (3-4x speedup)
            num_workers: Number of parallel workers (None = auto-detect based on CPU cores)
        """
        self.model_size = model_size
        self.language = language
        self.verbose = verbose
        self.use_vad = use_vad
        self.beam_size = beam_size
        self.use_parallel = use_parallel
        self.num_workers = num_workers
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
            import scipy.io.wavfile
            import scipy.signal
            
            # Load audio using scipy (works without additional dependencies)
            sample_rate, audio_data = scipy.io.wavfile.read(audio_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Handle stereo by converting to mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz if needed (VAD expects 16kHz)
            if sample_rate != 16000:
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = scipy.signal.resample(audio_data, num_samples)
                sample_rate = 16000
            
            # Convert to torch tensor for VAD
            import torch
            wav_tensor = torch.from_numpy(audio_data).float()
            
            # Get speech timestamps
            (get_speech_timestamps, _, read_audio, *_) = self.vad_utils
            speech_timestamps = get_speech_timestamps(
                wav_tensor,
                self.vad_model,
                sampling_rate=sample_rate,
                return_seconds=True
            )
            
            if self.verbose and len(speech_timestamps) > 0:
                total_duration = len(audio_data) / sample_rate
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
        
        If use_parallel is enabled, will use parallel processing for faster transcription.
        
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
        # Use parallel processing if enabled
        if self.use_parallel:
            return self.transcribe_video_parallel(
                video_path=video_path,
                task=task,
                temperature=temperature,
                best_of=best_of,
                num_workers=self.num_workers
            )
        
        # Sequential processing (original code)
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
                    beam_size=self.beam_size,
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
    

    
    def transcribe_video_parallel(
        self,
        video_path: str,
        task: str = "transcribe",
        temperature: float = 0.0,
        best_of: int = 5,
        num_workers: Optional[int] = None,
        chunk_duration: float = 300.0  # 5 minutes per chunk
    ) -> Dict[str, Any]:
        """
        Transcribe a video file using parallel processing.
        
        Args:
            video_path: Path to video file
            task: 'transcribe' or 'translate'
            temperature: Sampling temperature
            best_of: Number of candidates
            num_workers: Number of parallel workers (None = auto-detect)
            chunk_duration: Duration of each chunk in seconds (default: 300s = 5min)
            
        Returns:
            Dictionary containing transcription results with segments
        """
        # Validate video file
        validate_video_file(video_path)
        
        if self.verbose:
            console.print(f"\n[bold cyan]Transcribing (Parallel Mode):[/bold cyan] {video_path}\n")
        
        # Get video duration
        import av
        container = av.open(video_path)
        duration = float(container.duration / av.time_base)
        container.close()
        
        if self.verbose:
            console.print(f"[cyan]Video duration:[/cyan] {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # Determine number of workers
        if num_workers is None:
            num_workers = self.num_workers if self.num_workers else max(1, mp.cpu_count() - 1)
        
        # Calculate chunks
        num_chunks = math.ceil(duration / chunk_duration)
        num_workers = min(num_workers, num_chunks)  # Don't use more workers than chunks
        
        if self.verbose:
            console.print(f"[cyan]Splitting into {num_chunks} chunks of ~{chunk_duration:.0f}s each[/cyan]")
            console.print(f"[cyan]Using {num_workers} parallel workers[/cyan]")
        
        # Create chunk info with all necessary parameters
        chunks = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            chunks.append({
                'video_path': video_path,
                'start_time': start_time,
                'end_time': end_time,
                'chunk_id': i,
                'task': task,
                'temperature': temperature,
                'best_of': best_of,
                'model_size': self.model_size,
                'language': self.language,
                'device': self.device,
                'compute_type': self.compute_type,
                'beam_size': self.beam_size
            })
        
        # Process chunks in parallel
        if self.verbose:
            console.print(f"\n[cyan]Processing chunks in parallel...[/cyan]\n")
        
        # Use multiprocessing pool with explicit cleanup
        pool = None
        try:
            pool = Pool(processes=num_workers)
            chunk_results = pool.map(_transcribe_chunk_worker, chunks)
        except KeyboardInterrupt:
            if self.verbose:
                console.print("\n[yellow]⚠[/yellow] Interrupted by user, cleaning up...")
            if pool:
                pool.terminate()
                pool.join()
            raise
        except Exception as e:
            if pool:
                pool.terminate()
                pool.join()
            raise
        finally:
            # Explicit cleanup to prevent semaphore leaks
            if pool:
                pool.close()
                pool.join()
        
        # Sort by chunk_id to ensure correct order
        chunk_results.sort(key=lambda x: x['chunk_id'])
        
        # Merge results
        if self.verbose:
            console.print(f"\n[cyan]Merging {len(chunk_results)} chunks...[/cyan]")
        
        merged_result = {
            'text': '',
            'segments': [],
            'language': chunk_results[0]['language'] if chunk_results else None
        }
        
        segment_id = 0
        for chunk_result in chunk_results:
            # Append text with space
            if merged_result['text'] and chunk_result['text']:
                merged_result['text'] += ' '
            merged_result['text'] += chunk_result['text']
            
            # Append segments with renumbered IDs
            for segment in chunk_result['segments']:
                segment['id'] = segment_id
                merged_result['segments'].append(segment)
                segment_id += 1
        
        if self.verbose:
            console.print("[green]✓[/green] Parallel transcription complete")
        
        return merged_result
    
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
