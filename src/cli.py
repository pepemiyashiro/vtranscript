#!/usr/bin/env python3
"""
Command-line interface for Video Transcriptor.
"""

import click
import yaml
from pathlib import Path
from typing import Optional

from .transcriptor import VideoTranscriptor
from .utils import validate_video_file


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Video Transcriptor - Extract text transcriptions from video files."""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--output-dir', '-o',
    default='./transcriptions',
    help='Output directory for transcription files'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['txt', 'srt', 'json', 'all'], case_sensitive=False),
    default='txt',
    help='Output format'
)
@click.option(
    '--model', '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'], case_sensitive=False),
    default='base',
    help='Whisper model size (larger = more accurate but slower)'
)
@click.option(
    '--language', '-l',
    default=None,
    help='Language code (e.g., en, es, fr) or auto-detect if not specified'
)
@click.option(
    '--task', '-t',
    type=click.Choice(['transcribe', 'translate'], case_sensitive=False),
    default='transcribe',
    help='Task: transcribe in original language or translate to English'
)
@click.option(
    '--no-timestamps',
    is_flag=True,
    help='Exclude timestamps from text output'
)
@click.option(
    '--no-gpu',
    is_flag=True,
    help='Disable GPU acceleration'
)
@click.option(
    '--no-vad',
    is_flag=True,
    help='Disable Voice Activity Detection (VAD) for silent section skipping'
)
@click.option(
    '--compute-type',
    type=click.Choice(['int8', 'float16', 'float32', 'auto'], case_sensitive=False),
    default='auto',
    help='Computation precision (int8=fastest, float32=most accurate, auto=recommended)'
)
@click.option(
    '--fast',
    is_flag=True,
    help='Fast mode: use beam_size=1 for ~3x speed (already default)'
)
@click.option(
    '--accurate',
    is_flag=True,
    help='Accurate mode: use beam_size=5 for best quality (~3x slower)'
)
@click.option(
    '--parallel',
    is_flag=True,
    help='Use parallel processing for faster transcription (3-4x speedup)'
)
@click.option(
    '--workers',
    type=int,
    default=None,
    help='Number of parallel workers (default: CPU cores - 1)'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def transcribe(
    video_path: str,
    output_dir: str,
    format: str,
    model: str,
    language: Optional[str],
    task: str,
    no_timestamps: bool,
    no_gpu: bool,
    no_vad: bool,
    compute_type: str,
    fast: bool,
    accurate: bool,
    parallel: bool,
    workers: Optional[int],
    config: Optional[str]
):
    """Transcribe a single video file."""
    
    # Check for conflicting flags
    if fast and accurate:
        click.echo("Error: --fast and --accurate are mutually exclusive", err=True)
        raise click.Abort()
    
    # Determine beam_size based on flags
    beam_size = 5 if accurate else 1  # Default is 1 (fast mode)
    
    # Load configuration if provided
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Override with config values if not specified in CLI
        if not model:
            model = config_data.get('whisper', {}).get('model_size', 'base')
        if not language:
            language = config_data.get('whisper', {}).get('language')
        if not task:
            task = config_data.get('whisper', {}).get('task', 'transcribe')
        if not output_dir:
            output_dir = config_data.get('output', {}).get('output_dir', './transcriptions')
        if format == 'txt':
            format = config_data.get('output', {}).get('default_format', 'txt')
    
    try:
        # Validate video file
        validate_video_file(video_path)
        
        # Initialize transcriptor
        transcriptor = VideoTranscriptor(
            model_size=model,
            language=language,
            use_gpu=not no_gpu,
            use_vad=not no_vad,
            beam_size=beam_size,
            use_parallel=parallel,
            num_workers=workers,
            verbose=True
        )
        
        # Transcribe and save
        saved_files = transcriptor.transcribe_and_save(
            video_path=video_path,
            output_dir=output_dir,
            output_format=format,
            include_timestamps=not no_timestamps,
            task=task
        )
        
        click.echo(f"\n✓ Transcription saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('video_paths', nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    '--output-dir', '-o',
    default='./transcriptions',
    help='Output directory for transcription files'
)
@click.option(
    '--format', '-f',
    type=click.Choice(['txt', 'srt', 'json', 'all'], case_sensitive=False),
    default='txt',
    help='Output format'
)
@click.option(
    '--model', '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large'], case_sensitive=False),
    default='base',
    help='Whisper model size'
)
@click.option(
    '--language', '-l',
    default=None,
    help='Language code or auto-detect'
)
@click.option(
    '--task', '-t',
    type=click.Choice(['transcribe', 'translate'], case_sensitive=False),
    default='transcribe',
    help='Task type'
)
@click.option(
    '--no-timestamps',
    is_flag=True,
    help='Exclude timestamps from text output'
)
@click.option(
    '--no-gpu',
    is_flag=True,
    help='Disable GPU acceleration'
)
@click.option(
    '--no-vad',
    is_flag=True,
    help='Disable Voice Activity Detection (VAD)'
)
@click.option(
    '--compute-type',
    type=click.Choice(['int8', 'float16', 'float32', 'auto'], case_sensitive=False),
    default='auto',
    help='Computation precision'
)
@click.option(
    '--fast',
    is_flag=True,
    help='Fast mode: use beam_size=1 for ~3x speed (already default)'
)
@click.option(
    '--accurate',
    is_flag=True,
    help='Accurate mode: use beam_size=5 for best quality (~3x slower)'
)
@click.option(
    '--parallel',
    is_flag=True,
    help='Use parallel processing for faster transcription (3-4x speedup)'
)
@click.option(
    '--workers',
    type=int,
    default=None,
    help='Number of parallel workers (default: CPU cores - 1)'
)
def batch(
    video_paths: tuple,
    output_dir: str,
    format: str,
    model: str,
    language: Optional[str],
    task: str,
    no_timestamps: bool,
    no_gpu: bool,
    no_vad: bool,
    compute_type: str,
    fast: bool,
    accurate: bool,
    parallel: bool,
    workers: Optional[int]
):
    """Transcribe multiple video files."""
    
    # Check for conflicting flags
    if fast and accurate:
        click.echo("Error: --fast and --accurate are mutually exclusive", err=True)
        raise click.Abort()
    
    # Determine beam_size based on flags
    beam_size = 5 if accurate else 1  # Default is 1 (fast mode)
    
    try:
        # Initialize transcriptor with optimizations
        transcriptor = VideoTranscriptor(
            model_size=model,
            language=language,
            use_gpu=not no_gpu,
            use_vad=not no_vad,
            compute_type=compute_type,
            beam_size=beam_size,
            use_parallel=parallel,
            num_workers=workers,
            verbose=True
        )
        
        # Transcribe batch
        results = transcriptor.transcribe_batch(
            video_paths=list(video_paths),
            output_dir=output_dir,
            output_format=format,
            include_timestamps=not no_timestamps,
            task=task
        )
        
        # Summary
        successful = sum(1 for r in results.values() if 'error' not in r)
        failed = len(results) - successful
        
        click.echo(f"\n✓ Batch transcription complete!")
        click.echo(f"  Successful: {successful}")
        if failed > 0:
            click.echo(f"  Failed: {failed}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
def models():
    """List available Whisper models and their characteristics."""
    
    models_info = [
        ("tiny", "~39M params", "~1GB VRAM", "Fastest, least accurate"),
        ("base", "~74M params", "~1GB VRAM", "Good balance for most uses"),
        ("small", "~244M params", "~2GB VRAM", "Better accuracy"),
        ("medium", "~769M params", "~5GB VRAM", "High accuracy"),
        ("large", "~1550M params", "~10GB VRAM", "Best accuracy, slowest"),
    ]
    
    click.echo("\nAvailable Whisper Models:\n")
    for name, params, vram, description in models_info:
        click.echo(f"  {name:8} - {params:12} {vram:10} - {description}")
    click.echo()


@cli.command()
def info():
    """Display system information and configuration."""
    
    import torch
    
    click.echo("\nSystem Information:\n")
    click.echo(f"  PyTorch version: {torch.__version__}")
    click.echo(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        click.echo(f"  CUDA version: {torch.version.cuda}")
        click.echo(f"  GPU device: {torch.cuda.get_device_name(0)}")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        click.echo(f"  Apple Silicon MPS: Available")
    
    # Check for faster-whisper
    try:
        import faster_whisper
        click.echo(f"  faster-whisper: {faster_whisper.__version__} (optimized)")
    except ImportError:
        click.echo(f"  faster-whisper: Not installed (using openai-whisper)")
    
    # Check for VAD
    try:
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, verbose=False)
        click.echo(f"  Silero VAD: Available")
    except:
        click.echo(f"  Silero VAD: Not available")
    
    click.echo()


if __name__ == '__main__':
    cli()
