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
    config: Optional[str]
):
    """Transcribe a single video file."""
    
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
def batch(
    video_paths: tuple,
    output_dir: str,
    format: str,
    model: str,
    language: Optional[str],
    task: str,
    no_timestamps: bool,
    no_gpu: bool
):
    """Transcribe multiple video files."""
    
    try:
        # Initialize transcriptor
        transcriptor = VideoTranscriptor(
            model_size=model,
            language=language,
            use_gpu=not no_gpu,
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
    
    click.echo()


if __name__ == '__main__':
    cli()
