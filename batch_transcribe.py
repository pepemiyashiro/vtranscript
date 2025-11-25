#!/usr/bin/env python3
"""Batch transcribe all videos in a directory."""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from src.transcriptor import VideoTranscriptor

console = Console()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Batch transcribe all videos in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe videos, output to same directory
  python batch_transcribe.py "/path/to/videos"
  
  # Specify separate output directory
  python batch_transcribe.py "/path/to/videos" "/path/to/transcripts"
  
  # Use different model and language
  python batch_transcribe.py "/path/to/videos" "/path/to/transcripts" --model medium --language en
        """
    )
    
    parser.add_argument(
        'video_dir',
        type=str,
        help='Directory containing video files'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        default=None,
        help='Directory to save transcripts (default: same as video directory)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='es',
        help='Language code for transcription (default: es for Spanish)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (slower but uses less memory)'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else video_dir
    
    if not video_dir.exists():
        console.print(f"[red]✗[/red] Directory not found: {video_dir}")
        sys.exit(1)
    
    if not video_dir.is_dir():
        console.print(f"[red]✗[/red] Not a directory: {video_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .mp4 files, excluding macOS metadata files
    video_files = [str(f) for f in video_dir.glob("*.mp4") if not f.name.startswith("._")]
    
    if not video_files:
        console.print(f"[yellow]⚠[/yellow] No video files found in: {video_dir}")
        sys.exit(1)
    
    console.print(f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]")
    console.print(f"[cyan]Configuration:[/cyan]")
    console.print(f"  Video directory: {video_dir}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Language: {args.language}")
    console.print(f"  Parallel processing: {'No' if args.no_parallel else 'Yes'}")
    console.print(f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]")
    console.print(f"\n[cyan]ℹ[/cyan] Found {len(video_files)} video files to transcribe\n")
    
    # Initialize transcriptor
    transcriptor = VideoTranscriptor(
        model_size=args.model,
        language=args.language,
        use_gpu=True,
        verbose=True,
        use_vad=True,
        use_parallel=not args.no_parallel,
        num_workers=None  # Auto-detect CPU cores
    )
    
    # Use built-in batch transcription
    results = transcriptor.transcribe_batch(
        video_paths=video_files,
        output_dir=str(output_dir),
        output_format="all",  # Save as txt, srt, and json
        include_timestamps=True,
        task="transcribe"
    )
    
    # Additional summary
    console.print(f"\n[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]")
    console.print(f"[cyan]Transcription Summary:[/cyan]")
    
    success_count = sum(1 for r in results.values() if 'error' not in r)
    failed_count = len(results) - success_count
    
    console.print(f"[green]✓[/green] Successful: {success_count}/{len(results)}")
    
    if failed_count > 0:
        console.print(f"[red]✗[/red] Failed: {failed_count}")
        for video_path, result in results.items():
            if 'error' in result:
                console.print(f"  [red]→[/red] {Path(video_path).name}: {result['error']}")
    
    console.print(f"\n[cyan]Output location:[/cyan] {output_dir}")
    console.print(f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]")

if __name__ == "__main__":
    main()
