# Agent Guidelines for Video Transcriptor

## Project Overview

Video Transcriptor is a Python CLI tool for transcribing video files using Whisper (optimized with faster-whisper).

## CLI Commands

### Primary Commands (Use vtranscribe)

```bash
# Single transcription
vtranscribe transcribe video.mp4 --format srt --model base --language en

# Batch processing
vtranscribe batch video1.mp4 video2.mp4 --format all --language en

# Check system and optimization status
vtranscribe info

# List models
vtranscribe models
```

### Testing

```bash
# Test installation
python test_installation.py

# Manual test with example
python examples/example_usage.py
```

**Note**: No formal test suite. Test manually using example files.

## Code Style Guidelines

### Python Version
- **Required**: Python 3.11 or later (specified in `.python-version`)
- **Tested**: 3.11, 3.12, 3.13

### Import Organization

```python
# Standard library imports
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Third-party imports
import torch
import click
from rich.console import Console

# Local imports
from .transcriptor import VideoTranscriptor
from .utils import save_as_txt
```

**Rules**:
- Group imports: stdlib → third-party → local
- Separate groups with blank lines
- Use absolute imports from `src.` for local modules

### Type Hints

Required for all function parameters and returns:

```python
def transcribe_video(
    self,
    video_path: str,
    language: Optional[str] = None,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """Transcribe a video file."""
    pass
```

Use typing module: `Optional[str]`, `List[Dict[str, Any]]`, `Tuple[float, float]`

### Docstrings

Required for all functions and classes. Use Google style:

```python
def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save audio (optional, creates temp file if None)
        
    Returns:
        Path to extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If extraction fails
    """
    pass
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private members**: `_leading_underscore`

```python
# Good
def transcribe_video(video_path: str) -> Dict[str, Any]:
    MAX_RETRIES = 3
    temp_file_path = "/tmp/audio.wav"
    
class VideoTranscriptor:
    def __init__(self):
        self._model = None  # Private
```

### Error Handling

Use specific exceptions with clear messages:

```python
# Good
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

if sample_rate < 1000:
    raise ValueError(f"Invalid sample rate: {sample_rate} (must be >= 1000)")

try:
    result = self.model.transcribe(audio_path)
except Exception as e:
    raise RuntimeError(f"Transcription failed: {str(e)}")
finally:
    # Always clean up resources
    if os.path.exists(temp_file):
        os.remove(temp_file)
```

### Formatting

- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~100 characters (soft limit)
- **Blank lines**: One between functions, two between classes
- **Quotes**: Use double quotes for strings (unless escaping needed)

### Rich Console Output

Use `rich.console.Console()` for CLI output:

```python
from rich.console import Console

console = Console()

# Color codes
console.print("[green]✓[/green] Success message")
console.print("[cyan]ℹ[/cyan] Info message")
console.print("[yellow]⚠[/yellow] Warning message")
console.print("[red]✗[/red] Error message")
console.print("[bold cyan]Processing...[/bold cyan]")
```

### Path Handling

Always use `pathlib.Path` for cross-platform compatibility:

```python
from pathlib import Path

# Good
video_path = Path("videos/sample.mp4")
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Avoid
output_dir = "videos/" + filename  # Platform-specific
```

## Key Dependencies

- **faster-whisper**: Optimized Whisper implementation (4-10x faster)
- **openai-whisper**: Fallback if faster-whisper unavailable
- **torch/torchaudio**: ML framework and VAD support
- **moviepy**: Video processing (with bundled FFmpeg via imageio-ffmpeg)
- **click**: CLI framework
- **rich**: Beautiful CLI output
- **pyyaml**: Configuration files

## Project Structure

```
video-transcriptor/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── transcriptor.py       # Main transcription logic (Whisper model)
│   ├── audio_extractor.py    # Audio extraction (MoviePy)
│   ├── utils.py              # Utilities (save formats, validation)
│   └── cli.py                # CLI interface (Click)
├── examples/
│   └── example_usage.py      # Python API examples
├── .github/
│   └── AGENTS.md            # This file (development guidelines)
├── requirements.txt          # Dependencies
├── setup.py                 # Package setup (enables vtranscribe command)
├── config.yaml              # Default configuration
└── README.md                # Main documentation
```

## Key Files Explained

### `src/transcriptor.py`
- Core transcription logic
- Handles both faster-whisper and openai-whisper
- Device management (CUDA, MPS, CPU)
- VAD integration for silence skipping
- Compute type optimization (int8, float16, float32)

### `src/audio_extractor.py`
- Extracts audio from video using MoviePy
- Converts to 16kHz mono WAV (Whisper requirement)
- Handles temporary file management

### `src/utils.py`
- File format utilities (save_as_txt, save_as_srt, save_as_json)
- Video file validation
- Path management

### `src/cli.py`
- Click-based CLI interface
- Commands: transcribe, batch, models, info
- Option parsing and validation

## Development Workflow

### Making Changes

1. **Activate environment**: `source venv/bin/activate`
2. **Make code changes** following style guidelines above
3. **Test manually**: `vtranscribe transcribe test_video.mp4`
4. **Check with test script**: `python test_installation.py`

### Adding New Features

1. Update appropriate module (transcriptor, audio_extractor, utils, cli)
2. Add docstrings with Google style
3. Update CLI if needed (src/cli.py)
4. Update README.md and QUICKSTART.md if user-facing
5. Test thoroughly with different video formats

### Common Pitfalls

- **Don't forget error handling**: Always use try/finally for resource cleanup
- **Use Path objects**: Don't use string concatenation for paths
- **Type hints are required**: Functions need proper type annotations
- **Rich console everywhere**: Use console.print() not print() in CLI code
- **Clean up temp files**: Always remove temporary audio files in finally blocks

## Optimization Guidelines

### Performance Priorities (v0.2.0+)

1. **faster-whisper**: Default, fallback to openai-whisper
2. **Language specification**: Encourage users to use `--language`
3. **VAD**: Enabled by default, skip silence
4. **Compute type**: Auto-select based on hardware
5. **Device selection**: CUDA > MPS > CPU (automatic)

### When to Disable Optimizations

- **Disable VAD**: Music videos, continuous audio, no silence
- **Use float32**: Maximum accuracy needed, research purposes
- **Force CPU**: GPU memory issues, debugging

## Useful Commands Reference

```bash
# Development
source venv/bin/activate                    # Activate environment
pip install -e .                             # Install in editable mode
python test_installation.py                  # Test dependencies

# CLI Usage
vtranscribe transcribe video.mp4 --language en            # Standard
vtranscribe transcribe video.mp4 --language en --compute-type int8  # Fast
vtranscribe transcribe video.mp4 --no-vad --compute-type float32    # Accurate
vtranscribe batch *.mp4 --format all --language en        # Batch
vtranscribe info                                           # System check
vtranscribe models                                         # List models

# Upgrading
./UPGRADE.sh                                 # Upgrade to optimized version
pip install --upgrade -r requirements.txt    # Upgrade dependencies
```

## Questions?

- **Main docs**: [README.md](../README.md)
- **Quick start**: [QUICKSTART.md](../QUICKSTART.md)
- **Optimizations**: [OPTIMIZATION_UPGRADE.md](../OPTIMIZATION_UPGRADE.md)
- **Changelog**: [CHANGES.md](../CHANGES.md)
