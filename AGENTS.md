# Agent Guidelines for Video Transcriptor

## Commands
- **Run CLI**: `python -m src.cli <command>`
- **Single transcription**: `python -m src.cli transcribe video.mp4 --format srt --model base`
- **Batch processing**: `python -m src.cli batch video1.mp4 video2.mp4 --format all`
- **No formal test suite**: Test manually using example files or `python examples/example_usage.py`

## Code Style
- **Python version**: 3.11.9
- **Imports**: Group stdlib, third-party, local (separated by blank lines); use absolute imports from `src.`
- **Type hints**: Use typing module annotations for function parameters and returns (e.g., `Optional[str]`, `List[Dict[str, Any]]`)
- **Docstrings**: Required for all functions/classes; use Google style with Args/Returns/Raises sections
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error handling**: Raise specific exceptions (FileNotFoundError, ValueError, RuntimeError) with clear messages; clean up resources in finally blocks
- **Formatting**: 4 spaces indentation, max line length ~100 chars, blank line between functions
- **Rich console**: Use `rich.console.Console()` for CLI output with color codes: `[green]`, `[cyan]`, `[yellow]`, `[red]`
- **Path handling**: Use `pathlib.Path` for cross-platform compatibility

## Key Dependencies
- OpenAI Whisper for transcription, PyTorch for ML, MoviePy for video processing, Click for CLI, Rich for output
