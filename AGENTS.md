# Agent Guidelines for Video Transcriptor

## Commands
```bash
# Test: python test_parallel.py video.mp4 (no formal test suite)
# Build: pip install -e . (no separate build step)
# CLI: vtranscribe transcribe video.mp4 --language en --model base
# Single test: Use test_parallel.py or examples/example_usage.py
```

## Code Style
- **Python**: 3.11+ required (.python-version)
- **Imports**: stdlib → third-party → local (separated by blank lines)
- **Types**: Required for all function parameters/returns using `typing` module
- **Docstrings**: Google style, required for all functions/classes
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants, `_leading_underscore` private
- **Errors**: Use specific exceptions with clear messages (FileNotFoundError, ValueError, RuntimeError)
- **Paths**: Always use `pathlib.Path` for cross-platform compatibility
- **CLI Output**: Use `rich.console.Console()` with color codes: `[green]✓[/green]`, `[cyan]ℹ[/cyan]`, `[yellow]⚠[/yellow]`, `[red]✗[/red]`
- **Formatting**: 4 spaces, ~100 char lines, double quotes
- **Cleanup**: Always use try/finally for temp file removal

## Key Patterns
- **Error handling**: Always clean up resources in finally blocks
- **Device selection**: CUDA > MPS > CPU (automatic in VideoTranscriptor)
- **Optimization**: faster-whisper (default), VAD enabled, compute_type="auto"
- **Module-level workers**: Parallel processing workers must be module-level functions (picklable)
