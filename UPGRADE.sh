#!/bin/bash
# Upgrade script for Video Transcriptor optimizations

set -e

echo "🚀 Video Transcriptor - Optimization Upgrade"
echo "============================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate venv
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install/upgrade dependencies
echo "📥 Installing optimized dependencies..."
echo "   - faster-whisper (4-10x speedup)"
echo "   - torch & torchaudio (for VAD)"
echo "   - All other requirements"
pip install --upgrade -r requirements.txt

# Reinstall package in editable mode
echo ""
echo "🔧 Reinstalling video-transcriptor package..."
pip install -e .

echo ""
echo "✅ Upgrade complete!"
echo ""
echo "📊 Checking system capabilities..."
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {\"Available\" if torch.cuda.is_available() else \"Not available\"}')
if hasattr(torch.backends, 'mps'):
    print(f'   MPS (Apple Silicon): {\"Available\" if torch.backends.mps.is_available() else \"Not available\"}')
try:
    import faster_whisper
    print(f'   faster-whisper: {faster_whisper.__version__} ✨ (optimized)')
except ImportError:
    print(f'   faster-whisper: Not installed (using openai-whisper)')
"

echo ""
echo "🎯 Test the optimizations with:"
echo "   vtranscribe info"
echo "   vtranscribe transcribe your-video.mp4 --language en"
echo ""
echo "📖 Read OPTIMIZATION_UPGRADE.md for detailed usage guide"
