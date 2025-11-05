#!/bin/bash
# Installation script for Video Transcriptor

echo "=========================================="
echo "Video Transcriptor - Installation"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing video-transcriptor in editable mode..."
pip install -e .

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "To use the CLI:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run commands:"
echo "     vtranscribe transcribe video.mp4"
echo "     vtranscribe batch *.mp4 --format srt"
echo "     vtranscribe models"
echo ""
echo "Test the installation:"
echo "  python3 test_installation.py"
echo ""
