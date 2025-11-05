"""
Setup script for Video Transcriptor.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="video-transcriptor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python tool for extracting text transcriptions from video files using OpenAI Whisper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-transcriptor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vtranscribe=src.cli:cli",
            "video-transcriptor=src.cli:cli",
        ],
    },
    include_package_data=True,
)
