#!/usr/bin/env python3
"""
Example usage of the Video Transcriptor library.
"""

from pathlib import Path
from src.transcriptor import VideoTranscriptor


def example_single_video():
    """Transcribe a single video file."""
    
    print("=" * 60)
    print("Example 1: Transcribe a single video")
    print("=" * 60)
    
    # Initialize transcriptor with base model
    transcriptor = VideoTranscriptor(
        model_size="base",
        language="en",  # Set to None for auto-detection
        use_gpu=True,
        verbose=True
    )
    
    # Transcribe and save (replace with your video path)
    video_path = "path/to/your/video.mp4"
    
    if Path(video_path).exists():
        saved_files = transcriptor.transcribe_and_save(
            video_path=video_path,
            output_dir="./transcriptions",
            output_format="all",  # Save in all formats
            include_timestamps=True
        )
        
        print("\nSaved files:")
        for format_type, file_path in saved_files.items():
            print(f"  {format_type}: {file_path}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with your video file.")


def example_batch_processing():
    """Transcribe multiple video files."""
    
    print("\n" + "=" * 60)
    print("Example 2: Batch transcription")
    print("=" * 60)
    
    # Initialize transcriptor
    transcriptor = VideoTranscriptor(
        model_size="base",
        use_gpu=True,
        verbose=True
    )
    
    # List of videos to transcribe
    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        "path/to/video3.mp4"
    ]
    
    # Filter existing videos
    existing_videos = [v for v in video_paths if Path(v).exists()]
    
    if existing_videos:
        results = transcriptor.transcribe_batch(
            video_paths=existing_videos,
            output_dir="./transcriptions",
            output_format="txt",
            include_timestamps=True
        )
        
        print("\nBatch processing results:")
        for video, files in results.items():
            print(f"  {Path(video).name}: {files}")
    else:
        print("No video files found.")
        print("Please update the video_paths list with your video files.")


def example_custom_settings():
    """Transcribe with custom settings."""
    
    print("\n" + "=" * 60)
    print("Example 3: Custom settings")
    print("=" * 60)
    
    # Initialize with larger model for better accuracy
    transcriptor = VideoTranscriptor(
        model_size="medium",  # Better accuracy, slower
        language=None,  # Auto-detect language
        use_gpu=True,
        verbose=True
    )
    
    video_path = "path/to/your/video.mp4"
    
    if Path(video_path).exists():
        # Just get the transcription result without saving
        result = transcriptor.transcribe_video(
            video_path=video_path,
            task="transcribe",  # or "translate" to translate to English
        )
        
        # Access transcription data
        print(f"\nDetected language: {result['language']}")
        print(f"Number of segments: {len(result['segments'])}")
        print("\nFirst few words:")
        print(result['text'][:200] + "...")
        
        # Access individual segments with timestamps
        print("\nFirst 3 segments:")
        for i, segment in enumerate(result['segments'][:3], 1):
            print(f"\n  Segment {i}:")
            print(f"    Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"    Text: {segment['text'].strip()}")
    else:
        print(f"Video file not found: {video_path}")


def example_translation():
    """Transcribe and translate to English."""
    
    print("\n" + "=" * 60)
    print("Example 4: Translate to English")
    print("=" * 60)
    
    transcriptor = VideoTranscriptor(
        model_size="base",
        use_gpu=True,
        verbose=True
    )
    
    video_path = "path/to/foreign_language_video.mp4"
    
    if Path(video_path).exists():
        saved_files = transcriptor.transcribe_and_save(
            video_path=video_path,
            output_dir="./transcriptions",
            output_format="all",
            task="translate",  # Translate to English
            include_timestamps=True
        )
        
        print("\nTranslated transcription saved!")
    else:
        print(f"Video file not found: {video_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Video Transcriptor - Example Usage")
    print("=" * 60)
    print("\nNote: Update the video paths in this file with your actual videos.")
    print("=" * 60)
    
    # Run examples (uncomment the ones you want to try)
    example_single_video()
    # example_batch_processing()
    # example_custom_settings()
    # example_translation()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")
