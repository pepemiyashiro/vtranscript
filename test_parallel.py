#!/usr/bin/env python3
"""
Test script for parallel transcription feature.
This demonstrates the new parallel processing capability.

Usage:
    python test_parallel.py video.mp4

Note: Requires installation first:
    pip install -e .
"""

import sys
import time
from pathlib import Path

# Try importing the package
try:
    from src.transcriptor import VideoTranscriptor
except ImportError:
    print("❌ Error: Package not installed or dependencies missing")
    print("\nPlease install first:")
    print("  1. source venv/bin/activate  # Activate virtual environment")
    print("  2. pip install -r requirements.txt")
    print("  3. pip install -e .")
    sys.exit(1)


def test_parallel_transcription(video_path: str):
    """Test parallel vs sequential transcription."""
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("PARALLEL TRANSCRIPTION TEST")
    print("=" * 70)
    
    # Test 1: Sequential (original)
    print("\n📊 TEST 1: Sequential Processing (Original)")
    print("-" * 70)
    transcriptor_seq = VideoTranscriptor(
        model_size="base",
        language="en",
        use_gpu=True,
        beam_size=1,  # Fast mode
        use_parallel=False,  # Sequential
        verbose=True
    )
    
    start_time = time.time()
    result_seq = transcriptor_seq.transcribe_video(video_path)
    seq_time = time.time() - start_time
    
    print(f"\n⏱️  Sequential Time: {seq_time:.2f} seconds")
    print(f"📝 Text length: {len(result_seq['text'])} characters")
    print(f"🎯 Segments: {len(result_seq['segments'])}")
    
    # Test 2: Parallel (new feature)
    print("\n" + "=" * 70)
    print("📊 TEST 2: Parallel Processing (New Feature)")
    print("-" * 70)
    transcriptor_par = VideoTranscriptor(
        model_size="base",
        language="en",
        use_gpu=True,
        beam_size=1,  # Fast mode
        use_parallel=True,  # Parallel!
        num_workers=None,  # Auto-detect (7 workers on 8-core Mac)
        verbose=True
    )
    
    start_time = time.time()
    result_par = transcriptor_par.transcribe_video(video_path)
    par_time = time.time() - start_time
    
    print(f"\n⏱️  Parallel Time: {par_time:.2f} seconds")
    print(f"📝 Text length: {len(result_par['text'])} characters")
    print(f"🎯 Segments: {len(result_par['segments'])}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {seq_time/par_time:.2f}x faster")
    print(f"Time saved: {seq_time - par_time:.2f}s ({(1 - par_time/seq_time)*100:.1f}% reduction)")
    
    # Verify accuracy
    print("\n" + "=" * 70)
    print("✅ ACCURACY VERIFICATION")
    print("=" * 70)
    print(f"Text length difference: {abs(len(result_seq['text']) - len(result_par['text']))} chars")
    print(f"Segment count difference: {abs(len(result_seq['segments']) - len(result_par['segments']))}")
    
    # Show first few segments
    print("\n📄 First 3 segments (Sequential):")
    for i, seg in enumerate(result_seq['segments'][:3]):
        print(f"  [{i+1}] {seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text'][:50]}...")
    
    print("\n📄 First 3 segments (Parallel):")
    for i, seg in enumerate(result_par['segments'][:3]):
        print(f"  [{i+1}] {seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text'][:50]}...")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_parallel.py VIDEO_PATH")
        print("\nExample:")
        print("  python test_parallel.py video.mp4")
        print("\nYou can also use the CLI directly:")
        print("  vtranscribe transcribe video.mp4 --parallel")
        print("  vtranscribe transcribe video.mp4 --parallel --workers 4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_parallel_transcription(video_path)


if __name__ == "__main__":
    main()
