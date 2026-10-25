import subprocess
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extract 16kHz mono WAV audio from a video using FFmpeg.
    This format is optimized for Whisper/WhisperX models.
    """
    Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_audio_path)
    ]
    logger.info(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg audio extraction error: {result.stderr}")
    return output_audio_path

def get_video_duration(video_path: str) -> float:
    """Get duration of video in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not determine video duration via ffprobe: {e}")
    return 0.0
