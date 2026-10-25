import json
import logging
from typing import Dict, Any, List
from app.core.gpu import gpu_lock, clean_memory, get_device_and_compute_type
from app.config import Config

logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, language: str = "en") -> Dict[str, Any]:
    """
    Transcribe audio with word-level timestamps using WhisperX.
    Uses gpu_lock to ensure exclusive GPU usage and unloads model after completion.
    """
    device, compute_type = get_device_and_compute_type(Config.DEVICE, Config.COMPUTE_TYPE)
    model_name = Config.WHISPER_MODEL

    logger.info(f"Starting transcription on {device} ({compute_type}) using model '{model_name}'...")

    with gpu_lock:
        try:
            import whisperx
            # 1. Load WhisperX model
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                language=language
            )

            # 2. Transcribe audio
            audio = whisperx.load_audio(audio_path)
            raw_result = model.transcribe(audio, batch_size=16, language=language)

            # Free the transcription model before loading the alignment model
            del model
            clean_memory()

            # 3. Align whisper output to get precise word-level timestamps
            align_model, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            aligned_result = whisperx.align(
                raw_result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )

            del align_model
            clean_memory()

            full_text = " ".join([seg.get("text", "").strip() for seg in aligned_result["segments"]])

            return {
                "language": language,
                "full_text": full_text,
                "segments": aligned_result["segments"]
            }

        except ImportError:
            logger.warning("WhisperX is not installed. Generating structured mock transcript for testing pipeline.")
            return _generate_mock_transcript(audio_path, language)
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise e
        finally:
            clean_memory()

def _generate_mock_transcript(audio_path: str, language: str) -> Dict[str, Any]:
    """
    Fallback mock generator with word timestamps for development and testing
    when WhisperX / CUDA weights are being downloaded or configured.
    """
    sample_sentences = [
        "Welcome to this episode where we explore the future of artificial intelligence.",
        "The most critical breakthrough in recent years is how multi-agent architectures work together.",
        "Instead of relying on a single prompt, you have specialized agents evaluating and refining outputs.",
        "This completely changes productivity and allows long video content to be repurposed effortlessly.",
        "Thank you for watching and make sure to apply these strategies in your next project."
    ]

    segments = []
    current_time = 0.5
    for i, sentence in enumerate(sample_sentences):
        words = sentence.split()
        word_list = []
        seg_start = current_time
        for w in words:
            w_duration = 0.35 + (len(w) * 0.03)
            word_list.append({
                "word": w,
                "start": round(current_time, 2),
                "end": round(current_time + w_duration, 2),
                "score": 0.95
            })
            current_time += w_duration + 0.05
        seg_end = current_time
        segments.append({
            "id": i,
            "seek": int(seg_start * 100),
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": sentence,
            "words": word_list
        })
        current_time += 0.8  # pause between sentences

    full_text = " ".join(sample_sentences)
    return {
        "language": language,
        "full_text": full_text,
        "segments": segments
    }
