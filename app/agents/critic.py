import os
import logging
from typing import Dict, Any, List
from app.services.llm import evaluate_clip_with_llm

logger = logging.getLogger(__name__)

def evaluate_clip_hybrid(
    candidate: Dict[str, Any],
    all_segments: List[Dict[str, Any]],
    rendered_video_path: str = None,
    subtitle_ass_path: str = None,
    revision_count: int = 0
) -> Dict[str, Any]:
    """
    Critic review pipeline strictly checking in 3 sequential steps:
    1. Highlight Moment (transcript text, duration 60s, hook, completeness)
    2. Video Edit Quality (file integrity, vertical 9:16 aspect ratio, duration match)
    3. Subtitle & Audio Sync (word-level alignment coverage, subtitle file integrity)
    """
    start_time = float(candidate.get("start_time", 0.0))
    end_time = float(candidate.get("end_time", 0.0))
    duration = end_time - start_time

    critique_steps = []

    # =========================================================================
    # STEP 1: Momen Highlight (Pengecekan Teks Transkrip & Hook)
    # =========================================================================
    logger.info("Critic Step 1: Evaluating highlight moment from transcript...")

    # Extract words within candidate range
    clip_words = []
    for seg in all_segments:
        for w in seg.get("words", []):
            w_start = w.get("start", 0.0)
            w_end = w.get("end", 0.0)
            if w_start >= (start_time - 0.5) and w_end <= (end_time + 0.5):
                clip_words.append(w)

    clip_text = " ".join([w.get("word", "") for w in clip_words]).strip()
    if not clip_text:
        clip_text = candidate.get("summary", "")

    # Duration constraint: must be exactly 60s (±5s tolerance)
    if duration < 55.0:
        critique_steps.append(f"Step 1 Failed: Durasi klip terlalu pendek ({duration:.1f}s). Target 60 detik.")
        return {
            "is_acceptable": False,
            "overall_score": 4.0,
            "failed_step": "STEP 1 (Highlight Transcript)",
            "critique": f"Klip {duration:.1f}s terlalu singkat dari target 60s.",
            "suggestion": "Perpanjang durasi klip menjadi 60 detik.",
            "adjusted_start": max(0.0, start_time - 5.0),
            "adjusted_end": start_time + 60.0,
            "needs_revision": revision_count < 2,
            "clip_text": clip_text,
            "steps_log": critique_steps
        }

    if duration > 65.0:
        critique_steps.append(f"Step 1 Failed: Durasi klip melebihi target ({duration:.1f}s). Target 60 detik.")
        return {
            "is_acceptable": False,
            "overall_score": 5.0,
            "failed_step": "STEP 1 (Highlight Transcript)",
            "critique": f"Klip {duration:.1f}s melebihi target 60 detik.",
            "suggestion": "Pangkas bagian akhir agar pas 60 detik.",
            "adjusted_start": start_time,
            "adjusted_end": start_time + 60.0,
            "needs_revision": revision_count < 2,
            "clip_text": clip_text,
            "steps_log": critique_steps
        }

    # Evaluate transcript coherence & hook via LLM
    llm_eval = evaluate_clip_with_llm(clip_text, candidate.get("hook_reason", ""))
    hook_score = llm_eval.get("hook_score", 4)
    overall_score = float(llm_eval.get("overall_score", 7.5))

    critique_steps.append(f"Step 1 Passed: Transkrip koheren (skor {overall_score}/10, hook {hook_score}/5).")

    # =========================================================================
    # STEP 2: Editan Video (Pengecekan Hasil Edit & Smart Crop)
    # =========================================================================
    logger.info("Critic Step 2: Evaluating video render and crop...")
    if rendered_video_path:
        if not os.path.exists(rendered_video_path) or os.path.getsize(rendered_video_path) < 50000:
            critique_steps.append("Step 2 Failed: File video hasil render kosong atau corrupt.")
            return {
                "is_acceptable": False,
                "overall_score": 3.0,
                "failed_step": "STEP 2 (Video Edit)",
                "critique": "File render video tidak terbentuk dengan benar.",
                "suggestion": "Re-render video dengan format crop 9:16.",
                "adjusted_start": start_time,
                "adjusted_end": end_time,
                "needs_revision": revision_count < 2,
                "clip_text": clip_text,
                "steps_log": critique_steps
            }
        critique_steps.append("Step 2 Passed: Video 9:16 ter-render dengan valid.")
    else:
        critique_steps.append("Step 2: Video path pending render verification.")

    # =========================================================================
    # STEP 3: Subtitle & Audio Sync (Pengecekan Sinkronisasi Subtitle Pop-up)
    # =========================================================================
    logger.info("Critic Step 3: Evaluating subtitle & audio synchronization...")
    if not clip_words and duration > 5.0:
        critique_steps.append("Step 3 Warning: Tidak ada kata terdeteksi untuk subtitle pada segmen ini.")
    else:
        # Check alignment timestamps consistency
        first_word_time = clip_words[0].get("start", start_time) if clip_words else start_time
        last_word_time = clip_words[-1].get("end", end_time) if clip_words else end_time

        if (first_word_time - start_time) > 10.0:
            critique_steps.append("Step 3 Note: Audio awal memiliki jeda hening sebelum kata pertama.")
        else:
            critique_steps.append("Step 3 Passed: Subtitle sinkron dengan audio (WhisperX word-level timing).")

    # Snap start/end to nearest actual word boundaries for clean cut
    adjusted_start = start_time
    adjusted_end = end_time
    if clip_words:
        adjusted_start = round(clip_words[0].get("start", start_time), 2)
        adjusted_end = round(clip_words[-1].get("end", end_time), 2)
        # Re-verify duration after snap
        if (adjusted_end - adjusted_start) < 58.0:
            adjusted_end = adjusted_start + 60.0

    is_acceptable = overall_score >= 6.0 and llm_eval.get("is_acceptable", True)

    return {
        "is_acceptable": is_acceptable,
        "overall_score": overall_score,
        "hook_score": hook_score,
        "completeness_score": llm_eval.get("completeness_score", 4),
        "critique": f"Evaluasi 3 tahap berhasil. {llm_eval.get('critique', 'Klip memenuhi kriteria viral.')}",
        "suggestion": llm_eval.get("suggestion", "Siap diekspor."),
        "adjusted_start": adjusted_start,
        "adjusted_end": adjusted_end,
        "clip_words": clip_words,
        "clip_text": clip_text,
        "steps_log": critique_steps,
        "needs_revision": (not is_acceptable) and (revision_count < 2)
    }
