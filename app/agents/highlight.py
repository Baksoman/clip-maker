import logging
from typing import List, Dict, Any
from app.services.llm import find_candidate_highlights

logger = logging.getLogger(__name__)

def find_highlights_from_transcript(transcript: Dict[str, Any], max_clips: int = 20) -> List[Dict[str, Any]]:
    """
    Chunk transcript into ~10 minute blocks, find candidate highlights via LLM (max 20 clips, 60s each),
    and remove overlaps.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return []

    # Chunk segments by ~10 minutes (600 seconds)
    chunks = []
    current_chunk = []
    chunk_start_time = 0.0

    for seg in segments:
        current_chunk.append(seg)
        seg_end = seg.get("end", 0.0)
        if seg_end - chunk_start_time >= 600.0:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start_time = seg_end

    if current_chunk:
        chunks.append(current_chunk)

    all_candidates = []
    for chunk in chunks:
        chunk_text = "\n".join([f"[{s.get('start', 0):.1f}s - {s.get('end', 0):.1f}s] {s.get('text', '')}" for s in chunk])
        start_offset = chunk[0].get("start", 0.0)
        candidates = find_candidate_highlights(chunk_text, start_offset=start_offset)
        all_candidates.extend(candidates)

    # Validate duration (must be between 50 and 90 seconds) & deduplicate overlaps
    deduped = []
    for cand in all_candidates:
        c_start = float(cand.get("start_time", 0.0))
        c_end = float(cand.get("end_time", 0.0))
        dur = c_end - c_start

        if dur < 58.0 or dur > 62.0:
            c_end = c_start + 60.0  # Clamp to exactly 60s
            cand["end_time"] = round(c_end, 2)

        # Check if overlaps with existing selected candidate
        overlap = False
        for existing in deduped:
            e_start = existing.get("start_time", 0.0)
            e_end = existing.get("end_time", 0.0)
            # Overlap condition: start is before existing end and end is after existing start
            if max(c_start, e_start) < min(c_end, e_end):
                overlap = True
                break
        if not overlap:
            deduped.append(cand)
        if len(deduped) >= max_clips:
            break

    return deduped
