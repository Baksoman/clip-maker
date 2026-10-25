import os
import json
import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from app.agents.state import ClipSmithState
from app.services.audio import extract_audio, get_video_duration
from app.services.transcriber import transcribe_audio
from app.services.video import generate_ass_subtitles, render_vertical_clip
from app.services.llm import generate_clip_social_metadata
from app.agents.highlight import find_highlights_from_transcript
from app.agents.critic import evaluate_clip_hybrid
from app.config import Config

logger = logging.getLogger(__name__)

# --- Graph Nodes ---

def node_transcribe(state: ClipSmithState) -> Dict[str, Any]:
    """Node 1: Extract audio to temp and transcribe using WhisperX."""
    video_path = state["video_path"]
    project_id = state["project_id"]
    audio_path = os.path.join(Config.STORAGE_TEMP, f"{project_id}_audio.wav")

    logger.info(f"[{project_id}] Extracting audio to temp...")
    extract_audio(video_path, audio_path)
    duration = get_video_duration(video_path)

    logger.info(f"[{project_id}] Transcribing audio with word alignment...")
    transcript_data = transcribe_audio(audio_path, language=state.get("language", "en"))

    # Save full transcript into project results folder
    proj_name = state.get("project_title", project_id)
    safe_folder = "".join([c for c in proj_name if c.isalnum() or c in (' ', '-', '_')]).strip() or project_id
    proj_dir = os.path.join(Config.STORAGE_RESULTS, safe_folder)
    os.makedirs(proj_dir, exist_ok=True)
    try:
        with open(os.path.join(proj_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript_data.get("full_text", ""))
    except Exception as e:
        logger.warning(f"Could not save project transcript: {e}")

    return {
        "audio_path": audio_path,
        "duration": duration,
        "transcript": transcript_data,
        "status_message": "Transcription complete."
    }

def node_find_highlights(state: ClipSmithState) -> Dict[str, Any]:
    """Node 2: Identify top candidate segments."""
    transcript = state.get("transcript", {})
    project_id = state["project_id"]
    logger.info(f"[{project_id}] Finding engaging highlights from transcript...")

    candidates = find_highlights_from_transcript(transcript, max_clips=20)
    if not candidates:
        logger.warning(f"[{project_id}] No candidate highlights found, creating fallback segment.")
        candidates = [{
            "start_time": 0.0,
            "end_time": min(state.get("duration", 60.0), 60.0),
            "summary": "Introduction clip",
            "hook_reason": "Opening video hook"
        }]

    return {
        "candidate_highlights": candidates,
        "current_index": 0,
        "current_candidate": candidates[0],
        "revision_count": 0,
        "approved_clips": [],
        "status_message": f"Found {len(candidates)} highlight candidates."
    }

def node_editor(state: ClipSmithState) -> Dict[str, Any]:
    """Node 3: Cut clip, generate ASS subtitles, and render 9:16 vertical video into results/project/clip_x/."""
    project_id = state["project_id"]
    candidate = state["current_candidate"]
    idx = state.get("current_index", 0) + 1
    rev = state.get("revision_count", 0)

    start_t = float(candidate["start_time"])
    end_t = float(candidate["end_time"])

    logger.info(f"[{project_id}] Editing clip #{idx} (rev {rev}): {start_t:.1f}s to {end_t:.1f}s")

    # Gather words for this time segment
    segments = state.get("transcript", {}).get("segments", [])
    words_in_range = []
    for seg in segments:
        for w in seg.get("words", []):
            if w.get("start", 0) >= (start_t - 0.2) and w.get("end", 0) <= (end_t + 0.2):
                words_in_range.append(w)

    # Destination folder: storage/results/<project_title>/clip_<idx>/
    proj_name = state.get("project_title", project_id)
    safe_folder = "".join([c for c in proj_name if c.isalnum() or c in (' ', '-', '_')]).strip() or project_id
    clip_dir = os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{idx}")
    os.makedirs(clip_dir, exist_ok=True)

    sub_path = os.path.join(clip_dir, "subtitle.ass")
    out_video = os.path.join(clip_dir, "video.mp4")

    if words_in_range:
        generate_ass_subtitles(words_in_range, start_t, sub_path)
    else:
        sub_path = None

    render_vertical_clip(
        video_path=state["video_path"],
        start_time=start_t,
        end_time=end_t,
        output_path=out_video,
        subtitle_ass_path=sub_path
    )

    return {
        "clip_index": idx,
        "rendered_clip_path": out_video,
        "subtitle_ass_path": sub_path or "",
        "status_message": f"Rendered clip #{idx}."
    }


def node_critic(state: ClipSmithState) -> Dict[str, Any]:
    """Node 4: Hybrid Critic reviews boundaries, video render, and subtitle sync."""
    project_id = state["project_id"]
    candidate = state["current_candidate"]
    idx = state.get("current_index", 0) + 1
    rev = state.get("revision_count", 0)
    segments = state.get("transcript", {}).get("segments", [])

    logger.info(f"[{project_id}] Critic evaluating clip #{idx} (rev {rev})...")
    critic_result = evaluate_clip_hybrid(
        candidate,
        segments,
        rendered_video_path=state.get("rendered_clip_path"),
        subtitle_ass_path=state.get("subtitle_ass_path"),
        revision_count=rev
    )

    return {
        "critic_result": critic_result,
        "status_message": f"Critic score: {critic_result.get('overall_score', 0)}/10"
    }

def node_apply_revision(state: ClipSmithState) -> Dict[str, Any]:
    """Adjust start/end boundaries based on Critic's feedback and re-run editor."""
    critic_res = state["critic_result"]
    current = state["current_candidate"].copy()
    current["start_time"] = critic_res.get("adjusted_start", current["start_time"])
    current["end_time"] = critic_res.get("adjusted_end", current["end_time"])

    logger.info(f"Applying Critic revision: new range {current['start_time']} - {current['end_time']}")
    return {
        "current_candidate": current,
        "revision_count": state.get("revision_count", 0) + 1,
        "status_message": "Critic requested boundary adjustment. Re-editing clip..."
    }

def node_metadata(state: ClipSmithState) -> Dict[str, Any]:
    """Node 5: Generate viral titles, captions, and save metadata.json + transcript.txt for approved clip."""
    project_id = state["project_id"]
    candidate = state["current_candidate"]
    critic_res = state["critic_result"]
    idx = state.get("current_index", 0) + 1

    logger.info(f"[{project_id}] Generating social metadata for approved clip #{idx}...")
    clip_text = critic_res.get("clip_text", candidate.get("summary", ""))
    meta = generate_clip_social_metadata(clip_text, candidate.get("summary", ""))

    out_video = state["rendered_clip_path"]
    clip_dir = os.path.dirname(out_video)

    # Save transcript.txt in clip folder
    try:
        with open(os.path.join(clip_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(clip_text)
    except Exception as e:
        logger.warning(f"Could not save clip transcript.txt: {e}")

    # Save metadata.json in clip folder
    meta_payload = {
        "title": meta.get("title", f"Clip #{idx}"),
        "caption": meta.get("caption", ""),
        "hashtags": meta.get("hashtags", "#Shorts #AI"),
        "clip_index": idx,
        "start_time": candidate["start_time"],
        "end_time": candidate["end_time"],
        "duration": round(candidate["end_time"] - candidate["start_time"], 2),
        "critic_score": critic_res.get("overall_score", 0.0),
        "hook_reason": candidate.get("hook_reason", ""),
        "summary": candidate.get("summary", "")
    }
    try:
        with open(os.path.join(clip_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not save clip metadata.json: {e}")

    approved_clip = {
        "clip_index": idx,
        "start_time": candidate["start_time"],
        "end_time": candidate["end_time"],
        "duration": candidate["end_time"] - candidate["start_time"],
        "summary": candidate.get("summary", ""),
        "hook_reason": candidate.get("hook_reason", ""),
        "critic_score": critic_res.get("overall_score", 0.0),
        "critic_feedback": json.dumps(critic_res),
        "revision_count": state.get("revision_count", 0),
        "video_path": out_video,
        "subtitle_path": state.get("subtitle_ass_path", ""),
        "title": meta.get("title", f"Clip #{idx}"),
        "caption": meta.get("caption", ""),
        "hashtags": meta.get("hashtags", "#Shorts #AI")
    }

    updated_approved = list(state.get("approved_clips", []))
    updated_approved.append(approved_clip)

    return {
        "approved_clips": updated_approved,
        "status_message": f"Clip #{idx} approved and packaged."
    }

def node_next_candidate(state: ClipSmithState) -> Dict[str, Any]:
    """Move to next candidate in the queue."""
    next_idx = state.get("current_index", 0) + 1
    candidates = state.get("candidate_highlights", [])
    if next_idx < len(candidates):
        return {
            "current_index": next_idx,
            "current_candidate": candidates[next_idx],
            "revision_count": 0,
            "status_message": f"Processing next candidate ({next_idx + 1}/{len(candidates)})..."
        }
    return {
        "current_index": next_idx,
        "status_message": "All candidates processed."
    }

# --- Routing Logic ---

def route_after_critic(state: ClipSmithState) -> Literal["apply_revision", "metadata", "next_candidate"]:
    critic = state.get("critic_result", {})
    if critic.get("needs_revision", False):
        return "apply_revision"
    elif critic.get("is_acceptable", False):
        return "metadata"
    else:
        # Rejected after max revisions
        return "next_candidate"

def route_next_or_end(state: ClipSmithState) -> Literal["editor", "__end__"]:
    curr = state.get("current_index", 0)
    total = len(state.get("candidate_highlights", []))
    if curr < total:
        return "editor"
    return END

# --- Graph Assembly ---

def build_clipsmith_graph():
    builder = StateGraph(ClipSmithState)

    builder.add_node("transcriber", node_transcribe)
    builder.add_node("highlighter", node_find_highlights)
    builder.add_node("editor", node_editor)
    builder.add_node("critic", node_critic)
    builder.add_node("apply_revision", node_apply_revision)
    builder.add_node("metadata", node_metadata)
    builder.add_node("next_candidate", node_next_candidate)

    builder.set_entry_point("transcriber")
    builder.add_edge("transcriber", "highlighter")
    builder.add_edge("highlighter", "editor")
    builder.add_edge("editor", "critic")

    # Conditional branch from Critic
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "apply_revision": "apply_revision",
            "metadata": "metadata",
            "next_candidate": "next_candidate"
        }
    )

    # Revision loops back to Editor
    builder.add_edge("apply_revision", "editor")

    # Metadata moves to next candidate
    builder.add_edge("metadata", "next_candidate")

    # Next candidate loops or ends
    builder.add_conditional_edges(
        "next_candidate",
        route_next_or_end,
        {
            "editor": "editor",
            END: END
        }
    )

    return builder.compile()
