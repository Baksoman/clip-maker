from typing import TypedDict, List, Dict, Any, Optional

class HighlightCandidate(TypedDict):
    start_time: float
    end_time: float
    summary: str
    hook_reason: str
    words: Optional[List[Dict[str, Any]]]

class CriticEvaluation(TypedDict):
    is_acceptable: bool
    overall_score: float
    hook_score: float
    completeness_score: float
    critique: str
    suggestion: str
    needs_revision: bool

class ApprovedClip(TypedDict):
    clip_index: int
    start_time: float
    end_time: float
    duration: float
    summary: str
    hook_reason: str
    critic_score: float
    critic_feedback: str
    revision_count: int
    video_path: str
    subtitle_path: str
    title: str
    caption: str
    hashtags: str

class ClipSmithState(TypedDict, total=False):
    project_id: str
    video_path: str
    audio_path: str
    duration: float
    language: str

    # Transcript data
    transcript: Dict[str, Any]

    # Highlight finding
    candidate_highlights: List[Dict[str, Any]]
    current_index: int
    current_candidate: Dict[str, Any]

    # Editor output
    rendered_clip_path: str
    subtitle_ass_path: str

    # Critic evaluation
    critic_result: Dict[str, Any]
    revision_count: int

    # Final outputs
    approved_clips: List[Dict[str, Any]]
    status_message: str
    error: Optional[str]
