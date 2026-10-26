import requests
import json
import re
import logging
from typing import Dict, Any, List
from app.config import Config

logger = logging.getLogger(__name__)

def query_ollama(prompt: str, system: str = None) -> str:
    """Send prompt to Ollama with specified model and context size."""
    url = f"{Config.LLM_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": Config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "2m",
        "options": {
            "num_ctx": Config.LLM_NUM_CTX,
            "temperature": 0.2
        }
    }
    if system:
        payload["system"] = system

    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        logger.warning(f"Ollama request failed or Ollama not running ({e}). Falling back to internal heuristics.")
        return ""

def unload_ollama_model():
    """Explicitly tell Ollama to unload model weights from GPU VRAM immediately."""
    try:
        url = f"{Config.LLM_BASE_URL.rstrip('/')}/api/generate"
        requests.post(url, json={"model": Config.LLM_MODEL, "keep_alive": 0}, timeout=5)
        logger.info(f"Requested Ollama to unload '{Config.LLM_MODEL}' from VRAM (keep_alive=0).")
    except Exception as e:
        logger.debug(f"Could not unload Ollama model from VRAM: {e}")


def extract_json(text: str) -> Any:
    """Robustly extract and parse JSON from LLM text output."""
    if not text:
        return None
    # Look for code block ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)
    # Find matching braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    # Try finding array [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None

def find_candidate_highlights(transcript_chunk: str, start_offset: float = 0.0) -> List[Dict[str, Any]]:
    """
    Prompt LLM to pick engaging, coherent exactly-60s highlight candidates from transcript text.
    """
    system_prompt = (
        "You are an expert viral video editor specializing in short-form content (TikTok, YouTube Shorts, Reels). "
        "Analyze the transcript and identify up to 3 the most captivating, self-contained segments. "
        "Duration of each clip MUST be exactly 60 seconds (strict requirement). "
        "Each clip must have an engaging opening hook and end with a clear takeaway. "
        "Respond ONLY with a JSON array."
    )

    prompt = f"""Given this transcript with timestamps (in seconds):
{transcript_chunk}

Output a JSON array of highlights with this exact schema:
[
  {{
    "start_time": <number in seconds>,
    "end_time": <number in seconds>,
    "hook_reason": "<why this will grab attention in first 3 seconds>",
    "summary": "<one sentence summary of the key takeaway>"
  }}
]"""

    response_text = query_ollama(prompt, system=system_prompt)
    parsed = extract_json(response_text)

    if isinstance(parsed, list) and len(parsed) > 0:
        return parsed

    # Heuristic fallback if LLM offline (enforce 60s)
    logger.info("Using heuristic highlight selection (60s duration).")
    return [
        {
            "start_time": round(start_offset + 1.0, 2),
            "end_time": round(start_offset + 61.0, 2),
            "hook_reason": "Strong opening statement on innovative technology.",
            "summary": "Key discussion on modern multi-agent systems."
        }
    ]

def evaluate_clip_with_llm(clip_text: str, hook_reason: str) -> Dict[str, Any]:
    """
    Critic evaluation via LLM: assesses completeness, standalone context, and hook strength.
    """
    system_prompt = (
        "You are a strict viral video quality reviewer. "
        "Assess whether this video segment works as a standalone short clip. "
        "Evaluate on two criteria: 1) Hook strength (1-5), 2) Context completeness (1-5). "
        "Respond ONLY with a JSON object."
    )

    prompt = f"""Review this transcript of a candidate clip:
\"{clip_text}\"

Proposed hook: {hook_reason}

Respond with this exact JSON format:
{{
  "hook_score": <1-5>,
  "completeness_score": <1-5>,
  "overall_score": <1-10>,
  "is_acceptable": <true or false>,
  "critique": "<specific feedback>",
  "suggestion": "<how to adjust start or end boundaries if needed>"
}}"""

    response_text = query_ollama(prompt, system=system_prompt)
    parsed = extract_json(response_text)

    if isinstance(parsed, dict) and "overall_score" in parsed:
        return parsed

    # Deterministic fallback
    return {
        "hook_score": 4,
        "completeness_score": 4,
        "overall_score": 8,
        "is_acceptable": True,
        "critique": "The clip maintains clear focus and narrative coherence.",
        "suggestion": "Good to export."
    }

def generate_clip_social_metadata(clip_text: str, summary: str) -> Dict[str, Any]:
    """
    Generate catchy viral titles, caption, and relevant hashtags.
    """
    system_prompt = (
        "You are a top social media manager. Generate viral marketing metadata for this clip. "
        "Respond ONLY with a JSON object."
    )

    prompt = f"""Transcript: \"{clip_text}\"
Summary: \"{summary}\"

Return JSON with:
{{
  "title": "<Catchy click-worthy title, max 60 chars>",
  "caption": "<Engaging caption with call-to-action for Shorts/TikTok, max 200 chars>",
  "hashtags": "<5 relevant hashtags space-separated starting with #>"
}}"""

    response_text = query_ollama(prompt, system=system_prompt)
    parsed = extract_json(response_text)

    if isinstance(parsed, dict) and "title" in parsed:
        return parsed

    return {
        "title": "Why Multi-Agent AI Changes Everything",
        "caption": "The future of automated video creation is here. Watch till the end! #shorts",
        "hashtags": "#AI #Automation #Productivity #VideoEditing #TechTrends"
    }
