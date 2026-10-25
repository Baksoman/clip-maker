import subprocess
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
from app.config import Config

logger = logging.getLogger(__name__)

YUNET_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "face_detection_yunet_2023mar.onnx"

def _make_center_crop_layout(width: int, height: int) -> Dict[str, Any]:
    target_aspect = 9.0 / 16.0
    current_aspect = width / max(1, height)

    if current_aspect >= target_aspect:
        crop_h = height if height % 2 == 0 else height - 1
        crop_w = int(round(crop_h * target_aspect))
        crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
        crop_x = max(0, (width - crop_w) // 2)
        crop_x = crop_x if crop_x % 2 == 0 else crop_x - 1
        crop_y = 0
    else:
        crop_w = width if width % 2 == 0 else width - 1
        crop_h = int(round(crop_w / target_aspect))
        crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
        crop_x = 0
        crop_y = max(0, (height - crop_h) // 2)
        crop_y = crop_y if crop_y % 2 == 0 else crop_y - 1

    return {
        "mode": "CENTER_CROP",
        "filtergraph": f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920:flags=lanczos[cropped]",
        "crop_box": (crop_x, crop_y, crop_w, crop_h),
        "description": f"Tanpa wajah; crop tengah 9:16 ({crop_w}x{crop_h} at x={crop_x}, y={crop_y})."
    }

def _make_single_speaker_layout(width: int, height: int, median_x: float) -> Dict[str, Any]:
    target_aspect = 9.0 / 16.0
    current_aspect = width / max(1, height)

    if current_aspect >= target_aspect:
        crop_h = height if height % 2 == 0 else height - 1
        crop_w = int(round(crop_h * target_aspect))
        crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
        crop_x = max(0, min(int(round(median_x - crop_w / 2.0)), width - crop_w))
        crop_x = crop_x if crop_x % 2 == 0 else crop_x - 1
        crop_y = 0
    else:
        crop_w = width if width % 2 == 0 else width - 1
        crop_h = int(round(crop_w / target_aspect))
        crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
        crop_x = 0
        crop_y = max(0, (height - crop_h) // 2)
        crop_y = crop_y if crop_y % 2 == 0 else crop_y - 1

    return {
        "mode": "SINGLE_SPEAKER",
        "filtergraph": f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920:flags=lanczos[cropped]",
        "crop_box": (crop_x, crop_y, crop_w, crop_h),
        "description": f"Satu pembicara terdeteksi; smart-crop 9:16 fokus ke wajah (x={crop_x})."
    }

def _make_podcast_split_layout(width: int, height: int, left_cx: float, right_cx: float) -> Dict[str, Any]:
    # Each split tile in 1080x1920 is 1080x960 (aspect ratio 9:8 = 1.125)
    ratio = 9.0 / 8.0
    sp_h = height if height % 2 == 0 else height - 1
    sp_w = int(round(sp_h * ratio))
    if sp_w > width:
        sp_w = width if width % 2 == 0 else width - 1
        sp_h = int(round(sp_w / ratio))
    sp_w = sp_w if sp_w % 2 == 0 else sp_w - 1
    sp_h = sp_h if sp_h % 2 == 0 else sp_h - 1

    x1 = max(0, min(int(round(left_cx - sp_w / 2.0)), width - sp_w))
    x1 = x1 if x1 % 2 == 0 else x1 - 1
    x2 = max(0, min(int(round(right_cx - sp_w / 2.0)), width - sp_w))
    x2 = x2 if x2 % 2 == 0 else x2 - 1
    y1 = max(0, (height - sp_h) // 2)
    y1 = y1 if y1 % 2 == 0 else y1 - 1

    fg = (
        f"[0:v]split=2[sp1][sp2];"
        f"[sp1]crop={sp_w}:{sp_h}:{x1}:{y1},scale=1080:960:flags=lanczos[top];"
        f"[sp2]crop={sp_w}:{sp_h}:{x2}:{y1},scale=1080:960:flags=lanczos[bot];"
        f"[top][bot]vstack[cropped]"
    )

    return {
        "mode": "PODCAST_SPLIT",
        "filtergraph": fg,
        "crop_box": (x1, y1, sp_w, sp_h),
        "description": f"Dua pembicara podcast (kiri x={x1}, kanan x={x2}); split-screen atas-bawah (1080x960)."
    }

def _make_streaming_layout(width: int, height: int, cam_cx: float, cam_cy: float, face_w: float) -> Dict[str, Any]:
    # Top facecam: 1080x720 (ratio 1.5)
    cam_ratio = 1080.0 / 720.0
    cam_w = int(max(face_w * 2.4, width * 0.28))
    cam_h = int(round(cam_w / cam_ratio))
    if cam_h > height:
        cam_h = height
        cam_w = int(round(cam_h * cam_ratio))
    cam_w = cam_w if cam_w % 2 == 0 else cam_w - 1
    cam_h = cam_h if cam_h % 2 == 0 else cam_h - 1

    cam_x = max(0, min(int(round(cam_cx - cam_w / 2.0)), width - cam_w))
    cam_x = cam_x if cam_x % 2 == 0 else cam_x - 1
    cam_y = max(0, min(int(round(cam_cy - cam_h / 2.0)), height - cam_h))
    cam_y = cam_y if cam_y % 2 == 0 else cam_y - 1

    # Bottom gameplay: 1080x1200 (ratio 0.9)
    game_ratio = 1080.0 / 1200.0
    game_h = height if height % 2 == 0 else height - 1
    game_w = int(round(game_h * game_ratio))
    if game_w > width:
        game_w = width if width % 2 == 0 else width - 1
        game_h = int(round(game_w / game_ratio))
    game_w = game_w if game_w % 2 == 0 else game_w - 1
    game_h = game_h if game_h % 2 == 0 else game_h - 1

    game_x = max(0, (width - game_w) // 2)
    game_x = game_x if game_x % 2 == 0 else game_x - 1
    game_y = max(0, (height - game_h) // 2)
    game_y = game_y if game_y % 2 == 0 else game_y - 1

    fg = (
        f"[0:v]split=2[cam][game];"
        f"[cam]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},scale=1080:720:flags=lanczos[top];"
        f"[game]crop={game_w}:{game_h}:{game_x}:{game_y},scale=1080:1200:flags=lanczos[bot];"
        f"[top][bot]vstack[cropped]"
    )

    return {
        "mode": "STREAMING",
        "filtergraph": fg,
        "crop_box": (cam_x, cam_y, cam_w, cam_h),
        "description": f"Layout streaming (Facecam {cam_w}x{cam_h} atas, Gameplay {game_w}x{game_h} bawah)."
    }

def detect_video_layout(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_count: int = 5
) -> Dict[str, Any]:
    """
    Automatic heuristic layout detection based on speaker/face composition:
    1. Zero faces -> CENTER_CROP (9:16 center crop)
    2. Small face in corner -> STREAMING (Top: facecam 1080x720, Bottom: gameplay 1080x1200)
    3. Two separate faces (left & right) -> PODCAST_SPLIT (Top: Speaker 1, Bottom: Speaker 2, each 1080x960)
    4. One prominent face in center/side -> SINGLE_SPEAKER (Smart Crop 9:16 focused on speaker)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video {video_path}, fallback to CENTER_CROP.")
        return _make_center_crop_layout(1920, 1080)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    detector = None
    if YUNET_MODEL_PATH.exists():
        try:
            detector = cv2.FaceDetectorYN_create(
                str(YUNET_MODEL_PATH),
                "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            logger.warning(f"YuNet face detector init error: {e}")
            detector = None

    duration = max(1.0, end_time - start_time)
    sample_intervals = [start_time + duration * (i + 1) / (sample_count + 1) for i in range(sample_count)]

    frames_faces = []
    for t in sample_intervals:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None or not detector:
            continue

        try:
            det_w, det_h = 320, int(320 * (height / width))
            det_h = det_h if det_h % 2 == 0 else det_h + 1
            detector.setInputSize((det_w, det_h))
            resized = cv2.resize(frame, (det_w, det_h))
            _, faces = detector.detect(resized)

            current_frame_faces = []
            if faces is not None and len(faces) > 0:
                scale_x = width / det_w
                scale_y = height / det_h
                for f in faces:
                    fx, fy, fw, fh = float(f[0]) * scale_x, float(f[1]) * scale_y, float(f[2]) * scale_x, float(f[3]) * scale_y
                    current_frame_faces.append({
                        "x": fx, "y": fy, "w": fw, "h": fh,
                        "cx": fx + fw / 2.0, "cy": fy + fh / 2.0,
                        "area_ratio": (fw * fh) / (width * height),
                        "w_ratio": fw / width,
                        "h_ratio": fh / height
                    })
            if current_frame_faces:
                frames_faces.append(current_frame_faces)
        except Exception as e:
            logger.debug(f"Face detect error on frame: {e}")

    cap.release()

    all_faces = [f for f_list in frames_faces for f in f_list]

    # Heuristic 1: Nol Wajah -> Crop Tengah
    if not all_faces:
        return _make_center_crop_layout(width, height)

    # Heuristic 2: Wajah Kecil di Pojok -> Mode Streaming
    avg_w_ratio = sum(f["w_ratio"] for f in all_faces) / len(all_faces)
    avg_area_ratio = sum(f["area_ratio"] for f in all_faces) / len(all_faces)
    avg_cx = sum(f["cx"] for f in all_faces) / len(all_faces)
    avg_cy = sum(f["cy"] for f in all_faces) / len(all_faces)

    is_small_face = (avg_w_ratio < 0.20 and avg_area_ratio < 0.05)
    is_in_corner = (
        (avg_cx < 0.35 * width or avg_cx > 0.65 * width) and
        (avg_cy < 0.45 * height or avg_cy > 0.55 * height)
    )

    if is_small_face and is_in_corner:
        max_face_w = max(f["w"] for f in all_faces)
        return _make_streaming_layout(width, height, avg_cx, avg_cy, max_face_w)

    # Heuristic 3: Dua Wajah Terpisah Kiri & Kanan -> Podcast Split
    left_faces = [f for f in all_faces if f["cx"] < 0.48 * width and f["w_ratio"] >= 0.05]
    right_faces = [f for f in all_faces if f["cx"] > 0.52 * width and f["w_ratio"] >= 0.05]

    if left_faces and right_faces:
        left_cx = sorted(f["cx"] for f in left_faces)[len(left_faces) // 2]
        right_cx = sorted(f["cx"] for f in right_faces)[len(right_faces) // 2]
        separation = (right_cx - left_cx) / width
        if separation >= 0.22:
            return _make_podcast_split_layout(width, height, left_cx, right_cx)

    # Heuristic 4: Wajah Besar / Normal -> Single Speaker Smart Crop 9:16
    median_x = sorted(f["cx"] for f in all_faces)[len(all_faces) // 2]
    return _make_single_speaker_layout(width, height, median_x)

def detect_speaker_focus(video_path: str, start_time: float, end_time: float, sample_count: int = 5) -> Tuple[int, int, int, int]:
    """Compatibility helper returning single crop box tuple (x, y, w, h)."""
    layout = detect_video_layout(video_path, start_time, end_time, sample_count)
    return layout.get("crop_box", (0, 0, 1080, 1920))


def generate_ass_subtitles(words: List[Dict[str, Any]], clip_start_time: float, output_ass_path: str):
    """
    Generate ASS subtitle with word-by-word pop-up animation:
    - Text: Pure white (&H00FFFFFF)
    - Stroke: Solid black outline (&H00000000)
    - Pop-up animation: scale up to 125% then snap to 100% per word
    - Perfect sync: matches WhisperX word timestamps
    """
    Path(output_ass_path).parent.mkdir(parents=True, exist_ok=True)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: PopUp,Arial Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,460,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    for w in words:
        word_text = w.get("word", "").strip()
        if not word_text:
            continue

        raw_start = w.get("start", clip_start_time)
        raw_end = w.get("end", clip_start_time)

        rel_start = max(0.0, raw_start - clip_start_time)
        rel_end = max(rel_start + 0.15, raw_end - clip_start_time)

        start_str = _format_ass_timestamp(rel_start)
        end_str = _format_ass_timestamp(rel_end)

        # Viral word popup animation:
        # \\fscx125\\fscy125 (starts 125% zoomed) -> \\t(0,70,\\fscx100\\fscy100) (animates down to 100% in 70ms)
        popup_tag = "{\\fscx125\\fscy125\\t(0,70,\\fscx100\\fscy100)}"
        events.append(f"Dialogue: 0,{start_str},{end_str},PopUp,,0,0,0,,{popup_tag}{word_text}")

    content = header + "\n".join(events) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_ass_path

def _format_ass_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis = 0
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"

def render_vertical_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    subtitle_ass_path: str = None
) -> str:
    """
    Renders 9:16 vertical clip cropped to active speaker recognition box,
    scales to 1080x1920, and burns in word-by-word animated subtitles.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(1.0, end_time - start_time)

    # 1. Automatic heuristic layout detection
    layout = detect_video_layout(video_path, start_time, end_time)
    logger.info(f"Framing strategy: [{layout['mode']}] {layout['description']}")

    encoder = Config.VIDEO_ENCODER
    filter_complex = layout["filtergraph"]

    if subtitle_ass_path and os.path.exists(subtitle_ass_path):
        escaped_sub_path = str(Path(subtitle_ass_path).resolve()).replace("\\", "/").replace(":", "\\:")
        filter_complex += f";[cropped]subtitles='{escaped_sub_path}'[outv]"
        map_video = "[outv]"
    else:
        map_video = "[cropped]"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0.0, start_time)),
        "-t", str(duration),
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", map_video,
        "-map", "0:a?",
        "-sn", "-dn",
        "-map_metadata", "-1",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(output_path)
    ]

    if encoder == "libx264":
        cmd.extend(["-preset", "veryfast", "-crf", "22"])

    logger.info(f"Rendering 9:16 smart-crop clip: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        if encoder != "libx264":
            logger.warning(f"Encoder '{encoder}' failed, retrying with 'libx264'...")
            cmd[cmd.index(encoder)] = "libx264"
            cmd.extend(["-preset", "veryfast", "-crf", "23"])
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg render failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg render error: {result.stderr}")

    return output_path
