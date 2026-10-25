import os
import io
import zipfile
import json
import time
import logging
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
from app.db import db
from app.models import Project, Job, Transcript, Clip
from app.config import Config

logger = logging.getLogger(__name__)

web_bp = Blueprint("web", __name__)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@web_bp.route("/")
def index():
    recent_projects = Project.query.order_by(Project.created_at.desc()).limit(8).all()
    return render_template("index.html", projects=recent_projects)

@web_bp.route("/clips")
def view_clips():
    """Clips gallery page: lists all projects as cards and inspects clips when selected."""
    selected_project_id = request.args.get("project")
    projects = Project.query.order_by(Project.created_at.desc()).all()
    selected_project = None
    clips = []
    
    if selected_project_id:
        selected_project = Project.query.get(selected_project_id)
        if selected_project:
            # Fix 0 clips bug: retrieve all clips generated for this project
            clips = Clip.query.filter_by(project_id=selected_project.id).order_by(Clip.clip_index.asc()).all()

    return render_template(
        "clips.html",
        projects=projects,
        selected_project=selected_project,
        clips=clips
    )

@web_bp.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Format video tidak valid. Gunakan MP4, MOV, AVI, atau MKV"}), 400

    title = request.form.get("title", "").strip() or file.filename.rsplit(".", 1)[0]
    safe_name = secure_filename(file.filename)

    # Save to storage temp as requested in requirement #1
    project = Project(
        title=title,
        original_filename=safe_name,
        video_path="",
        status="QUEUED"
    )
    db.session.add(project)
    db.session.flush()

    save_filename = f"{project.id}_{safe_name}"
    save_path = os.path.join(Config.STORAGE_TEMP, save_filename)
    file.save(save_path)

    project.video_path = save_path

    # Create Job
    job = Job(
        project_id=project.id,
        step="PENDING",
        progress=0,
        message="Menunggu antrean worker..."
    )
    db.session.add(job)
    db.session.commit()

    return redirect(url_for("web.view_job", project_id=project.id))

@web_bp.route("/projects/<project_id>")
def view_job(project_id):
    project = Project.query.get_or_404(project_id)
    job = Job.query.filter_by(project_id=project.id).order_by(Job.id.desc()).first()
    return render_template("job.html", project=project, job=job)

@web_bp.route("/projects/<project_id>/results")
def view_results(project_id):
    """Redirect to the unified Clips gallery with project selected."""
    return redirect(url_for("web.view_clips", project=project_id))

@web_bp.route("/api/projects/<project_id>/status")
def project_status(project_id):
    db.session.expire_all()
    job = Job.query.filter_by(project_id=project_id).order_by(Job.id.desc()).first()
    if not job:
        return jsonify({"error": "Job not found"}), 404
    resp = jsonify(job.to_dict())
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@web_bp.route("/api/projects/<project_id>/stream")
def status_stream(project_id):
    """Server-Sent Events (SSE) stream for real-time live progress updates."""
    def event_stream():
        last_step = None
        last_progress = None
        last_message = None
        last_detail_len = 0
        while True:
            db.session.expire_all()
            job = Job.query.filter_by(project_id=project_id).order_by(Job.id.desc()).first()
            if job:
                cur_detail_len = len(job.status_detail or "")
                if job.step != last_step or job.progress != last_progress or job.message != last_message or cur_detail_len != last_detail_len:
                    last_step = job.step
                    last_progress = job.progress
                    last_message = job.message
                    last_detail_len = cur_detail_len
                    data = json.dumps(job.to_dict())
                    yield f"data: {data}\n\n"
                if job.step in ("DONE", "ERROR"):
                    break
            time.sleep(1)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no"
        }
    )

@web_bp.route("/api/projects/<project_id>/clips/<int:clip_index>/video")
def get_project_clip_video(project_id, clip_index):
    """Serve clip video directly by project_id and clip_index (supports live pipeline preview before DB insert)."""
    # 1. Check database record
    clip = Clip.query.filter_by(project_id=project_id, clip_index=clip_index).first()
    if clip and clip.video_path and os.path.exists(clip.video_path):
        return send_file(clip.video_path, mimetype="video/mp4", conditional=True)

    # 2. Check storage/results/ folder directly on disk (both safe_folder and project.id)
    project = Project.query.get(project_id)
    if project:
        safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
        candidate_paths = [
            os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{clip_index}", "video.mp4"),
            os.path.join(Config.STORAGE_RESULTS, project.id, f"clip_{clip_index}", "video.mp4"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return send_file(p, mimetype="video/mp4", conditional=True)

        # 3. Check active job status_detail for exact video_path
        job = Job.query.filter_by(project_id=project.id).order_by(Job.id.desc()).first()
        if job and job.status_detail:
            try:
                sdata = json.loads(job.status_detail)
                for ec in sdata.get("edited_clips", []):
                    if (ec.get("index") == clip_index or ec.get("video_url", "").endswith(f"/{clip_index}/video")) and ec.get("video_path") and os.path.exists(ec.get("video_path")):
                        return send_file(ec.get("video_path"), mimetype="video/mp4", conditional=True)
            except Exception:
                pass

    return "Video not found", 404

@web_bp.route("/api/projects/<project_id>/clips/<int:clip_index>/download")
def download_project_clip(project_id, clip_index):
    """Download individual clip directly by project_id and clip_index."""
    clip = Clip.query.filter_by(project_id=project_id, clip_index=clip_index).first()
    if clip and clip.video_path and os.path.exists(clip.video_path):
        safe_title = secure_filename(clip.title or f"clip_{clip_index}")
        if not safe_title.endswith(".mp4"):
            safe_title = f"{safe_title}.mp4"
        return send_file(clip.video_path, mimetype="video/mp4", as_attachment=True, download_name=safe_title, conditional=True)

    project = Project.query.get(project_id)
    if project:
        safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
        candidate_paths = [
            os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{clip_index}", "video.mp4"),
            os.path.join(Config.STORAGE_RESULTS, project.id, f"clip_{clip_index}", "video.mp4"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return send_file(p, mimetype="video/mp4", as_attachment=True, download_name=f"clip_{clip_index}.mp4", conditional=True)

        job = Job.query.filter_by(project_id=project.id).order_by(Job.id.desc()).first()
        if job and job.status_detail:
            try:
                sdata = json.loads(job.status_detail)
                for ec in sdata.get("edited_clips", []):
                    if (ec.get("index") == clip_index or ec.get("video_url", "").endswith(f"/{clip_index}/video")) and ec.get("video_path") and os.path.exists(ec.get("video_path")):
                        return send_file(ec.get("video_path"), mimetype="video/mp4", as_attachment=True, download_name=f"clip_{clip_index}.mp4", conditional=True)
            except Exception:
                pass

    return "Video not found", 404

@web_bp.route("/api/clips/<int:clip_id>/video")
def get_clip_video(clip_id):
    clip = Clip.query.get(clip_id)
    if clip and clip.video_path and os.path.exists(clip.video_path):
        return send_file(clip.video_path, mimetype="video/mp4", conditional=True)

    # Fallback: check recent projects for matching clip_index, folder on disk, or job status_detail
    for project in Project.query.order_by(Project.created_at.desc()).limit(5).all():
        # A. Check DB clip
        clip_match = Clip.query.filter_by(project_id=project.id, clip_index=clip_id).first()
        if clip_match and clip_match.video_path and os.path.exists(clip_match.video_path):
            return send_file(clip_match.video_path, mimetype="video/mp4", conditional=True)

        # B. Check storage/results/ directory (both safe_folder and project.id)
        safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
        candidate_paths = [
            os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{clip_id}", "video.mp4"),
            os.path.join(Config.STORAGE_RESULTS, project.id, f"clip_{clip_id}", "video.mp4"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return send_file(p, mimetype="video/mp4", conditional=True)

        # C. Check job.status_detail edited_clips
        job = Job.query.filter_by(project_id=project.id).order_by(Job.id.desc()).first()
        if job and job.status_detail:
            try:
                sdata = json.loads(job.status_detail)
                for ec in sdata.get("edited_clips", []):
                    if (ec.get("index") == clip_id or ec.get("video_url", "").endswith(f"/{clip_id}/video")) and ec.get("video_path") and os.path.exists(ec.get("video_path")):
                        return send_file(ec.get("video_path"), mimetype="video/mp4", conditional=True)
            except Exception:
                pass

    return "Video not found", 404

@web_bp.route("/api/clips/<int:clip_id>/download")
def download_clip(clip_id):
    """Download individual clip directly with clean filename and range support."""
    clip = Clip.query.get(clip_id)
    if clip and clip.video_path and os.path.exists(clip.video_path):
        safe_title = secure_filename(clip.title or f"clip_{clip.clip_index}")
        if not safe_title.endswith(".mp4"):
            safe_title = f"{safe_title}.mp4"
        return send_file(
            clip.video_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=safe_title,
            conditional=True
        )

    # Fallback to recent projects
    for project in Project.query.order_by(Project.created_at.desc()).limit(5).all():
        safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
        candidate_paths = [
            os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{clip_id}", "video.mp4"),
            os.path.join(Config.STORAGE_RESULTS, project.id, f"clip_{clip_id}", "video.mp4"),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return send_file(p, mimetype="video/mp4", as_attachment=True, download_name=f"clip_{clip_id}.mp4", conditional=True)

        job = Job.query.filter_by(project_id=project.id).order_by(Job.id.desc()).first()
        if job and job.status_detail:
            try:
                sdata = json.loads(job.status_detail)
                for ec in sdata.get("edited_clips", []):
                    if (ec.get("index") == clip_id or ec.get("video_url", "").endswith(f"/{clip_id}/video")) and ec.get("video_path") and os.path.exists(ec.get("video_path")):
                        return send_file(ec.get("video_path"), mimetype="video/mp4", as_attachment=True, download_name=f"clip_{clip_id}.mp4", conditional=True)
            except Exception:
                pass

    return "Video not found", 404

@web_bp.route("/api/clips/<int:clip_id>", methods=["DELETE", "POST"])
def delete_clip(clip_id):
    """Delete a generated clip folder and database record."""
    clip = Clip.query.get_or_404(clip_id)
    project = Project.query.get(clip.project_id)
    
    import shutil
    deleted_paths = []
    # 1. Delete clip directory from storage/results/
    if clip.video_path:
        clip_dir = os.path.dirname(clip.video_path)
        if os.path.isdir(clip_dir):
            try:
                shutil.rmtree(clip_dir, ignore_errors=True)
                deleted_paths.append(clip_dir)
                logger.info(f"Deleted clip folder: {clip_dir}")
            except Exception as e:
                logger.warning(f"Error removing clip folder: {e}")

    # 2. Check candidate folders based on project title and id
    if project:
        safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
        candidate_dirs = [
            os.path.join(Config.STORAGE_RESULTS, safe_folder, f"clip_{clip.clip_index}"),
            os.path.join(Config.STORAGE_RESULTS, project.id, f"clip_{clip.clip_index}")
        ]
        for c_dir in candidate_dirs:
            if os.path.isdir(c_dir) and c_dir not in deleted_paths:
                try:
                    shutil.rmtree(c_dir, ignore_errors=True)
                    logger.info(f"Deleted candidate clip folder: {c_dir}")
                except Exception as e:
                    logger.warning(f"Error removing candidate clip folder {c_dir}: {e}")

    db.session.delete(clip)
    db.session.commit()
    return jsonify({"success": True, "message": f"Klip #{clip.clip_index} beserta file storage berhasil dihapus."})

@web_bp.route("/api/projects/<project_id>", methods=["DELETE", "POST"])
def delete_project(project_id):
    """Delete a whole project folder and its database record."""
    project = Project.query.get_or_404(project_id)
    
    import shutil
    # 1. Remove project directories inside STORAGE_RESULTS (both safe_folder and project_id)
    safe_folder = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or project.id
    candidate_proj_dirs = [
        os.path.join(Config.STORAGE_RESULTS, safe_folder),
        os.path.join(Config.STORAGE_RESULTS, project.id)
    ]
    for p_dir in candidate_proj_dirs:
        if os.path.isdir(p_dir):
            try:
                shutil.rmtree(p_dir, ignore_errors=True)
                logger.info(f"Deleted project storage directory: {p_dir}")
            except Exception as e:
                logger.warning(f"Error removing project folder {p_dir}: {e}")

    # 2. Also ensure any individual clip folder not caught above is removed
    for c in project.clips:
        if c.video_path:
            c_dir = os.path.dirname(c.video_path)
            if os.path.isdir(c_dir):
                try:
                    shutil.rmtree(c_dir, ignore_errors=True)
                except Exception:
                    pass

    # 3. Clean up any temporary files related to project in STORAGE_TEMP
    try:
        if project.video_path and os.path.exists(project.video_path):
            os.remove(project.video_path)
    except Exception:
        pass

    try:
        if os.path.isdir(Config.STORAGE_TEMP):
            for fname in os.listdir(Config.STORAGE_TEMP):
                if project.id in fname:
                    fpath = os.path.join(Config.STORAGE_TEMP, fname)
                    try:
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                        elif os.path.isdir(fpath):
                            shutil.rmtree(fpath, ignore_errors=True)
                    except Exception:
                        pass
    except Exception as temp_err:
        logger.warning(f"Error cleaning temp storage for project {project.id}: {temp_err}")

    db.session.delete(project)
    db.session.commit()
    return jsonify({"success": True, "message": f"Project '{project.title}' beserta semua file storage berhasil dihapus."})

@web_bp.route("/api/projects/<project_id>/export")
def export_package(project_id):
    """Download ZIP package from results/project/."""
    project = Project.query.get_or_404(project_id)
    clips = Clip.query.filter_by(project_id=project.id).order_by(Clip.clip_index.asc()).all()

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in clips:
            if c.video_path and os.path.exists(c.video_path):
                clip_folder = os.path.dirname(c.video_path)
                arc_prefix = f"clip_{c.clip_index}"
                zf.write(c.video_path, arcname=f"{arc_prefix}/video.mp4")
                if c.subtitle_path and os.path.exists(c.subtitle_path):
                    zf.write(c.subtitle_path, arcname=f"{arc_prefix}/subtitle.ass")
                clip_txt = os.path.join(clip_folder, "transcript.txt")
                if os.path.exists(clip_txt):
                    zf.write(clip_txt, arcname=f"{arc_prefix}/transcript.txt")
                clip_meta = os.path.join(clip_folder, "metadata.json")
                if os.path.exists(clip_meta):
                    zf.write(clip_meta, arcname=f"{arc_prefix}/metadata.json")

        if project.transcript:
            zf.writestr("full_transcript.txt", project.transcript.full_text)

    memory_file.seek(0)
    safe_title = "".join([c for c in project.title if c.isalnum() or c in (' ', '-', '_')]).strip() or "clips"
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_title}_export.zip"
    )
