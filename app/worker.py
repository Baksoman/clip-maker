import time
import os
import json
import logging
import traceback
from threading import Thread
from app.db import db
from app.models import Project, Job, Transcript, Clip
from app.agents.graph import build_clipsmith_graph
from app.config import Config

logger = logging.getLogger(__name__)

class ClipWorker:
    def __init__(self, app):
        self.app = app
        self.is_running = False
        self.thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            logger.info("Background ClipWorker thread started.")

    def _run_loop(self):
        while self.is_running:
            try:
                self._process_next_job()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}\n{traceback.format_exc()}")
            time.sleep(2)

    def _process_next_job(self):
        with self.app.app_context():
            job = Job.query.filter_by(step="PENDING").order_by(Job.created_at.asc()).first()
            if not job:
                return

            project = Project.query.get(job.project_id)
            if not project:
                job.step = "ERROR"
                job.message = "Project not found"
                db.session.commit()
                return

            logger.info(f"Starting processing for Project: {project.id} ({project.title})")
            project.status = "PROCESSING"
            job.step = "TRANSCRIBING"
            job.progress = 10
            job.message = "Mengekstrak audio dan transkripsi dengan WhisperX..."

            status_data = {
                "transcript": "",
                "highlights": [],
                "edited_clips": [],
                "critic_reviews": []
            }
            job.status_detail = json.dumps(status_data)
            db.session.commit()

            try:
                graph = build_clipsmith_graph()
                initial_state = {
                    "project_id": project.id,
                    "project_title": project.title,
                    "video_path": project.video_path,
                    "language": "en"
                }

                # Stream execution step-by-step
                for output_step in graph.stream(initial_state):
                    for node_name, node_val in output_step.items():
                        if node_name == "transcriber":
                            transcript_data = node_val.get("transcript", {})
                            if transcript_data:
                                full_text = transcript_data.get("full_text", "")
                                status_data["transcript"] = full_text
                                if not project.transcript:
                                    trans_record = Transcript(
                                        project_id=project.id,
                                        language=transcript_data.get("language", "en"),
                                        full_text=full_text,
                                        segments_json=json.dumps(transcript_data.get("segments", []))
                                    )
                                    db.session.add(trans_record)
                            job.step = "TRANSCRIBING"
                            job.progress = 25
                            job.message = "Transkripsi selesai. Mencari momen viral..."
                            job.status_detail = json.dumps(status_data)
                            db.session.commit()

                        elif node_name == "highlighter":
                            candidates = node_val.get("candidate_highlights", [])
                            status_data["highlights"] = [
                                {
                                    "index": i + 1,
                                    "start_time": c.get("start_time", 0.0),
                                    "end_time": c.get("end_time", 0.0),
                                    "duration": round(float(c.get("end_time", 0.0)) - float(c.get("start_time", 0.0)), 1),
                                    "summary": c.get("summary", ""),
                                    "hook_reason": c.get("hook_reason", "")
                                }
                                for i, c in enumerate(candidates)
                            ]
                            job.step = "FINDING_HIGHLIGHTS"
                            job.progress = 40
                            job.message = f"Ditemukan {len(candidates)} momen highlight potensial."
                            job.status_detail = json.dumps(status_data)
                            db.session.commit()

                        elif node_name == "editor":
                            out_video = node_val.get("rendered_clip_path")
                            sub_path = node_val.get("subtitle_ass_path")
                            
                            # Determine real clip index from graph or folder path
                            idx = node_val.get("clip_index")
                            if not idx and out_video:
                                import re
                                m = re.search(r"clip_(\d+)", out_video)
                                idx = int(m.group(1)) if m else None
                            if not idx:
                                idx = len(status_data["edited_clips"]) + 1

                            if out_video:
                                clip_entry = {
                                    "index": idx,
                                    "video_path": out_video,
                                    "subtitle_path": sub_path,
                                    "video_url": f"/api/projects/{project.id}/clips/{idx}/video"
                                }
                                # Update existing entry for this clip index (e.g. revision) or append
                                existing_pos = next((i for i, c in enumerate(status_data["edited_clips"]) if c.get("index") == idx), None)
                                if existing_pos is not None:
                                    status_data["edited_clips"][existing_pos] = clip_entry
                                else:
                                    status_data["edited_clips"].append(clip_entry)

                            job.step = "EDITING"
                            num_h = max(1, len(status_data["highlights"]))
                            job.progress = min(85, 40 + int(45 * idx / num_h))
                            job.message = f"Smart-crop 9:16 & subtitle pop-up selesai untuk klip #{idx}."
                            job.status_detail = json.dumps(status_data)
                            db.session.commit()

                        elif node_name == "critic":
                            c_res = node_val.get("critic_result", {})
                            idx = len(status_data["critic_reviews"]) + 1
                            is_acc = c_res.get("is_acceptable", False)
                            needs_rev = c_res.get("needs_revision", False)
                            status_str = "APPROVED" if is_acc else ("REVISION" if needs_rev else "REJECTED")

                            review_item = {
                                "index": idx,
                                "overall_score": c_res.get("overall_score", 0.0),
                                "is_acceptable": is_acc,
                                "needs_revision": needs_rev,
                                "status": status_str,
                                "critique": c_res.get("critique", ""),
                                "suggestion": c_res.get("suggestion", ""),
                                "steps_log": c_res.get("steps_log", [])
                            }
                            status_data["critic_reviews"].append(review_item)
                            job.step = "CRITIQUING"
                            job.message = f"Critic mengevaluasi klip #{idx}: {status_str} ({c_res.get('overall_score', 0)}/10)"
                            job.status_detail = json.dumps(status_data)
                            db.session.commit()

                        elif node_name == "metadata":
                            approved_list = node_val.get("approved_clips", [])
                            if approved_list:
                                latest_clip = approved_list[-1]
                                c_idx = latest_clip.get("clip_index", 1)
                                existing_clip = Clip.query.filter_by(
                                    project_id=project.id,
                                    clip_index=c_idx
                                ).first()

                                if not existing_clip:
                                    clip_obj = Clip(
                                        project_id=project.id,
                                        clip_index=c_idx,
                                        start_time=latest_clip.get("start_time", 0.0),
                                        end_time=latest_clip.get("end_time", 0.0),
                                        duration=latest_clip.get("duration", 0.0),
                                        summary=latest_clip.get("summary", ""),
                                        hook_reason=latest_clip.get("hook_reason", ""),
                                        critic_score=latest_clip.get("critic_score", 0.0),
                                        critic_feedback=latest_clip.get("critic_feedback", "{}"),
                                        revision_count=latest_clip.get("revision_count", 0),
                                        is_approved=True,
                                        clip_status="APPROVED",
                                        video_path=latest_clip.get("video_path"),
                                        subtitle_path=latest_clip.get("subtitle_path"),
                                        title=latest_clip.get("title"),
                                        caption=latest_clip.get("caption"),
                                        hashtags=latest_clip.get("hashtags")
                                    )
                                    db.session.add(clip_obj)
                                    db.session.commit()

                # Completion
                saved_clips_count = Clip.query.filter_by(project_id=project.id).count()
                project.status = "COMPLETED"
                job.step = "DONE"
                job.progress = 100
                job.message = f"Berhasil membuat {saved_clips_count} klip video vertikal!"
                job.status_detail = json.dumps(status_data)
                db.session.commit()
                logger.info(f"Project {project.id} finished with {saved_clips_count} clips.")

            except Exception as e:
                db.session.rollback()
                err_trace = traceback.format_exc()
                logger.error(f"Pipeline error for project {project.id}: {err_trace}")
                project.status = "FAILED"
                job.step = "ERROR"
                job.message = f"Pipeline gagal: {str(e)}"
                job.error_log = err_trace
                db.session.commit()

            finally:
                # Requirement 6: Hapus video yang diupload dari folder temp untuk hemat storage
                try:
                    if project.video_path and os.path.exists(project.video_path):
                        os.remove(project.video_path)
                        logger.info(f"Cleaned up temporary uploaded video: {project.video_path}")
                except Exception as del_err:
                    logger.warning(f"Could not remove temp video: {del_err}")

                # Clean up temporary audio file
                try:
                    temp_audio = os.path.join(Config.STORAGE_TEMP, f"{project.id}_audio.wav")
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                except Exception:
                    pass

                # Free VRAM: unload Ollama model weights & clear PyTorch CUDA cache
                try:
                    from app.services.llm import unload_ollama_model
                    unload_ollama_model()
                except Exception:
                    pass

                try:
                    from app.core.gpu import clean_memory
                    clean_memory()
                except Exception:
                    pass

