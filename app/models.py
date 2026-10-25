from datetime import datetime, timezone
import uuid
import json
from app.db import db

def generate_uuid():
    return str(uuid.uuid4())

class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    title = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    video_path = db.Column(db.String(512), nullable=False)
    duration = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(32), default="QUEUED")  # QUEUED, PROCESSING, COMPLETED, FAILED
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    jobs = db.relationship("Job", backref="project", cascade="all, delete-orphan", lazy=True)
    transcript = db.relationship("Transcript", backref="project", uselist=False, cascade="all, delete-orphan")
    clips = db.relationship("Clip", backref="project", cascade="all, delete-orphan", lazy=True, order_by="Clip.clip_index")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "original_filename": self.original_filename,
            "duration": round(self.duration, 2),
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "clip_count": len(self.clips)
        }

class Job(db.Model):
    __tablename__ = "jobs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=False)
    step = db.Column(db.String(64), default="PENDING")
    progress = db.Column(db.Integer, default=0)
    message = db.Column(db.String(255), default="Job queued")
    status_detail = db.Column(db.Text, nullable=True)  # JSON: {transcript, highlights, clips_progress}
    error_log = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        parsed_detail = {}
        if self.status_detail:
            try:
                parsed_detail = json.loads(self.status_detail)
            except Exception:
                parsed_detail = {"raw": self.status_detail}
        return {
            "id": self.id,
            "project_id": self.project_id,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "status_detail": parsed_detail,
            "error_log": self.error_log,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Transcript(db.Model):
    __tablename__ = "transcripts"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=False, unique=True)
    language = db.Column(db.String(10), default="en")
    full_text = db.Column(db.Text, nullable=False)
    segments_json = db.Column(db.Text, nullable=False)  # JSON-encoded word/sentence segments
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def get_segments(self):
        try:
            return json.loads(self.segments_json)
        except Exception:
            return []

class Clip(db.Model):
    __tablename__ = "clips"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.String(36), db.ForeignKey("projects.id"), nullable=False)
    clip_index = db.Column(db.Integer, default=1)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    hook_reason = db.Column(db.Text, nullable=True)
    
    # Critic evaluation
    critic_score = db.Column(db.Float, default=0.0)
    critic_feedback = db.Column(db.Text, nullable=True)  # JSON formatted feedback
    revision_count = db.Column(db.Integer, default=0)
    is_approved = db.Column(db.Boolean, default=False)
    clip_status = db.Column(db.String(32), default="PENDING")  # PENDING, EDITING, CRITIQUING, APPROVED, REJECTED

    # Generated files & metadata
    video_path = db.Column(db.String(512), nullable=True)
    subtitle_path = db.Column(db.String(512), nullable=True)
    title = db.Column(db.String(255), nullable=True)
    caption = db.Column(db.Text, nullable=True)
    hashtags = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "clip_index": self.clip_index,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "summary": self.summary,
            "hook_reason": self.hook_reason,
            "critic_score": self.critic_score,
            "is_approved": self.is_approved,
            "revision_count": self.revision_count,
            "title": self.title,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "video_url": f"/api/clips/{self.id}/video" if self.video_path else None
        }
