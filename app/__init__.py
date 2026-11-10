import os
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from app.config import Config
from app.db import db

def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static"
    )
    app.config.from_object(config_class)

    # Ensure storage paths exist
    os.makedirs(app.config["STORAGE_RESULTS"], exist_ok=True)
    os.makedirs(app.config["STORAGE_TEMP"], exist_ok=True)

    # Initialize extensions
    db.init_app(app)

    # Enable WAL mode for SQLite to support concurrent worker & web reads
    with app.app_context():
        if "sqlite" in app.config["SQLALCHEMY_DATABASE_URI"]:
            with db.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("PRAGMA journal_mode=WAL;"))
        db.create_all()
        if "sqlite" in app.config["SQLALCHEMY_DATABASE_URI"]:
            with db.engine.connect() as conn:
                from sqlalchemy import text
                job_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(jobs);")).fetchall()]
                if "status_detail" not in job_cols:
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN status_detail TEXT;"))
                clip_cols = [r[1] for r in conn.execute(text("PRAGMA table_info(clips);")).fetchall()]
                if "clip_status" not in clip_cols:
                    conn.execute(text("ALTER TABLE clips ADD COLUMN clip_status TEXT DEFAULT 'PENDING';"))
                conn.commit()

    # Support reverse proxies like JupyterHub / jupyter-server-proxy
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Context processor to inject base_path into all templates
    @app.context_processor
    def inject_base_path():
        """
        Auto-detect base path from proxy headers or use configured APPLICATION_ROOT.
        This makes all URLs work with JupyterLab proxy deployment.
        
        Examples:
        - Local dev: base_path = ""
        - JupyterLab: base_path = "/user/username/proxy/5005"
        """
        # Priority 1: Check X-Forwarded-Prefix header (set by ProxyFix from jupyter-server-proxy)
        base_path = request.environ.get("HTTP_X_FORWARDED_PREFIX", "").rstrip("/")
        
        # Priority 2: Fallback to configured APPLICATION_ROOT
        if not base_path and app.config.get("APPLICATION_ROOT"):
            base_path = app.config["APPLICATION_ROOT"]
        
        # Priority 3: Extract from request.script_root (auto-detected by Flask)
        if not base_path and request.script_root:
            base_path = request.script_root.rstrip("/")
        
        return {
            "base_path": base_path,
            "url_prefix": base_path  # alias for backward compatibility
        }

    # Register web blueprint
    from app.web.routes import web_bp
    app.register_blueprint(web_bp)

    return app
