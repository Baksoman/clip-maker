import logging
import socket
from app import create_app
from app.config import Config
from app.worker import ClipWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = create_app()

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port

if __name__ == "__main__":
    # Start background job worker
    worker = ClipWorker(app)
    worker.start()

    active_port = find_available_port(Config.FLASK_PORT)

    print(f"\n========================================================")
    print(f"  ClipSmith Running on http://127.0.0.1:{active_port}")
    print(f"  Device: {Config.DEVICE} | Model: {Config.WHISPER_MODEL}")
    print(f"========================================================\n")

    app.run(
        host="0.0.0.0",
        port=active_port,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False  # Avoid double worker initialization
    )

