"""
ClipSmith - Setup Verification Test
Run: python tests/setup_test.py
Checks all dependencies and environment required to run ClipSmith.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(label, detail=""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  {GREEN}[OK]  {label}{RESET}{suffix}")

def fail(label, hint=""):
    suffix = f"\n     {YELLOW}-> {hint}{RESET}" if hint else ""
    print(f"  {RED}[!!] {label}{RESET}{suffix}")

def warn(label, detail=""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  {YELLOW}[??] {label}{RESET}{suffix}")

# ── Root dir (project root, one level up from tests/) ────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
def check_python_version():
    print(f"\n{BOLD}[1] Python Version{RESET}")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor >= 10:
        ok(f"Python {version_str}")
    else:
        fail(
            f"Python {version_str} — requires 3.10+",
            "Download from https://python.org/downloads"
        )


def check_cli_tool(name: str, args: list[str], label: str, install_hint: str):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5
        )
        version_line = (result.stdout or result.stderr).strip().splitlines()
        version = version_line[0] if version_line else "unknown"
        ok(f"{label}: {version}")
        return True
    except FileNotFoundError:
        fail(f"{label} not found", install_hint)
        return False
    except Exception as e:
        fail(f"{label} — {e}", install_hint)
        return False


def check_cli_tools():
    print(f"\n{BOLD}[2] External Tools{RESET}")
    check_cli_tool(
        "git", ["git", "--version"], "Git",
        "https://git-scm.com/downloads"
    )
    check_cli_tool(
        "ffmpeg", ["ffmpeg", "-version"], "FFmpeg",
        "Windows: winget install ffmpeg  |  Linux: sudo apt install ffmpeg  |  Mac: brew install ffmpeg"
    )
    check_cli_tool(
        "ollama", ["ollama", "--version"], "Ollama",
        "Download from https://ollama.com/download"
    )


def check_python_packages():
    print(f"\n{BOLD}[3] Python Packages{RESET}")

    packages = {
        "flask":            ("Flask",               "pip install flask>=3.0.0"),
        "flask_sqlalchemy": ("Flask-SQLAlchemy",    "pip install flask-sqlalchemy>=3.1.0"),
        "werkzeug":         ("Werkzeug",            "pip install werkzeug>=3.0.0"),
        "dotenv":           ("python-dotenv",       "pip install python-dotenv>=1.0.0"),
        "langgraph":        ("LangGraph",           "pip install langgraph>=0.2.0"),
        "langchain_core":   ("LangChain Core",      "pip install langchain-core>=0.3.0"),
        "pydantic":         ("Pydantic",            "pip install pydantic>=2.6.0"),
        "requests":         ("Requests",            "pip install requests>=2.31.0"),
        "cv2":              ("OpenCV",              "pip install opencv-python>=4.9.0"),
    }

    for module, (label, hint) in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "installed")
            ok(f"{label}", version)
        except ImportError:
            fail(f"{label} not installed", hint)

    # PyTorch — separate because it's not in requirements.txt
    print()
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_info = f"CUDA {'✓' if cuda_available else '✗ (CPU only)'}"
        ok(f"PyTorch {torch.__version__}", device_info)
    except ImportError:
        fail(
            "PyTorch not installed",
            "GPU:  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
            "     CPU:  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
        )

    # WhisperX
    try:
        import whisperx  # noqa: F401
        ok("WhisperX", getattr(whisperx, "__version__", "installed"))
    except ImportError:
        fail("WhisperX not installed", "pip install whisperx")


def check_env_file():
    print(f"\n{BOLD}[4] Environment File (.env){RESET}")
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"

    if not env_path.exists():
        fail(
            ".env file not found",
            f"cp .env.example .env  (then edit DEVICE, WHISPER_MODEL, VIDEO_ENCODER)"
        )
        return

    ok(".env exists")

    # Load and check required keys
    from dotenv import dotenv_values
    values = dotenv_values(env_path)

    required_keys = [
        "DEVICE", "COMPUTE_TYPE", "WHISPER_MODEL", "WHISPER_LANGUAGE",
        "LLM_BASE_URL", "LLM_MODEL",
        "VIDEO_ENCODER", "DATABASE_URL", "STORAGE_DIR",
        "FLASK_PORT", "SECRET_KEY",
    ]
    missing = [k for k in required_keys if not values.get(k)]
    if missing:
        fail(f"Missing keys in .env: {', '.join(missing)}", f"Check .env.example for reference")
    else:
        ok("All required keys present")

    # Warn if still using default secret key
    if values.get("SECRET_KEY") == "clipsmith-dev-secret-key-2026":
        warn("SECRET_KEY is still the default dev value — change it for production")


def check_storage_dirs():
    print(f"\n{BOLD}[5] Storage Directories{RESET}")
    dirs = [
        ROOT / "storage",
        ROOT / "storage" / "results",
        ROOT / "storage" / "temp",
    ]
    for d in dirs:
        if d.exists():
            ok(f"{d.relative_to(ROOT)}")
        else:
            fail(
                f"{d.relative_to(ROOT)} not found",
                f"mkdir {d.relative_to(ROOT)}"
            )


def check_model_assets():
    print(f"\n{BOLD}[6] Model Assets{RESET}")
    onnx_path = ROOT / "app" / "data" / "face_detection_yunet_2023mar.onnx"
    if onnx_path.exists():
        size_kb = onnx_path.stat().st_size // 1024
        ok(f"face_detection_yunet_2023mar.onnx", f"{size_kb} KB")
    else:
        fail(
            "face_detection_yunet_2023mar.onnx not found",
            "Make sure you cloned the full repository (use git lfs pull if needed)"
        )


def check_ollama_running():
    print(f"\n{BOLD}[7] Ollama Service{RESET}")
    try:
        import requests
        resp = requests.get("http://localhost:11434", timeout=3)
        if resp.status_code == 200:
            ok("Ollama is running at http://localhost:11434")
        else:
            warn(f"Ollama responded with status {resp.status_code}")
    except Exception:
        fail(
            "Ollama is not running",
            "Start it with: ollama run llama3.1:8b  (or check LLM_BASE_URL in .env)"
        )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*56}")
    print(f"  ClipSmith — Setup Verification")
    print(f"{'='*56}{RESET}")

    check_python_version()
    check_cli_tools()
    check_python_packages()
    check_env_file()
    check_storage_dirs()
    check_model_assets()
    check_ollama_running()

    print(f"\n{BOLD}{'='*56}")
    print(f"  Done. Fix any ❌ above, then run: python run.py")
    print(f"{'='*56}{RESET}\n")


if __name__ == "__main__":
    main()
