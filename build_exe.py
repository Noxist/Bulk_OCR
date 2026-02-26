import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ICON_DIR = BASE_DIR / "assets" / "icons"

# Import version from the shared config module
sys_path = str(BASE_DIR)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from config import APP_VERSION  # noqa: E402


def get_version() -> str:
    return APP_VERSION


def resolve_icon() -> Path | None:
    if not ICON_DIR.exists():
        return None
    ico_path = ICON_DIR / "app_icon.ico"
    if ico_path.exists():
        return ico_path
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = ICON_DIR / f"app_icon{ext}"
        if candidate.exists():
            generated = ICON_DIR / "app_icon_generated.ico"
            image = Image.open(candidate)
            image.save(generated, format="ICO")
            return generated
    return None


def build_exe() -> None:
    version = get_version()
    exe_name = f"BulkOCR-{version}"
    hidden_imports = [
        "--hidden-import",
        "addons.jsonl_upload",
        "--hidden-import",
        "addons.batch_manager",
        "--hidden-import",
        "addons.jsonl_to_txt",
        "--hidden-import",
        "addons.ai_text_refiner",
    ]
    command = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",
        "--name",
        exe_name,
        "--add-data",
        f"assets{os.pathsep}assets",
        "--add-data",
        f"config.py{os.pathsep}.",
        "--collect-all",
        "fitz",
        "--collect-all",
        "ebooklib",
        "--collect-all",
        "sv_ttk",
        "--collect-all",
        "tkinterdnd2",
        "main.py",
    ]
    command.extend(hidden_imports)
    icon_path = resolve_icon()
    if icon_path:
        command.extend(["--icon", icon_path.as_posix()])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    build_exe()
