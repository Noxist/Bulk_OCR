import re
import subprocess
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "epub_builder_gui.py"
ICON_DIR = BASE_DIR / "assets" / "icons"


def get_version() -> str:
    content = APP_FILE.read_text(encoding="utf-8")
    match = re.search(r'APP_VERSION\\s*=\\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "0.0.0"


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
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--name",
        exe_name,
        "main.py",
    ]
    command.extend(hidden_imports)
    icon_path = resolve_icon()
    if icon_path:
        command.extend(["--icon", icon_path.as_posix()])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    build_exe()
