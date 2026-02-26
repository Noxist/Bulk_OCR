import base64
import concurrent.futures
import datetime as dt
import html
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import multiprocessing
import fitz  # PyMuPDF
from ebooklib import epub
from openai import OpenAI
from PIL import Image, ImageTk
from rapidfuzz import fuzz
from tkinter import (
    BooleanVar,
    Canvas,
    IntVar,
    Listbox,
    Menu,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import importlib.util

# Sprint 4 imports (#31, #32)
try:
    import sv_ttk
    _HAS_SV_TTK = True
except ImportError:
    _HAS_SV_TTK = False

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

from config import (
    APP_VERSION,
    BATCH_MAX_FILE_BYTES,
    DEFAULT_OCR_PROMPT_DE,
    DEFAULT_OCR_PROMPT_EN,
    DEFAULT_TOC_PROMPT_DE,
    DEFAULT_TOC_PROMPT_EN,
    OPENAI_MODEL_VISION,
)

ADDONS_AVAILABLE = {}
BatchUploaderApp = None
BatchManagerApp = None
JsonlToTxtApp = None
AiTextRefinerApp = None

if importlib.util.find_spec("addons.jsonl_upload"):
    from addons.jsonl_upload import BatchUploaderApp

    ADDONS_AVAILABLE["uploader"] = True
else:
    ADDONS_AVAILABLE["uploader"] = False

if importlib.util.find_spec("addons.batch_manager"):
    from addons.batch_manager import BatchManagerApp

    ADDONS_AVAILABLE["manager"] = True
else:
    ADDONS_AVAILABLE["manager"] = False

if importlib.util.find_spec("addons.jsonl_to_txt"):
    from addons.jsonl_to_txt import JsonlToTxtApp

    ADDONS_AVAILABLE["converter"] = True
else:
    ADDONS_AVAILABLE["converter"] = False

if importlib.util.find_spec("addons.ai_text_refiner"):
    from addons.ai_text_refiner import AiTextRefinerApp

    ADDONS_AVAILABLE["refiner"] = True
else:
    ADDONS_AVAILABLE["refiner"] = False

# ============================================================
# Configuration – imported from config.py (#16)
# ============================================================
OPENAI_API_KEY = ""  # Loaded from secrets.json at runtime

# Marker used to detect failed/empty OCR pages – must NOT appear in real text
_OCR_ERR = "[[OCR-ERROR"


@dataclass
class AppSettings:
    api_key: str
    ocr_prompt_en: str
    ocr_prompt_de: str
    toc_prompt_en: str
    toc_prompt_de: str
    keep_pdf_images: bool
    keep_toc_images: bool
    # Sprint 2 additions
    last_pdf_path: str = ""
    last_txt_path: str = ""
    last_image_folder: str = ""
    output_format: str = "epub"   # epub | txt | html
    # Sprint 3 addition (#18)
    custom_output_dir: str = ""   # empty = default (app-data/output)
    # Sprint 4 addition (#31)
    theme_mode: str = "light"     # "light" | "dark"


@dataclass
class OcrState:
    pdf_path: str
    image_dir: str
    image_files: List[str]
    language: str
    current_index: int
    text_chunks: List[str]
    created_at: str


@dataclass
class TocEntry:
    title: str
    page: str


def get_app_data_dir() -> Path:
    """
    Returns the directory where app data (settings, temp images, logs) lives.
    Frozen EXE  -> next to the EXE (portable).
    Script mode -> next to the script.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def secrets_path() -> Path:
    """Path to secrets.json (API keys etc. – must NOT be committed to git)."""
    return get_app_data_dir() / "secrets.json"


def load_api_key_from_secrets() -> str:
    """Load OpenAI API key from secrets.json or environment."""
    env_key = os.environ.get("OPENAI_API_KEY", "")
    path = secrets_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            file_key = data.get("openai_api_key", "")
            if file_key:
                return file_key
        except (json.JSONDecodeError, OSError):
            pass
    return env_key


def save_api_key_to_secrets(api_key: str) -> None:
    """Persist OpenAI API key to secrets.json."""
    path = secrets_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data["openai_api_key"] = api_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def project_paths(project_root: Path) -> dict:
    images_dir = project_root / "_images"
    return {
        "base": project_root,
        "images": images_dir,
        "jsonl": project_root / "_jsonl",
        "logs": project_root / "_logs",
        "output": project_root / "_output",
        "pdfimgs": images_dir / "pdf",
        "tocimgs": images_dir / "toc",
    }


def ensure_project_dirs(project_root: Path) -> dict:
    paths = project_paths(project_root)
    try:
        paths["images"].mkdir(parents=True, exist_ok=True)
        paths["jsonl"].mkdir(parents=True, exist_ok=True)
        paths["logs"].mkdir(parents=True, exist_ok=True)
        paths["output"].mkdir(parents=True, exist_ok=True)
        paths["pdfimgs"].mkdir(parents=True, exist_ok=True)
        paths["tocimgs"].mkdir(parents=True, exist_ok=True)
    except PermissionError:
        messagebox.showerror(
            "Permission Error",
            f"Could not create folders in {project_root}.\n\n"
            "Try selecting a project folder you have write access to.",
        )

    return paths


def settings_path() -> Path:
    return get_app_data_dir() / "settings.json"


def ocr_state_path(log_dir: Path) -> Path:
    return log_dir / "ocr_state.json"


def ocr_progress_text_path(log_dir: Path) -> Path:
    return log_dir / "ocr_progress.txt"


def load_ocr_state(log_dir: Path) -> Optional[OcrState]:
    path = ocr_state_path(log_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return OcrState(
        pdf_path=data.get("pdf_path", ""),
        image_dir=data.get("image_dir", ""),
        image_files=list(data.get("image_files", [])),
        language=data.get("language", "english"),
        current_index=int(data.get("current_index", 0)),
        text_chunks=list(data.get("text_chunks", [])),
        created_at=data.get("created_at", ""),
    )


def save_ocr_state(state: OcrState, log_dir: Path) -> None:
    path = ocr_state_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pdf_path": state.pdf_path,
        "image_dir": state.image_dir,
        "image_files": state.image_files,
        "language": state.language,
        "current_index": state.current_index,
        "text_chunks": state.text_chunks,
        "created_at": state.created_at,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_ocr_state(log_dir: Path) -> None:
    path = ocr_state_path(log_dir)
    if path.exists():
        path.unlink()

def load_settings() -> AppSettings:
    api_key = load_api_key_from_secrets()
    defaults = AppSettings(
        api_key=api_key,
        ocr_prompt_en=DEFAULT_OCR_PROMPT_EN,
        ocr_prompt_de=DEFAULT_OCR_PROMPT_DE,
        toc_prompt_en=DEFAULT_TOC_PROMPT_EN,
        toc_prompt_de=DEFAULT_TOC_PROMPT_DE,
        keep_pdf_images=False,
        keep_toc_images=True,
    )
    path = settings_path()
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults
    return AppSettings(
        api_key=api_key or data.get("api_key", defaults.api_key),
        ocr_prompt_en=data.get("ocr_prompt_en", defaults.ocr_prompt_en),
        ocr_prompt_de=data.get("ocr_prompt_de", defaults.ocr_prompt_de),
        toc_prompt_en=data.get("toc_prompt_en", defaults.toc_prompt_en),
        toc_prompt_de=data.get("toc_prompt_de", defaults.toc_prompt_de),
        keep_pdf_images=data.get("keep_pdf_images", defaults.keep_pdf_images),
        keep_toc_images=data.get("keep_toc_images", defaults.keep_toc_images),
        last_pdf_path=data.get("last_pdf_path", ""),
        last_txt_path=data.get("last_txt_path", ""),
        last_image_folder=data.get("last_image_folder", ""),
        output_format=data.get("output_format", "epub"),
        custom_output_dir=data.get("custom_output_dir", ""),
        theme_mode=data.get("theme_mode", "light"),
    )


def save_settings(settings: AppSettings) -> None:
    # Persist API key to secrets.json (gitignored), everything else to settings.json
    if settings.api_key:
        save_api_key_to_secrets(settings.api_key)
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ocr_prompt_en": settings.ocr_prompt_en,
            "ocr_prompt_de": settings.ocr_prompt_de,
            "toc_prompt_en": settings.toc_prompt_en,
            "toc_prompt_de": settings.toc_prompt_de,
            "keep_pdf_images": settings.keep_pdf_images,
            "keep_toc_images": settings.keep_toc_images,
            "last_pdf_path": settings.last_pdf_path,
            "last_txt_path": settings.last_txt_path,
            "last_image_folder": settings.last_image_folder,
            "output_format": settings.output_format,
            "custom_output_dir": settings.custom_output_dir,
            "theme_mode": settings.theme_mode,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Failed to save settings: {e}")


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detect_language(text: str) -> str:
    sample = text[:2000].casefold()
    german_markers = ["ä", "ö", "ü", "ß", " der ", " die ", " das ", " und ", " ist "]
    english_markers = [" the ", " and ", " is ", " of ", " to ", " in "]
    g_score = sum(sample.count(m) for m in german_markers)
    e_score = sum(sample.count(m) for m in english_markers)
    return "german" if g_score >= e_score else "english"


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching (#20).

    Folds case, replaces common digraphs (ä->ae, ß->ss), then strips
    remaining accents via Unicode NFKD decomposition.
    """
    import unicodedata
    text = text.casefold()
    # Explicit German/French digraph replacements BEFORE accent stripping
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("æ", "ae")
        .replace("œ", "oe")
    )
    # Strip remaining accents (e.g. e with accent -> e)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_image_files(folder: Path) -> List[Path]:
    allowed = {".png", ".jpg", ".jpeg"}
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ],
        key=lambda item: item.name.casefold(),
    )


def open_output_folder(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.name == "posix":
        os.system(f'xdg-open "{path}"')


def open_file_in_editor(path: Path) -> None:
    """Open a file directly in the default text editor (#14)."""
    if not path.exists():
        return
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.name == "posix":
        os.system(f'xdg-open "{path}"')


def find_icon_file() -> Optional[Path]:
    # Look for icon relative to the executable or script
    if getattr(sys, "frozen", False):
         base = Path(sys.executable).parent
    else:
         base = get_app_data_dir()
         
    # Check assets/icons in various locations
    possible_dirs = [
        base / "assets" / "icons",
        base / "_internal" / "assets" / "icons", # PyInstaller _internal folder
    ]
    
    for icon_dir in possible_dirs:
        if not icon_dir.exists():
            continue
        for name in ("app_icon.ico", "app_icon.png", "app_icon.jpg", "app_icon.jpeg"):
            path = icon_dir / name
            if path.exists():
                return path
    return None


def build_openai_client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise ValueError(
            "OpenAI API key is missing. Please enter it in Settings."
        )
    return OpenAI(api_key=api_key.strip())


def chunk_list(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _mime_for_image(path: Path) -> str:
    """Return the correct MIME type for an image file (#24)."""
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def ocr_images_with_retry(
    client: OpenAI,
    image_paths: List[Path],
    prompt: str,
    max_batch_size: int = 4,
    log_fn=None,
) -> str:
    """Send images to Vision API with flat retry + batch splitting (#23, #24)."""
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def call_vision(paths: List[Path]) -> str:
        content = [{"type": "text", "text": prompt}]
        for path in paths:
            b64 = encode_image_to_base64(path)
            mime = _mime_for_image(path)  # #24
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        response = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    # --- Flat retry loop (no recursion) (#23) ---
    MAX_RETRIES = 3
    BACKOFFS = [6, 10, 20]

    queue: list[list[Path]] = list(chunk_list(image_paths, max_batch_size))
    outputs: List[str] = []

    while queue:
        batch = queue.pop(0)
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                log(f"OCR: batch of {len(batch)} image(s)  (attempt {attempt + 1}/{MAX_RETRIES})")
                result = call_vision(batch)
                if not result:
                    log("OCR returned empty content. Inserting placeholder.")
                    outputs.append(f"\n\n{_OCR_ERR}:LEER]]\n\n")
                else:
                    outputs.append(result)
                success = True
                break
            except Exception as exc:
                delay = BACKOFFS[attempt] if attempt < len(BACKOFFS) else BACKOFFS[-1]
                log(f"OCR error: {exc}. Waiting {delay}s before retry\u2026")
                time.sleep(delay)

        if not success:
            if len(batch) > 1:
                mid = len(batch) // 2
                log(f"Splitting failed batch of {len(batch)} into 2 sub-batches.")
                queue.insert(0, batch[mid:])
                queue.insert(0, batch[:mid])
            else:
                log("OCR failed permanently for single image. Inserting placeholder.")
                outputs.append(f"\n\n{_OCR_ERR}:FEHLER]]\n\n")

    return "\n".join(outputs)


def _write_batch_jsonl_entries(
    entries: Iterable[str], base_path: Path, max_bytes: int, log_fn=None
) -> List[Path]:
    output_paths: List[Path] = []
    base_stem = base_path.stem
    use_parts = False
    part_index = 1
    current_path = base_path
    current_size = 0

    def open_handle(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")

    handle = open_handle(current_path)
    output_paths.append(current_path)
    for entry in entries:
        entry_bytes = (entry + "\n").encode("utf-8")
        if current_size + len(entry_bytes) > max_bytes and current_size > 0:
            handle.close()
            part_index += 1
            if not use_parts:
                part1_path = base_path.with_name(
                    f"{base_stem}_part1{base_path.suffix}"
                )
                if part1_path != base_path:
                    base_path.rename(part1_path)
                    output_paths[0] = part1_path
                use_parts = True
            current_path = base_path.with_name(
                f"{base_stem}_part{part_index}{base_path.suffix}"
            )
            handle = open_handle(current_path)
            output_paths.append(current_path)
            current_size = 0
        handle.write(entry + "\n")
        current_size += len(entry_bytes)
    handle.close()
    if log_fn and use_parts:
        log_fn(
            f"Batch JSONL split into {len(output_paths)} files (limit {max_bytes / (1024 * 1024):.0f}MB)."
        )
    return output_paths


def _extract_batch_content(record: dict) -> Optional[str]:
    response = record.get("response") or record.get("result") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return message.get("content")


def parse_batch_result_files(paths: Sequence[Path], log_fn=None) -> str:
    results: List[tuple[str, str]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            if log_fn:
                log_fn(f"Failed to read batch file {path.name}: {exc}")
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if log_fn:
                    log_fn(f"Skipping invalid JSON line in {path.name}")
                continue
            custom_id = record.get("custom_id")
            content = _extract_batch_content(record)
            if not custom_id or content is None:
                if log_fn:
                    log_fn(f"Skipping incomplete batch record in {path.name}")
                continue
            results.append((custom_id, content))
    results.sort(key=lambda item: item[0])
    return "\n".join(content for _, content in results).strip()


def ocr_pages(
    client: OpenAI, image_paths: List[Path], language: str, prompt_override: str, log_fn=None
) -> str:
    if language == "german":
        prompt = prompt_override or DEFAULT_OCR_PROMPT_DE
    else:
        prompt = prompt_override or DEFAULT_OCR_PROMPT_EN
    return ocr_images_with_retry(client, image_paths, prompt, log_fn=log_fn)


def ocr_toc(
    client: OpenAI, image_paths: List[Path], language: str, prompt_override: str, log_fn=None
) -> List[TocEntry]:
    if language == "german":
        prompt = prompt_override or DEFAULT_TOC_PROMPT_DE
    else:
        prompt = prompt_override or DEFAULT_TOC_PROMPT_EN
    raw = ocr_images_with_retry(client, image_paths, prompt, log_fn=log_fn)
    entries: List[TocEntry] = []
    seen = set()
    for line in raw.splitlines():
        if line.strip().startswith("[[!! ERROR"):
            if log_fn:
                log_fn("TOC OCR returned an error placeholder; skipping line.")
            continue
        if ":::" not in line:
            continue
        title, page = [part.strip() for part in line.split(":::", 1)]
        key = (title.casefold(), page.casefold())
        if title and key not in seen:
            entries.append(TocEntry(title=title, page=page))
            seen.add(key)
    return entries


def extract_pdf_images(
    pdf_path: Path, output_dir: Path, cleanup: bool, log_fn=None
) -> List[Path]:
    if log_fn:
        log_fn(f"Rendering PDF pages to images: {pdf_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if cleanup:
        for item in output_dir.glob("*.png"):
            item.unlink()
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()
    if page_count == 0:
        return []
    zoom = 2
    cpu_count = max(1, os.cpu_count() or 1)
    worker_count = min(cpu_count, page_count)
    page_indices = list(range(page_count))
    if worker_count == 1:
        paths = _render_pdf_page_range(
            pdf_path.as_posix(), output_dir.as_posix(), page_indices, zoom
        )
    else:
        chunk_size = math.ceil(page_count / worker_count)
        chunks = [
            page_indices[i : i + chunk_size]
            for i in range(0, page_count, chunk_size)
        ]
        paths: List[str] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [
                executor.submit(
                    _render_pdf_page_range,
                    pdf_path.as_posix(),
                    output_dir.as_posix(),
                    chunk,
                    zoom,
                )
                for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                paths.extend(future.result())
    image_paths = sorted((Path(path) for path in paths), key=lambda p: p.name)
    if log_fn:
        for img_path in image_paths:
            log_fn(f"Saved image: {img_path.name}")
    return image_paths


def _render_pdf_page_range(
    pdf_path: str, output_dir: str, page_indices: Sequence[int], zoom: float
) -> List[str]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    output_paths: List[str] = []
    output_base = Path(output_dir)
    for page_index in page_indices:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = output_base / f"page_{page_index + 1:04d}.png"
        pix.save(img_path.as_posix())
        output_paths.append(img_path.as_posix())
    doc.close()
    return output_paths


def render_pdf_page_thumbnail(page: fitz.Page, zoom: float = 0.3) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def render_pdf_pages_to_images(
    pdf_path: Path,
    page_indices: List[int],
    output_dir: Path,
    zoom: float = 3.0,
    prefix: str = "toc_page",
    cleanup: bool = True,
    log_fn=None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cleanup:
        for item in output_dir.glob(f"{prefix}_*.png"):
            item.unlink()
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []
    mat = fitz.Matrix(zoom, zoom)
    for page_index in page_indices:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = output_dir / f"{prefix}_{page_index + 1:04d}.png"
        pix.save(img_path.as_posix())
        image_paths.append(img_path)
        if log_fn:
            log_fn(f"Rendered TOC image: {img_path.name}")
    doc.close()
    return image_paths


def split_into_paragraphs(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer))
    return paragraphs


def build_chapters_from_toc(
    full_text: str,
    toc_entries: List[TocEntry],
    threshold: int = 83,
    log_fn=None,
) -> List[dict]:
    lines = full_text.splitlines()
    normalized_lines = [normalize_text(line) for line in lines]
    chapters: List[dict] = []
    last_index = 0

    for entry in toc_entries:
        target = normalize_text(entry.title)
        best_index = None
        best_score = 0
        for idx in range(last_index, len(lines)):
            line_norm = normalized_lines[idx]
            if not line_norm:
                continue
            score = max(
                fuzz.ratio(target, line_norm),
                fuzz.partial_ratio(target, line_norm),
            )
            if score > best_score:
                best_score = score
                best_index = idx
            if score >= 98:
                break
        if best_index is None or best_score < threshold:
            if log_fn:
                log_fn(
                    f"Chapter heading not found for '{entry.title}' (best score {best_score})."
                )
            continue
        if log_fn:
            log_fn(
                f"Matched chapter '{entry.title}' at line {best_index + 1} (score {best_score})."
            )
        chapters.append({"title": entry.title, "start": best_index})
        last_index = best_index + 1

    for i, chapter in enumerate(chapters):
        start_line = chapter["start"] + 1
        end_line = chapters[i + 1]["start"] if i + 1 < len(chapters) else len(lines)
        body_lines = lines[start_line:end_line]
        chapter["text"] = "\n".join(body_lines).strip()

    return chapters


def create_epub(
    title: str,
    chapters: List[dict],
    output_path: Path,
    language: str,
    log_fn=None,
) -> None:
    """Build an EPUB file from *chapters* list (#22 – error handling + fallback)."""
    try:
        book = epub.EpubBook()
        book.set_identifier(f"bulk-ocr-{int(time.time())}")
        book.set_title(title)
        book.set_language("de" if language == "german" else "en")

        epub_chapters = []
        for idx, chapter in enumerate(chapters, start=1):
            chapter_title = chapter["title"]
            paragraphs = split_into_paragraphs(chapter.get("text", ""))
            body_html = "".join(
                f"<p>{html.escape(p)}</p>" for p in paragraphs if p.strip()
            )
            if not body_html:
                body_html = "<p></p>"
            chapter_filename = f"chapter_{idx}.xhtml"
            epub_chapter = epub.EpubHtml(
                title=chapter_title,
                file_name=chapter_filename,
                lang="de" if language == "german" else "en",
            )
            epub_chapter.content = f"<h1>{html.escape(chapter_title)}</h1>{body_html}"
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)

        book.toc = tuple(epub_chapters)
        book.spine = ["nav"] + epub_chapters
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(output_path.as_posix(), book, {})
    except Exception as exc:
        # Fallback: dump as Markdown so the user never loses their text
        fallback = output_path.with_suffix(".md")
        if log_fn:
            log_fn(f"EPUB creation failed ({exc}). Saving Markdown fallback: {fallback.name}")
        fallback.write_text(build_markdown(chapters), encoding="utf-8")
        raise  # re-raise so the caller knows it failed


def build_markdown(chapters: List[dict]) -> str:
    sections = []
    for chapter in chapters:
        title = chapter.get("title") or "Untitled"
        body = chapter.get("text", "").strip()
        sections.append(f"# {title}\n\n{body}".strip())
    return "\n\n".join(sections).strip() + "\n"


def build_plain_text(chapters: List[dict]) -> str:
    """Build a simple plain-text output from chapters."""
    sections = []
    for chapter in chapters:
        title = chapter.get("title") or "Untitled"
        body = chapter.get("text", "").strip()
        sections.append(f"{title}\n{'=' * len(title)}\n\n{body}")
    return "\n\n\n".join(sections).strip() + "\n"


def build_html_output(chapters: List[dict], title: str, language: str) -> str:
    """Build a self-contained HTML file from chapters (#10)."""
    lang_code = "de" if language == "german" else "en"
    parts = [
        f'<!DOCTYPE html>\n<html lang="{lang_code}">\n<head>',
        f'<meta charset="utf-8"><title>{html.escape(title)}</title>',
        '<style>body{font-family:Georgia,serif;max-width:42em;margin:2em auto;'
        'line-height:1.6;padding:0 1em}h1{margin-top:2em}</style>',
        '</head><body>',
    ]
    for chapter in chapters:
        ch_title = chapter.get("title") or "Untitled"
        paragraphs = split_into_paragraphs(chapter.get("text", ""))
        parts.append(f"<h1>{html.escape(ch_title)}</h1>")
        for p in paragraphs:
            if p.strip():
                parts.append(f"<p>{html.escape(p)}</p>")
    parts.append('</body></html>')
    return "\n".join(parts)


def detect_book_title(client: OpenAI, text: str, log_fn=None) -> str:
    """Ask the AI to detect the book title from the first ~2000 chars (#9)."""
    sample = text[:2000]
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[{
                "role": "user",
                "content": (
                    "Analysiere den folgenden Textanfang eines Buches und gib NUR den "
                    "Buchtitel zurück. Kein zusätzlicher Text, keine Anführungszeichen, "
                    "nur den Titel. Falls kein Titel erkennbar, antworte 'Untitled'.\n\n"
                    f"{sample}"
                ),
            }],
            temperature=0.0,
        )
        title = (response.choices[0].message.content or "Untitled").strip().strip('"').strip("'")
        if log_fn:
            log_fn(f"AI detected book title: {title}")
        return title if title else "Untitled"
    except Exception as exc:
        if log_fn:
            log_fn(f"Could not detect book title: {exc}")
        return "Untitled"


class EpubBuilderApp:
    def __init__(self, root: Tk):
        self.root = root
        self.settings = load_settings()
        self.project_root = get_app_data_dir()
        self.paths = ensure_project_dirs(self.project_root)
        self.log_file = self.paths["logs"] / "bulk_ocr.log"
        self.progress_text_file = ocr_progress_text_path(self.paths["logs"])
        self._icon_photo: Optional[ImageTk.PhotoImage] = None

        self._configure_style()
        self._set_app_icon()
        self.root.title(f"Assistive OCR EPUB Builder v{APP_VERSION}")

        self.mode_var = StringVar(value="pdf")
        self.skip_toc_var = BooleanVar(value=False)
        self.output_format_var = StringVar(value=self.settings.output_format or "epub")
        self.stack_images_var = BooleanVar(value=True)
        self.stack_batch_size_var = IntVar(value=4)
        self.pdf_path: Optional[Path] = (
            Path(self.settings.last_pdf_path) if self.settings.last_pdf_path else None
        )
        self.txt_path: Optional[Path] = (
            Path(self.settings.last_txt_path) if self.settings.last_txt_path else None
        )
        self.batch_result_paths: List[Path] = []
        self.toc_paths: List[Path] = []
        self.image_folder: Optional[Path] = (
            Path(self.settings.last_image_folder) if self.settings.last_image_folder else None
        )

        # --- Thread-safety primitives (#4, #5) ---
        self._run_lock = threading.Lock()
        self.is_running = False
        self.stop_event = threading.Event()     # raised when user clicks Stop
        self.pause_event = threading.Event()

        self.progress_var = IntVar(value=0)
        self.progress_label_var = StringVar(value="OCR progress: 0%")
        self.eta_label_var = StringVar(value="ETA: --")
        self.pause_button: Optional[ttk.Button] = None
        self.stop_button: Optional[ttk.Button] = None
        self.start_time: Optional[float] = None

        self._build_ui()
        if not self.settings.api_key or not self.settings.api_key.strip():
            self.root.after(200, self.open_settings_dialog)

    def _configure_style(self) -> None:
        """Apply sv-ttk theme or fall back to clam (#31)."""
        self.root.option_add("*Font", "{Segoe UI} 10")
        if _HAS_SV_TTK:
            theme = self.settings.theme_mode if self.settings.theme_mode in ("light", "dark") else "light"
            sv_ttk.set_theme(theme)
        else:
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("TButton", padding=6)
            style.configure("TLabel", padding=2)
            style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
            self.root.configure(background="#f4f4f6")

    def _toggle_theme(self) -> None:
        """Toggle between light and dark mode (#31)."""
        if not _HAS_SV_TTK:
            return
        current = sv_ttk.get_theme()
        new_theme = "dark" if current == "light" else "light"
        sv_ttk.set_theme(new_theme)
        self.settings.theme_mode = new_theme
        save_settings(self.settings)

    # ------------------------------------------------------------------ #
    # Drag & Drop handler (#32)                                           #
    # ------------------------------------------------------------------ #
    def _on_drop_files(self, event) -> None:
        """Handle files/folders dropped onto the Source Files area."""
        raw = event.data
        # Tkdnd wraps paths with spaces in {}, split carefully
        paths: list[Path] = []
        for token in re.findall(r'\{([^}]+)\}|(\S+)', raw):
            p = Path(token[0] or token[1])
            if p.exists():
                paths.append(p)
        if not paths:
            return
        for p in paths:
            suffix = p.suffix.lower()
            if p.is_dir():
                images = [
                    f for f in p.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
                ]
                if images:
                    self.image_folder = p
                    self._set_project_root(p)
                    self.mode_var.set("images")
                    self._log(f"Dropped image folder: {p.name} ({len(images)} images)")
                else:
                    self._log(f"Dropped folder has no images: {p.name}", level="WARN")
            elif suffix == ".pdf":
                self.pdf_path = p
                self._set_project_root(p)
                self.mode_var.set("pdf")
                self._log(f"Dropped PDF: {p.name}")
            elif suffix == ".txt":
                self.txt_path = p
                self._set_project_root(p)
                self.mode_var.set("text")
                self._log(f"Dropped text file: {p.name}")
            elif suffix == ".jsonl":
                self.batch_result_paths = [p]
                self._set_project_root(p)
                self.mode_var.set("batch")
                self._log(f"Dropped JSONL: {p.name}")
            elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
                folder = p.parent
                self.image_folder = folder
                self._set_project_root(folder)
                self.mode_var.set("images")
                self._log(f"Dropped image: {p.name} → using folder {folder.name}")

    # ------------------------------------------------------------------ #
    # Batch Workflow Status helpers (#34)                                  #
    # ------------------------------------------------------------------ #
    def _update_batch_status(self, status: str, step: str = "",
                             files: str = "", progress: int = 0,
                             log_message: str = "") -> None:
        """Update the batch status panel from any thread."""
        def _apply():
            self._batch_status_var.set(status)
            if step:
                self._batch_step_var.set(f"Step: {step}")
            if files:
                self._batch_files_var.set(f"Files: {files}")
            self._batch_progress_var.set(max(0, min(100, progress)))
            if log_message:
                ts = timestamp()
                self._batch_log.insert("end", f"[{ts}] {log_message}\n")
                self._batch_log.see("end")
        self.root.after(0, _apply)

    def _refresh_batch_status(self) -> None:
        """Scan the JSONL directory for batch files and show summary (#34)."""
        jsonl_dir = self.paths.get("jsonl")
        if not jsonl_dir or not jsonl_dir.exists():
            self._update_batch_status("No JSONL directory found", progress=0)
            return
        jsonl_files = list(jsonl_dir.glob("*.jsonl"))
        # Also check for result JSONL in the jsonl dir and the output dir
        output_dir = self._get_output_dir()
        result_files = []
        for search_dir in (jsonl_dir, output_dir):
            if search_dir.exists():
                result_files.extend(
                    f for f in search_dir.glob("*.jsonl")
                    if "result" in f.stem.lower() or "output" in f.stem.lower()
                )
        # Deduplicate by resolved path
        result_files = list({f.resolve(): f for f in result_files}.values())
        total = len(jsonl_files)
        done = len(result_files)
        if total == 0:
            self._update_batch_status(
                "No batch files found",
                step="Upload JSONL first",
                files=f"{total} JSONL",
                progress=0,
                log_message="Refreshed: no batch files in project folder.",
            )
        else:
            pct = int((done / max(total, 1)) * 100)
            self._update_batch_status(
                f"Found {total} JSONL file(s)",
                step=f"{done} result(s) downloaded" if done else "Awaiting results",
                files=f"{total} JSONL, {done} results",
                progress=pct,
                log_message=f"Refreshed: {total} JSONL, {done} results ({pct}%).",
            )

    def _clear_batch_status(self) -> None:
        """Reset the batch status panel (#34). Thread-safe."""
        def _apply():
            self._batch_status_var.set("No batch job active")
            self._batch_step_var.set("Step: --")
            self._batch_files_var.set("Files: --")
            self._batch_progress_var.set(0)
            self._batch_log.delete("1.0", "end")
        self.root.after(0, _apply)

    def _set_app_icon(self) -> None:
        icon_path = find_icon_file()
        if not icon_path:
            return
        try:
            if icon_path.suffix.lower() == ".ico" and os.name == "nt":
                self.root.iconbitmap(icon_path.as_posix())
                return
            image = Image.open(icon_path)
            self._icon_photo = ImageTk.PhotoImage(image)
            self.root.iconphoto(True, self._icon_photo)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # Thread-safe messagebox wrappers (#1)                                #
    # ------------------------------------------------------------------ #
    def _show_error(self, title: str, message: str) -> None:
        """Show error dialog from ANY thread – delegates to the main thread."""
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def _show_info(self, title: str, message: str) -> None:
        self.root.after(0, lambda: messagebox.showinfo(title, message))

    def _show_warning(self, title: str, message: str) -> None:
        self.root.after(0, lambda: messagebox.showwarning(title, message))

    def _build_ui(self) -> None:
        self._build_menu()
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        dashboard_tab = ttk.Frame(self.notebook, padding=10)
        tools_tab = ttk.Frame(self.notebook, padding=10)
        settings_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(dashboard_tab, text="Dashboard")
        self.notebook.add(tools_tab, text="Tools")
        self.notebook.add(settings_tab, text="Settings & Logs")

        dashboard_tab.columnconfigure(0, weight=1)
        dashboard_tab.columnconfigure(1, weight=1)

        mode_frame = ttk.Labelframe(dashboard_tab, text="Input Mode", padding=8)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        mode_frame.columnconfigure(0, weight=1)
        mode_frame.columnconfigure(1, weight=1)

        mode_pdf = ttk.Radiobutton(
            mode_frame,
            text="Mode A: PDF + TOC images → OCR → EPUB",
            variable=self.mode_var,
            value="pdf",
        )
        mode_txt = ttk.Radiobutton(
            mode_frame,
            text="Mode B: Textfile + TOC images → EPUB",
            variable=self.mode_var,
            value="text",
        )
        mode_batch = ttk.Radiobutton(
            mode_frame,
            text="Mode C: Import Batch Result → TOC images → EPUB",
            variable=self.mode_var,
            value="batch",
        )
        mode_images = ttk.Radiobutton(
            mode_frame,
            text="Mode D: Image folder → OCR → EPUB",
            variable=self.mode_var,
            value="images",
        )
        mode_pdf.grid(row=0, column=0, columnspan=2, sticky="w")
        mode_txt.grid(row=1, column=0, columnspan=2, sticky="w")
        mode_batch.grid(row=2, column=0, columnspan=2, sticky="w")
        mode_images.grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Checkbutton(
            mode_frame,
            text="Skip TOC (single chapter)",
            variable=self.skip_toc_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        action_frame = ttk.Labelframe(dashboard_tab, text="Source Files", padding=8)
        action_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        ttk.Button(action_frame, text="Select PDF", command=self.select_pdf).grid(
            row=0, column=0, sticky="ew", pady=4, padx=(0, 6)
        )
        ttk.Button(action_frame, text="Select text file", command=self.select_txt).grid(
            row=0, column=1, sticky="ew", pady=4, padx=(6, 0)
        )
        ttk.Button(
            action_frame,
            text="Create Batch JSONL",
            command=self.create_batch_jsonl,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            action_frame,
            text="Select batch result JSONL",
            command=self.select_batch_results,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            action_frame,
            text="Select image folder",
            command=self.select_image_folder,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)

        # --- Drag & Drop zone (#32) ---
        if _HAS_DND:
            self._dnd_label = ttk.Label(
                action_frame,
                text="\u2193  Drop PDF / TXT / Images / Folder here  \u2193",
                anchor="center",
                relief="groove",
                padding=10,
            )
            self._dnd_label.grid(
                row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0)
            )
            self._dnd_label.drop_target_register(DND_FILES)
            self._dnd_label.dnd_bind("<<Drop>>", self._on_drop_files)

        options_frame = ttk.Labelframe(dashboard_tab, text="Options", padding=8)
        options_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        options_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            options_frame,
            text="Stack multiple images per request",
            variable=self.stack_images_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(options_frame, text="Max images per request").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Spinbox(
            options_frame,
            from_=1,
            to=20,
            textvariable=self.stack_batch_size_var,
            width=6,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(options_frame, text="Output format").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        format_combo = ttk.Combobox(
            options_frame,
            textvariable=self.output_format_var,
            values=["epub", "txt", "html"],
            state="readonly",
            width=10,
        )
        format_combo.grid(row=2, column=1, sticky="w", pady=(6, 0))

        action_buttons = ttk.Frame(dashboard_tab)
        action_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        action_buttons.columnconfigure(0, weight=1)
        action_buttons.columnconfigure(1, weight=1)
        action_buttons.columnconfigure(2, weight=1)

        ttk.Button(action_buttons, text="Start", command=self.start).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self.pause_button = ttk.Button(
            action_buttons, text="Pause", command=self.toggle_pause, state="disabled"
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stop_button = ttk.Button(
            action_buttons, text="Stop", command=self.request_stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        progress_frame = ttk.Frame(dashboard_tab)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_label_var)
        progress_label.grid(row=0, column=0, sticky="w")

        progress_bar_frame = ttk.Frame(progress_frame)
        progress_bar_frame.grid(row=1, column=0, sticky="ew")
        progress_bar_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(
            progress_bar_frame, maximum=100, variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_bar_frame, textvariable=self.eta_label_var).grid(
            row=0, column=1, sticky="e", padx=(10, 0)
        )

        version_label = ttk.Label(
            dashboard_tab, text=f"Version {APP_VERSION}", foreground="#666666"
        )
        version_label.grid(row=5, column=0, columnspan=2, sticky="e", pady=(6, 0))

        tools_tab.columnconfigure(0, weight=1)
        tool_frame = ttk.LabelFrame(tools_tab, text="Workbench Tools", padding=12)
        tool_frame.grid(row=0, column=0, sticky="nsew")
        tool_frame.columnconfigure(0, weight=1)
        tool_frame.columnconfigure(1, weight=1)

        def tool_button(row, column, text, app_class, key):
            state = "normal" if ADDONS_AVAILABLE.get(key) else "disabled"
            btn = ttk.Button(
                tool_frame,
                text=text,
                command=lambda: self._launch_addon_window(app_class, text),
                state=state,
            )
            btn.grid(row=row, column=column, sticky="ew", padx=6, pady=6, ipady=12)
            return btn

        tool_button(0, 0, "Upload JSONL Batches", BatchUploaderApp, "uploader")
        tool_button(0, 1, "Manage & Download Batches", BatchManagerApp, "manager")
        tool_button(1, 0, "Convert JSONL to TXT", JsonlToTxtApp, "converter")
        tool_button(1, 1, "AI Text Refiner (Post-Process)", AiTextRefinerApp, "refiner")

        # --- Batch Workflow Status panel (#34) ---
        tools_tab.rowconfigure(1, weight=1)
        batch_status_frame = ttk.LabelFrame(tools_tab, text="Batch Workflow Status", padding=12)
        batch_status_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        batch_status_frame.columnconfigure(0, weight=1)
        batch_status_frame.columnconfigure(1, weight=1)

        self._batch_status_var = StringVar(value="No batch job active")
        self._batch_files_var = StringVar(value="Files: --")
        self._batch_progress_var = IntVar(value=0)
        self._batch_step_var = StringVar(value="Step: --")

        ttk.Label(batch_status_frame, textvariable=self._batch_status_var, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(batch_status_frame, textvariable=self._batch_step_var).grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(batch_status_frame, textvariable=self._batch_files_var).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        self._batch_progress_bar = ttk.Progressbar(
            batch_status_frame, maximum=100, variable=self._batch_progress_var
        )
        self._batch_progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 4))

        ttk.Button(
            batch_status_frame, text="Refresh Status", command=self._refresh_batch_status
        ).grid(row=4, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            batch_status_frame, text="Clear", command=self._clear_batch_status
        ).grid(row=4, column=1, sticky="ew", padx=(4, 0))

        # Batch status log (compact)
        self._batch_log = ScrolledText(batch_status_frame, height=6, font=("Consolas", 9))
        self._batch_log.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        batch_status_frame.rowconfigure(5, weight=1)

        settings_tab.columnconfigure(0, weight=1)
        settings_frame = ttk.LabelFrame(settings_tab, text="Settings", padding=8)
        settings_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(0, weight=1)

        ttk.Label(settings_frame, text="OpenAI API Key").grid(
            row=0, column=0, sticky="w"
        )
        self.api_key_var = StringVar(value=self.settings.api_key)
        api_entry = ttk.Entry(
            settings_frame, textvariable=self.api_key_var, show="*", width=60
        )
        api_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(settings_frame, text="OCR Prompt (English)").grid(
            row=2, column=0, sticky="w"
        )
        self.ocr_en_text = ScrolledText(settings_frame, height=4, width=60)
        self.ocr_en_text.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.ocr_en_text.insert("1.0", self.settings.ocr_prompt_en)

        ttk.Label(settings_frame, text="OCR Prompt (German)").grid(
            row=4, column=0, sticky="w"
        )
        self.ocr_de_text = ScrolledText(settings_frame, height=4, width=60)
        self.ocr_de_text.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.ocr_de_text.insert("1.0", self.settings.ocr_prompt_de)

        ttk.Label(settings_frame, text="TOC Prompt (English)").grid(
            row=6, column=0, sticky="w"
        )
        self.toc_en_text = ScrolledText(settings_frame, height=4, width=60)
        self.toc_en_text.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        self.toc_en_text.insert("1.0", self.settings.toc_prompt_en)

        ttk.Label(settings_frame, text="TOC Prompt (German)").grid(
            row=8, column=0, sticky="w"
        )
        self.toc_de_text = ScrolledText(settings_frame, height=4, width=60)
        self.toc_de_text.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        self.toc_de_text.insert("1.0", self.settings.toc_prompt_de)

        self.keep_pdf_var = BooleanVar(value=self.settings.keep_pdf_images)
        self.keep_toc_var = BooleanVar(value=self.settings.keep_toc_images)
        ttk.Checkbutton(
            settings_frame,
            text="Keep PDF images after OCR",
            variable=self.keep_pdf_var,
        ).grid(row=10, column=0, sticky="w")
        ttk.Checkbutton(
            settings_frame,
            text="Keep TOC images after OCR",
            variable=self.keep_toc_var,
        ).grid(row=11, column=0, sticky="w")

        # Output directory chooser (#18)
        out_dir_frame = ttk.Frame(settings_frame)
        out_dir_frame.grid(row=12, column=0, sticky="ew", pady=(8, 0))
        out_dir_frame.columnconfigure(1, weight=1)
        ttk.Label(out_dir_frame, text="Output folder:").grid(row=0, column=0, sticky="w")
        self.output_dir_var = StringVar(value=self.settings.custom_output_dir)
        ttk.Entry(out_dir_frame, textvariable=self.output_dir_var, width=40).grid(
            row=0, column=1, sticky="ew", padx=(6, 6)
        )
        ttk.Button(
            out_dir_frame, text="Browse\u2026", width=8,
            command=self._choose_output_dir,
        ).grid(row=0, column=2)
        ttk.Label(
            out_dir_frame, text="(Leave empty for default)", foreground="#888",
        ).grid(row=1, column=1, sticky="w", padx=(6, 0))

        ttk.Button(
            settings_frame, text="Save Settings", command=self._save_settings_from_ui
        ).grid(row=13, column=0, sticky="e", pady=(8, 0))

        log_frame = ttk.Labelframe(settings_tab, text="Logs", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=1)
        log_frame.rowconfigure(0, weight=1)
        settings_tab.rowconfigure(1, weight=1)

        self.log = ScrolledText(log_frame, height=18)
        self.log.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        self.log.configure(font=("Consolas", 10))

        ttk.Button(log_frame, text="Clear log", command=self.clear_log).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(log_frame, text="Open output folder", command=self.open_output).grid(
            row=1, column=1, sticky="ew", padx=(6, 0)
        )
        ttk.Button(log_frame, text="Export log", command=self.export_log).grid(
            row=2, column=0, sticky="ew", pady=(6, 0), padx=(0, 6)
        )
        ttk.Button(log_frame, text="Open log file", command=self.open_log_file).grid(
            row=2, column=1, sticky="ew", pady=(6, 0), padx=(6, 0)
        )
        ttk.Button(
            log_frame, text="Open OCR progress text", command=self.open_progress_text
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self._log(f"Project folder: {self.project_root}")
        self._log(f"Output folder: {self.paths['output']}")
        self._log(f"Log file: {self.log_file}")

    def _build_menu(self) -> None:
        menu_bar = Menu(self.root)
        settings_menu = Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Settings", command=self.open_settings_dialog)
        settings_menu.add_command(label="Open output folder", command=self.open_output)
        if _HAS_SV_TTK:
            settings_menu.add_separator()
            settings_menu.add_command(
                label="Toggle Dark / Light Mode", command=self._toggle_theme
            )
        menu_bar.add_cascade(label="Options", menu=settings_menu)
        log_menu = Menu(menu_bar, tearoff=0)
        log_menu.add_command(label="Open log file", command=self.open_log_file)
        log_menu.add_command(label="Export log", command=self.export_log)
        menu_bar.add_cascade(label="Logs", menu=log_menu)
        self.root.config(menu=menu_bar)

    def open_settings_dialog(self) -> None:
        if hasattr(self, "notebook"):
            self.notebook.select(2)
            self.root.focus_force()

    def _choose_output_dir(self) -> None:
        """Let the user pick a custom output folder (#18)."""
        initial = self.output_dir_var.get() or str(self.paths["output"])
        chosen = filedialog.askdirectory(initialdir=initial, title="Select output folder")
        if chosen:
            self.output_dir_var.set(chosen)

    def _get_output_dir(self) -> Path:
        """Return the effective output directory, respecting custom override (#18)."""
        custom = self.settings.custom_output_dir
        if custom and Path(custom).is_dir():
            return Path(custom)
        return self.paths["output"]

    def _save_settings_from_ui(self) -> None:
        self.settings = AppSettings(
            api_key=self.api_key_var.get().strip(),
            ocr_prompt_en=self.ocr_en_text.get("1.0", "end").strip(),
            ocr_prompt_de=self.ocr_de_text.get("1.0", "end").strip(),
            toc_prompt_en=self.toc_en_text.get("1.0", "end").strip(),
            toc_prompt_de=self.toc_de_text.get("1.0", "end").strip(),
            keep_pdf_images=self.keep_pdf_var.get(),
            keep_toc_images=self.keep_toc_var.get(),
            last_pdf_path=str(self.pdf_path) if self.pdf_path else "",
            last_txt_path=str(self.txt_path) if self.txt_path else "",
            last_image_folder=str(self.image_folder) if self.image_folder else "",
            output_format=self.output_format_var.get(),
            custom_output_dir=self.output_dir_var.get().strip(),
            theme_mode=self.settings.theme_mode,
        )
        save_settings(self.settings)
        self._log("Settings saved.")

    def _set_project_root(self, path: Path) -> None:
        if path.is_file():
            project_root = path.parent
        else:
            project_root = path
        if project_root == self.project_root:
            return
        self.project_root = project_root
        self.paths = ensure_project_dirs(self.project_root)
        self.log_file = self.paths["logs"] / "bulk_ocr.log"
        self.progress_text_file = ocr_progress_text_path(self.paths["logs"])
        self._log(f"Project folder set to: {self.project_root}")

    def _launch_addon_window(self, AppClass, title_prefix="Tool"):
        try:
            if AppClass is None:
                raise RuntimeError("Addon is not available in this build.")
            top = Toplevel(self.root)
            app = AppClass(top)
            if hasattr(app, "api_key_var") and self.settings.api_key:
                app.api_key_var.set(self.settings.api_key)
            top.focus_force()
        except Exception as e:
            self._show_error("Error", f"Could not launch {title_prefix}:\n{e}")

    def _format_eta(self, seconds: float) -> str:
        if seconds < 0 or math.isinf(seconds):
            return "ETA: --"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"ETA: {hours:02d}:{minutes:02d}:{secs:02d}"

    def _log(self, message: str, level: str = "INFO") -> None:
        """Write a log entry with optional level (#21).

        *level* should be one of ``INFO``, ``WARN``, ``ERROR``.
        """
        entry = f"[{timestamp()}] [{level}] {message}"
        # Schedule UI update on main thread (#1 thread-safety)
        def _insert():
            try:
                self.log.insert("end", f"{entry}\n")
                self.log.see("end")
            except Exception:
                pass
        self.root.after(0, _insert)
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{entry}\n")
        except Exception:
            return

    def _update_progress(self, current: int, total: int) -> None:
        percent = int((current / total) * 100) if total else 0
        eta_text = "ETA: --"
        if self.start_time and current > 0 and total > 0:
            elapsed = time.time() - self.start_time
            time_per_page = elapsed / current
            remaining = total - current
            eta_text = self._format_eta(time_per_page * remaining)

        def apply() -> None:
            self.progress_var.set(percent)
            self.progress_label_var.set(
                f"OCR progress: {percent}% ({current}/{total})"
            )
            self.eta_label_var.set(eta_text)

        self.root.after(0, apply)

    def _write_progress_text(self, text_chunks: List[str]) -> None:
        try:
            self.progress_text_file.parent.mkdir(parents=True, exist_ok=True)
            text = "\n".join(text_chunks).strip()
            self.progress_text_file.write_text(text, encoding="utf-8")
        except Exception:
            return

    def _wait_if_paused(self) -> None:
        """Block the worker thread while paused. Also checks for stop."""
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.2)

    def _check_stopped(self) -> bool:
        """Return True if the user requested a stop."""
        return self.stop_event.is_set()

    def request_stop(self) -> None:
        """Called from the UI when the user clicks Stop."""
        if not self.is_running:
            return
        self.stop_event.set()
        self.pause_event.clear()  # un-pause so the thread can exit
        self._log("Stop requested – finishing current page then stopping.")
        if self.stop_button:
            self.stop_button.configure(state="disabled")

    def toggle_pause(self) -> None:
        if not self.is_running:
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            if self.pause_button:
                self.pause_button.configure(text="Pause")
            self._log("Resumed OCR processing.")
        else:
            self.pause_event.set()
            if self.pause_button:
                self.pause_button.configure(text="Resume")
            self._log("Paused OCR processing.")

    def open_progress_text(self) -> None:
        if not self.progress_text_file.exists():
            self._log("OCR progress text does not exist yet.", level="WARN")
            return
        open_file_in_editor(self.progress_text_file)

    def _ocr_prompt_for_language(self, language: str) -> str:
        if language == "german":
            return self.settings.ocr_prompt_de
        return self.settings.ocr_prompt_en

    def _toc_prompt_for_language(self, language: str) -> str:
        if language == "german":
            return self.settings.toc_prompt_de
        return self.settings.toc_prompt_en

    def clear_log(self) -> None:
        self.log.delete("1.0", "end")
        try:
            if self.log_file.exists():
                self.log_file.unlink()
        except Exception:
            return

    def export_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export log",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
        )
        if not path:
            return
        content = self.log.get("1.0", "end").strip()
        try:
            Path(path).write_text(content, encoding="utf-8")
            self._log(f"Log exported to: {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not export log: {exc}")

    def open_log_file(self) -> None:
        if not self.log_file.exists():
            self._log("Log file does not exist yet.", level="WARN")
            return
        open_output_folder(self.log_file)

    def select_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            self.pdf_path = Path(path)
            self._set_project_root(self.pdf_path)
            self._log(f"Selected PDF: {self.pdf_path.name}")
            if self.skip_toc_var.get():
                self._log("Skip TOC enabled. Not selecting TOC pages.")
            else:
                self._select_toc_pages_from_pdf()

    def select_txt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt")],
        )
        if path:
            self.txt_path = Path(path)
            self._set_project_root(self.txt_path)
            self._log(f"Selected text file: {self.txt_path.name}")

    def select_batch_results(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Batch Result JSONL",
            filetypes=[("JSONL files", "*.jsonl")],
        )
        if paths:
            self.batch_result_paths = [Path(path) for path in paths]
            if self.batch_result_paths:
                self._set_project_root(self.batch_result_paths[0])
            names = ", ".join(path.name for path in self.batch_result_paths)
            self._log(f"Selected batch result files: {names}")

    def select_image_folder(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            folder = Path(path)
            self.image_folder = folder
            self._set_project_root(folder)
            images = collect_image_files(folder)
            if not images:
                messagebox.showwarning(
                    "No images",
                    "No PNG/JPEG images found in the selected folder.",
                )
            self._log(
                f"Selected image folder: {folder} ({len(images)} image(s) found)"
            )

    def create_batch_jsonl(self) -> None:
        if not self.pdf_path:
            messagebox.showerror("Missing PDF", "Please select a PDF file first.")
            return
        use_german = messagebox.askyesno(
            "Batch Prompt",
            "Use the German OCR prompt for the batch file?\n"
            "(Select 'No' to use English.)",
        )
        prompt = (
            self.settings.ocr_prompt_de if use_german else self.settings.ocr_prompt_en
        )
        save_path = filedialog.asksaveasfilename(
            title="Save Batch JSONL",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl")],
            initialdir=self.paths["jsonl"],
        )
        if not save_path:
            return
        thread = threading.Thread(
            target=self._create_batch_jsonl_worker,
            args=(prompt, Path(save_path)),
            daemon=True,
        )
        thread.start()

    def _create_batch_jsonl_worker(self, prompt: str, save_path: Path) -> None:
        self._log("Generating images for batch JSONL export.")
        self._update_batch_status("Creating batch JSONL…", step="Extracting PDF images", progress=10)
        images = extract_pdf_images(
            self.pdf_path,
            self.paths["pdfimgs"],
            cleanup=False,
            log_fn=self._log,
        )
        if not images:
            self._show_error("Batch Export Failed", "No images extracted.")
            self._update_batch_status("Batch creation failed", step="No images", progress=0,
                                      log_message="Batch JSONL creation failed: no images.")
            return
        self._update_batch_status("Creating batch JSONL…", step="Building JSONL entries",
                                  files=f"{len(images)} pages", progress=40)
        pdf_stem = self.pdf_path.stem

        def iter_entries() -> Iterable[str]:
            for idx, image_path in enumerate(images, start=1):
                b64 = encode_image_to_base64(image_path)
                mime = _mime_for_image(image_path)  # #24
                body = {
                    "model": OPENAI_MODEL_VISION,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.0,
                }
                entry = {
                    "custom_id": f"{pdf_stem}_page_{idx:04d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                yield json.dumps(entry, ensure_ascii=False)

        output_paths = _write_batch_jsonl_entries(
            iter_entries(), save_path, BATCH_MAX_FILE_BYTES, log_fn=self._log
        )
        if len(output_paths) == 1:
            self._log(f"Batch JSONL saved: {output_paths[0]}")
        else:
            names = ", ".join(path.name for path in output_paths)
            self._log(f"Batch JSONL saved: {names}")
        self._update_batch_status(
            f"Batch JSONL ready ({len(output_paths)} file(s))",
            step="Upload next",
            files=f"{len(images)} pages → {len(output_paths)} JSONL",
            progress=100,
            log_message=f"Created {len(output_paths)} JSONL file(s) with {len(images)} pages.",
        )

    def _select_toc_pages_from_pdf(self) -> None:
        if not self.pdf_path:
            return
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as exc:
            messagebox.showerror("PDF Error", f"Could not open PDF: {exc}")
            return
        page_count = doc.page_count
        if page_count == 0:
            doc.close()
            messagebox.showerror("PDF Error", "PDF has no pages.")
            return
        selected_indices = self._open_toc_selector_dialog(doc, page_count)
        doc.close()
        if not selected_indices:
            self._log("No TOC pages selected.")
            self.toc_paths = []
            return
        self.toc_paths = render_pdf_pages_to_images(
            self.pdf_path,
            selected_indices,
            self.paths["tocimgs"],
            zoom=3.0,
            cleanup=not self.settings.keep_toc_images,
            log_fn=self._log,
        )
        self._log(f"Selected {len(self.toc_paths)} TOC page(s)")

    def _open_toc_selector_dialog(
        self, doc: fitz.Document, page_count: int
    ) -> List[int]:
        dialog = Toplevel(self.root)
        dialog.title("Select TOC Pages")
        dialog.geometry("1200x700")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="Select the pages that contain the Table of Contents.",
            padding=8,
        ).pack(anchor="w")

        canvas = Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_configure)

        thumbnail_refs: List[ImageTk.PhotoImage] = []
        vars_by_index: List[IntVar] = []
        columns = 4
        for index in range(page_count):
            page = doc.load_page(index)
            image = render_pdf_page_thumbnail(page, zoom=0.06)
            photo = ImageTk.PhotoImage(image)
            thumbnail_refs.append(photo)
            var = IntVar(value=0)
            vars_by_index.append(var)

            frame = ttk.Frame(inner, padding=4, relief="solid")
            frame.grid(row=index // columns, column=index % columns, padx=6, pady=6)
            ttk.Label(frame, image=photo).pack()
            ttk.Checkbutton(frame, text=f"Page {index + 1}", variable=var).pack()

        selected_indices: List[int] = []

        def confirm() -> None:
            selection = [idx for idx, var in enumerate(vars_by_index) if var.get() == 1]
            if not selection:
                messagebox.showwarning(
                    "No selection", "Please select at least one TOC page."
                )
                return
            selected_indices.extend(selection)
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        button_frame = ttk.Frame(dialog, padding=8)
        button_frame.pack(side="bottom", fill="x")
        ttk.Button(button_frame, text="Confirm selection", command=confirm).pack(
            side="right", padx=6
        )
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side="right")

        dialog.wait_window()
        return selected_indices

    def _open_toc_editor_dialog(
        self, toc_entries: List[str], full_text: str
    ) -> Optional[List[str]]:
        """Open a dialog to review / edit / reorder TOC entries (#13).

        Entries that cannot be found in the text are highlighted in red.
        Returns the (potentially modified) list or None if the user cancels.
        """
        result_holder: List[Optional[List[str]]] = [None]
        ready = threading.Event()

        def _show():
            dialog = Toplevel(self.root)
            dialog.title("Edit TOC Entries")
            dialog.geometry("600x500")
            dialog.transient(self.root)
            dialog.grab_set()

            ttk.Label(
                dialog,
                text="Review the detected chapters. Red = not found in text. Click to edit.",
                padding=8,
            ).pack(anchor="w")

            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill="both", expand=True, padx=8)

            from tkinter import Listbox, END, EXTENDED
            listbox = Listbox(
                list_frame, selectmode=EXTENDED, font=("Segoe UI", 10),
                activestyle="none",
            )
            listbox.pack(fill="both", expand=True, side="left")
            lb_scroll = ttk.Scrollbar(list_frame, command=listbox.yview)
            lb_scroll.pack(side="right", fill="y")
            listbox.configure(yscrollcommand=lb_scroll.set)

            text_lower = full_text.lower()
            entries = list(toc_entries)  # mutable copy

            def _refresh():
                listbox.delete(0, END)
                for entry in entries:
                    listbox.insert(END, entry)
                    idx = listbox.size() - 1
                    if entry.lower().strip() not in text_lower:
                        # fuzzy match fallback
                        best = fuzz.partial_ratio(entry.lower(), text_lower)
                        if best < 70:
                            listbox.itemconfig(idx, fg="red")

            _refresh()

            btn_frame = ttk.Frame(dialog, padding=8)
            btn_frame.pack(fill="x")

            def _edit_selected():
                sel = listbox.curselection()
                if not sel:
                    return
                idx = sel[0]
                from tkinter.simpledialog import askstring
                new_val = askstring(
                    "Edit entry", "Chapter title:", initialvalue=entries[idx],
                    parent=dialog,
                )
                if new_val is not None:
                    entries[idx] = new_val.strip()
                    _refresh()

            def _delete_selected():
                sel = sorted(listbox.curselection(), reverse=True)
                for idx in sel:
                    entries.pop(idx)
                _refresh()

            def _add_entry():
                from tkinter.simpledialog import askstring
                new_val = askstring("Add entry", "New chapter title:", parent=dialog)
                if new_val and new_val.strip():
                    entries.append(new_val.strip())
                    _refresh()

            def _move_up():
                sel = listbox.curselection()
                if not sel or sel[0] == 0:
                    return
                idx = sel[0]
                entries[idx - 1], entries[idx] = entries[idx], entries[idx - 1]
                _refresh()
                listbox.selection_set(idx - 1)

            def _move_down():
                sel = listbox.curselection()
                if not sel or sel[0] >= len(entries) - 1:
                    return
                idx = sel[0]
                entries[idx + 1], entries[idx] = entries[idx], entries[idx + 1]
                _refresh()
                listbox.selection_set(idx + 1)

            ttk.Button(btn_frame, text="\u25b2 Up", command=_move_up, width=6).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="\u25bc Down", command=_move_down, width=6).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Edit", command=_edit_selected, width=6).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Delete", command=_delete_selected, width=6).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Add", command=_add_entry, width=6).pack(side="left", padx=2)

            action_frame = ttk.Frame(dialog, padding=8)
            action_frame.pack(fill="x")

            def _confirm():
                result_holder[0] = list(entries)
                dialog.destroy()
            def _cancel():
                dialog.destroy()

            ttk.Button(action_frame, text="Confirm", command=_confirm).pack(side="right", padx=6)
            ttk.Button(action_frame, text="Cancel", command=_cancel).pack(side="right")

            dialog.protocol("WM_DELETE_WINDOW", _cancel)
            dialog.wait_window()
            ready.set()

        self.root.after(0, _show)
        ready.wait()
        return result_holder[0]

    # ------------------------------------------------------------------ #
    # OCR Text Preview before export (#33)                                #
    # ------------------------------------------------------------------ #
    def _show_text_preview(self, chapters: list[dict]) -> Optional[list[dict]]:
        """Show a preview dialog with the OCR text before writing output.

        The user can review and edit the text per chapter.
        Returns the (possibly edited) chapters or None if cancelled.
        """
        result_holder: list[Optional[list[dict]]] = [None]
        ready = threading.Event()

        def _show():
            dialog = Toplevel(self.root)
            dialog.title("OCR Text Preview (#33)")
            dialog.geometry("750x600")
            dialog.transient(self.root)
            dialog.grab_set()

            ttk.Label(
                dialog,
                text="Review and edit the OCR text before export. Select a chapter on the left.",
                padding=8,
            ).pack(fill="x")

            paned = ttk.PanedWindow(dialog, orient="horizontal")
            paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))

            # Chapter list
            list_frame = ttk.Frame(paned, padding=4)
            paned.add(list_frame, weight=1)
            chapter_listbox = Listbox(list_frame, width=30)
            chapter_listbox.pack(fill="both", expand=True)
            for i, ch in enumerate(chapters):
                title = ch.get("title", f"Chapter {i+1}")
                chapter_listbox.insert("end", title)

            # Text editor
            text_frame = ttk.Frame(paned, padding=4)
            paned.add(text_frame, weight=3)
            text_edit = ScrolledText(text_frame, wrap="word", font=("Consolas", 10))
            text_edit.pack(fill="both", expand=True)

            current_idx = [None]

            def _save_current():
                if current_idx[0] is not None:
                    chapters[current_idx[0]]["text"] = text_edit.get("1.0", "end").strip()

            def _on_select(event):
                sel = chapter_listbox.curselection()
                if not sel:
                    return
                _save_current()
                idx = sel[0]
                current_idx[0] = idx
                text_edit.delete("1.0", "end")
                text_edit.insert("1.0", chapters[idx].get("text", ""))

            chapter_listbox.bind("<<ListboxSelect>>", _on_select)

            # Select first chapter
            if chapters:
                chapter_listbox.selection_set(0)
                current_idx[0] = 0
                text_edit.insert("1.0", chapters[0].get("text", ""))

            # Word count label
            wc_var = StringVar(value="")

            def _update_wc(*_):
                _save_current()  # persist current edits so total is accurate
                total = sum(len(ch.get("text", "").split()) for ch in chapters)
                cur = len(text_edit.get("1.0", "end").split()) if current_idx[0] is not None else 0
                wc_var.set(f"Current: {cur} words | Total: {total} words")

            text_edit.bind("<KeyRelease>", _update_wc)
            ttk.Label(dialog, textvariable=wc_var, padding=4).pack(fill="x")
            _update_wc()

            # Buttons
            btn_frame = ttk.Frame(dialog, padding=8)
            btn_frame.pack(fill="x")

            def _confirm():
                _save_current()
                result_holder[0] = list(chapters)
                dialog.destroy()

            def _cancel():
                dialog.destroy()

            ttk.Button(btn_frame, text="Export", command=_confirm).pack(side="right", padx=6)
            ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side="right")

            dialog.protocol("WM_DELETE_WINDOW", _cancel)
            dialog.wait_window()
            ready.set()

        self.root.after(0, _show)
        ready.wait()
        return result_holder[0]

    def start(self) -> None:
        # --- Input validation before thread start (#7) ---
        mode = self.mode_var.get()
        if mode == "pdf" and self.pdf_path and not self.pdf_path.exists():
            self._show_error("File not found", f"PDF no longer exists:\n{self.pdf_path}")
            self.pdf_path = None
            return
        if mode == "text" and self.txt_path and not self.txt_path.exists():
            self._show_error("File not found", f"Text file no longer exists:\n{self.txt_path}")
            self.txt_path = None
            return
        if mode == "images" and self.image_folder and not self.image_folder.exists():
            self._show_error("Folder not found", f"Image folder no longer exists:\n{self.image_folder}")
            self.image_folder = None
            return
        if not self.settings.api_key or not self.settings.api_key.strip():
            self._show_error("Missing API Key", "Please configure your OpenAI API key in Settings.")
            return

        # Double-click guard with lock (#4)
        if not self._run_lock.acquire(blocking=False):
            self._show_warning("Already running", "OCR processing is already running.")
            return
        if self.is_running:
            self._run_lock.release()
            self._show_warning("Already running", "OCR processing is already running.")
            return
        self.is_running = True
        self.stop_event.clear()
        self._run_lock.release()
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def open_output(self) -> None:
        open_output_folder(self._get_output_dir())

    def _run_pipeline(self) -> None:
        # Enable buttons on the main thread
        def _enable_controls():
            if self.pause_button:
                self.pause_button.configure(state="normal")
            if self.stop_button:
                self.stop_button.configure(state="normal")
        self.root.after(0, _enable_controls)

        self._save_settings_from_ui()
        self.start_time = time.time()
        self.eta_label_var.set("ETA: --")
        self._update_progress(0, 1)
        try:
            client = build_openai_client(self.settings.api_key)
            mode = self.mode_var.get()
            skip_toc = self.skip_toc_var.get()
            if mode == "batch":
                if not self.batch_result_paths:
                    self._show_error(
                        "Missing batch result",
                        "Please select batch result JSONL files.",
                    )
                    return
                text, language = self._process_batch_results()
                if not skip_toc:
                    if not self.toc_paths:
                        if not self.pdf_path:
                            self._show_error(
                                "Missing PDF",
                                "Please select a PDF file to extract TOC images.",
                            )
                            return
                        self._select_toc_pages_from_pdf()
                    if not self.toc_paths:
                        self._show_error(
                            "Missing TOC", "Please select TOC images."
                        )
                        return
            elif mode == "pdf":
                if not self.pdf_path:
                    self._show_error("Missing PDF", "Please select a PDF file.")
                    return
                if not skip_toc and not self.toc_paths:
                    self._select_toc_pages_from_pdf()
                if not skip_toc and not self.toc_paths:
                    self._show_error("Missing TOC", "Please select TOC images.")
                    return
                if self._check_stopped():
                    return
                text, language = self._process_pdf(client)
            elif mode == "text":
                if not skip_toc and not self.toc_paths:
                    self._show_error("Missing TOC", "Please select TOC images.")
                    return
                if not self.txt_path:
                    self._show_error("Missing text", "Please select a text file.")
                    return
                text = self.txt_path.read_text(encoding="utf-8", errors="ignore")
                language = detect_language(text)
                self._log(f"Detected language: {language}")
            elif mode == "images":
                if not self.image_folder:
                    self._show_error(
                        "Missing images", "Please select an image folder."
                    )
                    return
                if not skip_toc and not self.toc_paths:
                    self._show_error("Missing TOC", "Please select TOC images.")
                    return
                if self._check_stopped():
                    return
                text, language = self._process_image_folder(client)
            else:
                self._show_error("Invalid mode", "Please select a valid mode.")
                return

            if self._check_stopped():
                self._log("Processing stopped by user.", level="WARN")
                return

            if skip_toc:
                self._log("Skipping TOC. Creating a single-chapter EPUB.")
                chapters = [{"title": "Full Text", "text": text}]
            else:
                toc_entries = ocr_toc(
                    client,
                    self.toc_paths,
                    language,
                    prompt_override=self._toc_prompt_for_language(language),
                    log_fn=self._log,
                )
                if not self.settings.keep_toc_images:
                    for image_path in self.toc_paths:
                        try:
                            image_path.unlink()
                        except Exception:
                            continue
                if not toc_entries:
                    self._show_error("TOC OCR failed", "No TOC entries detected.")
                    return
                self._log(f"Detected {len(toc_entries)} TOC entries")

                # --- TOC editor dialog (#13) ---
                edited = self._open_toc_editor_dialog(toc_entries, text)
                if edited is None:
                    self._log("TOC editing cancelled by user.", level="WARN")
                    return
                toc_entries = edited
                self._log(f"Using {len(toc_entries)} TOC entries after editing")

                chapters = build_chapters_from_toc(
                    text, toc_entries, threshold=83, log_fn=self._log
                )
                if not chapters:
                    self._show_error(
                        "Chapter detection failed", "No chapters could be matched."
                    )
                    return

            if self._check_stopped():
                self._log("Processing stopped by user.", level="WARN")
                return

            # --- Detect book title via AI (#9) ---
            book_title = detect_book_title(client, text, log_fn=self._log)

            # --- OCR Text Preview before export (#33) ---
            self._log("Opening text preview for review…")
            edited_chapters = self._show_text_preview(chapters)
            if edited_chapters is None:
                self._log("Export cancelled by user in preview.", level="WARN")
                return
            chapters = edited_chapters
            self._log("Text preview confirmed – proceeding to export.")

            # --- Output in chosen format (#10) ---
            fmt = self.output_format_var.get()
            ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = self._get_output_dir()  # #18
            out_dir.mkdir(parents=True, exist_ok=True)
            if fmt == "txt":
                output_name = f"output_{ts}.md"
                output_path = out_dir / output_name
                output_path.write_text(build_markdown(chapters), encoding="utf-8")
            elif fmt == "html":
                output_name = f"output_{ts}.html"
                output_path = out_dir / output_name
                output_path.write_text(
                    build_html_output(chapters, book_title, language), encoding="utf-8"
                )
            else:  # epub
                output_name = f"epub_{ts}.epub"
                output_path = out_dir / output_name
                create_epub(book_title, chapters, output_path, language, log_fn=self._log)
            self._log(f"Output created: {output_path.name}")
            self._show_info("Done", f"Output created: {output_path.name}")
            self._update_progress(1, 1)
        except Exception as exc:
            self._log(f"Error: {exc}", level="ERROR")
            self._show_error("Error", str(exc))
        finally:
            self.is_running = False
            self.pause_event.clear()
            self.stop_event.clear()
            def _reset_controls():
                if self.pause_button:
                    self.pause_button.configure(text="Pause", state="disabled")
                if self.stop_button:
                    self.stop_button.configure(state="disabled")
            self.root.after(0, _reset_controls)
            self.start_time = None
            self.eta_label_var.set("ETA: --")

    def _process_pdf(self, client: OpenAI) -> tuple[str, str]:
        pdfimgs_dir = self.paths["pdfimgs"]
        images = extract_pdf_images(
            self.pdf_path,
            pdfimgs_dir,
            cleanup=not self.settings.keep_pdf_images,
            log_fn=self._log,
        )
        if not images:
            raise ValueError("No images extracted from PDF.")
        state = load_ocr_state(self.paths["logs"])
        use_state = False
        if state and self.pdf_path:
            if (
                state.pdf_path == self.pdf_path.as_posix()
                and state.image_dir == pdfimgs_dir.as_posix()
                and state.image_files == [path.name for path in images]
            ):
                # Thread-safe resume dialog (#1)
                resume_event = threading.Event()
                resume_result = [False]
                def _ask_resume():
                    resume_result[0] = messagebox.askyesno(
                        "Resume OCR",
                        "An unfinished OCR session was found. Resume from last progress?",
                    )
                    resume_event.set()
                self.root.after(0, _ask_resume)
                resume_event.wait()  # block worker until user decides
                if resume_result[0]:
                    use_state = True
                else:
                    clear_ocr_state(self.paths["logs"])
        if use_state and state:
            language = state.language
            text_chunks = list(state.text_chunks)
            start_index = state.current_index
            self._log(
                f"Resuming OCR at page {start_index + 1} of {len(images)} (language: {language})."
            )
            self._write_progress_text(text_chunks)
        else:
            self._log("Running quick OCR for language detection")
            sample_text = ocr_pages(
                client,
                [images[0]],
                "english",
                prompt_override=self._ocr_prompt_for_language("english"),
                log_fn=self._log,
            )
            language = detect_language(sample_text)
            self._log(f"Detected language: {language}")
            text_chunks = []
            start_index = 0
            clear_ocr_state(self.paths["logs"])
            self._write_progress_text([])

        self._log("Starting OCR on PDF images")
        total = len(images)
        prompt = self._ocr_prompt_for_language(language)
        failed_pages: list[str] = []
        for idx in range(start_index, total):
            if self._check_stopped():
                self._log("OCR stopped by user. State saved for resume.", level="WARN")
                break
            self._wait_if_paused()
            page_num = idx + 1
            self._log(f"OCR page {page_num}/{total}")
            page_text = ocr_images_with_retry(
                client,
                [images[idx]],
                prompt,
                max_batch_size=1,
                log_fn=self._log,
            )
            # Replace generic error marker with page-numbered placeholder
            if _OCR_ERR in page_text:
                reason = "Leere Antwort" if ":LEER]]" in page_text else "API-Fehler"
                page_text = (
                    f"\n\n\u26a0\ufe0f [SEITE {page_num}: Konnte nicht ausgewertet werden"
                    f" ({reason}) \u2013 bitte manuell pr\u00fcfen]\n\n"
                )
                failed_pages.append(f"Seite {page_num} ({reason})")
                self._log(f"OCR-Fehler auf Seite {page_num}: {reason} – Platzhalter eingef\u00fcgt.", level="WARN")
            if len(text_chunks) <= idx:
                text_chunks.extend([""] * (idx + 1 - len(text_chunks)))
            text_chunks[idx] = page_text.strip()
            state = OcrState(
                pdf_path=self.pdf_path.as_posix(),
                image_dir=pdfimgs_dir.as_posix(),
                image_files=[path.name for path in images],
                language=language,
                current_index=idx + 1,
                text_chunks=text_chunks,
                created_at=timestamp(),
            )
            save_ocr_state(state, self.paths["logs"])
            self._write_progress_text(text_chunks)
            self._update_progress(idx + 1, total)

        clear_ocr_state(self.paths["logs"])
        full_text = "\n".join(text_chunks).strip()

        # Append error summary at end of document
        if failed_pages:
            summary = (
                "\n\n---\n\n"
                "## \u26a0\ufe0f OCR-Fehlerzusammenfassung\n\n"
                f"{len(failed_pages)} Seite(n) konnten nicht korrekt ausgewertet werden:\n\n"
                + "\n".join(f"- {p}" for p in failed_pages)
                + "\n\n_Bitte diese Seiten manuell pr\u00fcfen und korrigieren._\n\n---\n"
            )
            full_text += summary
            self._log(
                f"OCR-Fehlerzusammenfassung: {len(failed_pages)} Seite(n) fehlgeschlagen: "
                + ", ".join(failed_pages),
                level="WARN",
            )

        if not self.settings.keep_pdf_images:
            for image_path in images:
                try:
                    image_path.unlink()
                except Exception:
                    continue
        return full_text, language

    def _process_image_folder(self, client: OpenAI) -> tuple[str, str]:
        if not self.image_folder:
            raise ValueError("No image folder selected.")
        images = collect_image_files(self.image_folder)
        if not images:
            raise ValueError("No PNG/JPEG images found in the selected folder.")
        self._log("Running quick OCR for language detection")
        sample_text = ocr_pages(
            client,
            [images[0]],
            "english",
            prompt_override=self._ocr_prompt_for_language("english"),
            log_fn=self._log,
        )
        language = detect_language(sample_text)
        self._log(f"Detected language: {language}")

        prompt = self._ocr_prompt_for_language(language)
        stack_images = self.stack_images_var.get()
        batch_size = self.stack_batch_size_var.get()
        if not stack_images or batch_size < 1:
            batch_size = 1
        total = len(images)
        batches = chunk_list(images, batch_size)
        text_chunks: List[str] = []
        self._log(
            f"Starting OCR on {total} images (batch size: {batch_size})"
        )
        processed = 0
        failed_batches: list[str] = []
        num_batches = len(batches)
        for idx, batch in enumerate(batches, start=1):
            if self._check_stopped():
                self._log("OCR stopped by user.", level="WARN")
                break
            self._wait_if_paused()
            start_img = processed + 1
            end_img = processed + len(batch)
            batch_label = f"Bild {start_img}" if start_img == end_img else f"Bilder {start_img}\u2013{end_img}"
            self._log(f"OCR batch {idx}/{num_batches} ({batch_label})")
            batch_text = ocr_images_with_retry(
                client,
                batch,
                prompt,
                max_batch_size=batch_size,
                log_fn=self._log,
            )
            # Replace generic error marker with image-numbered placeholder
            if _OCR_ERR in batch_text:
                reason = "Leere Antwort" if ":LEER]]" in batch_text else "API-Fehler"
                batch_text = (
                    f"\n\n\u26a0\ufe0f [{batch_label}: Konnte nicht ausgewertet werden"
                    f" ({reason}) \u2013 bitte manuell pr\u00fcfen]\n\n"
                )
                failed_batches.append(f"{batch_label} ({reason})")
                self._log(f"OCR-Fehler bei {batch_label}: {reason} – Platzhalter eingef\u00fcgt.", level="WARN")
            text_chunks.append(batch_text.strip())
            processed += len(batch)
            self._write_progress_text(text_chunks)
            self._update_progress(processed, total)
        full_text = "\n".join(text_chunks).strip()

        # Append error summary at end of document
        if failed_batches:
            summary = (
                "\n\n---\n\n"
                "## \u26a0\ufe0f OCR-Fehlerzusammenfassung\n\n"
                f"{len(failed_batches)} Bild-Batch(es) konnten nicht korrekt ausgewertet werden:\n\n"
                + "\n".join(f"- {b}" for b in failed_batches)
                + "\n\n_Bitte diese Bilder manuell pr\u00fcfen und korrigieren._\n\n---\n"
            )
            full_text += summary
            self._log(
                f"OCR-Fehlerzusammenfassung: {len(failed_batches)} Batch(es) fehlgeschlagen: "
                + ", ".join(failed_batches),
                level="WARN",
            )

        return full_text, language

    def _process_batch_results(self) -> tuple[str, str]:
        if not self.batch_result_paths:
            raise ValueError("No batch result files selected.")
        self._log("Importing batch results and reconstructing text.")
        text = parse_batch_result_files(self.batch_result_paths, log_fn=self._log)
        if not text:
            raise ValueError("No OCR text found in batch results.")
        language = detect_language(text)
        self._log(f"Detected language: {language}")
        return text, language


def _global_thread_exception_handler(args):
    """Global handler for uncaught exceptions in threads (#2).
    Logs them to a crash file so they are never silently lost.
    """
    import traceback
    crash_log = get_app_data_dir() / "_logs" / "crash.log"
    crash_log.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"[{timestamp()}] Unhandled exception in thread '{args.thread.name}':\n"
        f"{''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))}\n"
    )
    try:
        with crash_log.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass
    # Also print to stderr for dev convenience
    print(entry, file=sys.stderr)


def main() -> None:
    # Install global thread exception handler (#2)
    threading.excepthook = _global_thread_exception_handler

    # Use TkinterDnD root for drag & drop support (#32)
    if _HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = Tk()
    app = EpubBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # <--- Add this line
    main()
