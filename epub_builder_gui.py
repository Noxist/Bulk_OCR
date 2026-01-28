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
# OpenAI API KEY (replace this with your own key)
# ============================================================
APP_VERSION = "0.3.1"
OPENAI_API_KEY = "REPLACE_WITH_YOUR_API_KEY"
OPENAI_MODEL_VISION = "gpt-4o-mini"
BATCH_MAX_FILE_BYTES = 140 * 1024 * 1024
DEFAULT_OCR_PROMPT_EN = (
    "You are an OCR engine for audiobook-friendly text. Extract only the main body text visible in the "
    "image. Remove citation markers like [12], (Smith, 1999), or ¹, and drop footnote lines entirely. "
    "Expand abbreviations (e.g. -> for example, Dr. -> Doctor). Ignore running headers, footers, page "
    "numbers, and marginalia. Fix line-break hyphenations. Keep paragraphs intact; keep list items on "
    "separate lines. Return pure running text in the original language."
)
DEFAULT_OCR_PROMPT_DE = (
    "Du bist eine OCR-Engine für hörbuchfreundlichen Text. Extrahiere ausschließlich den Haupttext, "
    "der im Bild sichtbar ist. Entferne Zitationsmarker wie [12], (Smith, 1999) oder ¹ und lasse "
    "Fußnotenzeilen komplett weg. Schreibe Abkürzungen aus (z. B. -> zum Beispiel, Dr. -> Doktor). "
    "Ignoriere Kopf-/Fußzeilen, Seitenzahlen und Randnotizen. Korrigiere Worttrennungen am Zeilenende. "
    "Erhalte Absätze; Listenpunkte als eigene Zeilen. Gib reinen Fließtext in der Originalsprache zurück."
)
DEFAULT_TOC_PROMPT_EN = (
    "Extract the table of contents from the image(s). Return only top-level entries that have a page "
    "number. Ignore dotted leader lines (e.g., Chapter 1 ........ 5) and ignore section headers without "
    "page numbers. Format each line as: TITLE ::: PAGE. Keep titles exactly as shown, no subchapters. "
    "Support roman and arabic page numbers. No extra explanations."
)
DEFAULT_TOC_PROMPT_DE = (
    "Extrahiere das Inhaltsverzeichnis aus dem/den Bild(ern). Gib ausschließlich Einträge mit "
    "Seitenzahl aus. Ignoriere Punktlinien (z. B. Kapitel 1 .... 5) und Abschnittsüberschriften ohne "
    "Seitenzahl. Format pro Zeile: TITEL ::: SEITE. Behalte Titel exakt bei, keine Unterkapitel. "
    "Unterstütze römische und arabische Zahlen. Keine zusätzlichen Erklärungen."
)


@dataclass
class AppSettings:
    api_key: str
    ocr_prompt_en: str
    ocr_prompt_de: str
    toc_prompt_en: str
    toc_prompt_de: str
    keep_pdf_images: bool
    keep_toc_images: bool


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


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


def get_app_data_dir() -> Path:
    """
    Returns the directory where data/logs/output should be stored.
    If running as a frozen EXE, uses the EXE's directory (Portable style).
    If running as a script, uses the script's directory.
    """
    if getattr(sys, "frozen", False):
        # Running as compiled EXE - store data next to the EXE
        return Path(sys.executable).parent
    
    # Running as Python script
    return Path(__file__).resolve().parent


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
    default_api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    defaults = AppSettings(
        api_key=default_api_key,
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
        api_key=data.get("api_key", defaults.api_key),
        ocr_prompt_en=data.get("ocr_prompt_en", defaults.ocr_prompt_en),
        ocr_prompt_de=data.get("ocr_prompt_de", defaults.ocr_prompt_de),
        toc_prompt_en=data.get("toc_prompt_en", defaults.toc_prompt_en),
        toc_prompt_de=data.get("toc_prompt_de", defaults.toc_prompt_de),
        keep_pdf_images=data.get("keep_pdf_images", defaults.keep_pdf_images),
        keep_toc_images=data.get("keep_toc_images", defaults.keep_toc_images),
    )


def save_settings(settings: AppSettings) -> None:
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "api_key": settings.api_key,
            "ocr_prompt_en": settings.ocr_prompt_en,
            "ocr_prompt_de": settings.ocr_prompt_de,
            "toc_prompt_en": settings.toc_prompt_en,
            "toc_prompt_de": settings.toc_prompt_de,
            "keep_pdf_images": settings.keep_pdf_images,
            "keep_toc_images": settings.keep_toc_images,
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
    text = text.casefold()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
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


def find_icon_file() -> Optional[Path]:
    # Look for icon relative to the executable or script
    if getattr(sys, "frozen", False):
         base = Path(sys.executable).parent
    else:
         base = get_base_dir()
         
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
    if not api_key or api_key == "REPLACE_WITH_YOUR_API_KEY":
        raise ValueError(
            "OpenAI API key is missing. Please enter it in Settings."
        )
    return OpenAI(api_key=api_key)


def chunk_list(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def ocr_images_with_retry(
    client: OpenAI,
    image_paths: List[Path],
    prompt: str,
    max_batch_size: int = 4,
    log_fn=None,
) -> str:
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def call_vision(paths: List[Path]) -> str:
        content = [{"type": "text", "text": prompt}]
        for path in paths:
            b64 = encode_image_to_base64(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        response = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    outputs: List[str] = []
    for batch in chunk_list(image_paths, max_batch_size):
        attempts = 0
        backoffs = [6, 10, 20]
        while True:
            try:
                log(f"OCR: processing batch of {len(batch)} image(s)")
                result = call_vision(batch)
                if not result:
                    log("OCR returned empty content. Inserting placeholder.")
                    outputs.append("\n\n[[!! ERROR: EMPTY RESPONSE FROM AI !!]]\n\n")
                else:
                    outputs.append(result)
                break
            except Exception as exc:
                if attempts < len(backoffs):
                    delay = backoffs[attempts]
                    log(f"OCR rate limit/overload. Retrying in {delay}s...")
                    time.sleep(delay)
                    attempts += 1
                    continue
                if len(batch) > 1:
                    mid = len(batch) // 2
                    log("OCR batch failed repeatedly. Splitting batch...")
                    outputs.append(
                        ocr_images_with_retry(
                            client,
                            batch[:mid],
                            prompt,
                            max_batch_size=max_batch_size,
                            log_fn=log_fn,
                        )
                    )
                    outputs.append(
                        ocr_images_with_retry(
                            client,
                            batch[mid:],
                            prompt,
                            max_batch_size=max_batch_size,
                            log_fn=log_fn,
                        )
                    )
                    break
                log("OCR failed permanently for this batch. Inserting placeholder.")
                return "\n\n[[!! ERROR: OCR FAILED FOR THIS SECTION (Check Image) !!]]\n\n"
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
) -> None:
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


def build_markdown(chapters: List[dict]) -> str:
    sections = []
    for chapter in chapters:
        title = chapter.get("title") or "Untitled"
        body = chapter.get("text", "").strip()
        sections.append(f"# {title}\n\n{body}".strip())
    return "\n\n".join(sections).strip() + "\n"


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
        self.skip_epub_var = BooleanVar(value=False)
        self.stack_images_var = BooleanVar(value=True)
        self.stack_batch_size_var = IntVar(value=4)
        self.pdf_path: Optional[Path] = None
        self.txt_path: Optional[Path] = None
        self.batch_result_paths: List[Path] = []
        self.toc_paths: List[Path] = []
        self.image_folder: Optional[Path] = None
        self.pause_event = threading.Event()
        self.is_running = False
        self.progress_var = IntVar(value=0)
        self.progress_label_var = StringVar(value="OCR progress: 0%")
        self.eta_label_var = StringVar(value="ETA: --")
        self.pause_button: Optional[ttk.Button] = None
        self.start_time: Optional[float] = None

        self._build_ui()
        if (
            not self.settings.api_key
            or self.settings.api_key == "REPLACE_WITH_YOUR_API_KEY"
        ):
            self.root.after(200, self.open_settings_dialog)

    def _configure_style(self) -> None:
        self.root.option_add("*Font", "{Segoe UI} 10")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=6)
        style.configure("TLabel", padding=2)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        self.root.configure(background="#f4f4f6")

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
        ttk.Checkbutton(
            options_frame,
            text="Generate Markdown/TXT only (Skip EPUB)",
            variable=self.skip_epub_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        action_buttons = ttk.Frame(dashboard_tab)
        action_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        action_buttons.columnconfigure(0, weight=1)
        action_buttons.columnconfigure(1, weight=1)

        ttk.Button(action_buttons, text="Start", command=self.start).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        self.pause_button = ttk.Button(
            action_buttons, text="Pause", command=self.toggle_pause, state="disabled"
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

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

        ttk.Button(
            settings_frame, text="Save Settings", command=self._save_settings_from_ui
        ).grid(row=12, column=0, sticky="e", pady=(8, 0))

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

    def _save_settings_from_ui(self) -> None:
        self.settings = AppSettings(
            api_key=self.api_key_var.get().strip(),
            ocr_prompt_en=self.ocr_en_text.get("1.0", "end").strip(),
            ocr_prompt_de=self.ocr_de_text.get("1.0", "end").strip(),
            toc_prompt_en=self.toc_en_text.get("1.0", "end").strip(),
            toc_prompt_de=self.toc_de_text.get("1.0", "end").strip(),
            keep_pdf_images=self.keep_pdf_var.get(),
            keep_toc_images=self.keep_toc_var.get(),
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
                if self.settings.api_key != "REPLACE_WITH_YOUR_API_KEY":
                    app.api_key_var.set(self.settings.api_key)
            top.focus_force()
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch {title_prefix}:\n{e}")

    def _format_eta(self, seconds: float) -> str:
        if seconds < 0 or math.isinf(seconds):
            return "ETA: --"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"ETA: {hours:02d}:{minutes:02d}:{secs:02d}"

    def _log(self, message: str) -> None:
        entry = f"[{timestamp()}] {message}"
        self.log.insert("end", f"{entry}\n")
        self.log.see("end")
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
        while self.pause_event.is_set():
            time.sleep(0.2)

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
            self._log("OCR progress text does not exist yet.")
            return
        open_output_folder(self.progress_text_file)

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
            self._log("Log file does not exist yet.")
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
        images = extract_pdf_images(
            self.pdf_path,
            self.paths["pdfimgs"],
            cleanup=False,
            log_fn=self._log,
        )
        if not images:
            messagebox.showerror("Batch Export Failed", "No images extracted.")
            return
        pdf_stem = self.pdf_path.stem

        def iter_entries() -> Iterable[str]:
            for idx, image_path in enumerate(images, start=1):
                b64 = encode_image_to_base64(image_path)
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
                                        "url": f"data:image/png;base64,{b64}"
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

    def _select_toc_pages_from_pdf(self) -> None:
        if not self.pdf_path:
            return
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as exc:
            messagebox.showerror("PDF Error", f"Could not open PDF: {exc}")
            return
        page_count = min(50, doc.page_count)
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

    def start(self) -> None:
        if self.is_running:
            messagebox.showwarning(
                "Already running", "OCR processing is already running."
            )
            return
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def open_output(self) -> None:
        open_output_folder(self.paths["output"])

    def _run_pipeline(self) -> None:
        self.is_running = True
        if self.pause_button:
            self.pause_button.configure(state="normal")
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
                    messagebox.showerror(
                        "Missing batch result",
                        "Please select batch result JSONL files.",
                    )
                    return
                text, language = self._process_batch_results()
                if not skip_toc:
                    if not self.toc_paths:
                        if not self.pdf_path:
                            messagebox.showerror(
                                "Missing PDF",
                                "Please select a PDF file to extract TOC images.",
                            )
                            return
                        self._select_toc_pages_from_pdf()
                    if not self.toc_paths:
                        messagebox.showerror(
                            "Missing TOC", "Please select TOC images."
                        )
                        return
            elif mode == "pdf":
                if not self.pdf_path:
                    messagebox.showerror("Missing PDF", "Please select a PDF file.")
                    return
                if not skip_toc and not self.toc_paths:
                    self._select_toc_pages_from_pdf()
                if not skip_toc and not self.toc_paths:
                    messagebox.showerror("Missing TOC", "Please select TOC images.")
                    return
                text, language = self._process_pdf(client)
            elif mode == "text":
                if not skip_toc and not self.toc_paths:
                    messagebox.showerror("Missing TOC", "Please select TOC images.")
                    return
                if not self.txt_path:
                    messagebox.showerror("Missing text", "Please select a text file.")
                    return
                text = self.txt_path.read_text(encoding="utf-8", errors="ignore")
                language = detect_language(text)
                self._log(f"Detected language: {language}")
            elif mode == "images":
                if not self.image_folder:
                    messagebox.showerror(
                        "Missing images", "Please select an image folder."
                    )
                    return
                if not skip_toc and not self.toc_paths:
                    messagebox.showerror("Missing TOC", "Please select TOC images.")
                    return
                text, language = self._process_image_folder(client)
            else:
                messagebox.showerror("Invalid mode", "Please select a valid mode.")
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
                    messagebox.showerror("TOC OCR failed", "No TOC entries detected.")
                    return
                self._log(f"Detected {len(toc_entries)} TOC entries")

                chapters = build_chapters_from_toc(
                    text, toc_entries, threshold=83, log_fn=self._log
                )
                if not chapters:
                    messagebox.showerror(
                        "Chapter detection failed", "No chapters could be matched."
                    )
                    return

            if self.skip_epub_var.get():
                output_name = f"output_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                output_path = self.paths["output"] / output_name
                output_path.write_text(build_markdown(chapters), encoding="utf-8")
                self._log(f"Text output created: {output_path.name}")
                messagebox.showinfo("Done", f"Text output created: {output_path.name}")
            else:
                output_name = f"epub_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.epub"
                output_path = self.paths["output"] / output_name
                create_epub("Generated Book", chapters, output_path, language)
                self._log(f"EPUB created: {output_path.name}")
                messagebox.showinfo("Done", f"EPUB created: {output_path.name}")
            self._update_progress(1, 1)
        except Exception as exc:
            self._log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))
        finally:
            self.is_running = False
            self.pause_event.clear()
            if self.pause_button:
                self.pause_button.configure(text="Pause", state="disabled")
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
                resume = messagebox.askyesno(
                    "Resume OCR",
                    "An unfinished OCR session was found. Resume from last progress?",
                )
                if resume:
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
        for idx in range(start_index, total):
            self._wait_if_paused()
            self._log(f"OCR page {idx + 1}/{total}")
            page_text = ocr_images_with_retry(
                client,
                [images[idx]],
                prompt,
                max_batch_size=1,
                log_fn=self._log,
            )
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
        for idx, batch in enumerate(batches, start=1):
            self._wait_if_paused()
            self._log(f"OCR batch {idx}/{len(batches)}")
            batch_text = ocr_images_with_retry(
                client,
                batch,
                prompt,
                max_batch_size=batch_size,
                log_fn=self._log,
            )
            text_chunks.append(batch_text.strip())
            processed += len(batch)
            self._write_progress_text(text_chunks)
            self._update_progress(processed, total)
        full_text = "\n".join(text_chunks).strip()
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


def main() -> None:
    root = Tk()
    app = EpubBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # <--- Add this line
    main()
