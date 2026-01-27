import base64
import datetime as dt
import json
import html
import os
import re
import threading
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

# ============================================================
# OpenAI API KEY (replace this with your own key)
# ============================================================
APP_VERSION = "0.2.0"
OPENAI_API_KEY = "REPLACE_WITH_YOUR_API_KEY"
OPENAI_MODEL_VISION = "gpt-4o-mini"
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
    pdf_images_dir: str
    toc_images_dir: str
    keep_pdf_images: bool
    keep_toc_images: bool


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


def default_paths() -> dict:
    base = get_app_data_dir()
    return {
        "base": base,
        "input": base / "input",
        "output": base / "output",
        "pdfimgs": base / "input" / "_pdfimgs",
        "tocimgs": base / "input" / "_tocimgs",
        "logs": base / "logs",
    }


def ensure_dirs(settings: AppSettings) -> dict:
    paths = default_paths()
    input_dir = paths["input"]
    output_dir = paths["output"]
    pdfimgs_dir = (
        Path(settings.pdf_images_dir)
        if settings.pdf_images_dir
        else paths["pdfimgs"]
    )
    tocimgs_dir = (
        Path(settings.toc_images_dir)
        if settings.toc_images_dir
        else paths["tocimgs"]
    )
    log_dir = paths["logs"]
    
    # Attempt to create directories. If we are in a read-only location (like Program Files)
    # and not running as admin, this might fail. We wrap it to provide a helpful error 
    # or fallback could be implemented here (but per request, we keep it 'in location').
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdfimgs_dir.mkdir(parents=True, exist_ok=True)
        tocimgs_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        messagebox.showerror(
            "Permission Error", 
            f"Could not create folders in {paths['base']}.\n\n"
            "If you installed to 'Program Files', try running the app as Administrator "
            "or install to a user folder (e.g. C:\\Users\\Name\\BulkOCR)."
        )

    return {
        "base": paths["base"],
        "input": input_dir,
        "output": output_dir,
        "pdfimgs": pdfimgs_dir,
        "tocimgs": tocimgs_dir,
        "logs": log_dir,
    }


def settings_path() -> Path:
    return get_app_data_dir() / "settings.json"


def load_settings() -> AppSettings:
    default_api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    default_dirs = default_paths()
    defaults = AppSettings(
        api_key=default_api_key,
        ocr_prompt_en=DEFAULT_OCR_PROMPT_EN,
        ocr_prompt_de=DEFAULT_OCR_PROMPT_DE,
        toc_prompt_en=DEFAULT_TOC_PROMPT_EN,
        toc_prompt_de=DEFAULT_TOC_PROMPT_DE,
        pdf_images_dir=str(default_dirs["pdfimgs"]),
        toc_images_dir=str(default_dirs["tocimgs"]),
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
        pdf_images_dir=data.get("pdf_images_dir", defaults.pdf_images_dir),
        toc_images_dir=data.get("toc_images_dir", defaults.toc_images_dir),
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
            "pdf_images_dir": settings.pdf_images_dir,
            "toc_images_dir": settings.toc_images_dir,
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
            "OpenAI API key is missing. Please edit OPENAI_API_KEY in epub_builder_gui.py."
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
                outputs.append(call_vision(batch))
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
                raise exc
    return "\n".join(outputs)


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
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    if cleanup:
        for item in output_dir.glob("*.png"):
            item.unlink()
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = output_dir / f"page_{page_index + 1:04d}.png"
        pix.save(img_path.as_posix())
        image_paths.append(img_path)
        if log_fn:
            log_fn(f"Saved image: {img_path.name}")
    doc.close()
    return image_paths


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


class EpubBuilderApp:
    def __init__(self, root: Tk):
        self.root = root
        self.settings = load_settings()
        self.paths = ensure_dirs(self.settings)
        self.log_file = self.paths["logs"] / "bulk_ocr.log"
        self._icon_photo: Optional[ImageTk.PhotoImage] = None

        self._configure_style()
        self._set_app_icon()
        self.root.title(f"Assistive OCR EPUB Builder v{APP_VERSION}")

        self.mode_var = StringVar(value="pdf")
        self.pdf_path: Optional[Path] = None
        self.txt_path: Optional[Path] = None
        self.toc_paths: List[Path] = []

        self._build_ui()
        if not self.settings.api_key or self.settings.api_key == "REPLACE_WITH_YOUR_API_KEY":
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
        frame.columnconfigure(1, weight=1)

        mode_frame = ttk.Labelframe(frame, text="Input Mode", padding=8)
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
        mode_pdf.grid(row=0, column=0, columnspan=2, sticky="w")
        mode_txt.grid(row=1, column=0, columnspan=2, sticky="w")

        action_frame = ttk.Labelframe(frame, text="Source Files", padding=8)
        action_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        ttk.Button(action_frame, text="Select PDF", command=self.select_pdf).grid(
            row=0, column=0, sticky="ew", pady=4, padx=(0, 6)
        )
        ttk.Button(action_frame, text="Select text file", command=self.select_txt).grid(
            row=0, column=1, sticky="ew", pady=4, padx=(6, 0)
        )

        ttk.Button(frame, text="Start", command=self.start).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        log_frame = ttk.Labelframe(frame, text="Logs", padding=8)
        log_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=1)
        log_frame.rowconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

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

        version_label = ttk.Label(
            frame, text=f"Version {APP_VERSION}", foreground="#666666"
        )
        version_label.grid(row=4, column=0, columnspan=2, sticky="e", pady=(6, 0))

        self._log(f"Input folder: {self.paths['input']}")
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
        dialog = Toplevel(self.root)
        dialog.title("Settings")
        dialog.resizable(False, False)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="OpenAI API Key").grid(row=0, column=0, sticky="w")
        api_entry = ttk.Entry(frame, width=60)
        api_entry.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        api_entry.insert(0, self.settings.api_key)

        ttk.Label(frame, text="OCR Prompt (English)").grid(row=2, column=0, sticky="w")
        ocr_en = ScrolledText(frame, height=4, width=60)
        ocr_en.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ocr_en.insert("1.0", self.settings.ocr_prompt_en)

        ttk.Label(frame, text="OCR Prompt (German)").grid(row=4, column=0, sticky="w")
        ocr_de = ScrolledText(frame, height=4, width=60)
        ocr_de.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        ocr_de.insert("1.0", self.settings.ocr_prompt_de)

        ttk.Label(frame, text="TOC Prompt (English)").grid(row=6, column=0, sticky="w")
        toc_en = ScrolledText(frame, height=4, width=60)
        toc_en.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        toc_en.insert("1.0", self.settings.toc_prompt_en)

        ttk.Label(frame, text="TOC Prompt (German)").grid(row=8, column=0, sticky="w")
        toc_de = ScrolledText(frame, height=4, width=60)
        toc_de.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        toc_de.insert("1.0", self.settings.toc_prompt_de)

        ttk.Label(frame, text="PDF image folder").grid(row=10, column=0, sticky="w")
        pdf_dir_frame = ttk.Frame(frame)
        pdf_dir_frame.grid(row=11, column=0, sticky="ew", pady=(0, 10))
        pdf_dir_frame.columnconfigure(0, weight=1)
        pdf_dir_entry = ttk.Entry(pdf_dir_frame, width=60)
        pdf_dir_entry.grid(row=0, column=0, sticky="ew")
        pdf_dir_entry.insert(0, self.settings.pdf_images_dir)

        def choose_pdf_dir() -> None:
            path = filedialog.askdirectory(title="Select PDF image folder")
            if path:
                pdf_dir_entry.delete(0, "end")
                pdf_dir_entry.insert(0, path)

        ttk.Button(pdf_dir_frame, text="Browse", command=choose_pdf_dir).grid(
            row=0, column=1, padx=(6, 0)
        )

        ttk.Label(frame, text="TOC image folder").grid(row=12, column=0, sticky="w")
        toc_dir_frame = ttk.Frame(frame)
        toc_dir_frame.grid(row=13, column=0, sticky="ew", pady=(0, 10))
        toc_dir_frame.columnconfigure(0, weight=1)
        toc_dir_entry = ttk.Entry(toc_dir_frame, width=60)
        toc_dir_entry.grid(row=0, column=0, sticky="ew")
        toc_dir_entry.insert(0, self.settings.toc_images_dir)

        def choose_toc_dir() -> None:
            path = filedialog.askdirectory(title="Select TOC image folder")
            if path:
                toc_dir_entry.delete(0, "end")
                toc_dir_entry.insert(0, path)

        ttk.Button(toc_dir_frame, text="Browse", command=choose_toc_dir).grid(
            row=0, column=1, padx=(6, 0)
        )

        keep_pdf_var = BooleanVar(value=self.settings.keep_pdf_images)
        keep_toc_var = BooleanVar(value=self.settings.keep_toc_images)
        ttk.Checkbutton(
            frame,
            text="Keep PDF images after OCR",
            variable=keep_pdf_var,
        ).grid(row=14, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Keep TOC images after OCR",
            variable=keep_toc_var,
        ).grid(row=15, column=0, sticky="w")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=16, column=0, sticky="e", pady=(10, 0))

        def save_and_close() -> None:
            self.settings = AppSettings(
                api_key=api_entry.get().strip(),
                ocr_prompt_en=ocr_en.get("1.0", "end").strip(),
                ocr_prompt_de=ocr_de.get("1.0", "end").strip(),
                toc_prompt_en=toc_en.get("1.0", "end").strip(),
                toc_prompt_de=toc_de.get("1.0", "end").strip(),
                pdf_images_dir=pdf_dir_entry.get().strip(),
                toc_images_dir=toc_dir_entry.get().strip(),
                keep_pdf_images=keep_pdf_var.get(),
                keep_toc_images=keep_toc_var.get(),
            )
            save_settings(self.settings)
            self.paths = ensure_dirs(self.settings)
            self.log_file = self.paths["logs"] / "bulk_ocr.log"
            self._log("Settings saved.")
            dialog.destroy()

        ttk.Button(button_frame, text="Save", command=save_and_close).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(
            row=0, column=1
        )

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
            self._log(f"Selected PDF: {self.pdf_path.name}")
            self._select_toc_pages_from_pdf()

    def select_txt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt")],
        )
        if path:
            self.txt_path = Path(path)
            self._log(f"Selected text file: {self.txt_path.name}")

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
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def open_output(self) -> None:
        open_output_folder(self.paths["output"])

    def _run_pipeline(self) -> None:
        try:
            if not self.toc_paths:
                messagebox.showerror("Missing TOC", "Please select TOC images.")
                return

            client = build_openai_client(self.settings.api_key)

            if self.mode_var.get() == "pdf":
                if not self.pdf_path:
                    messagebox.showerror("Missing PDF", "Please select a PDF file.")
                    return
                text, language = self._process_pdf(client)
            else:
                if not self.txt_path:
                    messagebox.showerror("Missing text", "Please select a text file.")
                    return
                text = self.txt_path.read_text(encoding="utf-8", errors="ignore")
                language = detect_language(text)
                self._log(f"Detected language: {language}")

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

            output_name = f"epub_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.epub"
            output_path = self.paths["output"] / output_name
            create_epub("Generated Book", chapters, output_path, language)
            self._log(f"EPUB created: {output_path.name}")
            messagebox.showinfo("Done", f"EPUB created: {output_path.name}")
        except Exception as exc:
            self._log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))

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
        self._log("Starting OCR on PDF images")
        full_text = ocr_pages(
            client,
            images,
            language,
            prompt_override=self._ocr_prompt_for_language(language),
            log_fn=self._log,
        )
        if not self.settings.keep_pdf_images:
            for image_path in images:
                try:
                    image_path.unlink()
                except Exception:
                    continue
        return full_text, language


def main() -> None:
    root = Tk()
    app = EpubBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
