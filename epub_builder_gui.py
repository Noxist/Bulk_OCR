import base64
import datetime as dt
import json
import html
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from ebooklib import epub
from openai import OpenAI
from rapidfuzz import fuzz
from tkinter import Tk, Toplevel, filedialog, messagebox, StringVar, Menu
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk

# ============================================================
# OpenAI API KEY (replace this with your own key)
# ============================================================
OPENAI_API_KEY = "REPLACE_WITH_YOUR_API_KEY"
OPENAI_MODEL_VISION = "gpt-4o-mini"
DEFAULT_OCR_PROMPT_EN = (
    "You are an OCR engine. Extract only the text visible in the image. "
    "No hallucinations, no headers/footers/page numbers/URLs/ads. "
    "Fix line-break hyphenations. Return pure running text, preserve original language."
)
DEFAULT_OCR_PROMPT_DE = (
    "Du bist eine OCR-Engine. Extrahiere ausschließlich den Text, der im Bild sichtbar ist. "
    "Keine Halluzinationen, keine Überschriften/Fußzeilen/Seitenzahlen/URLs/Werbungen. "
    "Korrigiere Worttrennungen am Zeilenende. Gib reinen Fließtext zurück, Sprache unverändert."
)
DEFAULT_TOC_PROMPT_EN = (
    "Extract the table of contents from the image(s). "
    "Return only top-level chapters. Format each line as: TITLE ::: PAGE. "
    "Keep titles exactly as shown, no subchapters. Support roman and arabic page numbers. "
    "No extra explanations."
)
DEFAULT_TOC_PROMPT_DE = (
    "Extrahiere das Inhaltsverzeichnis aus dem/den Bild(ern). "
    "Gib ausschließlich die Hauptkapitel aus. Format pro Zeile: TITEL ::: SEITE. "
    "Behalte Titel exakt bei, keine Unterkapitel. Unterstütze römische und arabische Zahlen. "
    "Keine zusätzlichen Erklärungen."
)


@dataclass
class AppSettings:
    api_key: str
    ocr_prompt_en: str
    ocr_prompt_de: str
    toc_prompt_en: str
    toc_prompt_de: str


@dataclass
class TocEntry:
    title: str
    page: str


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_dirs() -> dict:
    base = get_base_dir()
    input_dir = base / "input"
    output_dir = base / "output"
    pdfimgs_dir = input_dir / "_pdfimgs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfimgs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "input": input_dir,
        "output": output_dir,
        "pdfimgs": pdfimgs_dir,
    }


def settings_path() -> Path:
    return get_base_dir() / "settings.json"


def load_settings() -> AppSettings:
    default_api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    defaults = AppSettings(
        api_key=default_api_key,
        ocr_prompt_en=DEFAULT_OCR_PROMPT_EN,
        ocr_prompt_de=DEFAULT_OCR_PROMPT_DE,
        toc_prompt_en=DEFAULT_TOC_PROMPT_EN,
        toc_prompt_de=DEFAULT_TOC_PROMPT_DE,
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
    )


def save_settings(settings: AppSettings) -> None:
    path = settings_path()
    data = {
        "api_key": settings.api_key,
        "ocr_prompt_en": settings.ocr_prompt_en,
        "ocr_prompt_de": settings.ocr_prompt_de,
        "toc_prompt_en": settings.toc_prompt_en,
        "toc_prompt_de": settings.toc_prompt_de,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.name == "posix":
        os.system(f'xdg-open "{path}"')


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


def extract_pdf_images(pdf_path: Path, output_dir: Path, log_fn=None) -> List[Path]:
    if log_fn:
        log_fn(f"Rendering PDF pages to images: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
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
        self.root.title("Assistive OCR EPUB Builder")
        self.paths = ensure_dirs()
        self.settings = load_settings()

        self.mode_var = StringVar(value="pdf")
        self.pdf_path: Optional[Path] = None
        self.txt_path: Optional[Path] = None
        self.toc_paths: List[Path] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_menu()
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        mode_label = ttk.Label(frame, text="Mode")
        mode_label.grid(row=0, column=0, sticky="w")

        mode_pdf = ttk.Radiobutton(
            frame,
            text="Mode A: PDF + TOC images → OCR → EPUB",
            variable=self.mode_var,
            value="pdf",
        )
        mode_txt = ttk.Radiobutton(
            frame,
            text="Mode B: Textfile + TOC images → EPUB",
            variable=self.mode_var,
            value="text",
        )
        mode_pdf.grid(row=1, column=0, columnspan=2, sticky="w")
        mode_txt.grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Button(frame, text="Select PDF", command=self.select_pdf).grid(
            row=3, column=0, sticky="ew", pady=4
        )
        ttk.Button(frame, text="Select text file", command=self.select_txt).grid(
            row=3, column=1, sticky="ew", pady=4
        )
        ttk.Button(frame, text="Select TOC images", command=self.select_toc).grid(
            row=4, column=0, sticky="ew", pady=4
        )

        ttk.Button(frame, text="Start", command=self.start).grid(
            row=4, column=1, sticky="ew", pady=4
        )

        ttk.Button(frame, text="Clear log", command=self.clear_log).grid(
            row=5, column=0, sticky="ew", pady=4
        )
        ttk.Button(frame, text="Open output folder", command=self.open_output).grid(
            row=5, column=1, sticky="ew", pady=4
        )

        self.log = ScrolledText(frame, height=18)
        self.log.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=6)
        frame.rowconfigure(6, weight=1)

        self._log(f"Input folder: {self.paths['input']}")
        self._log(f"Output folder: {self.paths['output']}")

    def _build_menu(self) -> None:
        menu_bar = Menu(self.root)
        settings_menu = Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Settings", command=self.open_settings_dialog)
        menu_bar.add_cascade(label="Options", menu=settings_menu)
        self.root.config(menu=menu_bar)

    def open_settings_dialog(self) -> None:
        dialog = Toplevel(self.root)
        dialog.title("Settings")
        dialog.resizable(False, False)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

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

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, sticky="e")

        def save_and_close() -> None:
            self.settings = AppSettings(
                api_key=api_entry.get().strip(),
                ocr_prompt_en=ocr_en.get("1.0", "end").strip(),
                ocr_prompt_de=ocr_de.get("1.0", "end").strip(),
                toc_prompt_en=toc_en.get("1.0", "end").strip(),
                toc_prompt_de=toc_de.get("1.0", "end").strip(),
            )
            save_settings(self.settings)
            self._log("Settings saved.")
            dialog.destroy()

        ttk.Button(button_frame, text="Save", command=save_and_close).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(
            row=0, column=1
        )

    def _log(self, message: str) -> None:
        self.log.insert("end", f"[{timestamp()}] {message}\n")
        self.log.see("end")

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

    def select_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            self.pdf_path = Path(path)
            self._log(f"Selected PDF: {self.pdf_path.name}")

    def select_txt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt")],
        )
        if path:
            self.txt_path = Path(path)
            self._log(f"Selected text file: {self.txt_path.name}")

    def select_toc(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select TOC images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")],
        )
        if paths:
            self.toc_paths = [Path(p) for p in paths]
            self._log(f"Selected {len(self.toc_paths)} TOC image(s)")

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
        for item in pdfimgs_dir.glob("*.png"):
            item.unlink()
        images = extract_pdf_images(self.pdf_path, pdfimgs_dir, log_fn=self._log)
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
        return full_text, language


def main() -> None:
    root = Tk()
    app = EpubBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
