"""Pure OCR / EPUB pipeline – kein tkinter, läuft auf headless Servern.

Importiert Config-Konstanten aus dem Parent-Verzeichnis (config.py).
"""
from __future__ import annotations

import base64
import concurrent.futures
import datetime as dt
import html as _html
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import fitz  # PyMuPDF
from ebooklib import epub
from openai import OpenAI
from rapidfuzz import fuzz

# config.py aus dem Parent-Verzeichnis laden
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from config import (  # noqa: E402
    DEFAULT_OCR_PROMPT_DE,
    DEFAULT_OCR_PROMPT_EN,
    DEFAULT_TOC_PROMPT_DE,
    DEFAULT_TOC_PROMPT_EN,
    OPENAI_MODEL_VISION,
)

# Marker für fehlerhafte OCR-Seiten (darf nicht im echten Text vorkommen)
_OCR_ERR = "[[OCR-ERROR"


@dataclass
class TocEntry:
    title: str
    page: str


# ------------------------------------------------------------------ #
# Hilfsfunktionen                                                     #
# ------------------------------------------------------------------ #

def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detect_language(text: str) -> str:
    sample = text[:2000].casefold()
    german_markers = ["ä", "ö", "ü", "ß", " der ", " die ", " das ", " und ", " ist "]
    english_markers = [" the ", " and ", " is ", " of ", " to ", " in "]
    g = sum(sample.count(m) for m in german_markers)
    e = sum(sample.count(m) for m in english_markers)
    return "german" if g >= e else "english"


def normalize_text(text: str) -> str:
    import unicodedata
    text = text.casefold()
    text = (
        text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            .replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def collect_image_files(folder: Path) -> List[Path]:
    allowed = {".png", ".jpg", ".jpeg"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed],
        key=lambda p: p.name.casefold(),
    )


def chunk_list(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def _mime_for_image(path: Path) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")


def build_openai_client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API-Key fehlt.")
    return OpenAI(api_key=api_key.strip())


# ------------------------------------------------------------------ #
# OCR                                                                  #
# ------------------------------------------------------------------ #

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
            mime = _mime_for_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        response = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    MAX_RETRIES = 3
    BACKOFFS = [6, 10, 20]
    queue: list[list[Path]] = list(chunk_list(image_paths, max_batch_size))
    outputs: List[str] = []

    while queue:
        batch = queue.pop(0)
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                log(f"OCR: {len(batch)} Bild(er), Versuch {attempt + 1}/{MAX_RETRIES}")
                result = call_vision(batch)
                if not result:
                    log("OCR: Leere Antwort. Platzhalter eingefügt.")
                    outputs.append(f"\n\n{_OCR_ERR}:LEER]]\n\n")
                else:
                    outputs.append(result)
                success = True
                break
            except Exception as exc:
                delay = BACKOFFS[attempt] if attempt < len(BACKOFFS) else BACKOFFS[-1]
                log(f"OCR-Fehler: {exc}. Warte {delay}s…")
                time.sleep(delay)

        if not success:
            if len(batch) > 1:
                mid = len(batch) // 2
                log(f"Teile fehlgeschlagenen Batch ({len(batch)}) in 2.")
                queue.insert(0, batch[mid:])
                queue.insert(0, batch[:mid])
            else:
                log("OCR dauerhaft fehlgeschlagen. Platzhalter eingefügt.")
                outputs.append(f"\n\n{_OCR_ERR}:FEHLER]]\n\n")

    return "\n".join(outputs)


def ocr_toc(
    client: OpenAI,
    image_paths: List[Path],
    language: str,
    prompt_override: str = "",
    log_fn=None,
) -> List[TocEntry]:
    prompt = prompt_override or (DEFAULT_TOC_PROMPT_DE if language == "german" else DEFAULT_TOC_PROMPT_EN)
    raw = ocr_images_with_retry(client, image_paths, prompt, log_fn=log_fn)
    entries: List[TocEntry] = []
    seen: set = set()
    for line in raw.splitlines():
        if ":::" not in line:
            continue
        title, page = [p.strip() for p in line.split(":::", 1)]
        key = (title.casefold(), page.casefold())
        if title and key not in seen:
            entries.append(TocEntry(title=title, page=page))
            seen.add(key)
    return entries


# ------------------------------------------------------------------ #
# PDF → Bilder                                                         #
# ------------------------------------------------------------------ #

def extract_pdf_images(
    pdf_path: Path,
    output_dir: Path,
    cleanup: bool = True,
    log_fn=None,
) -> List[Path]:
    if log_fn:
        log_fn(f"PDF-Seiten werden gerendert: {pdf_path.name}")
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
        paths = _render_pdf_page_range(pdf_path.as_posix(), output_dir.as_posix(), page_indices, zoom)
    else:
        chunk_size = math.ceil(page_count / worker_count)
        chunks = [page_indices[i: i + chunk_size] for i in range(0, page_count, chunk_size)]
        paths: List[str] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_render_pdf_page_range, pdf_path.as_posix(), output_dir.as_posix(), chunk, zoom)
                for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                paths.extend(future.result())
    return sorted((Path(p) for p in paths), key=lambda p: p.name)


def _render_pdf_page_range(
    pdf_path: str,
    output_dir: str,
    page_indices: Sequence[int],
    zoom: float,
) -> List[str]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    output_paths: List[str] = []
    base = Path(output_dir)
    for idx in page_indices:
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = base / f"page_{idx + 1:04d}.png"
        pix.save(img_path.as_posix())
        output_paths.append(img_path.as_posix())
    doc.close()
    return output_paths


# ------------------------------------------------------------------ #
# Text-Verarbeitung                                                    #
# ------------------------------------------------------------------ #

def split_into_paragraphs(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: List[str] = []
    buf: List[str] = []
    for line in lines:
        if not line:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


def build_chapters_from_toc(
    full_text: str,
    toc_entries: List[TocEntry],
    threshold: int = 83,
    log_fn=None,
) -> List[dict]:
    lines = full_text.splitlines()
    norm = [normalize_text(line) for line in lines]
    chapters: List[dict] = []
    last_index = 0
    for entry in toc_entries:
        target = normalize_text(entry.title)
        best_index, best_score = None, 0
        for idx in range(last_index, len(lines)):
            if not norm[idx]:
                continue
            score = max(fuzz.ratio(target, norm[idx]), fuzz.partial_ratio(target, norm[idx]))
            if score > best_score:
                best_score = score
                best_index = idx
            if score >= 98:
                break
        if best_index is None or best_score < threshold:
            if log_fn:
                log_fn(f"Kapitel nicht gefunden: '{entry.title}' (Score {best_score})")
            continue
        chapters.append({"title": entry.title, "start": best_index})
        last_index = best_index + 1
    for i, ch in enumerate(chapters):
        start_line = ch["start"] + 1
        end_line = chapters[i + 1]["start"] if i + 1 < len(chapters) else len(lines)
        ch["text"] = "\n".join(lines[start_line:end_line]).strip()
    return chapters


# ------------------------------------------------------------------ #
# Output-Builder                                                       #
# ------------------------------------------------------------------ #

def build_markdown(chapters: List[dict]) -> str:
    parts = [f"# {ch.get('title') or 'Untitled'}\n\n{ch.get('text', '').strip()}".strip() for ch in chapters]
    return "\n\n".join(parts).strip() + "\n"


def build_html_output(chapters: List[dict], title: str, language: str) -> str:
    lang = "de" if language == "german" else "en"
    out = [
        f'<!DOCTYPE html>\n<html lang="{lang}">\n<head>',
        f'<meta charset="utf-8"><title>{_html.escape(title)}</title>',
        '<style>body{font-family:Georgia,serif;max-width:42em;margin:2em auto;line-height:1.6;padding:0 1em}h1{margin-top:2em}</style>',
        '</head><body>',
    ]
    for ch in chapters:
        out.append(f"<h1>{_html.escape(ch.get('title') or 'Untitled')}</h1>")
        for p in split_into_paragraphs(ch.get("text", "")):
            if p.strip():
                out.append(f"<p>{_html.escape(p)}</p>")
    out.append("</body></html>")
    return "\n".join(out)


def create_epub(
    title: str,
    chapters: List[dict],
    output_path: Path,
    language: str,
    log_fn=None,
) -> None:
    try:
        book = epub.EpubBook()
        book.set_identifier(f"bulk-ocr-{int(time.time())}")
        book.set_title(title)
        book.set_language("de" if language == "german" else "en")
        epub_chs = []
        for idx, ch in enumerate(chapters, start=1):
            ch_title = ch.get("title") or f"Kapitel {idx}"
            paragraphs = split_into_paragraphs(ch.get("text", ""))
            body = "".join(f"<p>{_html.escape(p)}</p>" for p in paragraphs if p.strip()) or "<p></p>"
            ec = epub.EpubHtml(
                title=ch_title,
                file_name=f"chapter_{idx}.xhtml",
                lang="de" if language == "german" else "en",
            )
            ec.content = f"<h1>{_html.escape(ch_title)}</h1>{body}"
            book.add_item(ec)
            epub_chs.append(ec)
        book.toc = tuple(epub_chs)
        book.spine = ["nav"] + epub_chs
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(output_path.as_posix(), book, {})
    except Exception as exc:
        fallback = output_path.with_suffix(".md")
        if log_fn:
            log_fn(f"EPUB fehlgeschlagen ({exc}). Speichere Markdown-Fallback: {fallback.name}")
        fallback.write_text(build_markdown(chapters), encoding="utf-8")
        raise


def detect_book_title(client: OpenAI, text: str, log_fn=None) -> str:
    sample = text[:2000]
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[{
                "role": "user",
                "content": (
                    "Analysiere den folgenden Textanfang und gib NUR den Buchtitel zurück. "
                    "Kein zusätzlicher Text, keine Anführungszeichen. Falls kein Titel erkennbar: 'Untitled'.\n\n"
                    f"{sample}"
                ),
            }],
            temperature=0.0,
            max_tokens=60,
        )
        title = (response.choices[0].message.content or "Untitled").strip().strip('"').strip("'")
        if log_fn:
            log_fn(f"Buchtitel erkannt: {title}")
        return title or "Untitled"
    except Exception as exc:
        if log_fn:
            log_fn(f"Titelerkennungfehler: {exc}")
        return "Untitled"
