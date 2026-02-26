"""FastAPI Web-Server für den Bulk OCR → EPUB Pipeline.

Starten:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from ocr_pipeline import (
    DEFAULT_OCR_PROMPT_DE,
    DEFAULT_OCR_PROMPT_EN,
    DEFAULT_TOC_PROMPT_DE,
    DEFAULT_TOC_PROMPT_EN,
    _OCR_ERR,
    build_chapters_from_toc,
    build_html_output,
    build_markdown,
    build_openai_client,
    create_epub,
    detect_book_title,
    detect_language,
    extract_pdf_images,
    ocr_images_with_retry,
    ocr_toc,
)

# ------------------------------------------------------------------ #
# Setup                                                                #
# ------------------------------------------------------------------ #

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Bulk OCR Web")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Job-Registry: job_id → dict
jobs: dict[str, dict] = {}

# Referenz auf den asyncio-Eventloop (wird in startup gesetzt)
_main_loop: Optional[asyncio.AbstractEventLoop] = None

JOB_TTL_SECONDS = 3600  # Jobs nach 1h löschen


@app.on_event("startup")
async def _startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    # Cleanup-Task starten
    asyncio.create_task(_cleanup_old_jobs())


async def _cleanup_old_jobs():
    """Alte Jobs + ihre Dateien stündlich bereinigen."""
    while True:
        await asyncio.sleep(600)
        now = time.time()
        expired = [jid for jid, j in list(jobs.items()) if now - j.get("created", 0) > JOB_TTL_SECONDS]
        for jid in expired:
            job_dir = JOBS_DIR / jid
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            jobs.pop(jid, None)


# ------------------------------------------------------------------ #
# Thread→asyncio Bridge                                               #
# ------------------------------------------------------------------ #

def _put_event(queue: asyncio.Queue, event: dict) -> None:
    """Legt ein Event thread-sicher in die asyncio-Queue."""
    if _main_loop and not _main_loop.is_closed():
        _main_loop.call_soon_threadsafe(queue.put_nowait, event)


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    output_format: str = Form("epub"),
    skip_toc: str = Form("true"),
    toc_pages: str = Form(""),
):
    """Nimmt ein PDF entgegen, startet OCR-Job, gibt job_id zurück."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"error": "OPENAI_API_KEY Umgebungsvariable ist nicht gesetzt."}

    # Dateivalidierung
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "Nur PDF-Dateien werden akzeptiert."}

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # PDF speichern (Dateiname sanitisieren)
    safe_name = re.sub(r"[^\w\-_\. ]", "_", file.filename or "upload.pdf")
    pdf_path = job_dir / safe_name
    with pdf_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # TOC-Seiten parsen (1-basierte Eingabe → 0-basierte Indexe)
    toc_page_indices: list[int] = []
    if toc_pages.strip():
        for part in re.split(r"[,\s]+", toc_pages.strip()):
            if part.isdigit():
                idx = int(part) - 1
                if idx >= 0:
                    toc_page_indices.append(idx)

    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {
        "status": "pending",
        "queue": queue,
        "output_path": None,
        "created": time.time(),
    }

    thread = threading.Thread(
        target=_run_ocr_job,
        args=(job_id, pdf_path, output_format, skip_toc == "true", toc_page_indices, api_key),
        daemon=True,
        name=f"ocr-{job_id[:8]}",
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """Server-Sent Events Stream mit Fortschritt und Logs."""
    if job_id not in jobs:
        async def _not_found():
            yield f"data: {json.dumps({'type': 'error', 'msg': 'Job nicht gefunden'})}\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    async def _event_gen():
        queue: asyncio.Queue = jobs[job_id]["queue"]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str):
    """Fertige Ausgabe-Datei herunterladen."""
    if job_id not in jobs:
        return {"error": "Job nicht gefunden"}
    if jobs[job_id]["status"] != "done":
        return {"error": f"Job noch nicht abgeschlossen (Status: {jobs[job_id]['status']})"}

    out = jobs[job_id].get("output_path")
    if not out:
        return {"error": "Keine Ausgabe vorhanden"}
    p = Path(out)
    if not p.exists():
        # EPUB-Fallback: Markdown?
        md = p.with_suffix(".md")
        if md.exists():
            return FileResponse(str(md), filename=md.name,
                                media_type="text/markdown")
        return {"error": "Ausgabe-Datei nicht gefunden"}

    media_types = {
        ".epub": "application/epub+zip",
        ".html": "text/html",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    media_type = media_types.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), filename=p.name, media_type=media_type)


@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str):
    """Einfacher Status-Endpoint (polling-Alternative zu SSE)."""
    if job_id not in jobs:
        return {"error": "Job nicht gefunden"}
    j = jobs[job_id]
    return {
        "status": j["status"],
        "output_filename": Path(j["output_path"]).name if j.get("output_path") else None,
    }


# ------------------------------------------------------------------ #
# OCR-Job Worker (läuft in eigenem Thread)                            #
# ------------------------------------------------------------------ #

def _run_ocr_job(
    job_id: str,
    pdf_path: Path,
    output_format: str,
    skip_toc: bool,
    toc_page_indices: list[int],
    api_key: str,
) -> None:
    queue = jobs[job_id]["queue"]

    def log(msg: str, level: str = "INFO") -> None:
        _put_event(queue, {"type": "log", "level": level, "msg": msg,
                           "ts": dt.datetime.now().strftime("%H:%M:%S")})

    def progress(pct: int, label: str = "") -> None:
        _put_event(queue, {"type": "progress", "pct": max(0, min(100, pct)), "label": label})

    jobs[job_id]["status"] = "running"

    try:
        client = build_openai_client(api_key)
        job_dir = pdf_path.parent
        images_dir = job_dir / "images"

        # --- 1. PDF → Bilder ---
        progress(2, "PDF wird gerendert…")
        images = extract_pdf_images(pdf_path, images_dir, cleanup=True, log_fn=log)
        if not images:
            raise ValueError("Keine Bilder aus PDF extrahiert. Ist die Datei beschädigt?")
        total = len(images)
        log(f"{total} Seiten extrahiert.")

        # --- 2. Spracherkennung (Seite 1) ---
        progress(5, "Spracherkennung…")
        sample = ocr_images_with_retry(client, [images[0]], DEFAULT_OCR_PROMPT_EN,
                                       max_batch_size=1, log_fn=log)
        language = detect_language(sample)
        log(f"Erkannte Sprache: {language}")
        prompt = DEFAULT_OCR_PROMPT_DE if language == "german" else DEFAULT_OCR_PROMPT_EN

        # --- 3. OCR Seite für Seite ---
        text_chunks: list[str] = []
        failed_pages: list[str] = []

        for idx, img in enumerate(images):
            page_num = idx + 1
            pct = int(5 + (idx / total) * 72)
            progress(pct, f"OCR Seite {page_num}/{total}")
            log(f"OCR Seite {page_num}/{total}")

            page_text = ocr_images_with_retry(client, [img], prompt, max_batch_size=1, log_fn=log)

            if _OCR_ERR in page_text:
                reason = "Leere Antwort" if ":LEER]]" in page_text else "API-Fehler"
                page_text = (
                    f"\n\n⚠️ [SEITE {page_num}: Konnte nicht ausgewertet werden"
                    f" ({reason}) – bitte manuell prüfen]\n\n"
                )
                failed_pages.append(f"Seite {page_num} ({reason})")
                log(f"⚠️ Seite {page_num}: {reason}", level="WARN")

            text_chunks.append(page_text.strip())

        full_text = "\n".join(text_chunks).strip()

        # Fehlerzusammenfassung ans Ende
        if failed_pages:
            full_text += (
                "\n\n---\n\n## ⚠️ OCR-Fehlerzusammenfassung\n\n"
                f"{len(failed_pages)} Seite(n) konnten nicht ausgewertet werden:\n\n"
                + "\n".join(f"- {p}" for p in failed_pages)
                + "\n\n_Bitte diese Seiten manuell prüfen._\n\n---\n"
            )
            log(f"Insgesamt {len(failed_pages)} Seite(n) mit Fehlern: {', '.join(failed_pages)}", level="WARN")

        # --- 4. TOC / Kapitel ---
        progress(80, "Kapitel werden erkannt…")
        if skip_toc or not toc_page_indices:
            log("Kein Inhaltsverzeichnis – Dokument als ein Kapitel.")
            chapters = [{"title": "Volltext", "text": full_text}]
        else:
            valid_toc = [i for i in toc_page_indices if i < len(images)]
            log(f"TOC-OCR auf {len(valid_toc)} Seite(n): {[i+1 for i in valid_toc]}")
            toc_images = [images[i] for i in valid_toc]
            toc_prompt = DEFAULT_TOC_PROMPT_DE if language == "german" else DEFAULT_TOC_PROMPT_EN
            toc_entries = ocr_toc(client, toc_images, language, toc_prompt, log_fn=log)
            if toc_entries:
                log(f"{len(toc_entries)} TOC-Einträge gefunden.")
                chapters = build_chapters_from_toc(full_text, toc_entries, log_fn=log)
                log(f"{len(chapters)} Kapitel erkannt.")
            else:
                log("Keine TOC-Einträge gefunden – Dokument als ein Kapitel.", level="WARN")
                chapters = [{"title": "Volltext", "text": full_text}]

        # --- 5. Titel erkennen ---
        progress(88, "Buchtitel wird erkannt…")
        book_title = detect_book_title(client, full_text, log_fn=log)

        # --- 6. Ausgabe erstellen ---
        progress(93, f"{output_format.upper()} wird erstellt…")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_format = output_format.lower()

        if out_format == "epub":
            out_path = job_dir / f"output_{ts}.epub"
            create_epub(book_title, chapters, out_path, language, log_fn=log)
        elif out_format == "html":
            out_path = job_dir / f"output_{ts}.html"
            out_path.write_text(build_html_output(chapters, book_title, language), encoding="utf-8")
        else:  # txt / md
            out_path = job_dir / f"output_{ts}.md"
            out_path.write_text(build_markdown(chapters), encoding="utf-8")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["output_path"] = str(out_path)
        log(f"✅ Ausgabe erstellt: {out_path.name}")
        progress(100, "Fertig!")
        _put_event(queue, {"type": "done", "filename": out_path.name, "job_id": job_id})

    except Exception as exc:
        jobs[job_id]["status"] = "error"
        import traceback
        log(f"❌ Fehler: {exc}\n{traceback.format_exc()}", level="ERROR")
        _put_event(queue, {"type": "error", "msg": str(exc)})
