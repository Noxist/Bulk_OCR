# Assistive OCR EPUB Builder

A Windows desktop GUI application for converting books into well-structured EPUB files using OpenAI Vision OCR only (no Tesseract). The output is optimized for accessibility and audio readers.

## Features

- Tkinter single-window GUI
- Mode A: PDF + TOC images → OCR → EPUB
- Mode B: Textfile + TOC images → EPUB
- Automatic folder creation (`input/`, `output/`, `input/_pdfimgs/`)
- OpenAI Vision-based OCR with retry/backoff
- TOC OCR and fuzzy chapter matching
- Clean EPUB output with navigation (NCX + Nav)

## Setup

1. Install Python 3.10+ on Windows.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `epub_builder_gui.py` and replace the `OPENAI_API_KEY` value with your key.

## Usage

Run the app:

```bash
python epub_builder_gui.py
```

### Mode A (PDF)
1. Select PDF.
2. Select TOC images (1–N).
3. Click **Start**.

### Mode B (Textfile)
1. Select text file (.txt).
2. Select TOC images (1–N).
3. Click **Start**.

Outputs are saved in `output/` relative to the script location.

## Build Windows EXE

```bash
python build_exe.py
```

This uses:

```bash
pyinstaller --onefile --noconsole epub_builder_gui.py
```

The EXE is created in the `dist/` folder.
