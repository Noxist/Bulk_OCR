# Testing Checklist

## Textfile Mode Test

**Input:**
- Structured `.txt` with known chapters and headings.
- TOC images containing matching titles.

**Steps:**
1. Launch the app.
2. Select the text file.
3. Select TOC images.
4. Start the process.
5. Open the generated EPUB in Calibre or Apple Books.

**Expected:**
- All chapters detected.
- Text visible in EPUB.
- Correct navigation and chapter order.

## PDF Mode Test

**Input:**
- PDF with 5–10 pages.
- TOC images with main-level chapters.

**Steps:**
1. Launch the app.
2. Select the PDF.
3. Select TOC images.
4. Start the process.
5. Open the generated EPUB in Calibre.

**Expected:**
- OCR output quality is readable.
- No missing text blocks.
- Headers, footers, and page numbers are removed.

## Edge Cases

### Multiple TOC Pages
- Use more than one TOC image.
- Expected: entries are deduplicated and ordered.

### OCR Without Blank Lines
- Use a PDF with dense text and few line breaks.
- Expected: paragraphs remain readable with running text.

### Umlauts and Special Characters
- Use German content with umlauts and ß.
- Expected: text preserved accurately.

### Retry Logic Trigger
- Temporarily set the OpenAI account to a low rate limit or run multiple OCRs quickly.
- Expected: exponential backoff and batch splitting are triggered before failure.
