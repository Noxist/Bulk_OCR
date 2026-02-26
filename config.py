# ============================================================
# Centralised configuration constants (#16)
# ============================================================
APP_VERSION = "0.6.0"
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
