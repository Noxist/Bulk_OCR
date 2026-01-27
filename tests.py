import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from epub_builder_gui import (
    TocEntry,
    build_chapters_from_toc,
    detect_language,
    normalize_text,
    ocr_images_with_retry,
)


class LanguageDetectionTests(unittest.TestCase):
    def test_detect_language_english(self) -> None:
        text = "This is the story of the book and the hero."
        self.assertEqual(detect_language(text), "english")

    def test_detect_language_german(self) -> None:
        text = "Das ist der Anfang und das Ende."
        self.assertEqual(detect_language(text), "german")


class NormalizationTests(unittest.TestCase):
    def test_normalize_text_swiss_german(self) -> None:
        text = "Grüße aus Zürich: ä ö ü ß!"
        self.assertEqual(
            normalize_text(text),
            "gruesse aus zuerich ae oe ue ss",
        )


class TocSplitTests(unittest.TestCase):
    def test_build_chapters_from_toc_splits_text(self) -> None:
        full_text = "\n".join(
            [
                "Intro",
                "Chapter One",
                "Line 1",
                "Line 2",
                "Chapter Two",
                "Line A",
                "Line B",
            ]
        )
        toc_entries = [
            TocEntry(title="Chapter One", page="1"),
            TocEntry(title="Chapter Two", page="5"),
        ]
        chapters = build_chapters_from_toc(full_text, toc_entries, threshold=80)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["start"], 1)
        self.assertEqual(chapters[0]["text"], "Line 1\nLine 2")
        self.assertEqual(chapters[1]["start"], 4)
        self.assertEqual(chapters[1]["text"], "Line A\nLine B")


class OpenAIMockTests(unittest.TestCase):
    def test_ocr_images_with_retry_uses_mock(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"fake image bytes")
            tmp_path = Path(tmp.name)

        client = mock.Mock()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Mock OCR text"))]
        )
        client.chat.completions.create.return_value = response

        try:
            result = ocr_images_with_retry(
                client, [tmp_path], "prompt", max_batch_size=1
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(result, "Mock OCR text")
        client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
