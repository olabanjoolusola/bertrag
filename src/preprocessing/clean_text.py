"""
BERTRAG Text Cleaning Module
============================

This module performs text normalization and cleaning for indigenous
knowledge documents prior to segmentation and embedding generation.

Author:
    BERTRAG Project

License:
    MIT
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Clean and normalize raw corpus documents.
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def remove_html(text: str) -> str:
        """
        Remove HTML/XML tags.
        """
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs.
        """
        return re.sub(r"http\S+|www\S+", "", text)

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """
        Normalize quotation marks.
        """
        replacements = {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Collapse repeated whitespace.
        """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """
        Remove invisible control characters.
        """
        return "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C"
        )

    @staticmethod
    def remove_duplicate_blank_lines(text: str) -> str:
        """
        Remove repeated blank lines.
        """
        return re.sub(r"\n\s*\n+", "\n\n", text)

    def clean(self, text: str) -> str:
        """
        Full cleaning pipeline.
        """

        logger.info("Cleaning document...")

        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.normalize_quotes(text)
        text = self.remove_control_characters(text)
        text = self.remove_duplicate_blank_lines(text)
        text = self.normalize_whitespace(text)

        return text


def load_document(filepath: str | Path) -> str:
    """
    Load text document.
    """

    filepath = Path(filepath)

    with filepath.open(
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def save_document(text: str, filepath: str | Path):
    """
    Save cleaned document.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with filepath.open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)


def clean_file(
    input_file: str | Path,
    output_file: str | Path
):
    """
    Clean one document.
    """

    cleaner = TextCleaner()

    raw = load_document(input_file)

    cleaned = cleaner.clean(raw)

    save_document(cleaned, output_file)

    logger.info("Saved cleaned file -> %s", output_file)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    clean_file(
        "example.txt",
        "cleaned_example.txt"
    )