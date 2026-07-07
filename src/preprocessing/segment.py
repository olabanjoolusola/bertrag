"""
BERTRAG Passage Segmentation Module
===================================

Segments cleaned documents into semantically coherent passages suitable
for retrieval-augmented generation.

Author:
    BERTRAG Project

License:
    MIT
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    """
    Representation of a segmented passage.
    """

    passage_id: str
    document_id: str
    order: int
    text: str


class PassageSegmenter:
    """
    Segment documents into retrieval passages.

    Parameters
    ----------
    max_words : int
        Maximum words per passage.

    overlap : int
        Number of words to overlap between adjacent passages.
    """

    def __init__(
        self,
        max_words: int = 150,
        overlap: int = 30,
    ):

        self.max_words = max_words
        self.overlap = overlap

    @staticmethod
    def paragraph_split(text: str) -> List[str]:
        """
        Split document into paragraphs.
        """

        paragraphs = [
            p.strip()
            for p in text.split("\n")
            if p.strip()
        ]

        return paragraphs

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    def merge_short_paragraphs(
        self,
        paragraphs: List[str]
    ) -> List[str]:
        """
        Merge extremely short paragraphs.
        """

        merged = []

        buffer = ""

        for para in paragraphs:

            if self.word_count(para) < 40:

                buffer += " " + para

            else:

                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""

                merged.append(para)

        if buffer:
            merged.append(buffer.strip())

        return merged

    def split_long_paragraph(
        self,
        paragraph: str
    ) -> List[str]:
        """
        Split long paragraphs using a sliding window.
        """

        words = paragraph.split()

        passages = []

        start = 0

        while start < len(words):

            end = min(
                start + self.max_words,
                len(words)
            )

            chunk = " ".join(words[start:end])

            passages.append(chunk)

            if end == len(words):
                break

            start = end - self.overlap

        return passages

    def segment(
        self,
        document: str,
        document_id: str
    ) -> List[Passage]:

        logger.info("Segmenting document %s", document_id)

        paragraphs = self.paragraph_split(document)

        paragraphs = self.merge_short_paragraphs(paragraphs)

        segmented = []

        order = 1

        for para in paragraphs:

            chunks = self.split_long_paragraph(para)

            for chunk in chunks:

                segmented.append(

                    Passage(

                        passage_id=f"P-{uuid.uuid4().hex[:8]}",

                        document_id=document_id,

                        order=order,

                        text=chunk,

                    )

                )

                order += 1

        logger.info(
            "Created %d passages",
            len(segmented)
        )

        return segmented


def save_passages(
    passages: List[Passage],
    output_file: str | Path
):

    import json

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records = [
        p.__dict__
        for p in passages
    ]

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    sample = """
The Awori people are one of the indigenous Yoruba groups.

They traditionally occupied the coastal region of Lagos.

Their oral traditions describe migration from Ile-Ife.

Traditional festivals continue to preserve their cultural identity.
"""

    segmenter = PassageSegmenter(
        max_words=50,
        overlap=10
    )

    passages = segmenter.segment(
        sample,
        "DOC001"
    )

    save_passages(
        passages,
        "passages.json"
    )