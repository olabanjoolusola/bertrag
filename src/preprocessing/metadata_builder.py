"""
BERTRAG Metadata Builder
========================

Constructs structured metadata for indigenous knowledge passages.
The generated metadata support governance-aware retrieval,
provenance tracking, and cultural filtering.

Author:
    BERTRAG Project

License:
    MIT
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# Metadata Object
# ==========================================================

@dataclass
class Metadata:

    document_id: str

    passage_id: str

    title: str

    community: str

    knowledge_category: str

    document_genre: str

    source_type: str

    language: str

    translation_status: str

    geographic_location: str

    temporal_context: str

    cultural_sensitivity: str

    validation_status: str

    validator_role: str

    keywords: List[str]

    governance_flag: bool


# ==========================================================
# Builder
# ==========================================================

class MetadataBuilder:

    """
    Build metadata for BERTRAG corpus.
    """

    def __init__(self):

        logger.info("Metadata Builder initialized")

    # ------------------------------------------------------

    @staticmethod
    def extract_keywords(text: str) -> List[str]:

        """
        Simple keyword extraction.

        (Can later be replaced by KeyBERT.)
        """

        stopwords = {

            "the",
            "a",
            "of",
            "and",
            "to",
            "in",
            "is",
            "was",
            "for",
            "with"

        }

        words = [

            w.lower()

            for w in text.split()

            if len(w) > 3

        ]

        keywords = []

        for word in words:

            word = word.strip(".,!?;:'\"()")

            if word not in stopwords:

                keywords.append(word)

        keywords = list(dict.fromkeys(keywords))

        return keywords[:15]

    # ------------------------------------------------------

    @staticmethod
    def detect_community(text: str):

        text = text.lower()

        if "awori" in text:

            return "Awori"

        if "ogu" in text:

            return "Ogu"

        return "Unknown"

    # ------------------------------------------------------

    @staticmethod
    def detect_genre(text: str):

        text = text.lower()

        if "proverb" in text:

            return "Proverb"

        if "ritual" in text:

            return "Ritual Description"

        if "festival" in text:

            return "Community Narrative"

        return "Oral History"

    # ------------------------------------------------------

    @staticmethod
    def detect_category(text: str):

        text = text.lower()

        if "king" in text:

            return "History"

        if "ancestor" in text:

            return "Lineage"

        if "river" in text:

            return "Geography"

        if "festival" in text:

            return "Culture"

        return "General"

    # ------------------------------------------------------

    @staticmethod
    def detect_language(text: str):

        return "English"

    # ------------------------------------------------------

    @staticmethod
    def determine_sensitivity(text: str):

        text = text.lower()

        restricted = [

            "secret",

            "sacred",

            "restricted"

        ]

        for word in restricted:

            if word in text:

                return "Restricted"

        return "Public"

    # ------------------------------------------------------

    def build(

        self,

        passage_text: str,

        title: str,

        document_id: str,

        passage_id: Optional[str] = None,

    ) -> Metadata:

        if passage_id is None:

            passage_id = str(uuid.uuid4())

        sensitivity = self.determine_sensitivity(

            passage_text

        )

        metadata = Metadata(

            document_id=document_id,

            passage_id=passage_id,

            title=title,

            community=self.detect_community(

                passage_text

            ),

            knowledge_category=self.detect_category(

                passage_text

            ),

            document_genre=self.detect_genre(

                passage_text

            ),

            source_type="Community Archive",

            language=self.detect_language(

                passage_text

            ),

            translation_status="Original",

            geographic_location="Lagos",

            temporal_context="Unknown",

            cultural_sensitivity=sensitivity,

            validation_status="Validated",

            validator_role="Community Expert",

            keywords=self.extract_keywords(

                passage_text

            ),

            governance_flag=(

                sensitivity == "Public"

            )

        )

        return metadata


# ==========================================================
# Export
# ==========================================================

def save_metadata(

    metadata_list: List[Metadata],

    output_file: str

):

    Path(output_file).parent.mkdir(

        parents=True,

        exist_ok=True

    )

    records = [

        asdict(m)

        for m in metadata_list

    ]

    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            records,

            f,

            indent=4,

            ensure_ascii=False

        )

    logger.info(

        "Saved %d metadata records.",

        len(records)

    )


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    builder = MetadataBuilder()

    metadata = builder.build(

        passage_text="""
        The Awori people celebrate annual festivals
        preserving indigenous traditions.
        """,

        title="Awori Festival",

        document_id="DOC001"

    )

    print(

        json.dumps(

            asdict(metadata),

            indent=4

        )

    )