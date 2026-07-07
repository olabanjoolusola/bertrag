"""
BERTRAG Continual Domain Adaptation
===================================

Continued pretraining of BERT on indigenous knowledge corpora
using Masked Language Modeling (MLM).

Author:
    BERTRAG Project

License:
    MIT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class DomainPretrainer:
    """
    Continual pretraining using MLM.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str = "models/domainbert",
        max_length: int = 256,
    ):

        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length

        logger.info("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        logger.info("Loading pretrained model...")

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name
        )

    # --------------------------------------------------------

    def load_corpus(
        self,
        corpus: List[str]
    ) -> Dataset:

        return Dataset.from_dict(
            {
                "text": corpus
            }
        )

    # --------------------------------------------------------

    def tokenize(
        self,
        dataset: Dataset
    ) -> Dataset:

        return dataset.map(

            lambda x: self.tokenizer(

                x["text"],

                truncation=True,

                padding="max_length",

                max_length=self.max_length,

            ),

            batched=True,

            remove_columns=["text"],

        )

    # --------------------------------------------------------

    def train(

        self,

        corpus: List[str],

        epochs: int = 5,

        batch_size: int = 16,

        learning_rate: float = 2e-5,

    ):

        logger.info("Preparing dataset...")

        dataset = self.load_corpus(corpus)

        dataset = self.tokenize(dataset)

        collator = DataCollatorForLanguageModeling(

            tokenizer=self.tokenizer,

            mlm=True,

            mlm_probability=0.15,

        )

        args = TrainingArguments(

            output_dir=self.output_dir,

            overwrite_output_dir=True,

            num_train_epochs=epochs,

            learning_rate=learning_rate,

            per_device_train_batch_size=batch_size,

            logging_steps=50,

            save_steps=500,

            save_total_limit=2,

            report_to="none",

        )

        trainer = Trainer(

            model=self.model,

            args=args,

            train_dataset=dataset,

            tokenizer=self.tokenizer,

            data_collator=collator,

        )

        logger.info("Starting continual pretraining...")

        trainer.train()

        logger.info("Saving DomainBERT...")

        trainer.save_model(

            self.output_dir

        )

        self.tokenizer.save_pretrained(

            self.output_dir

        )

        logger.info("Training completed.")


# ------------------------------------------------------------

def load_processed_corpus(

    corpus_file: str

):

    import json

    with open(

        corpus_file,

        encoding="utf-8"

    ) as f:

        corpus = json.load(f)

    return [

        p["text"]

        for p in corpus

    ]


# ------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    corpus = load_processed_corpus(

        "data/processed/passages.json"

    )

    trainer = DomainPretrainer(

        model_name="bert-base-uncased"

    )

    trainer.train(

        corpus=corpus,

        epochs=5,

        batch_size=16,

        learning_rate=2e-5,

    )