"""
BERTRAG Contrastive Fine-Tuning
===============================

Supervised contrastive fine-tuning of DomainBERT.

Positive pairs:
    Same community
    Same knowledge category
    Same topic

Negative pairs:
    Different community
    Different category
    Random passages

Author:
    BERTRAG Project
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class ContrastiveDataset(Dataset):

    def __init__(

        self,

        anchors,

        positives,

        negatives,

        tokenizer,

        max_length=256,

    ):

        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.anchors)

    def encode(self, text):

        return self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt",

        )

    def __getitem__(self, idx):

        return (

            self.encode(self.anchors[idx]),

            self.encode(self.positives[idx]),

            self.encode(self.negatives[idx]),

        )


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

class DomainEncoder(nn.Module):

    def __init__(

        self,

        model_name="models/domainbert",

    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(

            model_name

        )

    def forward(self, batch):

        outputs = self.encoder(

            input_ids=batch["input_ids"].squeeze(1),

            attention_mask=batch["attention_mask"].squeeze(1),

        )

        cls = outputs.last_hidden_state[:, 0]

        cls = F.normalize(

            cls,

            p=2,

            dim=1,

        )

        return cls


# ---------------------------------------------------------
# Triplet Loss
# ---------------------------------------------------------

class ContrastiveLoss(nn.Module):

    """
    Margin-based triplet loss.
    """

    def __init__(

        self,

        margin=0.30,

    ):

        super().__init__()

        self.margin = margin

    def forward(

        self,

        anchor,

        positive,

        negative,

    ):

        pos = F.cosine_similarity(

            anchor,

            positive,

        )

        neg = F.cosine_similarity(

            anchor,

            negative,

        )

        loss = torch.relu(

            neg - pos + self.margin

        )

        return loss.mean()


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

class ContrastiveTrainer:

    def __init__(

        self,

        model_name="models/domainbert",

        lr=2e-5,

    ):

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

        logger.info(self.device)

        self.model = DomainEncoder(

            model_name

        ).to(self.device)

        self.loss_fn = ContrastiveLoss()

        self.optimizer = torch.optim.AdamW(

            self.model.parameters(),

            lr=lr,

        )

    def train(

        self,

        dataloader,

        epochs=5,

    ):

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0

            for anchor, positive, negative in dataloader:

                anchor = {

                    k: v.to(self.device)

                    for k, v in anchor.items()

                }

                positive = {

                    k: v.to(self.device)

                    for k, v in positive.items()

                }

                negative = {

                    k: v.to(self.device)

                    for k, v in negative.items()

                }

                z_anchor = self.model(anchor)

                z_positive = self.model(positive)

                z_negative = self.model(negative)

                loss = self.loss_fn(

                    z_anchor,

                    z_positive,

                    z_negative,

                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            logger.info(

                "Epoch %d Loss %.4f",

                epoch + 1,

                total_loss,

            )

    def save(

        self,

        output_dir="models/domainbert_contrastive",

    ):

        self.model.encoder.save_pretrained(

            output_dir

        )

        logger.info(

            "Saved model to %s",

            output_dir,

        )


# ---------------------------------------------------------

if __name__ == "__main__":

    print(

        "BERTRAG Contrastive Trainer"

    )