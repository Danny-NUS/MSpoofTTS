#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from sklearn.metrics import classification_report

from Discriminator import TokenDiscriminator

# ============================================================
# PATH CONFIG
# ============================================================

REAL_ROOT = "/data2/minh_duc/from_hf/libritts/train.clean.100"
SYN_ROOT  = "/data2/minh_duc/neutts/libritts/train.clean.100"
SAVE_ROOT = "/data2/minh_duc/TTS_spoofing"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# TOKEN PARSER
# ============================================================

CODE_RE = re.compile(r"<\|speech_(\d+)\|>")

def parse_code_str(codes: str) -> List[int]:
    return [int(num) for num in CODE_RE.findall(codes)]

# ============================================================
# KALDI LOADER (ORDER PRESERVED)
# ============================================================

def load_kaldi_dataset(
    root_dir: str,
    start: int,
    end: int,
) -> List[Dict]:
    """
    Load Kaldi-style dataset slice.
    Order strictly follows wav.scp.
    """
    root = Path(root_dir)

    # ---- read text ----
    texts = {}
    with open(root / "text") as f:
        for line in f:
            utt, text = line.rstrip("\n").split(maxsplit=1)
            texts[utt] = text

    # ---- read utt2codes ----
    utt2codes = {}
    with open(root / "utt2codes") as f:
        for line in f:
            utt, path = line.strip().split(maxsplit=1)
            utt2codes[utt] = path

    samples = []

    with open(root / "wav.scp") as f:
        wav_lines = f.readlines()[start:end]

    for line in wav_lines:
        utt = line.strip().split(maxsplit=1)[0]

        if utt not in texts or utt not in utt2codes:
            continue

        with open(utt2codes[utt]) as cf:
            code_str = cf.read().strip()

        samples.append({
            "id": utt,
            "text": texts[utt],
            "tokens": parse_code_str(code_str),
        })

    return samples


# ============================================================
# DATASET (PAIRWISE REAL / FAKE)
# ============================================================

class TokenSpoofDataset(Dataset):
    """
    Each item:
      tokens: LongTensor [T]
      label:  float (0 = fake, 1 = real)
    """

    def __init__(self, samples: List[Dict], label: int):
        self.samples = samples
        self.label = float(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = torch.tensor(
            self.samples[idx]["tokens"],
            dtype=torch.long
        )
        label = torch.tensor(self.label, dtype=torch.float)
        return tokens, label


def collate_fn(batch):
    tokens, labels = zip(*batch)
    return list(tokens), torch.stack(labels)


# ============================================================
# LIGHTNING MODULE
# ============================================================

class LitTokenDiscriminator(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy(threshold=0.5)
        self.val_acc = BinaryAccuracy(threshold=0.5)
        self.test_auc = BinaryAUROC()

        self.test_logits = []
        self.test_labels = []

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self.model(tokens)

        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        self.train_acc(probs, labels.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    # -------------------------
    # Validation (defined, unused)
    # -------------------------
    def validation_step(self, batch, batch_idx):
        pass

    # -------------------------
    # Test
    # -------------------------
    def test_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self.model(tokens)
        probs = torch.sigmoid(logits)

        self.test_auc(probs, labels.int())
        self.test_logits.append(probs.cpu())
        self.test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        probs = torch.cat(self.test_logits)
        labels = torch.cat(self.test_labels)

        preds = (probs >= 0.5).int()
        auc = self.test_auc.compute().item()

        report = classification_report(
            labels.numpy(),
            preds.numpy(),
            target_names=["synthetic", "real"],
            digits=4,
        )

        self.log("test_auc", auc)

        report_path = os.path.join(
            self.logger.log_dir,
            "test_classification_report.txt"
        )

        with open(report_path, "w") as f:
            f.write(f"Test AUC: {auc:.4f}\n\n")
            f.write(report)

        print("\n==== Test Classification Report ====")
        print(report)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


# ============================================================
# DATA MODULE
# ============================================================

class TokenSpoofDataModule(pl.LightningDataModule):
    def __init__(
        self,
        real_root: str,
        syn_root: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.real_root = real_root
        self.syn_root = syn_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        real_train = load_kaldi_dataset(self.real_root, 0, 8000)
        real_test  = load_kaldi_dataset(self.real_root, 8000, 9000)

        syn_train = load_kaldi_dataset(self.syn_root, 0, 24000)
        syn_test  = load_kaldi_dataset(self.syn_root, 24000, 27000)

        real_train_ds = TokenSpoofDataset(real_train, label=1)
        syn_train_ds  = TokenSpoofDataset(syn_train,  label=0)

        real_test_ds = TokenSpoofDataset(real_test, label=1)
        syn_test_ds  = TokenSpoofDataset(syn_test,  label=0)

        self.train_ds = torch.utils.data.ConcatDataset(
            [real_train_ds, syn_train_ds]
        )

        self.test_ds = torch.utils.data.ConcatDataset(
            [real_test_ds, syn_test_ds]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

# ============================================================
# MAIN
# ============================================================

def main():
    model = TokenDiscriminator()

    lit_model = LitTokenDiscriminator(model)

    dm = TokenSpoofDataModule(
        real_root=REAL_ROOT,
        syn_root=SYN_ROOT,
        batch_size=20,
    )

    logger = TensorBoardLogger(
        save_dir=SAVE_ROOT,
        name="Utt_discriminator",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        logger=logger,
        log_every_n_steps=10,
    )

    print("-----------TRAINING--------------")
    print("Number of samples: 4000 + 12000")

    trainer.fit(lit_model, dm)
    trainer.test(lit_model, dm)

    # ckpt_path = os.path.join(SAVE_ROOT, "final.ckpt")
    # trainer.save_checkpoint(ckpt_path)
    # print(f"Saved model to {ckpt_path}")

if __name__ == "__main__":
    main()
