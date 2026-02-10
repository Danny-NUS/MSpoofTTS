#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from sklearn.metrics import classification_report

from Discriminator import StridedSegmentTokenDiscriminator

# ============================================================
# PATH CONFIG
# ============================================================

REAL_TRAIN_ROOT = "/data2/minh_duc/from_hf/libritts/train.clean.100"
SYN_TRAIN_ROOT  = "/data2/minh_duc/neutts/libritts/train.clean.100"
REAL_TEST_ROOT = "/data2/minh_duc/from_hf/libritts/test.clean"
SYN_TEST_ROOT  = "/data2/minh_duc/neutts/libritts/test.clean"
SAVE_ROOT = "/data2/minh_duc/TTS_spoofing"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# SEGMENT CONFIG
# ============================================================

SEGMENT_LEN = 50     
SCALE = 10           

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
    min_len: int = SEGMENT_LEN,
) -> List[Dict]:
    """
    Load Kaldi-style dataset slice.
    Order strictly follows wav.scp.
    Filters out utterances with token length < min_len (no padding policy).
    """
    root = Path(root_dir)

    # ---- read text ----
    texts: Dict[str, str] = {}
    with open(root / "text", "r") as f:
        for line in f:
            utt, text = line.rstrip("\n").split(maxsplit=1)
            texts[utt] = text

    # ---- read utt2codes ----
    utt2codes: Dict[str, str] = {}
    with open(root / "utt2codes", "r") as f:
        for line in f:
            utt, path = line.strip().split(maxsplit=1)
            utt2codes[utt] = path

    samples: List[Dict] = []

    with open(root / "wav.scp", "r") as f:
        wav_lines = f.readlines()[start:end]

    for line in wav_lines:
        utt = line.strip().split(maxsplit=1)[0]

        if utt not in texts or utt not in utt2codes:
            continue

        with open(utt2codes[utt], "r") as cf:
            code_str = cf.read().strip()

        tokens = parse_code_str(code_str)
        if len(tokens) < min_len:
            continue

        samples.append({
            "id": utt,
            "text": texts[utt],
            "tokens": tokens,
        })

    return samples

# ============================================================
# DATASET (SEGMENT-LEVEL REAL / FAKE)
# ============================================================

class SegmentTokenSpoofDataset(Dataset):
    """
    Each item:
      seg_tokens: LongTensor [SEGMENT_LEN]
      label:      float (0 = fake, 1 = real)
    Randomly samples a 10/25/50-token segment per __getitem__ call.
    """

    def __init__(
        self,
        samples: List[Dict],
        label: int,
        segment_len: int = SEGMENT_LEN,
    ):
        self.samples = samples
        self.label = float(label)
        self.segment_len = segment_len

    def __len__(self):
        return len(self.samples)

    def _random_segment(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: LongTensor [T], T >= segment_len
        returns: LongTensor [segment_len]
        """
        T = tokens.size(0)
        if T == self.segment_len:
            return tokens
        # sample start uniformly
        start = torch.randint(0, T - self.segment_len + 1, (1,), device=tokens.device).item()
        return tokens[start:start + self.segment_len]

    def __getitem__(self, idx):
        tokens = torch.tensor(self.samples[idx]["tokens"], dtype=torch.long)
        seg = self._random_segment(tokens)  
        label = torch.tensor(self.label, dtype=torch.float)
        return seg, label


def collate_fn(batch):
    segs, labels = zip(*batch)
    # segs are [LEN] each -> stack to [B, LEN]
    return torch.stack(segs, dim=0), torch.stack(labels, dim=0)

# ============================================================
# LIGHTNING MODULE
# ============================================================

class LitSegmentTokenDiscriminator(pl.LightningModule):
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
        self.test_auc = BinaryAUROC()

        self.test_probs = []
        self.test_labels = []

    def training_step(self, batch, batch_idx):
        seg_tokens, labels = batch  # seg_tokens: [B, LEN]
        logits = self.model(seg_tokens)  # [B]

        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        self.train_acc(probs, labels.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # optional, you can implement if you want a val split
        pass

    def test_step(self, batch, batch_idx):
        seg_tokens, labels = batch
        logits = self.model(seg_tokens)
        probs = torch.sigmoid(logits)

        self.test_auc(probs, labels.int())
        self.test_probs.append(probs.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        probs = torch.cat(self.test_probs, dim=0)
        labels = torch.cat(self.test_labels, dim=0)

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

        # reset buffers (good hygiene if you test multiple times)
        self.test_probs.clear()
        self.test_labels.clear()
        self.test_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_state_dict"] = self.model.state_dict()

        # Optional: remove lightning junk if you want a clean file
        checkpoint.pop("state_dict", None)

# ============================================================
# DATA MODULE
# ============================================================

class TokenSpoofDataModule(pl.LightningDataModule):
    def __init__(
        self,
        real_train_root: str,
        syn_train_root: str,
        real_test_root: str,
        syn_test_root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        segment_len: int = SEGMENT_LEN,
    ):
        super().__init__()
        self.real_train_root = real_train_root
        self.syn_train_root = syn_train_root
        self.real_test_root = real_test_root
        self.syn_test_root = syn_test_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_len = segment_len

    def setup(self, stage: Optional[str] = None):
        # Same slicing as your original script (adjust as you like)
        real_train = load_kaldi_dataset(self.real_train_root, 0, 30000, min_len=self.segment_len)
        real_test  = load_kaldi_dataset(self.real_test_root, 0, 4500, min_len=self.segment_len)

        syn_train  = load_kaldi_dataset(self.syn_train_root, 0, 90000, min_len=self.segment_len)
        syn_test   = load_kaldi_dataset(self.syn_test_root, 0, 13500, min_len=self.segment_len)

        real_train_ds = SegmentTokenSpoofDataset(real_train, label=1, segment_len=self.segment_len)
        syn_train_ds  = SegmentTokenSpoofDataset(syn_train,  label=0, segment_len=self.segment_len)

        real_test_ds = SegmentTokenSpoofDataset(real_test, label=1, segment_len=self.segment_len)
        syn_test_ds  = SegmentTokenSpoofDataset(syn_test,  label=0, segment_len=self.segment_len)

        self.train_ds = torch.utils.data.ConcatDataset([real_train_ds, syn_train_ds])
        self.test_ds  = torch.utils.data.ConcatDataset([real_test_ds,  syn_test_ds])

        print(f"[Data] Train: real={len(real_train_ds)}, syn={len(syn_train_ds)}, total={len(self.train_ds)}")
        print(f"[Data] Test : real={len(real_test_ds)},  syn={len(syn_test_ds)},  total={len(self.test_ds)}")
        print(f"[Data] Segment length = {self.segment_len}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        # you currently reuse test_ds as "val"; keeping same behavior
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# ============================================================
# MAIN
# ============================================================

def main():
    model = StridedSegmentTokenDiscriminator(
        segment_len=SEGMENT_LEN,   # 50
        scale=SCALE,               # 50 / 25 / 10
        vocab_size=65536,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        )

    lit_model = LitSegmentTokenDiscriminator(model)

    dm = TokenSpoofDataModule(
        real_train_root=REAL_TRAIN_ROOT,
        real_test_root=REAL_TEST_ROOT,
        syn_train_root=SYN_TRAIN_ROOT,
        syn_test_root=SYN_TEST_ROOT,
        batch_size=20,
        segment_len=SEGMENT_LEN,
        num_workers=4
        )

    logger = TensorBoardLogger(
        save_dir=SAVE_ROOT,
        name=f"Strided_discriminator_seg{SEGMENT_LEN}_scale{SCALE}",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="epoch{epoch}",
        save_top_k=-1,          # <-- save ALL epochs
        every_n_epochs=1,       # <-- every epoch
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
    )

    print("-----------TRAINING--------------")
    trainer.fit(lit_model, dm)

    torch.manual_seed(0)
    trainer.test(lit_model, dm)

if __name__ == "__main__":
    main()
