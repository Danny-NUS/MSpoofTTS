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

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from sklearn.metrics import classification_report

from Discriminator import SuffixTokenDiscriminator  

# ============================================================
# PATH CONFIG
# ============================================================

REAL_ROOT = "/data2/minh_duc/from_hf/libritts/train.clean.100"
SYN_ROOT  = "/data2/minh_duc/neutts/libritts/train.clean.100"
SAVE_ROOT = "/data2/minh_duc/TTS_spoofing"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================
# SEGMENT CONFIG
# ============================================================

SEGMENT_LEN = 50
HORIZONS = [10, 25, 50]

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

        # ---- metrics per horizon ----
        self.train_acc = nn.ModuleDict({
            str(h): BinaryAccuracy(threshold=0.5)
            for h in self.model.horizons
        })

        self.test_acc = nn.ModuleDict({
            str(h): BinaryAccuracy(threshold=0.5)
            for h in self.model.horizons
        })

        self.test_auc = nn.ModuleDict({
            str(h): BinaryAUROC()
            for h in self.model.horizons
        })

        self.test_probs = {str(h): [] for h in self.model.horizons}
        self.test_labels = []

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        seg_tokens, labels = batch  # [B, 50]
        logits_dict = self.model(seg_tokens)  # {h: [B]}

        # ---- equal-weight loss ----
        losses = [
            self.loss_fn(logits, labels)
            for logits in logits_dict.values()
        ]
        loss = torch.stack(losses).mean()

        # ---- per-head accuracy ----
        for h, logits in logits_dict.items():
            probs = torch.sigmoid(logits)
            self.train_acc[str(h)](probs, labels.int())
            self.log(f"train_acc/h{h}", self.train_acc[str(h)], prog_bar=(h == max(self.model.horizons)))

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------
    # Validation (unused)
    # -------------------------
    def validation_step(self, batch, batch_idx):
        pass

    # -------------------------
    # Test
    # -------------------------
    def test_step(self, batch, batch_idx):
        seg_tokens, labels = batch
        logits_dict = self.model(seg_tokens)

        for h, logits in logits_dict.items():
            probs = torch.sigmoid(logits)
            self.test_auc[str(h)](probs, labels.int())
            self.test_acc[str(h)](probs, labels.int())
            self.test_probs[str(h)].append(probs.detach().cpu())

        self.test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        labels = torch.cat(self.test_labels, dim=0)

        for h in self.model.horizons:
            probs = torch.cat(self.test_probs[str(h)], dim=0)

            auc = self.test_auc[str(h)].compute().item()
            acc = self.test_acc[str(h)].compute().item()

            preds = (probs >= 0.5).int()

            report = classification_report(
                labels.numpy(),
                preds.numpy(),
                target_names=["synthetic", "real"],
                digits=4,
            )

            # ---- log scalars ----
            self.log(f"test_auc/h{h}", auc)
            self.log(f"test_acc/h{h}", acc)

            # ---- write report ----
            report_path = os.path.join(
                self.logger.log_dir,
                f"test_classification_report_h{h}.txt"
            )

            with open(report_path, "w") as f:
                f.write(f"Horizon {h}\n")
                f.write(f"Test AUC: {auc:.4f}\n")
                f.write(f"Test ACC: {acc:.4f}\n\n")
                f.write(report)

            print(f"\n==== Test Report (h={h}) ====")
            print(f"AUC={auc:.4f} | ACC={acc:.4f}")
            print(report)

        # ---- reset ----
        self.test_labels.clear()
        for h in self.model.horizons:
            self.test_probs[str(h)].clear()
            self.test_auc[str(h)].reset()
            self.test_acc[str(h)].reset()

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
        segment_len: int = SEGMENT_LEN,
    ):
        super().__init__()
        self.real_root = real_root
        self.syn_root = syn_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_len = segment_len

    def setup(self, stage: Optional[str] = None):
        # Same slicing as your original script (adjust as you like)
        real_train = load_kaldi_dataset(self.real_root, 0, 20000, min_len=self.segment_len)
        real_test  = load_kaldi_dataset(self.real_root, 30000, 33000, min_len=self.segment_len)

        syn_train  = load_kaldi_dataset(self.syn_root, 0, 60000, min_len=self.segment_len)
        syn_test   = load_kaldi_dataset(self.syn_root, 90000, 99000, min_len=self.segment_len)

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
    model = SuffixTokenDiscriminator(
        segment_len=SEGMENT_LEN,
        horizons=HORIZONS,
    )

    lit_model = LitSegmentTokenDiscriminator(model)

    dm = TokenSpoofDataModule(
        real_root=REAL_ROOT,
        syn_root=SYN_ROOT,
        batch_size=100,
        segment_len=SEGMENT_LEN,
        num_workers=4,
    )

    logger = TensorBoardLogger(
        save_dir=SAVE_ROOT,
        name="Suffix_discriminator_h50_25_10",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        logger=logger,
        log_every_n_steps=10,
    )

    print("-----------TRAINING--------------")
    trainer.fit(lit_model, dm)

    torch.manual_seed(0)
    trainer.test(lit_model, dm)


if __name__ == "__main__":
    main()
