import os
import sys
import time
import math
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# Project deps
from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from models import *
from utils import *

# ---------------------------
# Utilities
# ---------------------------

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def current_lr(optim: torch.optim.Optimizer) -> float:
    return optim.param_groups[0]["lr"]


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset: EMGDataset,
    device: str,
    seq_len: int,
    n_phones: int,
    phone_weight: float = 0.5,
    label_smoothing: float = 0.0,
    batch_size: int = 32,
    desc: str = "evaluate",
) -> Tuple[float, float, np.ndarray]:
    """Run evaluation loop and return (loss, phoneme_acc, confusion_matrix)."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_raw)
    losses, accs = [], []
    ph_conf = np.zeros((n_phones, n_phones))
    for batch in tqdm.tqdm(loader, desc=desc):
        X_btc = combine_fixed_length(
            [t.to(device, non_blocking=True) for t in batch["raw_emg"]],
            seq_len * 8,
        )
        # Tokenizer returns (codes, z_q, aux)
        codes, z_q, _ = tokenizer(X_btc)        # (B,L) , (B,C_lat,L)
        pred, ph_logits = model(z_q, X_btc)     # (B,L,80), (B,L,P)
        loss, acc = dtw_loss(
            pred, ph_logits, batch,
            phoneme_eval=True,
            phoneme_confusion=ph_conf,
            phone_weight=phone_weight,
            label_smoothing=label_smoothing
        )
        losses.append(loss.item())
        accs.append(acc)
    model.train()
    return float(np.mean(losses)) if losses else math.inf, float(np.mean(accs)) if accs else 0.0, ph_conf


def save_checkpoint(model: nn.Module, ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def append_metrics_csv(csv_path: Path, row: Dict):
    df_row = pd.DataFrame([row])
    header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row.to_csv(csv_path, mode="a", header=header, index=False)


def parse_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() in {"1", "true", "t", "yes", "y"}


# ---------------------------
# Main Training
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EMG→phoneme model with CSV logging and test eval.")
    # Repro & device
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help='Override device, e.g. "cpu"')

    # Optimization
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-7)
    parser.add_argument("--learning-rate-patience", type=int, default=5)

    # Model / data sizes
    parser.add_argument("--seq-len", type=int, default=256, help="Latent sequence length target (after /8).")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--audio-features", type=int, default=80)
    parser.add_argument("--combine-method", type=str, default="cross", choices=["None", "gated_sum", "cross"])

    # Tokenizer
    parser.add_argument("--tokenizer-ckpt", type=str, default="/")
    parser.add_argument("--freeze-tokenizer", type=parse_bool, default=True)
    parser.add_argument("--unfreeze-epoch", type=int, default=9)
    parser.add_argument("--tokenizer-vq-weight", type=float, default=0.25)

    # Loss
    parser.add_argument("--phoneme-loss-weight-init", type=float, default=0.5)
    parser.add_argument("--phoneme-loss-weight-final", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # Logging / output
    parser.add_argument("--output-dir", type=str, default="/")
    parser.add_argument("--run-name", type=str, default=None, help="Optional subfolder under output-dir.")
    parser.add_argument("--save-every-epoch", type=parse_bool, default=True)

    # Evaluation
    parser.add_argument("--eval-batch-size", type=int, default=32)

    args = parser.parse_args()

    # Paths
    run_dir = Path(args.output_dir) / (args.run_name if args.run_name else time.strftime("%Y%m%d-%H%M%S"))
    ckpt_dir = run_dir / "checkpoints"
    csv_path = run_dir / "metrics.csv"
    log_path = run_dir / "train.log"

    # Logging
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    )
    logging.info("Run directory: %s", run_dir)

    # Seed + device
    set_global_seed(args.seed)
    device = args.device if args.device else get_device()
    logging.info(f"Using device: {device}")

    # Config snapshot (for reproducibility)
    cfg_snapshot = vars(args).copy()
    cfg_snapshot["device_resolved"] = device
    (run_dir / "config.json").write_text(json.dumps(cfg_snapshot, indent=2))

    # Datasets & loaders
    n_phones = len(phoneme_inventory)
    trainset = EMGDataset(dev=False, test=False)
    devset   = EMGDataset(dev=True)
    testset  = EMGDataset(dev=False, test=True)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=SizeAwareSampler(trainset, max_len=256000),
        collate_fn=devset.collate_raw,
        pin_memory=(device == "cuda"),
        num_workers=0,
    )

    # Tokenizer
    tokenizer = load_tokenizer_for_lm(args.tokenizer_ckpt, device=device)
    if args.freeze_tokenizer:
        for p in tokenizer.parameters():
            p.requires_grad = False
    tokenizer.eval()
    latent_dim = tokenizer.vq.code_dim

    # Model
    model = create_emg2phoneme_model(
        latent_dim=latent_dim,
        n_phones=n_phones,
        audio_features=args.audio_features,
        d_model=args.d_model,
        n_layers=args.n_layers,
        combine_method=args.combine_method,
    ).to(device)

    # Optimizer & LR scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=args.learning_rate_patience)

    # Training loop
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        # Optionally unfreeze tokenizer after some epochs
        if args.freeze_tokenizer and epoch == args.unfreeze_epoch:
            logging.info("Unfreezing tokenizer parameters.")
            for p in tokenizer.parameters():
                p.requires_grad = True

        # Anneal phoneme loss weight if desired
        ph_w = args.phoneme_loss_weight_init if epoch <= 8 else args.phoneme_loss_weight_final

        model.train()
        epoch_losses = []

        for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch}"):
            X_btc = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch["raw_emg"]],
                args.seq_len * 8,
            )

            # Tokenizer forward
            codes, z_q, vq_loss = tokenizer(X_btc)

            # Model forward
            pred, ph_logits = model(z_q, X_btc)  # (B,L,80), (B,L,P)
            loss, _ = dtw_loss(
                pred, ph_logits, batch,
                phone_weight=ph_w,
                label_smoothing=args.label_smoothing
            )
            total_loss = loss  # + args.tokenizer_vq_weight * vq_loss['vq_loss'] (if needed)

            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_losses.append(total_loss.item())
            global_step += 1

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.inf

        # Tokenizer code usage stats from last batch codes
        stats = batch_code_stats(codes, K=tokenizer.vq.num_codes)
        code_ppl = float(stats.get("code_perplexity", float("nan")))
        code_usage = float(stats.get("code_usage_frac", float("nan")))

        # Validation
        val_loss, val_acc, _ = evaluate(
            model, tokenizer, devset, device,
            seq_len=args.seq_len,
            n_phones=n_phones,
            phone_weight=ph_w,
            label_smoothing=args.label_smoothing,
            batch_size=args.eval_batch_size,
            desc="validate",
        )
        lr_sched.step(val_loss)

        # Save checkpoint
        if args.save_every_epoch:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(model, ckpt_path)

        # Log
        lr_now = current_lr(optim)
        logging.info(
            f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"phoneme acc {100*val_acc:.2f}% | lr {lr_now:.6g} | [tok] ppl={code_ppl:.2f} usage={100*code_usage:.1f}%"
        )

        # CSV logging (per-epoch)
        append_metrics_csv(
            csv_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr_now,
                "tokenizer_code_perplexity": code_ppl,
                "tokenizer_code_usage_frac": code_usage,
                "phoneme_loss_weight": ph_w,
            },
        )

    # ---------------------------
    # Final Test Evaluation
    # ---------------------------
    test_loss, test_acc, test_conf = evaluate(
        model, tokenizer, testset, device,
        seq_len=args.seq_len,
        n_phones=n_phones,
        phone_weight=args.phoneme_loss_weight_final,
        label_smoothing=args.label_smoothing,
        batch_size=args.eval_batch_size,
        desc="test",
    )

    logging.info(f"[TEST] loss {test_loss:.4f} | phoneme acc {100*test_acc:.2f}%")

    # Save final checkpoint + test metrics
    final_ckpt = ckpt_dir / "final.pt"
    save_checkpoint(model, final_ckpt)

    # Append a final CSV row for test metrics (epoch = num_epochs + 1 marker)
    append_metrics_csv(
        csv_path,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": args.num_epochs + 1,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_acc": np.nan,
            "lr": current_lr(optim),
            "tokenizer_code_perplexity": np.nan,
            "tokenizer_code_usage_frac": np.nan,
            "phoneme_loss_weight": args.phoneme_loss_weight_final,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
    )

    # Also persist the confusion matrix as CSV for convenience
    conf_path = run_dir / "test_confusion.csv"
    pd.DataFrame(test_conf.astype(int)).to_csv(conf_path, index=False)
    logging.info("Saved final checkpoint to %s and metrics to %s", final_ckpt, csv_path)


if __name__ == "__main__":
    main()
