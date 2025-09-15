from __future__ import annotations
import os
import sys

import math
import json
import time
import logging
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# Ensure src is in sys.path for direct script execution
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Project imports
from read_emg import EMGDataset, SizeAwareSampler
from data_utils import decollate_tensor, combine_fixed_length
from models import Conv1dVAE
from utils import export_tokenizer_from_vae, batch_code_stats
from absl import flags
FLAGS = flags.FLAGS
FLAGS([''])

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def parse_enc_channels(s: str) -> List[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Invalid --enc-channels '{s}'. Use comma‑separated ints, e.g. 16,32,64") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name in {"cuda", "cpu"}:
        if name == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return name
    raise ValueError("--device must be one of: auto, cuda, cpu")


def make_dataloader(trainset: EMGDataset, dev_collate, device: str, batch_tokens: int, num_workers: int):
    return torch.utils.data.DataLoader(
        trainset,
        pin_memory=(device == "cuda"),
        collate_fn=dev_collate,
        num_workers=num_workers,
        batch_sampler=SizeAwareSampler(trainset, batch_tokens),
    )


def build_model(cfg: dict, device: str) -> nn.Module:
    vae = Conv1dVAE(
        in_channels=cfg["n_channels"],
        out_channels=cfg["n_channels"],  # reconstruct EMG
        channels=cfg["enc_channels"],
        latent_dim=cfg["latent_dim"],
        use_vq=True,
        codebook_size=cfg["codebook_size"],
        beta_kl=0.0,  # no KL for VQ‑VAE
    ).to(device)
    return vae


def recon_loss_fn(kind: str):
    if kind.lower() == "l1":
        return lambda y_hat, x: F.l1_loss(y_hat, x)
    if kind.lower() in {"l2", "mse"}:
        return lambda y_hat, x: F.mse_loss(y_hat, x)
    raise ValueError("--recon-type must be one of: l1, l2, mse")


def train(cfg: dict):
    # Prepare data
    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info("train/dev split: %d / %d", len(trainset), len(devset))
    logging.info("dev example index: %s", getattr(devset, "example_indices", [None])[0])

    device = cfg["device"]
    dataloader = make_dataloader(
        trainset=trainset,
        dev_collate=devset.collate_raw,
        device=device,
        batch_tokens=cfg["batch_tokens"],
        num_workers=cfg["num_workers"],
    )

    # Model & optim
    vae = build_model(cfg, device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) 

    down_factor = 2 ** len(cfg["enc_channels"])  # time downsampling
    window_len = cfg["seq_len"] * down_factor
    loss_recon = recon_loss_fn(cfg["recon_type"])

    def vae_loss(y_hat, x, vq_loss):
        recon = loss_recon(y_hat, x)
        loss = recon + vq_loss
        return loss, float(recon.item()), float(vq_loss.item())

    # Train
    vae.train()
    step = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        pbar = tqdm.tqdm(dataloader, desc=f"epoch {epoch}")
        for batch in pbar:
            # (B,T,C) windows with hop=win
            X_btc = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch["raw_emg"]], window_len
            )
            x = X_btc.transpose(1, 2).contiguous()  # (B,C,T)

            # Encode → quantize μ → decode
            z, mu, logvar = vae.encode(x)
            z_q, codes, vq_loss = vae.vq(mu)
            y_hat = vae.decode(z_q)

            loss, rec, vql = vae_loss(y_hat, x, vq_loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg["grad_clip"]) 
            opt.step()

            step += 1
            if step % cfg["log_every"] == 0:
                stats = batch_code_stats(codes, K=vae.vq.num_codes)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{rec:.4f}",
                    vq=f"{vql:.4f}",
                    ppl=f"{stats['code_perplexity']:.1f}",
                    usage=f"{100 * stats['code_usage_frac']:.1f}%",
                )
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{rec:.4f}", vq=f"{vql:.4f}")

    # Attach tokenizer config for export
    vae._tok_cfg = {
        "in_channels": cfg["n_channels"],
        "channels": cfg["enc_channels"],
        "latent_dim": cfg["latent_dim"],
        "codebook_size": cfg["codebook_size"],
        "vq_hparams": {
            "commitment_cost": cfg["commitment_cost"],
            "decay": cfg["decay"],
            "eps": 1e-5,
        },
    }

    os.makedirs(os.path.dirname(cfg["tokenizer_out"]) or ".", exist_ok=True)
    export_tokenizer_from_vae(vae, cfg["tokenizer_out"], device=device)
    logging.info("Saved tokenizer to: %s", cfg["tokenizer_out"])


def main(argv: List[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="Train and export EMG VQ‑VAE tokenizer")

    # Data/loader
    p.add_argument("--batch-tokens", type=int, default=256000, help="Approx. tokens per batch for SizeAwareSampler")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # Architecture
    p.add_argument("--n-channels", type=int, default=8, help="EMG channels (C)")
    p.add_argument("--enc-channels", type=parse_enc_channels, default="16,32,64", help="Comma‑separated encoder channel sizes")
    p.add_argument("--latent-dim", type=int, default=786, help="Latent embedding dim")
    p.add_argument("--codebook-size", type=int, default=512, help="VQ codebook size")

    # Training
    p.add_argument("--batch-size", type=int, default=16, help="(For devset")
    p.add_argument("--seq-len", type=int, default=256, help="Latent sequence length per window (before downsampling)")
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--recon-type", choices=["l1", "l2", "mse"], default="l1", help="Reconstruction loss")
    p.add_argument("--commitment-cost", type=float, default=0.25)
    p.add_argument("--decay", type=float, default=0.99, help="EMA decay for VQ")
    p.add_argument("--beta-kl", type=float, default=0.0, help="use kl loss")
    p.add_argument("--log-every", type=int, default=20)

    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # Output
    p.add_argument("--tokenizer-out", default="output/tokenizer/tokenizer_saved.pt")

    args = p.parse_args(argv)

    # Derived / runtime cfg
    cfg = dict(
        batch_tokens=args.batch_tokens,
        num_workers=args.num_workers,
        n_channels=args.n_channels,
        enc_channels=args.enc_channels if isinstance(args.enc_channels, list) else parse_enc_channels(args.enc_channels),
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        recon_type=args.recon_type,
        beta_kl=args.beta_kl,
        log_every=args.log_every,
        tokenizer_out=args.tokenizer_out,
        device=get_device(args.device),
    )

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seed(args.seed)
    logging.info("Config: %s", json.dumps({k: (v if not isinstance(v, list) else v) for k, v in cfg.items()}, indent=2))

    # Train & export
    train(cfg)


if __name__ == "__main__":
    main()



