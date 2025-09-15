import os, json
import math, time, torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from models import Conv1dTokenizer, Conv1dVAE



def export_tokenizer_from_vae(trained_vae: Conv1dVAE, save_path: str, device=None):
    assert trained_vae.use_vq, "Export expects a VQ-VAE (use_vq=True)."
    cfg = trained_vae._tok_cfg  # set earlier

    tok = Conv1dTokenizer(
        in_channels=cfg["in_channels"],
        channels=cfg["channels"],
        latent_dim=cfg["latent_dim"],
        use_vq=True,
        codebook_size=cfg["codebook_size"],
    ).to(next(trained_vae.parameters()).device)

    tok.encoder.load_state_dict(trained_vae.encoder.state_dict())

    with torch.no_grad():
        tok.to_latent.weight.copy_(trained_vae.to_mu.weight)
        tok.to_latent.bias.copy_(trained_vae.to_mu.bias)

        tok.vq.codebook.copy_(trained_vae.vq.codebook)
        tok.vq.ema_embed.copy_(trained_vae.vq.ema_embed)
        tok.vq.ema_cluster_size.copy_(trained_vae.vq.ema_cluster_size)

        vqh = cfg["vq_hparams"]
        tok.vq.commitment_cost = vqh["commitment_cost"]
        tok.vq.decay = vqh["decay"]
        tok.vq.eps = vqh["eps"]

    ckpt = {"state_dict": tok.state_dict(), "cfg": cfg}
    torch.save(ckpt, save_path)

    base, ext = os.path.splitext(save_path)
    with open(base + ".json", "w") as f:
        json.dump(cfg, f, indent=2)

    return save_path


def load_tokenizer_for_lm(ckpt_path: str, device="cuda" if torch.cuda.is_available() else "cpu") -> Conv1dTokenizer:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    tok = Conv1dTokenizer(
        in_channels=cfg["in_channels"],
        channels=cfg["channels"],
        latent_dim=cfg["latent_dim"],
        use_vq=True,
        codebook_size=cfg["codebook_size"],
    ).to(device)
    tok.load_state_dict(ckpt["state_dict"])
    vqh = cfg.get("vq_hparams", {})
    if vqh:
        tok.vq.commitment_cost = vqh.get("commitment_cost", tok.vq.commitment_cost)
        tok.vq.decay = vqh.get("decay", tok.vq.decay)
        tok.vq.eps = vqh.get("eps", tok.vq.eps)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad = False
    return tok

@torch.no_grad()
def batch_code_stats(codes, K):
    flat = codes.reshape(-1)
    counts = torch.bincount(flat, minlength=K).float()
    total = counts.sum().clamp(min=1.0)
    probs = counts / total
    mask = probs > 0
    H = -(probs[mask] * probs[mask].log()).sum()
    ppl = torch.exp(H).item()
    usage = (counts > 0).float().mean().item()
    return {"code_perplexity": ppl, "code_usage_frac": usage}



def dtw_loss(
    predictions: torch.Tensor,
    phoneme_predictions: torch.Tensor,
    example: Dict,
    phoneme_eval: bool = False,
    phoneme_confusion: Optional[torch.Tensor] = None,
    phone_weight: float = 1.0,
    label_smoothing: float = 0.0,
):
    """
    Computes sequence loss with DTW alignment when silent, or frame-aligned loss when voiced.

    Returns (loss_per_frame, frame_accuracy_fraction).
    """
    device = predictions.device
    predictions = decollate_tensor(predictions, example['lengths'])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths'])
    audio_features = [t.to(device, non_blocking=True) for t in example['audio_features']]
    phoneme_targets = example['phonemes']

    losses = []
    correct_phones = 0
    total_length = 0

    for pred, y, pred_phone, y_phone, silent in zip(predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent']):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)
        T = y.size(0)

        if silent:
            # cost matrix over (pred_t, y_t')
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0)).squeeze(0)  # (Tp, Ty)
            log_probs = F.log_softmax(pred_phone, dim=-1)[:, y_phone]                      # (Tp, P)
            #phone_cost = -log_probs[:, y_phone]                                 # (Tp,)
            # broadcast to (Tp, Ty): encourage matching frames whose phone matches
            costs = dists + phone_weight * (-log_probs)   # (Tp, Ty)
            # sum along the alignment path
            path = align_from_distances(costs.T.detach().cpu().numpy()) 
            loss = costs[path, range(len(path))].sum()

            if phoneme_eval:
                preds = log_probs.argmax(-1)
                correct_phones += (preds[path] == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(preds[path].tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)
            dists = F.pairwise_distance(y, pred)  # (T,)
            ce  = F.cross_entropy(pred_phone, y_phone, label_smoothing=label_smoothing)
            loss = dists.sum() + phone_weight * ce

            if phoneme_eval:
                preds = pred_phone.argmax(-1)
                correct_phones += (preds == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(preds.tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += T

    return sum(losses) / max(1, total_length), (correct_phones / max(1, total_length))



def dtw_loss(predictions, phoneme_predictions, example, 
             phoneme_eval=False, phoneme_confusion=None, phone_weight=0.5, label_smoothing=0.0):
    device_ = predictions.device
    preds   = decollate_tensor(predictions, example['lengths'])          # list[(T,80)]
    ppreds  = decollate_tensor(phoneme_predictions, example['lengths'])  # list[(T,P)]
    feats   = [t.to(device_, non_blocking=True) for t in example['audio_features']]
    labels  = example['phonemes']

    losses = []
    correct_phones = 0
    total_length = 0
    for pred, y, pred_phone, y_phone, silent in zip(preds, feats, ppreds, labels, example['silent']):
        y_phone = y_phone.to(device_)

        if silent:
            # cost: L2(pred,y) + λ * -log P_phone
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0)).squeeze(0)  # (L_pred, L_y)
            lprobs = F.log_softmax(pred_phone, -1)[:, y_phone]                  # (L_pred, L_y)
            costs = dists + phone_weight * (-lprobs)
            # DTW alignment (monotonic path)
            path = align_from_distances(costs.T.detach().cpu().numpy())         # (L_y,)
            loss = costs[path, range(len(path))].sum()

            if phoneme_eval:
                # FIXED: Use pred_phone.argmax(-1) instead of lprobs.argmax(-1)
                argmax = pred_phone.argmax(-1)  # (L_pred,) - indices in [0, n_phones-1]
                correct_phones += (argmax[path] == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(argmax[path].tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1
        else:
            # voiced: framewise L2 + CE
            assert y.size(0) == pred.size(0)
            l2s = F.pairwise_distance(y, pred)                                   # (T,)
            ce  = F.cross_entropy(pred_phone, y_phone, reduction='sum', label_smoothing=label_smoothing)
            loss = l2s.sum() + phone_weight * ce

            if phoneme_eval:
                argmax = pred_phone.argmax(-1)
                correct_phones += (argmax == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(argmax.tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y.size(0)
    return sum(losses) / max(1, total_length), correct_phones / max(1, total_length)