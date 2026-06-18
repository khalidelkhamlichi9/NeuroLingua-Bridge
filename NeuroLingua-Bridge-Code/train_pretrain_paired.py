"""
NeuroLingua-Bridge — train_pretrain_paired.py
Stage 2: LLM Alignment (F2F autoregressive + F2T cosine).

Loads frozen Stage-1 tokenizer, trains alignment heads.
Loss: L = L_F2T + 0.1 · L_F2F

Usage:
    python train_pretrain_paired.py --lm_name gpt2 --ckpt_tok ckpt_tok_gpt2.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_abide, inter_site_split, make_loaders
from model_fmrilm_abide import fMRITokenizer, LossWeights, AlignmentModel
from language_models import load_backend

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_protos(llm, descs, lbl_all, train_idx, d_llm, n=15):
    proto = {}
    for cls in [0, 1]:
        idxs = np.where(lbl_all[train_idx] == cls)[0][:n]
        proto[cls] = torch.stack([llm.get_embedding(descs[train_idx[i]]) for i in idxs]).mean(0).to(DEVICE)
    return proto


def train(args):
    (ts_all, fc_corr, fc_tan, fc_par,
     lbl_all, sex_all, site_list, descs) = load_abide(args.data_dir)
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)

    llm   = load_backend(args.lm_name)
    D_LLM = llm.D_LLM

    train_loader, val_loader, _ = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs,
        train_idx, val_idx, test_idx, batch_size=args.batch_size)

    protos = _build_protos(llm, descs, lbl_all, train_idx, D_LLM)

    # Load Stage-1
    tok = fMRITokenizer(D_LLM).to(DEVICE)
    lw  = LossWeights().to(DEVICE)
    if args.ckpt_tok:
        cp = torch.load(args.ckpt_tok, map_location=DEVICE)
        tok.load_state_dict(cp["tok"]); lw.load_state_dict(cp["lw"])
        print(f"Stage-1 loaded: {args.ckpt_tok}")

    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)
    opt = torch.optim.AdamW([p for p in s2.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=6e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"\nStage 2 — {args.epochs} ep | LLM={args.lm_name}")
    print(f"{'Ep':>3} {'Total':>8} {'F2F':>8} {'F2T':>8}")

    for ep in range(args.epochs):
        s2.train(); ep_f2f = ep_f2t = 0.
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(train_loader, desc=f"S2 Ep{ep+1}", leave=False)):
            ts    = batch["ts"].to(DEVICE)
            c_emb = torch.stack([protos.get(int(l), torch.zeros(D_LLM, device=DEVICE))
                                  for l in batch["label"]]).to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=torch.cuda.is_available()):
                out  = s2(ts, llm_emb=c_emb)
                loss = out["total"] / 4
            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(s2.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            ep_f2f += out["f2f"].item(); ep_f2t += out["f2t"].item()
            del ts, c_emb, out, loss
        sched.step()
        n = len(train_loader)
        print(f"{ep+1:>3} {(ep_f2f+ep_f2t)/n:>8.4f} {ep_f2f/n:>8.4f} {ep_f2t/n:>8.4f}")
        torch.save({"s2": s2.state_dict(), "d_llm": D_LLM}, f"ckpt_s2_{args.lm_name}.pt")

    print(f"\nStage 2 done → ckpt_s2_{args.lm_name}.pt")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--ckpt_tok",   default=None)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=6e-4)
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
