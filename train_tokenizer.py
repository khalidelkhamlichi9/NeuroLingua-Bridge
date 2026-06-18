"""
NeuroLingua-Bridge — train_tokenizer.py
Stage 1: fMRI Tokenizer training with automatic lambda grid search.

L_tok = w1·L_quant + w2·L_SigLIP + w3·L_GRL    (Σwi = 1, softmax)

Usage:
    python train_tokenizer.py --lm_name gpt2   --epochs 15
    python train_tokenizer.py --lm_name clinicalt5
    python train_tokenizer.py --lm_name deepseek --batch_size 2
"""

import gc, math, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_abide, inter_site_split, make_loaders
from model_fmrilm_abide import fMRITokenizer, LossWeights, grl_schedule
from language_models import load_backend

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_GRID = [
    (0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2),
    (0.4, 0.3, 0.3), (0.33, 0.33, 0.34), (0.5, 0.4, 0.1),
]


def _build_protos(llm, descs, lbl_all, train_idx, d_llm, n=15):
    proto = {}
    for cls, name in [(0, "Control"), (1, "ASD")]:
        idxs = np.where(lbl_all[train_idx] == cls)[0][:n]
        embs = [llm.get_embedding(descs[train_idx[i]]) for i in idxs]
        proto[cls] = torch.stack(embs).mean(0).to(DEVICE)
    return proto


def _quick_eval(w_init, train_loader, val_loader, protos, d_llm, n_epochs=3):
    tok = fMRITokenizer(d_llm).to(DEVICE)
    lw  = LossWeights(*w_init).to(DEVICE)
    opt = torch.optim.AdamW(list(tok.parameters()) + list(lw.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    last = 0.
    for ep in range(n_epochs):
        tok.train(); ep_tot = n_b = 0
        lam = grl_schedule(ep, n_epochs)
        for batch in train_loader:
            ts    = batch["ts"].to(DEVICE)
            t_emb = torch.stack([protos.get(int(l), torch.zeros(d_llm, device=DEVICE))
                                  for l in batch["label"]]).to(DEVICE)
            out  = tok(ts, text_emb=t_emb, lam=lam, loss_weights=lw)
            loss = out["total"] / 4
            loss.backward()
            nn.utils.clip_grad_norm_(list(tok.parameters()) + list(lw.parameters()), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            ep_tot += out["total"].item(); n_b += 1
            del ts, t_emb, out, loss
        last = ep_tot / max(n_b, 1)

    tok.eval(); val_loss = n_v = 0
    with torch.no_grad():
        for batch in (val_loader if val_loader else train_loader):
            ts    = batch["ts"].to(DEVICE)
            t_emb = torch.stack([protos.get(int(l), torch.zeros(d_llm, device=DEVICE))
                                  for l in batch["label"]]).to(DEVICE)
            out = tok(ts, text_emb=t_emb, lam=1.0, loss_weights=lw)
            val_loss += out["total"].item(); n_v += 1; del ts, t_emb, out
    val_loss = val_loss / max(n_v, 1) if n_v else last
    w_f = lw.weights.detach().cpu().numpy()
    del tok, lw; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return val_loss, w_f


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

    # ── Lambda grid search ──────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Lambda Grid Search  ({len(LAMBDA_GRID)} configs × {args.search_epochs} ep)\n{'='*60}")
    results = []
    for w_init in LAMBDA_GRID:
        vl, wf = _quick_eval(w_init, train_loader, val_loader, protos, D_LLM, args.search_epochs)
        results.append({"init": w_init, "val_loss": vl, "w_final": wf})
        print(f"  {w_init}  val_loss={vl:.4f}  w=({wf[0]:.3f},{wf[1]:.3f},{wf[2]:.3f})")
    best     = min(results, key=lambda r: r["val_loss"])
    BEST_W   = best["init"]
    print(f"\n  Best: {BEST_W}  val_loss={best['val_loss']:.4f}\n{'='*60}")

    # ── Full Stage-1 training ───────────────────────────────────────────────
    tok    = fMRITokenizer(D_LLM).to(DEVICE)
    lw_opt = LossWeights(*BEST_W).to(DEVICE)
    opt    = torch.optim.AdamW(list(tok.parameters()) + list(lw_opt.parameters()),
                               lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"\nStage 1 — {args.epochs} ep | LLM={args.lm_name} | D={D_LLM}")
    print(f"{'Ep':>3} {'Total':>7} {'Quant':>7} {'SigLIP':>7} {'GRL':>7} "
          f"{'w_q':>6} {'w_s':>6} {'w_g':>6} {'CB%':>5}")

    for ep in range(args.epochs):
        tok.train(); lw_opt.train()
        ep_tot = ep_q = ep_c = ep_d = 0.
        lam = grl_schedule(ep, args.epochs)
        for batch in tqdm(train_loader, desc=f"S1 Ep{ep+1}", leave=False):
            ts    = batch["ts"].to(DEVICE)
            t_emb = torch.stack([protos.get(int(l), torch.zeros(D_LLM, device=DEVICE))
                                  for l in batch["label"]]).to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=torch.cuda.is_available()):
                out  = tok(ts, text_emb=t_emb, lam=lam, loss_weights=lw_opt)
                loss = out["total"] / 4
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(tok.parameters()) + list(lw_opt.parameters()), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            ep_tot += out["total"].item(); ep_q += out["quant"].item()
            ep_c += out["contrast"].item(); ep_d += out["domain"].item()
            del ts, t_emb, out, loss
        sched.step()
        n  = len(train_loader)
        w  = lw_opt.weights.detach().cpu()
        print(f"{ep+1:>3} {ep_tot/n:>7.4f} {ep_q/n:>7.4f} {ep_c/n:>7.4f} {ep_d/n:>7.4f} "
              f"{w[0]:>6.3f} {w[1]:>6.3f} {w[2]:>6.3f} {tok.quantizer.utilisation*100:>4.1f}%")
        torch.save({"tok": tok.state_dict(), "lw": lw_opt.state_dict(),
                    "d_llm": D_LLM, "best_w": BEST_W},
                   f"ckpt_tok_{args.lm_name}.pt")

    print(f"\nStage 1 done → ckpt_tok_{args.lm_name}.pt")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",      default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--data_dir",     default="./abide_data")
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--search_epochs",type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
