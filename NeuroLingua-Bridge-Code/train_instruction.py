"""
NeuroLingua-Bridge — train_instruction.py
Stage 3: ASD Classifier fine-tuning + final evaluation on unseen test sites.

Multi-task: ASD (primary) + Sex (auxiliary, weight=0.3).
Target: Acc ≥ 76.56%  AUC ≥ 76.22%  (fMRI-LM-B(Q), Table 3, Wei et al. 2026)

Usage:
    python train_instruction.py --lm_name gpt2 --ckpt_s2 ckpt_s2_gpt2.pt
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)

from dataset import load_abide, inter_site_split, make_loaders
from model_fmrilm_abide import fMRITokenizer, LossWeights, AlignmentModel, ASDClassifier
from language_models import load_backend

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(clf, loader):
    clf.eval()
    probs, gts = [], []
    with torch.no_grad():
        for batch in loader:
            lg, _ = clf(batch["ts"].to(DEVICE))
            probs.extend(F.softmax(lg, 1)[:, 1].cpu().numpy())
            gts.extend(batch["label"].numpy())
    probs, gts = np.array(probs), np.array(gts)
    preds = (probs >= 0.5).astype(int)
    return {"acc":  accuracy_score(gts, preds) * 100,
            "bacc": balanced_accuracy_score(gts, preds) * 100,
            "auc":  roc_auc_score(gts, probs) * 100,
            "probs": probs, "gts": gts}


def train(args):
    (ts_all, fc_corr, fc_tan, fc_par,
     lbl_all, sex_all, site_list, descs) = load_abide(args.data_dir)
    train_idx, val_idx, test_idx, _, _ = inter_site_split(lbl_all, site_list)

    llm   = load_backend(args.lm_name)
    D_LLM = llm.D_LLM

    train_loader, val_loader, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs,
        train_idx, val_idx, test_idx, batch_size=args.batch_size)

    # Build model chain
    tok = fMRITokenizer(D_LLM).to(DEVICE)
    lw  = LossWeights().to(DEVICE)
    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)

    if args.ckpt_s2:
        cp = torch.load(args.ckpt_s2, map_location=DEVICE)
        s2.load_state_dict(cp["s2"], strict=False)
        print(f"Stage-2 loaded: {args.ckpt_s2}")

    clf   = ASDClassifier(s2, D_LLM).to(DEVICE)
    opt   = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=5e-5)
    ckpt  = f"ckpt_clf_{args.lm_name}.pt"
    best_acc = best_auc = 0.

    print(f"\nStage 3 — {args.epochs} ep | LLM={args.lm_name} | D={D_LLM}")
    print(f"{'Ep':>3} {'Loss':>8} {'Val Acc':>9} {'Val AUC':>9}")

    for ep in range(args.epochs):
        clf.train(); tot_loss = 0.
        for batch in tqdm(train_loader, desc=f"S3 Ep{ep+1}", leave=False):
            if batch["ts"].shape[0] < 2: continue
            ts  = batch["ts"].to(DEVICE)
            la  = batch["label"].to(DEVICE).long()
            ls  = batch["sex"].to(DEVICE).long()
            opt.zero_grad(set_to_none=True)
            al, sl = clf(ts)
            loss   = clf.loss(al, sl, la, ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 0.8)
            opt.step(); tot_loss += loss.item()
            del ts, la, ls, al, sl, loss
        sched.step()
        m = evaluate(clf, val_loader)
        if m["acc"] > best_acc:
            best_acc, best_auc = m["acc"], m["auc"]
            torch.save({"clf": clf.state_dict(), "d_llm": D_LLM}, ckpt)
        print(f"{ep+1:>3} {tot_loss/len(train_loader):>8.4f} "
              f"{m['acc']:>8.2f}% {m['auc']:>8.2f}%")

    # ── Final evaluation on unseen test sites ─────────────────────────────
    print(f"\nLoading best checkpoint {ckpt}...")
    cp  = torch.load(ckpt, map_location=DEVICE)
    clf2 = ASDClassifier(s2, cp["d_llm"]).to(DEVICE)
    clf2.load_state_dict(cp["clf"])
    tm   = evaluate(clf2, test_loader)

    print(f"\n{'='*62}")
    print(f"  NeuroLingua-Bridge — {args.lm_name.upper()} | ABIDE-I Unseen Sites")
    print(f"{'='*62}")
    print(f"  Test Accuracy        : {tm['acc']:.2f}%")
    print(f"  Test Balanced Acc    : {tm['bacc']:.2f}%")
    print(f"  Test AUC-ROC         : {tm['auc']:.2f}%")
    print(f"  Paper target (B(Q))  : Acc=76.56%  AUC=76.22%")
    print(f"{'='*62}")
    print(classification_report(tm["gts"], (tm["probs"] >= 0.5).astype(int),
                                target_names=["Control", "ASD"]))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--ckpt_s2",    default=None)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--epochs",     type=int,   default=25)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=8e-4)
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
