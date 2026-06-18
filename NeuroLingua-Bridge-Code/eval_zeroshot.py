"""
NeuroLingua-Bridge — eval_zeroshot.py
Zero-shot evaluation on ABIDE-I unseen test sites.

Usage:
    python eval_zeroshot.py --lm_name gpt2 --checkpoint ckpt_clf_gpt2.pt
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)

from dataset import load_abide, inter_site_split, make_loaders
from model_fmrilm_abide import fMRITokenizer, AlignmentModel, ASDClassifier
from language_models import load_backend

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    (ts_all, fc_corr, fc_tan, fc_par,
     lbl_all, sex_all, site_list, descs) = load_abide(args.data_dir)
    _, _, test_idx, _, small_sites = inter_site_split(lbl_all, site_list)

    llm   = load_backend(args.lm_name)
    D_LLM = llm.D_LLM

    _, _, test_loader = make_loaders(
        ts_all, fc_corr, fc_tan, fc_par, lbl_all, sex_all, descs,
        np.array([]), np.array([]), test_idx, batch_size=args.batch_size)

    tok = fMRITokenizer(D_LLM).to(DEVICE)
    s2  = AlignmentModel(tok, D_LLM).to(DEVICE)
    clf = ASDClassifier(s2, D_LLM).to(DEVICE)

    cp  = torch.load(args.checkpoint, map_location=DEVICE)
    clf.load_state_dict(cp["clf"])
    clf.eval()
    print(f"Loaded: {args.checkpoint}")

    probs, gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            lg, _ = clf(batch["ts"].to(DEVICE))
            probs.extend(F.softmax(lg, 1)[:, 1].cpu().numpy())
            gts.extend(batch["label"].numpy())
    probs, gts = np.array(probs), np.array(gts)
    preds = (probs >= 0.5).astype(int)

    print(f"\n{'='*62}")
    print(f"  Zero-shot Evaluation — {args.lm_name.upper()} | {len(small_sites)} unseen sites")
    print(f"{'='*62}")
    print(f"  Accuracy        : {accuracy_score(gts, preds)*100:.2f}%")
    print(f"  Balanced Acc    : {balanced_accuracy_score(gts, preds)*100:.2f}%")
    print(f"  AUC-ROC         : {roc_auc_score(gts, probs)*100:.2f}%")
    print(f"  Paper target    : Acc=76.56%  AUC=76.22%")
    print(f"{'='*62}")
    print(classification_report(gts, preds, target_names=["Control", "ASD"]))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lm_name",    default="gpt2",
                   choices=["gpt2", "clinicalt5", "deepseek", "grok", "gemini"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   default="./abide_data")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    main(get_args())
