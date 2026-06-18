"""
NeuroLingua-Bridge — model_fmrilm_abide.py
fMRI-LM adapted for ABIDE-I with pluggable LLM backends.

3-stage pipeline:
  Stage 1 — fMRI Tokenizer  (Encoder → VQ → Decoder)
             + L_tok = w1·L_quant + w2·L_SigLIP + w3·L_GRL   (Σwi = 1)
  Stage 2 — LLM Alignment   (F2F + F2T)
  Stage 3 — ASD Classifier  (attention pool + multi-task head)

Reference: Wei et al. arXiv:2511.21760v3 (2026)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_loss import siglip_loss

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (Appendix C, Table 6)
# ──────────────────────────────────────────────────────────────────────────────
D_ENC = 256   # encoder hidden dim
D_VQ  = 128   # VQ codebook dim
K_VQ  = 8192  # codebook size


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — fMRI Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class GradReversal(torch.autograd.Function):
    """Gradient Reversal Layer for domain-adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return grad.neg() * ctx.alpha, None

def grl_schedule(ep: int, total: int, gamma: float = 10.0) -> float:
    p = ep / max(total - 1, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class LossWeights(nn.Module):
    """
    Learnable softmax-normalised weights for the 3-component tokenizer loss.
    Guarantees w1 + w2 + w3 = 1 at all times.
    """
    def __init__(self, w_q=0.5, w_s=0.3, w_g=0.2):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor([
            math.log(w_q / (1 - w_q + 1e-8)),
            math.log(w_s / (1 - w_s + 1e-8)),
            math.log(w_g / (1 - w_g + 1e-8)),
        ]))
    @property
    def weights(self): return F.softmax(self.raw, dim=0)
    def forward(self, lq, ls, lg):
        w = self.weights; return w[0] * lq + w[1] * ls + w[2] * lg
    def __repr__(self):
        w = self.weights.detach()
        return f"LossWeights(q={w[0]:.3f} s={w[1]:.3f} g={w[2]:.3f} Σ={w.sum():.4f})"


class NormEMAVQ(nn.Module):
    """Normalised EMA VQ with dead-token reset (NeuroLM style)."""
    def __init__(self, K=K_VQ, D=D_VQ, beta=1.0, decay=0.99):
        super().__init__()
        self.K, self.D, self.beta, self.decay = K, D, beta, decay
        self.embed = nn.Embedding(K, D)
        nn.init.uniform_(self.embed.weight, -1/K, 1/K)
        self.register_buffer("ema_cnt", torch.ones(K))
        self.register_buffer("ema_w",   self.embed.weight.data.clone())

    def forward(self, z):
        zn = F.normalize(z, dim=-1)
        en = F.normalize(self.embed.weight, dim=-1)
        zf = zn.reshape(-1, self.D)
        d  = zf.pow(2).sum(1, True) + en.pow(2).sum(1) - 2 * (zf @ en.T)
        idx = d.argmin(1)
        zq  = self.embed(idx).view(z.shape)
        if self.training:
            oh = F.one_hot(idx, self.K).float()
            self.ema_cnt = self.decay * self.ema_cnt + (1 - self.decay) * oh.sum(0)
            self.ema_w   = self.decay * self.ema_w   + (1 - self.decay) * (oh.T @ zf)
            dead = self.ema_cnt < 0.5
            n_d  = int(dead.sum())
            if n_d:
                n_p  = min(n_d, zf.shape[0])
                perm = torch.randperm(zf.shape[0], device=zf.device)[:n_p]
                di   = torch.where(dead)[0][:n_p]
                self.ema_w.data[di]   = zf[perm]
                self.ema_cnt.data[di] = 1.
            self.embed.weight.data = F.normalize(
                self.ema_w / (self.ema_cnt.unsqueeze(1) + 1e-5), dim=-1)
        return z + (zq - z).detach(), self.beta * F.mse_loss(zq.detach(), z), idx

    @property
    def utilisation(self): return (self.ema_cnt > 0.5).float().mean().item()


class fMRIEncoder(nn.Module):
    def __init__(self, n_rois=450, T=160, d=D_ENC, nhead=8, nlayers=3, P=32):
        super().__init__()
        self.P = P
        self.roi_proj = nn.Linear(n_rois, d)
        self.t_proj   = nn.Linear(P, d)
        self.combine  = nn.Linear(d * 2, d)
        enc = nn.TransformerEncoderLayer(d, nhead, d*4, 0.1,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, nlayers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):          # x: (B, T, N)
        B, T, N = x.shape
        T2  = T // self.P
        xp  = x[:, :T2*self.P, :].reshape(B, T2, self.P, N)
        z   = self.combine(torch.cat([self.roi_proj(xp.mean(2)),
                                      self.t_proj(xp.mean(-1))], -1))
        return self.norm(self.transformer(z))  # (B, T2, D_ENC)


class fMRIDecoder(nn.Module):
    def __init__(self, d_in=D_ENC, P=32, N=450):
        super().__init__()
        self.P, self.N = P, N
        dec = nn.TransformerEncoderLayer(d_in, 4, d_in*2, 0.1,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(dec, 2)
        self.out = nn.Linear(d_in, P * N)

    def forward(self, z):          # z: (B, T2, d_in)
        B, T2, _ = z.shape
        return self.out(self.transformer(z)).reshape(B, T2*self.P, self.N)


class fMRITokenizer(nn.Module):
    """
    Stage-1 model: Encoder → Task Layer → VQ Codebook → Decoder.
    Computes L_tok = w1·L_quant + w2·L_SigLIP + w3·L_GRL.
    """
    def __init__(self, d_llm: int = 1024):
        super().__init__()
        self.encoder    = fMRIEncoder()
        self.task_layer = nn.Sequential(nn.Linear(D_ENC, D_ENC), nn.Tanh(),
                                        nn.Linear(D_ENC, D_VQ))
        self.quantizer  = NormEMAVQ()
        self.vq_to_enc  = nn.Linear(D_VQ, D_ENC)
        self.decoder    = fMRIDecoder(d_in=D_ENC)
        self.proj_out   = nn.Sequential(nn.Linear(D_VQ, d_llm), nn.GELU())
        self.dom_clf    = nn.Sequential(
            nn.Linear(D_ENC, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fmri_head  = nn.Sequential(nn.Linear(D_ENC, 256), nn.LayerNorm(256))
        self.text_head  = nn.Sequential(nn.Linear(d_llm, 256),  nn.LayerNorm(256))
        self.log_temp   = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode(self, x) -> torch.Tensor:
        z = self.encoder(x)
        zq, _, _ = self.quantizer(self.task_layer(z))
        return self.proj_out(zq)            # (B, T2, d_llm)

    def forward(self, x, text_emb=None, lam: float = 0.5, loss_weights=None):
        B = x.shape[0]
        z       = self.encoder(x)           # (B, T2, D_ENC)
        z_vq    = self.task_layer(z)
        zq, lc, _ = self.quantizer(z_vq)

        recon   = self.decoder(self.vq_to_enc(zq.detach()))
        l_quant = F.mse_loss(recon, x) + lc

        z_pool  = z.mean(1)                 # (B, D_ENC)
        l_dom   = F.binary_cross_entropy_with_logits(
            self.dom_clf(GradReversal.apply(z_pool, lam)),
            torch.ones(B, 1, device=x.device))

        l_contr = torch.tensor(0., device=x.device)
        if text_emb is not None:
            l_contr = siglip_loss(self.fmri_head(z_pool),
                                  self.text_head(text_emb), self.log_temp)

        total = (loss_weights(l_quant, l_contr, l_dom)
                 if loss_weights is not None
                 else l_quant + l_contr + lam * l_dom)

        return {"total": total, "quant": l_quant,
                "contrast": l_contr, "domain": l_dom, "z_pool": z_pool}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LLM Alignment  (F2F + F2T)
# ══════════════════════════════════════════════════════════════════════════════

ALPHA_F2F = 0.1

class AlignmentModel(nn.Module):
    """Frozen tokenizer + alignment heads (F2F autoregressive + F2T cosine)."""
    def __init__(self, tokenizer: fMRITokenizer, d_llm: int = 1024):
        super().__init__()
        self.tok = tokenizer
        for p in self.tok.parameters(): p.requires_grad = False
        self.projector  = nn.Sequential(nn.Linear(d_llm, d_llm), nn.GELU())
        self.f2f_head   = nn.Linear(d_llm, D_VQ)
        self.align_head = nn.Sequential(nn.Linear(d_llm, d_llm), nn.LayerNorm(d_llm))
        self.d_llm = d_llm

    def get_features(self, x): return self.projector(self.tok.encode(x))

    def forward(self, x, llm_emb=None):
        h = self.get_features(x)
        with torch.no_grad():
            z_vq = self.tok.task_layer(self.tok.encoder(x))
        l_f2f = F.mse_loss(self.f2f_head(h[:, :-1, :]), z_vq[:, 1:, :])
        l_f2t = torch.tensor(0., device=x.device)
        if llm_emb is not None:
            pool  = self.align_head(h.mean(1))
            l_f2t = 1.0 - F.cosine_similarity(
                F.normalize(pool, dim=-1), F.normalize(llm_emb, dim=-1)).mean()
        return {"total": l_f2t + ALPHA_F2F * l_f2f, "f2f": l_f2f, "f2t": l_f2t, "h": h}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ASD Classifier
# ══════════════════════════════════════════════════════════════════════════════

class ASDClassifier(nn.Module):
    """
    Multi-task ASD classifier on top of the Stage-2 alignment model.
    Primary task: ASD (1=ASD, 0=Control).
    Auxiliary task: sex prediction (weight=0.3).
    Target: Acc ≥ 76.56%, AUC ≥ 76.22% (Table 3, Wei et al. 2026).
    """
    def __init__(self, align_model: AlignmentModel, d_llm: int = 1024, hidden: int = 512):
        super().__init__()
        self.align = align_model
        for p in self.align.parameters(): p.requires_grad = False
        for p in self.align.projector.parameters():  p.requires_grad = True
        for p in self.align.align_head.parameters(): p.requires_grad = True

        self.attn_pool = nn.Sequential(nn.Linear(d_llm, 1), nn.Softmax(dim=1))
        self.feat = nn.Sequential(
            nn.Linear(d_llm, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU())
        self.asd_head = nn.Linear(hidden // 2, 2)
        self.sex_head = nn.Linear(hidden // 2, 2)

    def forward(self, x):
        h   = self.align.get_features(x)          # (B, T2, d_llm)
        atw = self.attn_pool(h)                    # (B, T2, 1)
        hp  = (h * atw).sum(1)                     # (B, d_llm)
        f   = self.feat(hp)
        return self.asd_head(f), self.sex_head(f)

    def loss(self, asd_lg, sex_lg, lbl_asd, lbl_sex):
        return (F.cross_entropy(asd_lg, lbl_asd, label_smoothing=0.08)
                + 0.3 * F.cross_entropy(sex_lg, lbl_sex, label_smoothing=0.05))
