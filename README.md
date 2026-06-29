# NeuroLingua-Bridge
## Diagnosis of Autism Spectrum Disorder using LLMs and Multimodal Brain Connectivity Analysis

**Author:** Mr. Khalid EL KHAMLICHI

---

## Overview

NeuroLingua-Bridge extends the fMRI-LM framework with three domain-generalization mechanisms for cross-site ASD classification on ABIDE-I (17 sites, CC200 atlas):

1. **Multi-class site-adversarial GRL** — 17-way site discriminator + gradient reversal → provably drives latent–site mutual information I(z;s) → 0
2. **Dynamic latent prototypes** — learnable EMA-updated geometric anchor points in the embedding space, site-balanced
3. **Episodic Group-DRO** — site-as-task meta-learning, optimizes worst-site risk

**Main model:** DeepSeek-Coder-1.3B  
**Benchmarks:** GPT-2, ClinicalT5, Grok, Gemini



---

## Repository Structure

```
NeuroLingua-Bridge/
├── dataset.py                      # ABIDE-I loader (CC200, 200 ROIs, 17 sites, inter-site split)
├── model_fmrilm_abide.py           # Full model: encoder + VQ + site-adversarial GRL + prototypes
├── train_tokenizer.py              # Stage 1: tokenizer training (Lquant + SigLIP + Lsite)
├── train_pretrain_paired.py        # Stage 2: LLM alignment (F2F + F2T)
├── train_instruction.py            # Stage 3: ASD classifier + Group-DRO
├── eval_zeroshot.py                # Evaluation on held-out unseen sites
├── brain_encoder/                  # Transformer encoder + patch embedding
├── language_models/                # 5 LLM backends (DeepSeek, GPT-2, ClinicalT5, Grok, Gemini)
├── quantizers/                     # VQ, FSQ, NormEMA quantizers
├── chatbot/                        # 3-paradigm clinical chatbot
├── metrics/                        # Evaluation metrics (AUC, F1, worst-site acc)
├── configs/                        # YAML configs + DeepSpeed
├── scripts/                        # Run scripts per backbone
├── nbs_data/                       # Descriptor generation + preprocessing
├── NeuroLingua_Bridge_Colab.ipynb  # ✅ Colab-ready runnable notebook (start here)
├── NeuroLingua_Bridge_DomainGen.ipynb  # Domain-gen focused notebook
├── requirements.txt
└── paper/                          # LaTeX source + compiled PDF
```


## Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/NeuroLingua-Bridge.git
cd NeuroLingua-Bridge
pip install -r requirements.txt
```

### Run Stage 1 (tokenizer)
```bash
python train_tokenizer.py --backbone deepseek --epochs 15
```

### Run Stage 3 (classifier + Group-DRO)
```bash
python train_instruction.py --backbone deepseek --epochs 25 --use_groupdro
```

### Evaluate on held-out sites
```bash
python eval_zeroshot.py --backbone deepseek --checkpoint checkpoints/stage3_deepseek.pt
```

---

## Key Scientific Contribution

This work addresses a specific limitation of **fMRI-LM** (Wei et al., arXiv:2511.21760):

| fMRI-LM | NeuroLingua-Bridge |
|---------|-------------------|
| Binary fMRI-vs-text adversary (no site label) | **17-way site adversary** → I(z;s) = 0 |
| No geometric structure in embedding space | **Dynamic latent prototypes** (EMA, site-balanced) |
| ERM training (dominated by large sites) | **Episodic Group-DRO** (worst-site risk) |
| Within-site / mixed-site evaluation | **Strict inter-site protocol** (unseen test sites) |
| Schaefer-400 + Tian (450 ROIs) | **CC200 atlas (200 ROIs, native)** |


## Citation

```bibtex
@article{elkhamlichi2026neurolinguabridge,
  title   = {NeuroLingua-Bridge: Diagnosis of Autism Spectrum Disorder 
             using LLMs and Multimodal Brain Connectivity Analysis},
  author  = {EL KHAMLICHI, Khalid},
  year    = {2026}
}
```

---

## Acknowledgements

Built on top of [fMRI-LM](https://arxiv.org/abs/2511.21760) (Wei et al., 2026).  
Dataset: [ABIDE-I](http://fcon_1000.projects.nitrc.org/indi/abide/) via Nilearn.  
Atlas: [CC200](https://www.nitrc.org/projects/bioimagesuite/) (Craddock et al., 2012).
