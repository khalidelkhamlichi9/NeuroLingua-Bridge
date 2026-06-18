#!/bin/bash
# NeuroLingua-Bridge — scripts/run_gpt2.sh
# Full 3-stage pipeline with GPT-2 Medium (local, no API key required).
set -e
LM="gpt2"
DATA="./abide_data"
echo "========================================"
echo "  NeuroLingua-Bridge  |  LLM: $LM"
echo "========================================"
echo "[Stage 1] Tokenizer + lambda search..."
python train_tokenizer.py       --lm_name $LM --data_dir $DATA --epochs 15 --batch_size 4
echo "[Stage 2] LLM alignment..."
python train_pretrain_paired.py --lm_name $LM --data_dir $DATA --ckpt_tok ckpt_tok_${LM}.pt --epochs 10
echo "[Stage 3] ASD classifier..."
python train_instruction.py     --lm_name $LM --data_dir $DATA --ckpt_s2  ckpt_s2_${LM}.pt  --epochs 25
echo "[Eval]    Zero-shot on unseen sites..."
python eval_zeroshot.py         --lm_name $LM --data_dir $DATA --checkpoint ckpt_clf_${LM}.pt
echo "Done → ckpt_clf_${LM}.pt"
