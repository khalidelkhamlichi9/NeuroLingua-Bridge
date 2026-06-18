#!/bin/bash
# run_grok.sh
set -e
LM="grok"
DATA="./abide_data"
[ -z "$XAI_API_KEY" ] && { echo "ERROR: set XAI_API_KEY first"; exit 1; }
python train_tokenizer.py       --lm_name $LM --data_dir $DATA --epochs 15 --batch_size 4
python train_pretrain_paired.py --lm_name $LM --data_dir $DATA --ckpt_tok ckpt_tok_${LM}.pt --epochs 10 --batch_size 4
python train_instruction.py     --lm_name $LM --data_dir $DATA --ckpt_s2  ckpt_s2_${LM}.pt  --epochs 25 --batch_size 4
python eval_zeroshot.py         --lm_name $LM --data_dir $DATA --checkpoint ckpt_clf_${LM}.pt
