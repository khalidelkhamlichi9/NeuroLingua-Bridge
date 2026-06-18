#!/bin/bash
# run_deepseek.sh
set -e
LM="deepseek"
DATA="./abide_data"
python train_tokenizer.py       --lm_name $LM --data_dir $DATA --epochs 15 --batch_size 2
python train_pretrain_paired.py --lm_name $LM --data_dir $DATA --ckpt_tok ckpt_tok_${LM}.pt --epochs 10 --batch_size 2
python train_instruction.py     --lm_name $LM --data_dir $DATA --ckpt_s2  ckpt_s2_${LM}.pt  --epochs 25 --batch_size 2
python eval_zeroshot.py         --lm_name $LM --data_dir $DATA --checkpoint ckpt_clf_${LM}.pt
