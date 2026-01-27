#!/usr/bin/env bash
set -euo pipefail

# ===== 建议先锁定单卡 =====
export CUDA_VISIBLE_DEVICES=0
# 可选：调试 CUDA 报错更直观
# export CUDA_LAUNCH_BLOCKING=1
# 可选：HF 缓存目录
# export HF_HOME="$HOME/.cache/huggingface"

# ===== 参数（按需修改路径）=====
# MODEL="liuhaotian/llava-v1.5-7b"
# MODEL="Qwen/Qwen2-VL-7B-Instruct"
MODEL="Salesforce/instructblip-vicuna-7b"
COCO="/home/ad/ch786084/VLM/VCD-master/data/COCO/val2014"
Q="0.10"
NOISE="50"
DEV="cuda:0"
# ALPHA="0.05"

SPLIT_RANDOM="/home/ad/ch786084/VLM/VCD-master/experiments/data/POPE/coco/coco_pope_random.json"
SPLIT_ADVERSARIAL="/home/ad/ch786084/VLM/VCD-master/experiments/data/POPE/coco/coco_pope_adversarial.json"
SPLIT_POPULAR="/home/ad/ch786084/VLM/VCD-master/experiments/data/POPE/coco/coco_pope_popular.json"

# ===== 运行 =====
echo ">>> POPE-RANDOM"
# python3 eval_pope_vcd_gm_llava.py \
python3 eval_pope_vcd_gm_instructblip.py \
  --model_id "$MODEL" \
  --pope_json "$SPLIT_RANDOM" \
  --coco_img_dir "$COCO" \
  --device "$DEV" \
  --q "$Q" \
  --noise_step "$NOISE" \
  --out_dir "./out_random_token" \
  # --out_dir "./out_qwen_random" \
  # --alpha "$ALPHA"

echo ">>> POPE-ADVERSARIAL"
# python3 eval_pope_vcd_gm_llava.py \
python3 eval_pope_vcd_gm_instructblip.py \
  --model_id "$MODEL" \
  --pope_json "$SPLIT_ADVERSARIAL" \
  --coco_img_dir "$COCO" \
  --device "$DEV" \
  --q "$Q" \
  --noise_step "$NOISE" \
  --out_dir "./out_adversarial_token" \
  # --out_dir "./out_qwen_adversarial" \
  # --alpha "$ALPHA"

echo ">>> POPE-POPULAR"
# python3 eval_pope_vcd_gm_llava.py \
python3 eval_pope_vcd_gm_instructblip.py \
  --model_id "$MODEL" \
  --pope_json "$SPLIT_POPULAR" \
  --coco_img_dir "$COCO" \
  --device "$DEV" \
  --q "$Q" \
  --noise_step "$NOISE" \
  --out_dir "./out_popular_token" \
  # --out_dir "./out_qwen_popular" \
  # --alpha "$ALPHA"

echo "All splits done."
