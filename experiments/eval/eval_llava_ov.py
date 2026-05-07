#!/usr/bin/env python3
"""
POPE evaluation with LLaVA-OneVision (2024) + CORAL GM mirror statistics.

LLaVA-OneVision uses:
  - SigLIP-SO400M vision encoder (patch_size=14, 384x384 → 729 patches)
  - Qwen2 language model
  - HF-native: no custom llava library required (transformers >= 4.45)

Usage:
    python eval_llava_ov.py \
        --model_id llava-hf/llava-onevision-qwen2-7b-ov-hf \
        --pope_json /path/to/pope_random.json \
        --coco_img_dir /path/to/coco/val2014 \
        --q 0.10 --out_dir ./out_llava_ov_pope
"""

import os
import json
import argparse
from collections import Counter
from math import sqrt, floor, ceil

import numpy as np
import torch
from PIL import Image

# ===== VCD patch & noise =====
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
evolve_vcd_sampling()

# ===== LLaVA-OneVision — HF native, no custom llava package needed =====
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor


# ------------------------------
# Robust POPE loader (JSON or JSONL)
# ------------------------------
def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "items", "annotations", "samples", "records"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
            return [obj]
    except Exception:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def get_image_filename(ex: dict) -> str:
    for k in ("image", "image_path", "img", "file_name", "filename"):
        if k in ex:
            return str(ex[k])
    raise KeyError("No image filename field in sample")


def get_answer_label(ex: dict) -> str:
    for k in ("answer", "label", "gt", "gt_answer"):
        if k in ex:
            return str(ex[k]).strip().lower()
    raise KeyError("No answer/label field in sample")


def get_question_text(ex: dict) -> str:
    for k in ("question", "prompt", "q", "text"):
        if k in ex:
            return str(ex[k])
    if "object" in ex:
        return f"Is there a {ex['object']} in the image?"
    raise KeyError("No question/prompt in sample")


# ------------------------------
# Yes/No token ids
# ------------------------------
def get_yes_no_ids(tokenizer):
    for y, n in [("Yes", "No"), ("yes", "no")]:
        y_ids = tokenizer.encode(y, add_special_tokens=False)
        n_ids = tokenizer.encode(n, add_special_tokens=False)
        if len(y_ids) == 1 and len(n_ids) == 1:
            return y_ids[0], n_ids[0]
    return (tokenizer.encode("Yes", add_special_tokens=False)[0],
            tokenizer.encode("No",  add_special_tokens=False)[0])


# ------------------------------
# Triplet logits: base / +noise / -noise
# LLaVA-OV pixel_values may be 4D [B,C,H,W] or 5D [B,tiles,C,H,W] (AnyRes)
# ------------------------------
@torch.no_grad()
def get_triplet_logits_llava_ov(
    model,
    processor,
    image_pil: Image.Image,
    question: str,
    noise_step: int = 50,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16,
):
    # Build prompt via Qwen2 chat template
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image_pil, text=text, return_tensors="pt")

    # Move all non-pixel tensors to device
    for k in list(inputs.keys()):
        if k != "pixel_values" and torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)

    # pixel_values: [1,C,H,W] (single-res) or [1,tiles,C,H,W] (AnyRes)
    pv = inputs["pixel_values"]
    orig_shape = pv.shape
    if pv.dim() == 5:
        B, tiles, C, H, W = pv.shape
        pv_flat = pv.reshape(B * tiles, C, H, W).to(device=device, dtype=dtype)
    else:
        pv_flat = pv.to(device=device, dtype=dtype)

    # Gaussian-mirror views
    noisy = add_diffusion_noise(pv_flat, noise_step=noise_step).to(device=device, dtype=dtype)
    z = noisy - pv_flat
    v_base = pv_flat.reshape(orig_shape).to(dtype=dtype)
    v_pos  = (pv_flat + z).reshape(orig_shape).to(dtype=dtype)
    v_neg  = (pv_flat - z).reshape(orig_shape).to(dtype=dtype)

    def _fwd(pixel_values):
        inp = dict(inputs)
        inp["pixel_values"] = pixel_values.to(device)
        return model(**inp)

    out_base = _fwd(v_base)
    l_base = out_base.logits[0, -1, :].clone()
    del out_base; torch.cuda.empty_cache()

    out_pos = _fwd(v_pos)
    l_pos = out_pos.logits[0, -1, :].clone()
    del out_pos; torch.cuda.empty_cache()

    out_neg = _fwd(v_neg)
    l_neg = out_neg.logits[0, -1, :].clone()
    del out_neg; torch.cuda.empty_cache()

    return l_base, l_pos, l_neg


# ------------------------------
# FDP estimator and threshold search (same as eval_llava.py)
# ------------------------------
def fdp_hat(deltas: np.ndarray, t: float) -> float:
    neg = np.sum(deltas <= -t)
    pos = np.sum(deltas >=  t)
    return float(neg + 1) / max(1, int(pos))

def pick_threshold(deltas: np.ndarray, q: float = 0.10) -> float:
    abs_max = np.quantile(np.abs(deltas), 0.999)
    grid = np.linspace(0.0, float(abs_max), 400)
    for t in grid:
        if fdp_hat(deltas, t) <= q:
            return float(t)
    return float(np.quantile(np.abs(deltas), 0.995))


# ------------------------------
# Power (normal approximation with continuity correction)
# ------------------------------
def _z_alpha(alpha_two_sided: float) -> float:
    if abs(alpha_two_sided - 0.05) < 1e-9:
        return 1.959963984540054
    return 1.959963984540054

def _z_alpha_one_sided(alpha_one_sided: float) -> float:
    if abs(alpha_one_sided - 0.05) < 1e-9:
        return 1.6448536269514722
    return 1.6448536269514722

def binom_kcrit_two_sided(n: int, alpha: float = 0.05):
    z = _z_alpha(alpha)
    mu0 = n * 0.5; sigma0 = sqrt(n * 0.25)
    return floor(mu0 - 0.5 - z * sigma0), ceil(mu0 + 0.5 + z * sigma0)

def power_two_sided_binom_normal(n: int, p_true: float, alpha: float = 0.05):
    k_lo, k_hi = binom_kcrit_two_sided(n, alpha)
    mu = n * p_true; sigma = sqrt(n * p_true * (1 - p_true))
    if sigma == 0:
        return (1.0 if (mu <= k_lo or mu >= k_hi) else 0.0), k_lo, k_hi
    from math import erf, sqrt as msqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / msqrt(2.0)))
    power = Phi((k_lo + 0.5 - mu) / sigma) + (1.0 - Phi((k_hi - 0.5 - mu) / sigma))
    return max(0.0, min(1.0, power)), k_lo, k_hi

def binom_kcrit_one_sided_upper(n: int, alpha: float = 0.05):
    z = _z_alpha_one_sided(alpha)
    return ceil(n * 0.5 + 0.5 + z * sqrt(n * 0.25))

def power_one_sided_upper_binom_normal(n: int, p_true: float, alpha: float = 0.05):
    k_hi = binom_kcrit_one_sided_upper(n, alpha)
    mu = n * p_true; sigma = sqrt(n * p_true * (1 - p_true))
    if sigma == 0:
        return (1.0 if mu >= k_hi else 0.0), k_hi
    from math import erf, sqrt as msqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / msqrt(2.0)))
    return max(0.0, min(1.0, 1.0 - Phi((k_hi - 0.5 - mu) / sigma))), k_hi


# ------------------------------
# Main
# ------------------------------
def main(args):
    if args.device == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = (torch.bfloat16 if (device.type == "cuda" and args.force_bf16)
             else (torch.float16 if device.type == "cuda" else torch.float32))

    # Load LLaVA-OneVision
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, dtype=dtype, device_map=None
    ).to(device).eval()

    yes_id, no_id = get_yes_no_ids(processor.tokenizer)

    data = load_json_any(args.pope_json)
    if not data:
        raise RuntimeError(f"No samples loaded from {args.pope_json}")

    deltas_gm, y_true, y_pred, recs = [], [], [], []

    for ex in data:
        try:
            img_name = get_image_filename(ex)
            question  = get_question_text(ex)
            yt        = get_answer_label(ex)
        except Exception as e:
            print("[WARN] Skip sample due to field error:", e); continue

        img_path = os.path.join(args.coco_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}"); continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image {img_path}: {e}"); continue

        try:
            logits_base, logits_pos, logits_neg = get_triplet_logits_llava_ov(
                model, processor, pil, question,
                noise_step=args.noise_step, device=device, dtype=dtype
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"[WARN] OOM for {img_name}, retrying with 336x336 resize...")
                try:
                    pil_small = pil.resize((336, 336), Image.LANCZOS)
                    logits_base, logits_pos, logits_neg = get_triplet_logits_llava_ov(
                        model, processor, pil_small, question,
                        noise_step=args.noise_step, device=device, dtype=dtype
                    )
                except Exception as e2:
                    torch.cuda.empty_cache()
                    print(f"[WARN] Forward failed even after resize for {img_name}: {e2}"); continue
            else:
                print(f"[WARN] Forward failed for {img_name}: {e}"); continue
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}"); continue

        y_base_yes = logits_base[yes_id].item()
        y_pos_yes  = logits_pos[yes_id].item()
        y_neg_yes  = logits_neg[yes_id].item()
        y_base_no  = logits_base[no_id].item()
        y_pos_no   = logits_pos[no_id].item()
        y_neg_no   = logits_neg[no_id].item()

        dpos_yes = y_base_yes - y_pos_yes
        dneg_yes = y_base_yes - y_neg_yes
        gm_yes   = abs(dpos_yes + dneg_yes) - abs(dpos_yes - dneg_yes)

        Sp_clean = y_base_yes - y_base_no
        yp = "yes" if Sp_clean >= 0.0 else "no"

        y_true.append(yt); y_pred.append(yp); deltas_gm.append(gm_yes)
        recs.append({
            "image": img_name, "question": question, "answer": yt,
            "token_step": "final",
            "logit_base_yes": float(y_base_yes), "logit_pos_yes": float(y_pos_yes),
            "logit_neg_yes":  float(y_neg_yes),  "logit_base_no":  float(y_base_no),
            "logit_pos_no":   float(y_pos_no),   "logit_neg_no":   float(y_neg_no),
            "Delta_plus_yes":  float(dpos_yes),  "Delta_minus_yes": float(dneg_yes),
            "GM_stat_yes":     float(gm_yes),    "Sp_clean":        float(Sp_clean),
        })

    if not deltas_gm:
        raise RuntimeError("No valid samples processed. Check paths and formats.")

    deltas_gm = np.array(deltas_gm, dtype=float)
    T = pick_threshold(deltas_gm, q=args.q)

    # POPE metrics
    cnt = Counter()
    for yt, yp in zip(y_true, y_pred):
        key = ("TP" if yt=='yes' and yp=='yes' else
               "TN" if yt=='no'  and yp=='no'  else
               "FP" if yt=='no'  and yp=='yes' else "FN")
        cnt[key] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    acc  = (TP+TN) / max(1, TP+TN+FP+FN)
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    f1   = 2*prec*rec / max(1e-9, prec+rec)

    yes_idx   = [i for i, yp in enumerate(y_pred) if yp == 'yes']
    vis_yes   = float(np.mean(deltas_gm[yes_idx] >= T)) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    neg_le = int(np.sum(deltas_gm <= -T))
    pos_ge = int(np.sum(deltas_gm >=  T))
    fdp_at_T = float(neg_le + 1) / max(1, pos_ge)

    print(f"\n#Samples={len(deltas_gm)}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg_le}, pos>=T={pos_ge})")

    # Power
    n_tot = len(deltas_gm)
    p_hat_sign = float(np.mean(deltas_gm > 0))
    pow_sign, k_lo, k_hi = power_two_sided_binom_normal(n_tot, p_hat_sign)
    pow_acc, k_hi_acc     = power_one_sided_upper_binom_normal(n_tot, acc)
    print("\n==== Power summary ====")
    print(f"Sign test: n={n_tot}, p_hat={p_hat_sign:.3f}, power≈{pow_sign:.3f}, "
          f"kcrit=({k_lo},{k_hi})")
    print(f"Accuracy test (vs 0.5): n={n_tot}, acc={acc:.3f}, power≈{pow_acc:.3f}, "
          f"kcrit_hi={k_hi_acc}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_random_llava_ov_gm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id, "T": T,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "vis_supported_yes": vis_yes, "hallucinated_yes": halluc_yes,
            "fdp_at_T": fdp_at_T,
            "fdp_counts": {"neg_le_minus_t": neg_le, "pos_ge_t": pos_ge},
            "power": {
                "sign_test": {"n": n_tot, "p_hat": p_hat_sign,
                              "power": pow_sign, "kcrit_lo": k_lo, "kcrit_hi": k_hi},
                "accuracy_test_vs_0p5": {"n": n_tot, "acc": acc,
                                         "power": pow_acc, "kcrit_hi": k_hi_acc},
            },
            "records": recs,
        }, f, indent=2)
    print(f"Saved results to: {out_path}")
    print("GM-stat stats:",
          f"mean={deltas_gm.mean():.6f}", f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}", f"max={deltas_gm.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                        help="LLaVA-OV model ID. "
                             "Also supports: llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--pope_json",    type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step",   type=int,   default=50)
    parser.add_argument("--q",            type=float, default=0.10)
    parser.add_argument("--alpha",        type=float, default=0.05)
    parser.add_argument("--out_dir",      type=str,   default="./pope_eval_out_llava_ov")
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--force_bf16",   action="store_true")
    args = parser.parse_args()
    main(args)
