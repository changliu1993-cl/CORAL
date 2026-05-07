#!/usr/bin/env python3
"""
POPE evaluation with Qwen2.5-VL (or Qwen3-VL) + CORAL GM mirror statistics.

Qwen2.5-VL improvements over Qwen2-VL:
  - Native dynamic resolution (any aspect ratio, no padding waste)
  - Stronger vision encoder (Window SigLIP)
  - Better instruction following and temporal grounding

Requires transformers >= 4.49 for Qwen2_5VLForConditionalGeneration.
Falls back to Qwen2VLForConditionalGeneration for older installs.

Usage:
    python eval_qwen25vl.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --pope_json /path/to/pope_random.json \
        --coco_img_dir /path/to/coco/val2014 \
        --q 0.10 --out_dir ./out_qwen25vl_pope
"""

import os, json, argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ===== VCD patch & noise =====
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
evolve_vcd_sampling()

# ===== Qwen2.5-VL — falls back to Qwen2-VL for older transformers =====
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    _QWEN_CLASS = "Qwen2_5_VLForConditionalGeneration"
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    _QWEN_CLASS = "Qwen2VLForConditionalGeneration (fallback)"
from transformers import AutoProcessor


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

def get_image_filename(ex):
    for k in ("image", "image_path", "img", "file_name", "filename"):
        if k in ex: return str(ex[k])
    raise KeyError("No image field")

def get_answer_label(ex):
    for k in ("answer", "label", "gt", "gt_answer"):
        if k in ex: return str(ex[k]).strip().lower()
    raise KeyError("No answer/label field")

def get_question_text(ex):
    for k in ("question", "prompt", "q", "text"):
        if k in ex: return str(ex[k])
    if "object" in ex: return f"Is there a {ex['object']} in the image?"
    raise KeyError("No question/prompt & no 'object'")


# ------------------------------
# Yes/No single-token ids
# ------------------------------
def get_single_token_id(tokenizer, text_list):
    for t in text_list:
        ids = tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if len(ids) == 1:
            return ids[0]
    return None


# ------------------------------
# Build processor inputs (Qwen chat template)
# ------------------------------
def build_inputs(processor, image_pil: Image.Image, question: str, device):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image_pil], return_tensors="pt", padding=True)
    for k in inputs:
        if torch.is_tensor(inputs[k]) and k != "pixel_values":
            inputs[k] = inputs[k].to(device)
    return inputs


# ------------------------------
# Gaussian-Mirror mirror views on pixel_values
# ------------------------------
@torch.no_grad()
def add_mirror_views(pixel_values: torch.FloatTensor, noise_step: int,
                     device: torch.device, dtype: torch.dtype):
    base  = pixel_values.to(device=device, dtype=dtype)
    noisy = add_diffusion_noise(base, noise_step=noise_step).to(device=device, dtype=dtype)
    z = noisy - base
    return base, (base + z).to(dtype=dtype), (base - z).to(dtype=dtype)


# ------------------------------
# Triplet logits: base / +noise / -noise
# ------------------------------
@torch.no_grad()
def get_triplet_logits_qwen25vl(model, processor, image_pil, question,
                                 noise_step, device, dtype):
    inputs = build_inputs(processor, image_pil, question, device)
    pixel  = inputs["pixel_values"].to(device=device, dtype=dtype)
    v_base, v_pos, v_neg = add_mirror_views(pixel, noise_step, device, dtype)

    def _fwd(pv):
        inp = dict(inputs)
        inp["pixel_values"] = pv
        return model(**inp).logits[:, -1, :]

    return _fwd(v_base)[0], _fwd(v_pos)[0], _fwd(v_neg)[0]


# ------------------------------
# Fallback: cumulative log-prob for "Yes." / "No."
# ------------------------------
@torch.no_grad()
def prefix_total_logprob(model, processor, pixel_values, question, prefix_text, device):
    messages = [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefix_text}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    dummy = Image.new("RGB", (1, 1), (0, 0, 0))
    encoded = processor(text=[text], images=[dummy], return_tensors="pt", padding=True)
    for k in encoded:
        if torch.is_tensor(encoded[k]) and k != "pixel_values":
            encoded[k] = encoded[k].to(device)
    encoded["pixel_values"] = pixel_values
    out = model(**encoded)
    log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    labels    = encoded["input_ids"][:, 1:]
    gathered  = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    k = min(gathered.shape[1], 16)
    return gathered[0, -k:].sum().item()


# ------------------------------
# FDR via GM-FDP and threshold search
# ------------------------------
def fdp_hat(deltas: np.ndarray, t: float) -> float:
    neg = np.sum(deltas <= -t); pos = np.sum(deltas >= t)
    return float(neg + 1) / max(1, int(pos))

def pick_threshold(deltas: np.ndarray, q: float = 0.10) -> float:
    abs_max = np.quantile(np.abs(deltas), 0.999)
    grid = np.linspace(0.0, float(abs_max), 400)
    for t in grid:
        if fdp_hat(deltas, t) <= q: return float(t)
    return float(np.quantile(np.abs(deltas), 0.995))


# ------------------------------
# Power estimates (exact binomial CDF, no scipy)
# ------------------------------
import math

def _log_binom_pmf(i, n, p):
    if p <= 0.0: return 0.0 if i == 0 else -float("inf")
    if p >= 1.0: return 0.0 if i == n else -float("inf")
    return (math.lgamma(n+1) - math.lgamma(i+1) - math.lgamma(n-i+1)
            + i*math.log(p) + (n-i)*math.log1p(-p))

def binom_cdf(k, n, p):
    k = max(-1, min(k, n))
    if k < 0: return 0.0
    logs = [_log_binom_pmf(i, n, p) for i in range(k+1)]
    m = max(logs)
    return math.exp(m) * sum(math.exp(li - m) for li in logs)

def binom_sf(k, n, p): return max(0.0, 1.0 - binom_cdf(k, n, p))

def _binom_critical_region_two_sided(n, alpha, p0=0.5):
    target = alpha / 2.0
    loL, loR = -1, n
    while loL + 1 < loR:
        mid = (loL + loR) // 2
        if binom_cdf(mid, n, p0) < target: loL = mid
        else: loR = mid
    k_lo = loL
    hiL, hiR = -1, n
    while hiL + 1 < hiR:
        mid = (hiL + hiR) // 2
        if binom_sf(mid, n, p0) < target: hiR = mid
        else: hiL = mid
    return k_lo, hiR

def binom_power_sign_test(n, p1, alpha=0.05):
    k_lo, k_hi = _binom_critical_region_two_sided(n, alpha)
    beta  = max(0.0, min(1.0, binom_cdf(k_hi-1, n, p1) - binom_cdf(k_lo, n, p1)))
    return 1.0 - beta, beta, (k_lo+1, k_hi-1)

def binom_power_accuracy_test(n, acc, alpha=0.05):
    lo, hi = 0, n+1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if binom_sf(mid-1, n, 0.5) <= alpha: hi = mid
        else: lo = mid
    kcrit = hi
    power = binom_sf(kcrit-1, n, acc)
    return power, 1.0 - power, kcrit


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

    print(f"[INFO] Using model class: {_QWEN_CLASS}")

    # Load Qwen2.5-VL (or Qwen2-VL fallback)
    model = QwenVLModel.from_pretrained(
        args.model_id, dtype=dtype, device_map=None, trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    yes_id = get_single_token_id(processor.tokenizer, ["Yes", "yes"])
    no_id  = get_single_token_id(processor.tokenizer, ["No",  "no"])

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
            print("[WARN] Skip:", e); continue

        img_path = os.path.join(args.coco_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing: {img_path}"); continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Open failed {img_path}: {e}"); continue

        try:
            lb, lp, ln = get_triplet_logits_qwen25vl(
                model, processor, pil, question, args.noise_step, device, dtype)
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}"); continue

        if yes_id is not None and no_id is not None:
            yb_yes, yp_yes, yn_yes = lb[yes_id].item(), lp[yes_id].item(), ln[yes_id].item()
            yb_no,  yp_no,  yn_no  = lb[no_id].item(),  lp[no_id].item(),  ln[no_id].item()
        else:
            # Fallback: prefix log-prob
            inputs = build_inputs(processor, pil, question, device)
            pixel  = inputs["pixel_values"].to(device=device, dtype=dtype)
            v_base, v_pos, v_neg = add_mirror_views(pixel, args.noise_step, device, dtype)
            yb_yes = prefix_total_logprob(model, processor, v_base, question, "Yes.", device)
            yp_yes = prefix_total_logprob(model, processor, v_pos,  question, "Yes.", device)
            yn_yes = prefix_total_logprob(model, processor, v_neg,  question, "Yes.", device)
            yb_no  = prefix_total_logprob(model, processor, v_base, question, "No.",  device)
            yp_no  = prefix_total_logprob(model, processor, v_pos,  question, "No.",  device)
            yn_no  = prefix_total_logprob(model, processor, v_neg,  question, "No.",  device)

        Sp_clean = yb_yes - yb_no
        yp = "yes" if Sp_clean >= 0.0 else "no"

        dpos = yb_yes - yp_yes; dneg = yb_yes - yn_yes
        gm   = abs(dpos + dneg) - abs(dpos - dneg)

        y_true.append(yt); y_pred.append(yp); deltas_gm.append(gm)
        recs.append({
            "image": img_name, "question": question, "answer": yt,
            "logit_base_yes": float(yb_yes), "logit_pos_yes": float(yp_yes),
            "logit_neg_yes":  float(yn_yes), "logit_base_no":  float(yb_no),
            "logit_pos_no":   float(yp_no),  "logit_neg_no":   float(yn_no),
            "Delta_plus_yes":  float(dpos),  "Delta_minus_yes": float(dneg),
            "GM_stat_yes":     float(gm),    "Sp_clean":        float(Sp_clean),
        })

    if not deltas_gm:
        raise RuntimeError("No valid samples processed")

    deltas_gm = np.array(deltas_gm, dtype=float)
    T = pick_threshold(deltas_gm, q=args.q)

    cnt = Counter()
    for yt, yhat in zip(y_true, y_pred):
        key = ("TP" if yt=='yes' and yhat=='yes' else
               "TN" if yt=='no'  and yhat=='no'  else
               "FP" if yt=='no'  and yhat=='yes' else "FN")
        cnt[key] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    acc  = (TP+TN) / max(1, TP+TN+FP+FN)
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    f1   = 2*prec*rec / max(1e-9, prec+rec)

    yes_idx   = [i for i, yhat in enumerate(y_pred) if yhat == 'yes']
    vis_yes   = float(np.mean(deltas_gm[yes_idx] >= T)) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    neg = int(np.sum(deltas_gm <= -T)); pos = int(np.sum(deltas_gm >= T))
    fdp_at_T = float(neg) / max(1, pos)

    print(f"\n#Samples={len(deltas_gm)}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg}, pos>=T={pos})")

    n = len(deltas_gm); p_hat = float((deltas_gm > 0).mean())
    pow_sign, beta_sign, acc_iv = binom_power_sign_test(n, p_hat, alpha=args.alpha)
    pow_acc,  beta_acc,  kcrit  = binom_power_accuracy_test(n, acc,   alpha=args.alpha)
    print("==== Power summary ====")
    print(f"Sign test: n={n}, p_hat={p_hat:.3f}, power={pow_sign:.6f}, accept={acc_iv}")
    print(f"Accuracy test (vs 0.5): n={n}, acc={acc:.3f}, power={pow_acc:.6f}, kcrit={kcrit}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_qwen25vl_gm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id, "model_class": _QWEN_CLASS,
            "T": T, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "vis_supported_yes": vis_yes, "hallucinated_yes": halluc_yes,
            "FDP_at_T": fdp_at_T,
            "records": recs,
        }, f, indent=2)
    print(f"Saved results to: {out_path}")
    print("GM-stat stats:",
          f"mean={deltas_gm.mean():.6f}", f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}", f"max={deltas_gm.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL or Qwen3-VL model ID. "
                             "E.g. Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--pope_json",    type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step",   type=int,   default=50)
    parser.add_argument("--q",            type=float, default=0.10)
    parser.add_argument("--out_dir",      type=str,   default="./out_qwen25vl_pope")
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--force_bf16",   action="store_true")
    parser.add_argument("--alpha",        type=float, default=0.05)
    args = parser.parse_args()
    main(args)
