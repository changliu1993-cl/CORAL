#!/usr/bin/env python3
"""
POPE evaluation with InternVL3 (OpenGVLab/InternVL3-8B) + CORAL GM mirror statistics.

Usage:
    python eval_internvl.py \
        --model_id OpenGVLab/InternVL3-8B \
        --pope_json /path/to/pope_random.json \
        --coco_img_dir /path/to/coco/val2014 \
        --q 0.10 --out_dir ./out_internvl_pope
"""

import os, json, argparse, math
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# ===== VCD patch & noise =====
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
evolve_vcd_sampling()

# ===== InternVL3 (trust_remote_code) =====
from transformers import AutoModel, AutoTokenizer

# InternVL image token constants
IMG_START_TOKEN  = "<img>"
IMG_END_TOKEN    = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 448   # InternVL3 default input resolution


# ------------------------------
# InternVL image preprocessing
# ------------------------------
def build_transform(image_size: int = IMG_SIZE):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def pil_to_pixel_values(pil_img: Image.Image, device, dtype,
                         image_size: int = IMG_SIZE) -> torch.Tensor:
    transform = build_transform(image_size)
    pv = transform(pil_img).unsqueeze(0)   # [1, C, H, W]
    return pv.to(device=device, dtype=dtype)


# ------------------------------
# Robust POPE loaders
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
    raise KeyError("No image field in sample")

def get_answer_label(ex):
    for k in ("answer", "label", "gt", "gt_answer"):
        if k in ex: return str(ex[k]).strip().lower()
    raise KeyError("No answer/label field")

def get_question_text(ex):
    for k in ("question", "prompt", "q", "text"):
        if k in ex: return str(ex[k])
    if "object" in ex:
        return f"Is there a {ex['object']} in the image?"
    raise KeyError("No question field")


# ------------------------------
# Build input_ids with image tokens for InternVL
# ------------------------------
def build_internvl_input(tokenizer, model, question: str, device):
    num_img_tokens = model.num_image_token  # e.g. 256 after compression
    img_placeholder = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_img_tokens + IMG_END_TOKEN
    prompt = img_placeholder + "\n" + question
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    return input_ids


# ------------------------------
# Yes/No single token ids
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
# Gaussian-Mirror mirror views
# ------------------------------
@torch.no_grad()
def add_mirror_views(pixel_values: torch.Tensor, noise_step: int,
                     device: torch.device, dtype: torch.dtype):
    base  = pixel_values.to(device=device, dtype=dtype)
    noisy = add_diffusion_noise(base, noise_step=noise_step).to(device=device, dtype=dtype)
    z = noisy - base
    v_pos = (base + z).to(dtype=dtype)
    v_neg = (base - z).to(dtype=dtype)
    return base, v_pos, v_neg


# ------------------------------
# Triplet logits: base / +noise / -noise
# ------------------------------
@torch.no_grad()
def get_triplet_logits_internvl(model, tokenizer, image_pil: Image.Image,
                                question: str, noise_step: int,
                                device: torch.device, dtype: torch.dtype):
    pixel_values = pil_to_pixel_values(image_pil, device, dtype)
    v_base, v_pos, v_neg = add_mirror_views(pixel_values, noise_step, device, dtype)
    input_ids = build_internvl_input(tokenizer, model, question, device)

    def _fwd(pv):
        out = model(pixel_values=pv, input_ids=input_ids, return_dict=True)
        return out.logits[0, -1, :]   # [V]

    return _fwd(v_base), _fwd(v_pos), _fwd(v_neg)


# ------------------------------
# FDR via GM-FDP and threshold search
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
# Binomial power helpers (exact CDF, no scipy)
# ------------------------------
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

def binom_power_sign_test(n, p1, alpha=0.05):
    target = alpha / 2.0
    loL, loR = -1, n
    while loL + 1 < loR:
        mid = (loL + loR) // 2
        (loL if binom_cdf(mid, n, 0.5) < target else loR).__class__  # noqa
        if binom_cdf(mid, n, 0.5) < target: loL = mid
        else: loR = mid
    k_lo = loL
    hiL, hiR = -1, n
    while hiL + 1 < hiR:
        mid = (hiL + hiR) // 2
        if binom_sf(mid, n, 0.5) < target: hiR = mid
        else: hiL = mid
    k_hi = hiR
    beta = max(0.0, min(1.0, binom_cdf(k_hi-1, n, p1) - binom_cdf(k_lo, n, p1)))
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
    dtype = torch.bfloat16 if (device.type == "cuda" and args.force_bf16) else (
            torch.float16 if device.type == "cuda" else torch.float32)

    # Load InternVL3
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)

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
            print("[WARN] Skip sample:", e); continue

        img_path = os.path.join(args.coco_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}"); continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}"); continue

        try:
            lb, lp, ln = get_triplet_logits_internvl(
                model, tokenizer, pil, question,
                noise_step=args.noise_step, device=device, dtype=dtype)
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}"); continue

        yb_yes, yp_yes, yn_yes = lb[yes_id].item(), lp[yes_id].item(), ln[yes_id].item()
        yb_no,  yp_no,  yn_no  = lb[no_id].item(),  lp[no_id].item(),  ln[no_id].item()

        Sp_clean = yb_yes - yb_no
        yp = "yes" if Sp_clean >= 0.0 else "no"

        dpos = yb_yes - yp_yes
        dneg = yb_yes - yn_yes
        gm   = abs(dpos + dneg) - abs(dpos - dneg)

        y_true.append(yt); y_pred.append(yp); deltas_gm.append(gm)
        recs.append({
            "image": img_name, "question": question, "answer": yt,
            "logit_base_yes": float(yb_yes), "logit_pos_yes": float(yp_yes),
            "logit_neg_yes": float(yn_yes), "logit_base_no": float(yb_no),
            "logit_pos_no": float(yp_no), "logit_neg_no": float(yn_no),
            "Delta_plus_yes": float(dpos), "Delta_minus_yes": float(dneg),
            "GM_stat_yes": float(gm), "Sp_clean": float(Sp_clean),
        })

    if not deltas_gm:
        raise RuntimeError("No valid samples processed.")

    deltas_gm = np.array(deltas_gm, dtype=float)
    T = pick_threshold(deltas_gm, q=args.q)

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

    yes_idx = [i for i, yp in enumerate(y_pred) if yp == 'yes']
    vis_yes   = float(np.mean(deltas_gm[yes_idx] >= T)) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    neg = int(np.sum(deltas_gm <= -T))
    pos = int(np.sum(deltas_gm >=  T))
    fdp_at_T = float(neg+1) / max(1, pos)

    print(f"\n#Samples={len(deltas_gm)}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg}, pos>=T={pos})")

    n = len(deltas_gm)
    p_hat = float((deltas_gm > 0).mean())
    pow_sign, _, acc_iv = binom_power_sign_test(n, p_hat, alpha=args.alpha)
    pow_acc,  _, kcrit  = binom_power_accuracy_test(n, acc, alpha=args.alpha)
    print(f"\n==== Power summary ====")
    print(f"Sign test: n={n}, p_hat={p_hat:.3f}, power={pow_sign:.4f}, accept={acc_iv}")
    print(f"Accuracy test (vs 0.5): n={n}, acc={acc:.3f}, power={pow_acc:.4f}, kcrit={kcrit}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_internvl_gm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "T": T, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "vis_supported_yes": vis_yes, "hallucinated_yes": halluc_yes,
            "fdp_at_T": fdp_at_T,
            "power": {
                "sign_test":  {"n": n, "p_hat": p_hat, "power": pow_sign},
                "acc_test":   {"n": n, "acc": acc, "power": pow_acc, "kcrit": kcrit},
            },
            "records": recs,
        }, f, indent=2)
    print(f"Saved results to: {out_path}")

    print("GM-stat stats:",
          f"mean={deltas_gm.mean():.6f}",
          f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}",
          f"max={deltas_gm.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3-8B",
                        help="InternVL3 model ID, e.g. OpenGVLab/InternVL3-8B or InternVL3-14B")
    parser.add_argument("--pope_json",    type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step",   type=int,   default=50)
    parser.add_argument("--q",            type=float, default=0.10)
    parser.add_argument("--out_dir",      type=str,   default="./out_internvl_pope")
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--force_bf16",   action="store_true")
    parser.add_argument("--alpha",        type=float, default=0.05)
    args = parser.parse_args()
    main(args)
