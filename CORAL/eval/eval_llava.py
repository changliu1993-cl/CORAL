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

# ===== LLaVA =====
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX


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
    raise KeyError("No image filename field in sample (expected 'image'/'image_path'/...)")


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
    raise KeyError("No question/prompt in sample and 'object' not found to build one")


# ------------------------------
# Gaussian-Mirror mirror views (strict same-noise +/-)
# ------------------------------
@torch.no_grad()
def add_mirror_views(
    image_tensor: torch.Tensor,
    noise_step: int = 50,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16,
):
    base = image_tensor.to(device=device, dtype=dtype)
    noisy = add_diffusion_noise(base, noise_step=noise_step)
    noisy = noisy.to(device=device, dtype=dtype)  # safety
    z = noisy - base
    v_pos = (base + z).to(device=device, dtype=dtype)
    v_neg = (base - z).to(device=device, dtype=dtype)
    return base, v_pos, v_neg


# ------------------------------
# Yes/No token ids
# ------------------------------
def get_yes_no_ids(tokenizer):
    pairs = [("Yes", "No"), ("yes", "no")]
    for y, n in pairs:
        y_ids = tokenizer.encode(y, add_special_tokens=False)
        n_ids = tokenizer.encode(n, add_special_tokens=False)
        if len(y_ids) == 1 and len(n_ids) == 1:
            return y_ids[0], n_ids[0]
    y_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    n_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return y_id, n_id


# ------------------------------
# Triplet logits at the final decoding step (base / pos / neg)
# ------------------------------
@torch.no_grad()
def get_triplet_logits_llava(
    model,
    tokenizer,
    image_processor,
    image_pil: Image.Image,
    question: str,
    noise_step: int = 50,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float16,
):
    # image -> tensor
    img_tensor = process_images([image_pil], image_processor, model.config).to(device=device, dtype=dtype)[0]
    # three views
    v_base, v_pos, v_neg = add_mirror_views(img_tensor, noise_step=noise_step, device=device, dtype=dtype)

    # prompt with <image>
    qs = f"USER: <image>\n{question}\nASSISTANT:"
    input_ids = (
        tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    # forward (LLaVA accepts list[Tensor] for images)
    out_base = model(input_ids=input_ids, images=[v_base], return_dict=True)
    out_pos  = model(input_ids=input_ids, images=[v_pos],  return_dict=True)
    out_neg  = model(input_ids=input_ids, images=[v_neg],  return_dict=True)

    last_base = out_base.logits[:, -1, :]
    last_pos  = out_pos.logits[:,  -1, :]
    last_neg  = out_neg.logits[:,  -1, :]

    return last_base[0], last_pos[0], last_neg[0]   # [V], [V], [V]


# ------------------------------
# FDP estimator (GM-FDP) and threshold search
# ------------------------------
def fdp_hat(deltas: np.ndarray, t: float) -> float:
    neg = np.sum(deltas <= -t)
    pos = np.sum(deltas >=  t)
    # +1 in numerator gives a slightly conservative estimate and avoids jitter for small pos
    return float(neg + 1) / max(1, int(pos))

def pick_threshold(deltas: np.ndarray, q: float = 0.10) -> float:
    abs_max = np.quantile(np.abs(deltas), 0.999)
    grid = np.linspace(0.0, float(abs_max), 400)
    for t in grid:
        if fdp_hat(deltas, t) <= q:
            return float(t)
    return float(np.quantile(np.abs(deltas), 0.995))


# ------------------------------
# Power (normal approximation with continuity correction; no SciPy)
# ------------------------------
def _z_alpha(alpha_two_sided: float) -> float:
    # z for common alpha=0.05 (two-sided)
    if abs(alpha_two_sided - 0.05) < 1e-9:
        return 1.959963984540054
    return 1.959963984540054

def _z_alpha_one_sided(alpha_one_sided: float) -> float:
    # z for common alpha=0.05 (one-sided)
    if abs(alpha_one_sided - 0.05) < 1e-9:
        return 1.6448536269514722
    return 1.6448536269514722

def binom_kcrit_two_sided(n: int, alpha: float = 0.05):
    """Two-sided H0: p=0.5 critical region with continuity correction."""
    z = _z_alpha(alpha)
    mu0 = n * 0.5
    sigma0 = sqrt(n * 0.25)
    kcrit_hi = ceil(mu0 + 0.5 + z * sigma0)
    kcrit_lo = floor(mu0 - 0.5 - z * sigma0)
    return kcrit_lo, kcrit_hi

def power_two_sided_binom_normal(n: int, p_true: float, alpha: float = 0.05):
    """Approximate two-sided power for H0: p=0.5 using normal with continuity correction."""
    k_lo, k_hi = binom_kcrit_two_sided(n, alpha)
    mu = n * p_true
    sigma = sqrt(n * p_true * (1 - p_true))
    if sigma == 0:
        return (1.0 if (mu <= k_lo or mu >= k_hi) else 0.0), k_lo, k_hi
    from math import erf, sqrt as msqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / msqrt(2.0)))
    z_lo = (k_lo + 0.5 - mu) / sigma
    z_hi = (k_hi - 0.5 - mu) / sigma
    power = Phi(z_lo) + (1.0 - Phi(z_hi))
    return max(0.0, min(1.0, power)), k_lo, k_hi

def binom_kcrit_one_sided_upper(n: int, alpha: float = 0.05):
    """One-sided H0: p=0.5 vs H1: p>0.5 critical k with continuity correction."""
    z = _z_alpha_one_sided(alpha)
    mu0 = n * 0.5
    sigma0 = sqrt(n * 0.25)
    kcrit_hi = ceil(mu0 + 0.5 + z * sigma0)
    return kcrit_hi

def power_one_sided_upper_binom_normal(n: int, p_true: float, alpha: float = 0.05):
    """Approximate one-sided power for H0: p=0.5 vs H1: p>0.5."""
    k_hi = binom_kcrit_one_sided_upper(n, alpha)
    mu = n * p_true
    sigma = sqrt(n * p_true * (1 - p_true))
    if sigma == 0:
        return (1.0 if mu >= k_hi else 0.0), k_hi
    from math import erf, sqrt as msqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / msqrt(2.0)))
    z_hi = (k_hi - 0.5 - mu) / sigma
    power = 1.0 - Phi(z_hi)
    return max(0.0, min(1.0, power)), k_hi


# ------------------------------
# Main
# ------------------------------
def main(args):
    # device & dtype
    if args.device == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and args.force_bf16) else (torch.float16 if device.type == "cuda" else torch.float32)

    # load LLaVA (single device)
    model_path = args.model_id
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, device_map=None, torch_dtype=dtype
    )
    model.to(device).eval()

    # POPE data
    data = load_json_any(args.pope_json)
    if not data:
        raise RuntimeError(f"No samples loaded from {args.pope_json}")

    yes_id, no_id = get_yes_no_ids(tokenizer)

    # results
    deltas_gm, y_true, y_pred, recs = [], [], [], []

    for ex in data:
        try:
            img_name = get_image_filename(ex)
            question = get_question_text(ex)
            yt = get_answer_label(ex)  # 'yes'/'no'
        except Exception as e:
            print("[WARN] Skip sample due to field error:", e)
            continue

        img_path = os.path.join(args.coco_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open image {img_path}: {e}")
            continue

        try:
            logits_base, logits_pos, logits_neg = get_triplet_logits_llava(
                model, tokenizer, image_processor, pil, question,
                noise_step=args.noise_step, device=device, dtype=dtype
            )
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}")
            continue

        # Yes/No logits at the final step (token y_i)
        y_base_yes, y_pos_yes, y_neg_yes = logits_base[yes_id].item(), logits_pos[yes_id].item(), logits_neg[yes_id].item()
        y_base_no,  y_pos_no,  y_neg_no  = logits_base[no_id].item(),  logits_pos[no_id].item(),  logits_neg[no_id].item()

        # Δ_v^+ and Δ_v^- for "Yes"
        dpos_yes = y_base_yes - y_pos_yes
        dneg_yes = y_base_yes - y_neg_yes

        # Gaussian Mirror statistic for "Yes"
        gm_yes = abs(dpos_yes + dneg_yes) - abs(dpos_yes - dneg_yes)

        # decision using clean view margin
        Sp_clean = y_base_yes - y_base_no
        yp = "yes" if Sp_clean >= 0.0 else "no"

        y_true.append(yt)
        y_pred.append(yp)
        deltas_gm.append(gm_yes)
        recs.append({
            "image": img_name,
            "question": question,
            "answer": yt,
            "token_step": "final",
            "logit_base_yes": float(y_base_yes),
            "logit_pos_yes":  float(y_pos_yes),
            "logit_neg_yes":  float(y_neg_yes),
            "logit_base_no":  float(y_base_no),
            "logit_pos_no":   float(y_pos_no),
            "logit_neg_no":   float(y_neg_no),
            "Delta_plus_yes":  float(dpos_yes),
            "Delta_minus_yes": float(dneg_yes),
            "GM_stat_yes":     float(gm_yes),
            "Sp_clean":        float(Sp_clean)
        })

    if len(deltas_gm) == 0:
        raise RuntimeError("No valid samples processed. Check paths and formats.")

    deltas_gm = np.array(deltas_gm, dtype=float)

    # FDR threshold (GM-stat)
    T = pick_threshold(deltas_gm, q=args.q)

    # Standard POPE metrics
    cnt = Counter()
    for yt, yp in zip(y_true, y_pred):
        key = (
            "TP" if yt == 'yes' and yp == 'yes' else
            "TN" if yt == 'no'  and yp == 'no'  else
            "FP" if yt == 'no'  and yp == 'yes' else
            "FN"
        )
        cnt[key] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    acc  = (TP + TN) / max(1, TP + TN + FP + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    f1   = 2 * prec * rec / max(1e-9, (prec + rec))

    # Visual support among predicted-Yes
    yes_idx = [i for i, yp in enumerate(y_pred) if yp == 'yes']
    vis_yes = float(np.mean((deltas_gm[yes_idx] >= T))) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    # FDP at the chosen T (and raw counts)
    neg_le_minus_t = int(np.sum(deltas_gm <= -T))
    pos_ge_t       = int(np.sum(deltas_gm >=  T))
    fdp_at_T       = float(neg_le_minus_t + 1) / max(1, pos_ge_t)  # same convention as fdp_hat()

    print(f"\n#Samples={len(deltas_gm)}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg_le_minus_t}, pos>=T={pos_ge_t})")

    # ===== Post-hoc power =====
    n_tot = len(deltas_gm)
    p_hat_sign = float(np.mean(deltas_gm > 0))  # for sign test on symmetry
    p_hat_acc  = float(acc)                     # accuracy vs 0.5

    alpha_sign = 0.05
    alpha_acc  = 0.05

    pow_sign, k_lo, k_hi   = power_two_sided_binom_normal(n_tot, p_hat_sign, alpha_sign)
    pow_acc,  k_hi_acc     = power_one_sided_upper_binom_normal(n_tot, p_hat_acc,  alpha_acc)

    print("\n==== Power summary ====")
    print(f"Sign test: n={n_tot}, p_hat=Pr(Δ>0)={p_hat_sign:.3f}, alpha={alpha_sign:.2f}, "
          f"two-sided, power≈{pow_sign:.3f}, kcrit_lo={k_lo}, kcrit_hi={k_hi}")
    print(f"Accuracy test (vs 0.5): n={n_tot}, acc={p_hat_acc:.3f}, alpha={alpha_acc:.2f}, "
          f"one-sided, power≈{pow_acc:.3f}, kcrit_hi={k_hi_acc}")

    # Save JSON
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_random_llava_gm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "T": T,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "vis_supported_yes": vis_yes,
                "hallucinated_yes": halluc_yes,
                "fdp_at_T": fdp_at_T,
                "fdp_counts": {"neg_le_minus_t": neg_le_minus_t, "pos_ge_t": pos_ge_t},
                "power": {
                    "sign_test": {
                        "n": n_tot,
                        "p_hat_Pr_delta_gt_0": p_hat_sign,
                        "alpha_two_sided": alpha_sign,
                        "power_normal_cc": pow_sign,
                        "kcrit_lo": k_lo,
                        "kcrit_hi": k_hi
                    },
                    "accuracy_test_vs_0p5": {
                        "n": n_tot,
                        "acc": p_hat_acc,
                        "alpha_one_sided": alpha_acc,
                        "power_normal_cc": pow_acc,
                        "kcrit_hi": k_hi_acc
                    }
                },
                "records": recs,
            },
            f,
            indent=2,
        )
    print(f"Saved results to: {out_path}")

    # GM-stat summary
    print("GM-stat (Δ_v) stats:",
          f"mean={deltas_gm.mean():.6f}",
          f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}",
          f"p99={np.percentile(deltas_gm,99):.6f}",
          f"max={deltas_gm.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--pope_json", type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step", type=int, default=50)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--out_dir", type=str, default="./pope_eval_out_llava")
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 | cpu | auto")
    parser.add_argument("--force_bf16", action="store_true", help="Use bfloat16 on supported GPUs")
    args = parser.parse_args()
    main(args)
