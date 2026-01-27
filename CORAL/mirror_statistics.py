#!/usr/bin/env python3
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

# ===== InstructBLIP =====
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ------------------------------
# Robust POPE loader (JSON or JSONL)
# ------------------------------
def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # try JSON
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
    # jsonl
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        recs.append(json.loads(line))
    return recs

def get_image_filename(ex: dict) -> str:
    for k in ("image", "image_path", "img", "file_name", "filename"):
        if k in ex:
            return str(ex[k])
    raise KeyError("No image filename field")

def get_answer_label(ex: dict) -> str:
    for k in ("answer", "label", "gt", "gt_answer"):
        if k in ex:
            return str(ex[k]).strip().lower()
    raise KeyError("No answer/label field")

def get_question_text(ex: dict) -> str:
    for k in ("question", "prompt", "q", "text"):
        if k in ex:
            return str(ex[k])
    if "object" in ex:
        return f"Is there a {ex['object']} in the image?"
    raise KeyError("No question/prompt & no 'object'")

# ------------------------------
# GM mirror views on pixel_values
# ------------------------------
@torch.no_grad()
def add_mirror_views(pixel_values: torch.FloatTensor,
                     noise_step: int,
                     device: torch.device,
                     dtype: torch.dtype):
    base = pixel_values.to(device=device, dtype=dtype)
    noisy = add_diffusion_noise(base, noise_step=noise_step).to(device=device, dtype=dtype)
    z = noisy - base
    v_pos = (base + z).to(device=device, dtype=dtype)
    v_neg = (base - z).to(device=device, dtype=dtype)
    return base, v_pos, v_neg

# ------------------------------

# ------------------------------
def try_get_one_token_id(tokenizer, txts):
    for t in txts:
        ids = tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if len(ids) == 1:
            return ids[0]
    return None

# ------------------------------

# ------------------------------
def build_inputs(processor, image_pil: Image.Image, question: str, device):
    inputs = processor(images=image_pil, text=question, return_tensors="pt")
    # 把非像素张量挪到 device；像素我们要做镜像
    for k in inputs:
        if torch.is_tensor(inputs[k]) and k != "pixel_values":
            inputs[k] = inputs[k].to(device)
    return inputs

# ------------------------------

# ------------------------------
@torch.no_grad()
def last_step_logits(model, inputs):
    out = model(**inputs)
    # 对于 encoder-decoder（Flan-T5）是 decoder logits；对于 Vicuna 变体是 LM logits
    logits = out.logits  # [B, L, V]
    return logits[:, -1, :]  # [B, V]

@torch.no_grad()
def get_triplet_logits_instructblip(model, processor, image_pil, question,
                                    noise_step, device, dtype):
    inputs = build_inputs(processor, image_pil, question, device)
    pixel = inputs["pixel_values"].to(device=device, dtype=dtype)

    v_base, v_pos, v_neg = add_mirror_views(pixel, noise_step, device, dtype)

    ib = {k: v for k, v in inputs.items()}; ib["pixel_values"] = v_base
    ip = {k: v for k, v in inputs.items()}; ip["pixel_values"] = v_pos
    ineg= {k: v for k, v in inputs.items()}; ineg["pixel_values"] = v_neg

    logits_base = last_step_logits(model, ib)[0]  # [V]
    logits_pos  = last_step_logits(model, ip)[0]
    logits_neg  = last_step_logits(model, ineg)[0]
    return logits_base, logits_pos, logits_neg

# ------------------------------
# FDP & threshold
# ------------------------------
def fdp_hat(deltas: np.ndarray, t: float) -> float:
    neg = np.sum(deltas <= -t)
    pos = np.sum(deltas >=  t)
    return float(neg + 1) / max(1, int(pos))  # +1 稍保守

def pick_threshold(deltas: np.ndarray, q: float = 0.10) -> float:
    if len(deltas) == 0:
        return 0.0
    abs_max = np.quantile(np.abs(deltas), 0.999)
    grid = np.linspace(0.0, float(abs_max), 400)
    for t in grid:
        if fdp_hat(deltas, t) <= q:
            return float(t)
    return float(np.quantile(np.abs(deltas), 0.995))

# ------------------------------

# ------------------------------
def normal_cdf(z):
    # Φ(z)
    return 0.5 * (1.0 + torch.erf(torch.tensor(z)/np.sqrt(2.0))).item()

def binom_power_sign_normal(n, p1, alpha=0.05):
    # 两侧检验，H0: p=0.5，Ha: p≠0.5
    if n <= 0:
        return 0.0, 1.0
    z_alpha_2 = 1.959963984540054  # ~ N^-1(1 - alpha/2)
    # 临界线（标准化）：
    # 以 p0=0.5 为基准，功效在 p1 下的近似
    # 这里给出保守近似：power ≈ 1 - [Φ(z_hi) - Φ(z_lo)]
    mu = n * 0.5
    sigma = np.sqrt(n * 0.5 * 0.5)
    # 将 p1 的均值差映射到标准化偏移
    shift = (p1 - 0.5) * np.sqrt(n) / 0.5  # = (p1-0.5)*sqrt(n)/(sqrt(p0(1-p0)))
    # 对应双侧：|Z| > z_a/2
    power = 1.0 - (normal_cdf(z_alpha_2 - shift) - normal_cdf(-z_alpha_2 - shift))
    power = float(np.clip(power, 0.0, 1.0))
    return power, 1.0 - power

def binom_power_acc_normal(n, acc, alpha=0.05):
    # 单侧检验，H0: p=0.5, Ha: p>0.5
    if n <= 0:
        return 0.0, 1.0
    z_alpha = 1.6448536269514722  # ~ N^-1(1 - alpha)
    shift = (acc - 0.5) * np.sqrt(n) / 0.5
    power = 1.0 - normal_cdf(z_alpha - shift)
    power = float(np.clip(power, 0.0, 1.0))
    return power, 1.0 - power

# ------------------------------
# Main
# ------------------------------
def main(args):
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        dtype = torch.bfloat16 if args.force_bf16 else torch.float16
    else:
        dtype = torch.float32


    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=None
    ).to(device).eval()
    processor = InstructBlipProcessor.from_pretrained(args.model_id)

    # Yes/No 单 token id（优先带空格，再不行用无空格）
    tkr = processor.tokenizer
    yes_id = try_get_one_token_id(tkr, [" Yes", "Yes"])
    no_id  = try_get_one_token_id(tkr, [" No", "No"])


    data = load_json_any(args.pope_json)
    if not data:
        raise RuntimeError(f"No samples loaded from {args.pope_json}")

    deltas_gm, y_true, y_pred, recs = [], [], [], []

    kept = 0
    for ex in data:
        try:
            img_name = get_image_filename(ex)
            question = get_question_text(ex)
            yt = get_answer_label(ex)
        except Exception as e:
            print("[WARN] Skip: bad fields:", e)
            continue

        img_path = os.path.join(args.coco_img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Open image failed: {img_path} ({e})")
            continue

        try:
            lb, lp, ln = get_triplet_logits_instructblip(
                model, processor, pil, question, args.noise_step, device, dtype
            )
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}")
            continue


        def score_for(token_id, logits):
            if token_id is not None:
                return logits[token_id].item()

            cand_txts = [" yes"," Yes","yes","Yes"," no"," No","no","No"]
            best = -1e9
            for s in cand_txts:
                ids = tkr(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
                if len(ids) >= 1:
                    best = max(best, logits[ids[0]].item())
            return best

        yb_yes = score_for(yes_id, lb); yp_yes = score_for(yes_id, lp); yn_yes = score_for(yes_id, ln)
        yb_no  = score_for(no_id,  lb); yp_no  = score_for(no_id,  lp); yn_no  = score_for(no_id,  ln)

        Sp_clean = yb_yes - yb_no
        dpos_yes = yb_yes - yp_yes
        dneg_yes = yb_yes - yn_yes

        gm_yes = abs(dpos_yes + dneg_yes) - abs(dpos_yes - dneg_yes)
        yp_label = "yes" if Sp_clean >= 0.0 else "no"

        y_true.append(yt)
        y_pred.append(yp_label)
        deltas_gm.append(gm_yes)
        kept += 1

        if args.save_records:
            recs.append({
                "image": img_name,
                "question": question,
                "answer": yt,
                "logit_base_yes": float(yb_yes),
                "logit_pos_yes":  float(yp_yes),
                "logit_neg_yes":  float(yn_yes),
                "logit_base_no":  float(yb_no),
                "logit_pos_no":   float(yp_no),
                "logit_neg_no":   float(yn_no),
                "Delta_plus_yes":  float(dpos_yes),
                "Delta_minus_yes": float(dneg_yes),
                "GM_stat_yes":     float(gm_yes),
                "Sp_clean":        float(Sp_clean)
            })

    if kept == 0:
        raise RuntimeError("No valid samples processed")

    deltas_gm = np.array(deltas_gm, dtype=float)
    T = pick_threshold(deltas_gm, q=args.q)


    cnt = Counter()
    for yt, yp in zip(y_true, y_pred):
        key = ("TP" if yt=='yes' and yp=='yes' else
               "TN" if yt=='no'  and yp=='no'  else
               "FP" if yt=='no'  and yp=='yes' else
               "FN")
        cnt[key] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    n = TP + TN + FP + FN
    acc  = (TP + TN) / max(1, n)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    f1   = 2 * prec * rec / max(1e-9, (prec + rec))


    yes_idx = [i for i, yp in enumerate(y_pred) if yp == 'yes']
    vis_yes = float(np.mean((deltas_gm[yes_idx] >= T))) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    # FDP@T
    neg = int(np.sum(deltas_gm <= -T))
    pos = int(np.sum(deltas_gm >=  T))
    fdp_at_T = float(neg) / max(1, pos)


    # 1) Sign test: p_hat = Pr(Δ>0)
    p_hat = float(np.mean(deltas_gm > 0.0))
    power_sign, beta_sign = binom_power_sign_normal(n=len(deltas_gm), p1=p_hat, alpha=args.alpha)
    # 2) Accuracy test: H0: acc=0.5 vs Ha: acc>0.5
    power_acc, beta_acc = binom_power_acc_normal(n=n, acc=acc, alpha=args.alpha)

    print(f"\n#Samples={n}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg}, pos>=T={pos})")

    print("\n==== Power summary ====")
    print(f"Sign test: n={len(deltas_gm)}, p_hat=Pr(Δ>0)={p_hat:.3f}, alpha={args.alpha}, two-sided, power≈{power_sign:.3f}, beta≈{beta_sign:.3f}")
    print(f"Accuracy test (vs 0.5): n={n}, acc={acc:.3f}, alpha={args.alpha}, one-sided, power≈{power_acc:.3f}, beta≈{beta_acc:.3f}")


    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_instructblip_gm.json")
    payload = {
        "T": T,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "vis_supported_yes": vis_yes,
        "hallucinated_yes": halluc_yes,
        "FDP_at_T": fdp_at_T,
        "power": {
            "sign_test": {"p_hat": p_hat, "alpha": args.alpha, "power": power_sign, "beta": beta_sign},
            "acc_test":  {"acc": acc,    "alpha": args.alpha, "power": power_acc,  "beta": beta_acc}
        }
    }
    if args.save_records:
        payload["records"] = recs

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to: {out_path}")


    print("GM-stat (Δ_v) stats:",
          f"mean={deltas_gm.mean():.6f}",
          f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}",
          f"p99={np.percentile(deltas_gm,99):.6f}",
          f"max={deltas_gm.max():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Salesforce/instructblip-vicuna-7b")
    parser.add_argument("--pope_json", type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step", type=int, default=50)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out_dir", type=str, default="./out_instructblip_pope")
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 | cpu | auto")
    parser.add_argument("--force_bf16", action="store_true")
    parser.add_argument("--save_records", action="store_true")
    args = parser.parse_args()
    main(args)
