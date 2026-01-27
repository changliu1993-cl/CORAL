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

# ===== Qwen-VL =====
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
# Gaussian-Mirror mirror views on pixel_values (same device/dtype)
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
# Single-token id helper (Yes/No)
# ------------------------------
def get_single_token_id(tokenizer, text_list):
    for t in text_list:
        ids = tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if len(ids) == 1:
            return ids[0]
    return None

# yes_id = get_single_token_id(processor.tokenizer, ["Yes", "yes"])
# no_id  = get_single_token_id(processor.tokenizer, ["No", "no"])


# ------------------------------
# Qwen preprocess: build inputs via chat messages
# ------------------------------
def build_inputs(processor, image_pil: Image.Image, question: str, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }]
    # 先把对话转成文本模板（会包含 <image> 占位）
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,   # 末尾加 <assistant> 提示
        tokenize=False
    )
    # 再把 text + images 编码成张量
    inputs = processor(
        text=[text],                  # 用列表以便 batch 维度稳定
        images=[image_pil],
        return_tensors="pt",
        padding=True
    )
    # 移到 device；像素先保持原状，后面三视图再改
    for k in inputs:
        if torch.is_tensor(inputs[k]) and k != "pixel_values":
            inputs[k] = inputs[k].to(device)
    return inputs


# ------------------------------
# Forward once per view to get last-step logits
# ------------------------------
@torch.no_grad()
def get_triplet_logits_qwen(model, processor, image_pil, question,
                            noise_step, device, dtype):
    inputs = build_inputs(processor, image_pil, question, device)

    # processor 产生的像素（float32），我们做 GM 三视图替换
    pixel = inputs["pixel_values"].to(device=device, dtype=dtype)
    v_base, v_pos, v_neg = add_mirror_views(pixel, noise_step, device, dtype)

    # base
    inputs_base = {k: v for k, v in inputs.items()}
    inputs_base["pixel_values"] = v_base
    out_base = model(**inputs_base)
    logits_base = out_base.logits[:, -1, :]  # [1, V]

    # pos
    inputs_pos = {k: v for k, v in inputs.items()}
    inputs_pos["pixel_values"] = v_pos
    out_pos = model(**inputs_pos)
    logits_pos = out_pos.logits[:, -1, :]

    # neg
    inputs_neg = {k: v for k, v in inputs.items()}
    inputs_neg["pixel_values"] = v_neg
    out_neg = model(**inputs_neg)
    logits_neg = out_neg.logits[:, -1, :]

    return logits_base[0], logits_pos[0], logits_neg[0]


# ------------------------------
# FDR via GM-FDP
# ------------------------------
def fdp_hat(deltas: np.ndarray, t: float) -> float:
    neg = np.sum(deltas <= -t)
    pos = np.sum(deltas >=  t)
    return float(neg + 1) / max(1, int(pos))  # +1 for a slightly conservative estimate

def pick_threshold(deltas: np.ndarray, q: float = 0.10) -> float:
    abs_max = np.quantile(np.abs(deltas), 0.999)
    grid = np.linspace(0.0, float(abs_max), 400)
    for t in grid:
        if fdp_hat(deltas, t) <= q:
            return float(t)
    return float(np.quantile(np.abs(deltas), 0.995))

# ------------------------------
# Fallback: cumulative log-prob for "Yes." / "No."
# ------------------------------
@torch.no_grad()
def prefix_total_logprob_qwen(model, processor, pixel_values, question, prefix_text, device):
    # user + assistant 前缀
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefix_text}]}
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,   # 已经有人设了 assistant 文本
        tokenize=False
    )
    # 用任意一张“占位图”让 processor 正常工作（随便 1x1 的黑图即可）
    dummy = Image.new("RGB", (1, 1), (0, 0, 0))
    encoded = processor(text=[text], images=[dummy], return_tensors="pt", padding=True)

    # 把张量搬上 device，并覆盖 pixel_values 为我们传入的三视图之一
    for k in encoded:
        if torch.is_tensor(encoded[k]) and k != "pixel_values":
            encoded[k] = encoded[k].to(device)
    encoded["pixel_values"] = pixel_values

    out = model(**encoded)
    logits = out.logits[:, :-1, :]
    labels = encoded["input_ids"][:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # 仅累计 assistant 段的 token。简化：按 prefix_text 的 token 数回溯
    # （更严谨可解析 role spans，这里用保守窗口）
    k = min(gathered.shape[1], 16)
    total = gathered[0, -k:].sum().item()
    return total


# ------------------------------
# Simple power estimates (binomial tests)
# ------------------------------
import math

# ---- Stable log PMF for Binomial ----
def _log_binom_pmf(i: int, n: int, p: float) -> float:
    # log( C(n,i) p^i (1-p)^(n-i) )
    if p <= 0.0:
        return 0.0 if i == 0 else -float("inf")
    if p >= 1.0:
        return 0.0 if i == n else -float("inf")
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))

# ---- Stable CDF via log-sum-exp, works for large n ----
def binom_cdf(k: int, n: int, p: float) -> float:
    k = max(-1, min(k, n))
    if k < 0:
        return 0.0
    # accumulate logs for i = 0..k, then log-sum-exp
    logs = [_log_binom_pmf(i, n, p) for i in range(k + 1)]
    m = max(logs)
    # sum exp(log_i - m) safely
    s = sum(math.exp(li - m) for li in logs)
    return math.exp(m) * s  # equals exp( m + log(s) ), always in (0,1]

def binom_sf(k: int, n: int, p: float) -> float:
    # survival function P[X > k] = 1 - CDF(k)
    c = binom_cdf(k, n, p)
    return max(0.0, 1.0 - c)

# ---- Find two-sided critical region for sign test under p0 = 0.5 ----
def _binom_critical_region_two_sided(n: int, alpha: float, p0: float = 0.5):
    # smallest k_lo with CDF(k_lo; n, 0.5) >= alpha/2  => accept region starts above it
    # largest k_hi with SF(k_hi; n, 0.5) >= alpha/2     => accept region ends below it
    # We return acceptance interval [k_lo+1, k_hi-1]; rejection is <=k_lo or >=k_hi
    target = alpha / 2.0

    # search k_lo
    loL, loR = -1, n
    while loL + 1 < loR:
        mid = (loL + loR) // 2
        if binom_cdf(mid, n, p0) < target:
            loL = mid
        else:
            loR = mid
    k_lo = loL  # largest with CDF < alpha/2

    # search k_hi
    hiL, hiR = -1, n
    while hiL + 1 < hiR:
        mid = (hiL + hiR) // 2
        if binom_sf(mid, n, p0) < target:
            hiR = mid
        else:
            hiL = mid
    k_hi = hiR  # smallest with SF < alpha/2  => rejection when k >= k_hi

    return k_lo, k_hi

# ---- Power for sign test: H0 p=0.5 vs H1 p=p1 (two-sided) ----
def binom_power_sign_test(n: int, p1: float, alpha: float = 0.05):
    # critical region under H0
    k_lo, k_hi = _binom_critical_region_two_sided(n, alpha, p0=0.5)
    # Type II error beta under true p1: probability to fall in acceptance region
    # acceptance region = {k in [k_lo+1, k_hi-1]}
    # beta = CDF(k_hi-1; n, p1) - CDF(k_lo; n, p1)
    cdf_hi_minus_1 = binom_cdf(k_hi - 1, n, p1)
    cdf_lo = binom_cdf(k_lo, n, p1)
    beta = max(0.0, min(1.0, cdf_hi_minus_1 - cdf_lo))
    power = 1.0 - beta
    return power, beta, (k_lo + 1, k_hi - 1)

# ---- Power for "accuracy > 0.5" one-sided test ----
def binom_power_accuracy_test(n: int, acc: float, alpha: float = 0.05):
    # one-sided test: H0 p=0.5 vs H1 p>0.5
    # critical k_hi: smallest k with P_{0.5}[X >= k] <= alpha
    # i.e., find k s.t. SF(k-1; n, 0.5) <= alpha
    lo, hi = 0, n + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if binom_sf(mid - 1, n, 0.5) <= alpha:
            hi = mid
        else:
            lo = mid
    kcrit = hi  # reject H0 when X >= kcrit
    # Power at true p = acc: P[X >= kcrit] = SF(kcrit-1; n, acc)
    power = binom_sf(kcrit - 1, n, acc)
    beta = 1.0 - power
    return power, beta, kcrit

# ------------------------------
# Main
# ------------------------------
def main(args):
    # device & dtype
    if args.device == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if args.force_bf16 else torch.float16
    else:
        dtype = torch.float32

    # load Qwen2-VL
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,                # new HF uses 'dtype' (not torch_dtype)
        device_map=None,
        trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Yes/No single-token ids
    # yes_id = get_single_token_id(processor.tokenizer, "Yes")
    # no_id  = get_single_token_id(processor.tokenizer, "No")
    yes_id = get_single_token_id(processor.tokenizer, ["Yes", "yes"])
    no_id  = get_single_token_id(processor.tokenizer, ["No", "no"])

    # load POPE
    data = load_json_any(args.pope_json)
    if not data:
        raise RuntimeError(f"No samples loaded from {args.pope_json}")

    deltas_gm, y_true, y_pred, recs = [], [], [], []

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
            logits_base, logits_pos, logits_neg = get_triplet_logits_qwen(
                model, processor, pil, question, args.noise_step, device, dtype
            )
        except Exception as e:
            print(f"[WARN] Forward failed for {img_name}: {e}")
            continue

        # scores for Yes/No
        if yes_id is not None and no_id is not None:
            yb_yes = logits_base[yes_id].item(); yp_yes = logits_pos[yes_id].item(); yn_yes = logits_neg[yes_id].item()
            yb_no  = logits_base[no_id].item();  yp_no  = logits_pos[no_id].item();  yn_no  = logits_neg[no_id].item()
        else:
            # fallback to prefix logprob ("Yes." / "No.")
            inputs = build_inputs(processor, pil, question, device)
            pixel = inputs["pixel_values"].to(device=device, dtype=dtype)
            v_base, v_pos, v_neg = add_mirror_views(pixel, args.noise_step, device, dtype)
            # yb_yes = prefix_total_logprob_qwen(model, processor, v_base, question, "Yes.", device)
            # yp_yes = prefix_total_logprob_qwen(model, processor, v_pos,  question, "Yes.", device)
            # yn_yes = prefix_total_logprob_qwen(model, processor, v_neg,  question, "Yes.", device)
            # yb_no  = prefix_total_logprob_qwen(model, processor, v_base, question, "No.",  device)
            # yp_no  = prefix_total_logprob_qwen(model, processor, v_pos,  question, "No.",  device)
            # yn_no  = prefix_total_logprob_qwen(model, processor, v_neg,  question, "No.",  device)
            yb_yes = prefix_total_logprob_qwen(model, processor, v_base, question, "Yes.", device)
            yp_yes = prefix_total_logprob_qwen(model, processor, v_pos,  question, "Yes.", device)
            yn_yes = prefix_total_logprob_qwen(model, processor, v_neg,  question, "Yes.", device)

            yb_no  = prefix_total_logprob_qwen(model, processor, v_base, question, "No.",  device)
            yp_no  = prefix_total_logprob_qwen(model, processor, v_pos,  question, "No.",  device)
            yn_no  = prefix_total_logprob_qwen(model, processor, v_neg,  question, "No.",  device)


        # decision on clean view
        Sp_clean = yb_yes - yb_no
        yp = "yes" if Sp_clean >= 0.0 else "no"

        # GM statistic (Ke et al.): |Δ+ + Δ-| - |Δ+ - Δ-|
        dpos_yes = yb_yes - yp_yes
        dneg_yes = yb_yes - yn_yes
        gm_yes = abs(dpos_yes + dneg_yes) - abs(dpos_yes - dneg_yes)

        y_true.append(yt)
        y_pred.append(yp)
        deltas_gm.append(gm_yes)
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

    if len(deltas_gm) == 0:
        raise RuntimeError("No valid samples processed")

    deltas_gm = np.array(deltas_gm, dtype=float)
    T = pick_threshold(deltas_gm, q=args.q)

    # POPE metrics
    cnt = Counter()
    for yt, yhat in zip(y_true, y_pred):
        key = ("TP" if yt=='yes' and yhat=='yes' else
               "TN" if yt=='no'  and yhat=='no'  else
               "FP" if yt=='no'  and yhat=='yes' else
               "FN")
        cnt[key] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    acc  = (TP + TN) / max(1, TP + TN + FP + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    f1   = 2 * prec * rec / max(1e-9, (prec + rec))

    yes_idx = [i for i, yhat in enumerate(y_pred) if yhat == 'yes']
    vis_yes = float(np.mean((deltas_gm[yes_idx] >= T))) if yes_idx and np.isfinite(T) else 0.0
    halluc_yes = 1.0 - vis_yes if yes_idx else 0.0

    # FDP@T（报告用）
    neg = int(np.sum(deltas_gm <= -T))
    pos = int(np.sum(deltas_gm >=  T))
    fdp_at_T = float(neg) / max(1, pos)

    print(f"\n#Samples={len(deltas_gm)}  FDR threshold T={T:.6f}")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Visually supported Yes={vis_yes:.4f}  Hallucinated Yes={halluc_yes:.4f}")
    print(f"FDP@T: {fdp_at_T:.4f}  (neg<=-T={neg}, pos>=T={pos})")

    # ===== (Optional) Power summary =====
    if args.alpha is not None:
        alpha = args.alpha if hasattr(args, "alpha") else 0.05
        # sign test for Δ>0 proportion
        p_hat = float((deltas_gm > 0).mean())
        power_sign, beta_sign, accept_interval = binom_power_sign_test(n=len(deltas_gm), p1=p_hat, alpha=alpha)
        power_acc,  beta_acc,  kcrit_hi       = binom_power_accuracy_test(n=len(deltas_gm), acc=acc, alpha=alpha)

        # accuracy test vs 0.5
        # power_acc, kcrit_hi = binom_power_accuracy(n=len(deltas_gm), acc=acc, alpha=alpha)
        print("==== Power summary ====")
        print(f"Sign test: n={len(deltas_gm)}, p_hat=Pr(Δ>0)={p_hat:.3f}, alpha={alpha:.2f}, "
            f"power={power_sign:.6f}, beta={beta_sign:.6f}, accept=[{accept_interval[0]},{accept_interval[1]}]")
        print(f"Accuracy test (vs 0.5): n={len(deltas_gm)}, acc={acc:.3f}, alpha={alpha:.2f}, "
            f"power={power_acc:.6f}, kcrit_hi={kcrit_hi}")

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pope_qwen_gm.json")
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
                "FDP_at_T": fdp_at_T,
                "records": recs,
            },
            f, indent=2
        )
    print(f"Saved results to: {out_path}")

    # brief stats
    print("GM-stat (Δ_v) stats:",
          f"mean={deltas_gm.mean():.6f}",
          f"std={deltas_gm.std():.6f}",
          f"p95={np.percentile(deltas_gm,95):.6f}",
          f"p99={np.percentile(deltas_gm,99):.6f}",
          f"max={deltas_gm.max():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--pope_json", type=str, required=True)
    parser.add_argument("--coco_img_dir", type=str, required=True)
    parser.add_argument("--noise_step", type=int, default=50)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--out_dir", type=str, default="./out_qwen_pope")
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 | cpu | auto")
    parser.add_argument("--force_bf16", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05, help="optional power reporting (e.g., 0.05)")
    args = parser.parse_args()
    main(args)
