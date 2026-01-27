
import torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from fx import ModelAdapter, SelectionOut, token_kl
from bernoulli_mask import bernoulli_mask_tokens

@dataclass
class TestOut:
    H_v: torch.Tensor       # (B,T)  KL(p(v,t) || p(v',t))
    H_t_mean: torch.Tensor  # (B,T)  E_f[ KL(p(v,t) || p(v,t_f)) ]
    diff: torch.Tensor      # (B,T)  H_t_mean - H_v
    pval: torch.Tensor      # (B,T)  one-sided p-value for H_t_mean > H_v
    label: torch.Tensor     # (B,T)  0=ROBUST, 1=TEXT_DRIVEN, 2=VISION_DRIVEN

@torch.no_grad()
def run_hypothesis_tests_gX(
    model: ModelAdapter,
    sel: SelectionOut,                # outputs from VCD selection
    image_clean: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    B: int = 30,
    p_text: float = 0.15,
    rho: float = 2.0
) -> TestOut:

    # --- ensure proper dtypes/devices ---
    dev = next(model.parameters()).device
    image_clean = image_clean.to(dev)
    input_ids = input_ids.to(dev, dtype=torch.long)
    attention_mask = attention_mask.to(dev, dtype=torch.long)

    # --- baseline distributions ---
    P  = F.softmax(sel.logits_clean, dim=-1)  # (B,T,V)
    Qv = F.softmax(sel.logits_dist,  dim=-1)  # (B,T,V)
    H_v = token_kl(P, Qv)                     # (B,T)

    # --- bootstrap text masks ---
    Ht_samples = []
    for _ in range(B):
        ft_ids = bernoulli_mask_tokens(input_ids, attention_mask, model.tokenizer, p=float(p_text))
        ft_ids = torch.as_tensor(ft_ids, device=input_ids.device, dtype=torch.long)
        logits_tflip = model(image_clean, ft_ids, attention_mask)
        Qt = F.softmax(logits_tflip, dim=-1)
        Ht = token_kl(P, Qt)
        Ht_samples.append(Ht)

    # shape: (B, batch, T)
    Ht_stack = torch.stack(Ht_samples, dim=0)
    H_t_mean = Ht_stack.mean(dim=0)          # (B,T)
    diff = H_t_mean - H_v                    # (B,T)

    # --- one-sided p-value: Pr[ H_t >= H_v ] with +1 smoothing ---
    comp = Ht_stack >= H_v.unsqueeze(0)      # (B, batch, T)
    pval = (comp.float().sum(dim=0) + 1.0) / (Ht_stack.shape[0] + 1.0)  # (B,T)

    # --- labeling ---
    eps = 1e-8
    ratio_t_over_v = H_t_mean / (H_v + eps)
    ratio_v_over_t = H_v / (H_t_mean + eps)
    label = torch.zeros_like(H_v, dtype=torch.long)
    label[ratio_t_over_v > rho] = 1  # TEXT_DRIVEN
    label[ratio_v_over_t > rho] = 2  # VISION_DRIVEN

    return TestOut(H_v=H_v, H_t_mean=H_t_mean, diff=diff, pval=pval, label=label)


# your BH is good; keeping it as-is
import torch

def benjamini_hochberg_safe(pvals, alpha: float = 0.1):
    """
    Torch-safe BH procedure.
    Accepts list/np/tensor and returns a boolean tensor shaped like input.
    """
    if not isinstance(pvals, torch.Tensor):
        try:
            import numpy as np
            if isinstance(pvals, np.ndarray):
                p = torch.from_numpy(pvals)
            else:
                p = torch.as_tensor(pvals)
        except Exception:
            p = torch.as_tensor(pvals)
    else:
        p = p

    p = p.to(dtype=torch.float64)
    shape = p.shape
    flat = p.reshape(-1)
    n = flat.numel()
    if n == 0:
        return torch.zeros_like(p, dtype=torch.bool)

    sorted_p, idx = torch.sort(flat)  # ascending
    ranks = torch.arange(1, n + 1, dtype=torch.float64, device=sorted_p.device)
    thresh = (ranks / n) * float(alpha)

    reject_sorted = sorted_p <= thresh
    if reject_sorted.any():
        k = torch.nonzero(reject_sorted, as_tuple=False).max()
        reject_sorted[: k.item() + 1] = True

    reject = torch.zeros_like(reject_sorted, dtype=torch.bool)
    reject[idx] = reject_sorted
    return reject.reshape(shape)


