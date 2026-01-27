# fx.py
import torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

class ModelAdapter(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

# ---------- small helpers ----------
def _tt(x, device=None, dtype=None):
    """Coerce lists/ndarrays to torch.Tensor on the right device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

# ---------- numerics ----------
@torch.no_grad()
def distort_image_gaussian_pixels(img: torch.Tensor, sigma: float = 0.06) -> torch.Tensor:
    # ensure tensor & float
    img   = _tt(img, device=img.device, dtype=img.dtype)
    sigma = float(sigma)
    noise = torch.randn_like(img) * sigma
    # keep within original range (CLIP-normalized images are typically ~[-3, 3])
    lo, hi = img.amin().item(), img.amax().item()
    return (img + noise).clamp(lo, hi)

def vcd_logits(logits_clean: torch.Tensor, logits_dist: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    alpha = float(alpha)
    return (1.0 + alpha) * logits_clean - alpha * logits_dist  # (B,T,V)

def token_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: (B,T,V)
    p = (p + eps) / (p.sum(-1, keepdim=True) + eps)
    q = (q + eps) / (q.sum(-1, keepdim=True) + eps)
    return (p * (p.log() - q.log())).sum(-1)  # (B,T)

# ---------- outputs ----------
@dataclass
class SelectionOut:
    logits_clean: torch.Tensor   # (B,T,V)
    logits_dist: torch.Tensor    # (B,T,V)
    logits_vcd: torch.Tensor     # (B,T,V)
    delta_v: torch.Tensor        # (B,T)  ||logit(v)-logit(v')||_2
    image_dist: torch.Tensor     # (B,C,H,W)

# ---------- main selection ----------
@torch.no_grad()
def run_selection_vcd_fission(
    model: ModelAdapter,
    image: torch.Tensor,                 # (B,C,H,W)
    input_ids: torch.Tensor,             # (B,T)
    attention_mask: Optional[torch.Tensor],  # (B,T) or None
    alpha: float = 2.0,
    sigma: float = 0.06
) -> SelectionOut:
    model.eval()
    dev = next(model.parameters()).device

    # Coerce & move to the correct device/dtypes
    image          = _tt(image, device=dev)                   # keep dtype; adapter will cast to model dtype
    input_ids      = _tt(input_ids, device=dev, dtype=torch.long)
    attention_mask = None if attention_mask is None else _tt(attention_mask, device=dev, dtype=torch.long)
    alpha          = float(alpha)
    sigma          = float(sigma)

    # clean forward
    logits_clean = model(image, input_ids, attention_mask)           # (B,T,V)

    # distorted image forward
    image_dist  = distort_image_gaussian_pixels(image, sigma=sigma)  # (B,C,H,W)
    logits_dist = model(image_dist, input_ids, attention_mask)       # (B,T,V)

    # combine
    logits_vcd = vcd_logits(logits_clean, logits_dist, alpha=alpha)  # (B,T,V)
    delta_v    = torch.norm(logits_clean - logits_dist, p=2, dim=-1) # (B,T)

    return SelectionOut(
        logits_clean=logits_clean,
        logits_dist=logits_dist,
        logits_vcd=logits_vcd,
        delta_v=delta_v,
        image_dist=image_dist
    )
