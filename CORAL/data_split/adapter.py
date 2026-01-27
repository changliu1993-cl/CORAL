# adapter.py
import torch, types, inspect
from fx import ModelAdapter
from transformers import AutoTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor

MODEL_ID_OR_PATH = "liuhaotian/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_OR_PATH, trust_remote_code=True)
clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

vlm = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID_OR_PATH,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True,
    low_cpu_mem_usage=True,
)

def _ensure_image_token(tok, model):
    for k in ("image_token", "image_token_str"):
        s = getattr(model.config, k, None)
        if isinstance(s, str) and s:
            tok.add_special_tokens({"additional_special_tokens": [s]})
            try: model.resize_token_embeddings(len(tok))
            except Exception: pass
            return s
    for s in ["<image>", "<image_token>", "<image_placeholder>"]:
        if s in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [s]})
            try: model.resize_token_embeddings(len(tok))
            except Exception: pass
            return s
    s = "<image>"
    tok.add_special_tokens({"additional_special_tokens": [s]})
    try: model.resize_token_embeddings(len(tok))
    except Exception: pass
    return s

IMAGE_TOKEN = _ensure_image_token(tokenizer, vlm)
IMAGE_TOK_ID = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

# Patch metadata so downstream code can discover it
for obj in (vlm, getattr(vlm, "model", None), getattr(vlm, "config", None),
            getattr(getattr(vlm, "model", None), "config", None)):
    if obj is not None:
        setattr(obj, "image_token", IMAGE_TOKEN)
        setattr(obj, "image_token_index", IMAGE_TOK_ID)

# Keep a tolerant mask helper for other calls that might use it
def _compat_get_placeholder_mask(self, *args, **kwargs):
    ids = args[0] if args else kwargs.get("input_ids", None)
    if ids is None:
        raise RuntimeError("get_placeholder_mask: input_ids not provided")
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids)
    tok_id = kwargs.get("image_token_index",
                        getattr(getattr(self, "config", None), "image_token_index", IMAGE_TOK_ID))
    return (ids == tok_id)

try:
    from transformers.models.llava.modeling_llava import LlavaModel as _LM
    _LM.get_placeholder_mask = _compat_get_placeholder_mask
except Exception:
    pass

vlm.get_placeholder_mask = types.MethodType(_compat_get_placeholder_mask, vlm)
if hasattr(vlm, "model"):
    vlm.model.get_placeholder_mask = types.MethodType(_compat_get_placeholder_mask, vlm.model)

print(f"[PATCH] image_token_id = {IMAGE_TOK_ID}")

# ----------------- MANUAL FUSION ADAPTER -----------------
# adapter.py (replace the LlavaAdapter class with this version)
# at top of adapter.py you already have: import torch, types, inspect

class LlavaAdapter(ModelAdapter):
    def __init__(self, tokenizer, vlm):
        super().__init__(tokenizer)
        self.vlm = vlm
        if hasattr(self.vlm.config, "use_cache"):
            self.vlm.config.use_cache = False
        try:
            self.vlm.config._attn_implementation = "eager"
        except Exception:
            pass

    @torch.no_grad()
    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run the LLaVA vision tower in a signature-safe way and project to LM width.
        Returns (B, Npatch, C_text).
        """
        vm = self.vlm.model
        vt = vm.vision_tower
        # Some builds store the actual CLIP encoder in .vision_model, others *are* the encoder.
        vision = getattr(vt, "vision_model", vt)

        # Build kwargs only if supported by this build
        try:
            sig = inspect.signature(vision.forward)
        except (ValueError, TypeError):
            # Fallback: try calling directly with positional tensor
            sig = None

        kwargs = {}
        # pixel_values: some older builds accept positional only; newer accept keyword
        call_with_kw = sig is not None and ("pixel_values" in sig.parameters)
        # return_dict: only set if supported
        if sig is not None and ("return_dict" in sig.parameters):
            kwargs["return_dict"] = False  # get tensors/tuple back; works across versions

        # Forward pass
        if call_with_kw:
            vout = vision(pixel_values=pixel_values, **kwargs)
        else:
            # Positional form (older CLIPVisionTransformer)
            if kwargs.get("return_dict", None) is not None:
                kwargs.pop("return_dict")
            vout = vision(pixel_values)  # positional

        # Get hidden states robustly
        if isinstance(vout, (tuple, list)):
            feats = vout[0]
        else:
            feats = getattr(vout, "last_hidden_state", vout)

        # Drop CLS token when present (common for CLIP ViT-* that return 1+N tokens)
        B, L, Cv = feats.shape
        # expected patches for 336x336 with /14 patch = 24*24 = 576
        H, W = pixel_values.shape[-2], pixel_values.shape[-1]
        expected_tokens = (H // 14) * (W // 14)
        if L == expected_tokens + 1:
            feats = feats[:, 1:, :]  # remove CLS
        elif L != expected_tokens:
            # Some builds already return N=576; others may return pooled features.
            # If pooled (L == 1), just repeat to K when masking; we leave as-is here.
            pass

        # Project vision width -> language width
        feats = vm.multi_modal_projector(feats)  # (B, Npatch, C_text)
        return feats

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        dev = next(self.vlm.parameters()).device
        image = image.to(dev, dtype=self.vlm.dtype)
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)

        vm = self.vlm.model
        lm = vm.language_model
        tok_embed = lm.embed_tokens

        # 1) text embeds
        inputs_embeds = tok_embed(input_ids)  # (B,T,C)

        # 2) vision → projector
        img_feats = self._encode_images(image)  # (B,N,C)
        B, N, C = img_feats.shape

        # 3) fuse at <image> positions
        img_id = getattr(self.vlm.config, "image_token_index", None)
        if img_id is None:
            img_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        for b in range(B):
            pos = (input_ids[b] == img_id).nonzero(as_tuple=False).squeeze(-1)
            K = int(pos.numel())
            if K == 0:
                continue
            if K == N:
                fused = img_feats[b]
            elif K > N:
                pad = torch.zeros((K - N, C), dtype=img_feats.dtype, device=img_feats.device)
                fused = torch.cat([img_feats[b], pad], dim=0)
            else:
                fused = img_feats[b, :K, :]
            inputs_embeds[b, pos, :] = fused.to(inputs_embeds.dtype)

        # 4) LM forward (hidden states)
        lm_out = lm(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True)
        hidden = getattr(lm_out, "last_hidden_state", lm_out[0])

        # 5) head → logits
        logits = self.vlm.lm_head(hidden)

        aux = {"image_token_mask": (input_ids == img_id), "num_image_patches": N}
        return logits, aux

                                    # (B,T,V)

adapter = LlavaAdapter(tokenizer, vlm).eval()

__all__ = ["adapter", "tokenizer", "clip_proc", "vlm", "IMAGE_TOKEN"]
