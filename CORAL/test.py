import os, json, csv, time
from typing import List, Dict
import torch
from PIL import Image
from transformers.generation.utils import GenerationMixin

# LLaVA
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token

# VCD sampling + HTest (your patch)
from vcd_htest_patch import patch_vcd_htest

# Official VCD noise util
from vcd_utils.vcd_add_noise_df import add_diffusion_noise

# ------------------- CONFIG -------------------
POPE_JSON = os.path.expanduser(
    "~/VLM/VCD-master/experiments/data/POPE/coco/coco_pope_adversarial.json"
)
COCO_VAL2014_DIR = os.path.expanduser(
    "~/VLM/VCD-master/data/COCO/val2014"
)
OUT_CSV = os.path.expanduser(
    "~/VLM/VCD-master/outputs/pope_coco_vcd_htest_adversarial_v2.csv"
)
MODEL_ID = "liuhaotian/llava-v1.5-7b"

# VCD + HTest hyperparams - IMPROVED
CD_ALPHA = 2.0
CD_BETA  = 0.10


FDR_Q    = 0.05  


DIFFUSION_STEP_1 = 10  
DIFFUSION_STEP_2 = 30  

SEED_VIEW1 = 1234
SEED_VIEW2 = 5678

MAX_SAMPLES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------


def load_jsonl(path: str):
    """Load JSONL file line by line."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def full_img_path(filename: str) -> str:
    """Get full path to COCO image."""
    return os.path.join(COCO_VAL2014_DIR, filename)


def build_prompt(question: str) -> str:
    """Build LLaVA conversation prompt."""
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def patch_llava_dtype():
    """Patch LLaVA's encode_images to ensure dtype consistency."""
    from llava.model.llava_arch import LlavaMetaForCausalLM
    
    original_encode_images = LlavaMetaForCausalLM.encode_images
    
    def encode_images_fixed(self, images):
        """Encode images with dtype consistency."""
        target_dtype = next(self.get_model().mm_projector.parameters()).dtype
        
        if images.dtype != target_dtype:
            images = images.to(target_dtype)
        
        if hasattr(self.get_vision_tower(), 'to'):
            vision_tower = self.get_vision_tower()
            if next(vision_tower.parameters()).dtype != target_dtype:
                vision_tower = vision_tower.to(target_dtype)
        
        image_features = original_encode_images(self, images)
        
        if image_features.dtype != target_dtype:
            image_features = image_features.to(target_dtype)
        
        return image_features
    
    LlavaMetaForCausalLM.encode_images = encode_images_fixed
    print("[Patch] Applied dtype consistency fix to encode_images")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Improved Configuration:")
    print(f"  - FDR_Q: {FDR_Q} (increased from 0.10)")
    print(f"  - Diffusion steps: {DIFFUSION_STEP_1} and {DIFFUSION_STEP_2} (different strengths)")
    
    # Apply patches
    patch_vcd_htest()
    patch_llava_dtype()

    # Load model
    print("\nLoading LLaVA...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_name=MODEL_ID,
        model_path=MODEL_ID,
        model_base=None,
        device=DEVICE
    )
    
    print("Converting model to float16...")
    model = model.to(torch.float16)
    
    if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        if model.model.vision_tower is not None:
            model.model.vision_tower = model.model.vision_tower.to(torch.float16)
    
    print(f"Model loaded: {type(model)}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Load dataset
    print(f"\nReading POPE dataset: {POPE_JSON}")
    items = load_jsonl(POPE_JSON)
    if MAX_SAMPLES is not None:
        items = items[:MAX_SAMPLES]
        print(f"Processing {MAX_SAMPLES} items...")

    rows = []

    # Process each sample
    for idx, it in enumerate(items, 1):
        qid = it.get("question_id", -1)
        img_file = it.get("image")
        qtype = it.get("type", "")
        question = it.get("question") or it.get("text") or "Describe this image briefly."
        label = it.get("label", "")

        img_path = full_img_path(img_file)
        
        if not os.path.isfile(img_path):
            print(f"[{idx}/{len(items)}] MISSING: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}/{len(items)}] Failed to open: {e}")
            continue

        # Preprocessing
        image_inputs = image_processor.preprocess(image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(
            device=model.device, 
            dtype=torch.float16
        )
        
        image_sizes = image_inputs.get("image_sizes", None)
        if image_sizes is not None:
            image_sizes = image_sizes.to(model.device)

        prompt = build_prompt(question)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

        # Create contrastive views with DIFFERENT noise strengths
        torch.manual_seed(SEED_VIEW1)
        images_cd = add_diffusion_noise(pixel_values, noise_step=DIFFUSION_STEP_1)
        images_cd = images_cd.to(device=model.device, dtype=torch.float16)

        torch.manual_seed(SEED_VIEW2)
        images_cd2 = add_diffusion_noise(pixel_values, noise_step=DIFFUSION_STEP_2)
        images_cd2 = images_cd2.to(device=model.device, dtype=torch.float16)

        # Sanity checks
        assert pixel_values.dtype == torch.float16
        assert images_cd.dtype == torch.float16
        assert images_cd2.dtype == torch.float16

        # Generate
        try:
            t0 = time.time()
            out = model.generate(
                input_ids,
                images=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                images_cd=images_cd,
                images_cd2=images_cd2,
                cd_alpha=CD_ALPHA,
                cd_beta=CD_BETA,
                htest=True,
                htest_q=FDR_Q,
                max_new_tokens=64,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            dt = time.time() - t0
        except Exception as e:
            print(f"[{idx}/{len(items)}] Generation failed: {e}")
            continue

        # Decode
        answer = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

        # Extract HTest statistics
        T = FDRq = None
        n_sig = n_tot = 0
        
        if hasattr(out, "htest"):
            W = out.htest.get("W_vis", None)
            T = out.htest.get("T", None)
            FDRq = out.htest.get("FDR_q", None)
            
            if W is not None:
                n_tot = int(W.numel())
            
            sig = out.htest.get("sig_mask", None)
            if sig is not None:
                n_sig = int(sig.sum().item())

        rows.append({
            "question_id": qid,
            "image": img_file,
            "type": qtype,
            "question": question,
            "label": label,
            "answer_text": answer,
            "T": T if T is not None else "",
            "FDR_q": FDRq if FDRq is not None else "",
            "num_sig_tokens": n_sig,
            "num_tokens": n_tot,
            "time_sec": round(dt, 3),
        })

        sig_pct = (n_sig / n_tot * 100) if n_tot > 0 else 0
        print(f"[{idx}/{len(items)}] {img_file[:30]:30} | {qtype:>10} | "
              f"t={dt:.2f}s | sig={n_sig}/{n_tot} ({sig_pct:.1f}%) | "
              f"ans={answer[:30]}...")

    # Save results
    print(f"\nSaving results to: {OUT_CSV}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "image", "type", "question", "label", "answer_text",
            "T", "FDR_q", "num_sig_tokens", "num_tokens", "time_sec"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Saved {len(rows)} results")
    
    # Summary statistics
    if rows:
        avg_time = sum(r["time_sec"] for r in rows) / len(rows)
        total_sig = sum(r["num_sig_tokens"] for r in rows)
        total_tokens = sum(r["num_tokens"] for r in rows)
        sig_ratio = total_sig / total_tokens if total_tokens > 0 else 0
        samples_with_sig = sum(1 for r in rows if r["num_sig_tokens"] > 0)
        
        print(f"\nSummary:")
        print(f"  - Total samples: {len(rows)}")
        print(f"  - Avg time per sample: {avg_time:.2f}s")
        print(f"  - Total significant tokens: {total_sig}/{total_tokens} ({sig_ratio:.2%})")
        print(f"  - Samples with sig>0: {samples_with_sig}/{len(rows)} ({samples_with_sig/len(rows):.1%})")


if __name__ == "__main__":
    main()