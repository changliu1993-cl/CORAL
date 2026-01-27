# vcd_htest_patch.py
import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import (
    GenerationMixin,
    SampleOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
)

# optional: auto-build cd views when user passes cd_sigma
try:
    from data_fission.core import fission_gaussian  # expects images in [0,1]
    _HAS_FISSION = True
except Exception:
    _HAS_FISSION = False

# vcd_htest_patch.py
# ... all other imports ...

def _htest_threshold(W: torch.Tensor, q: float) -> Tuple[float, torch.Tensor]:
    """FDR-controlled threshold based on VCD's simplified method."""
    if W.numel() == 0:
        return float("inf"), torch.zeros(0, dtype=torch.bool, device=W.device)

    # Sort the absolute values of W
    W_abs = torch.sort(W.abs())[0]
    
    # Calculate the FDR threshold
    T = float("inf")
    p_values = W_abs.flip(dims=(0,)) # p-values are based on the rank
    
    # A simplified, more robust FDR check
    for i in range(p_values.numel()):
        p = p_values[i]
        rank = p_values.numel() - i
        fdr_hat = (rank) / (i + 1)
        if fdr_hat <= q:
            T = p
            break
            
    # Create the significance mask
    sig_mask = W.abs() >= T
    
    return T, sig_mask

def _knockoff_threshold(W: torch.Tensor, q: float) -> Tuple[float, torch.Tensor]:
    """Knockoff-style threshold:
    T = min{ t>0 : (1 + #{W<=-t}) / max(1, #{W>=t}) <= q }.
    Returns (T, sig_mask).
    """
    if W.numel() == 0:
        return float("inf"), torch.zeros(0, dtype=torch.bool, device=W.device)
    abs_vals = torch.sort(torch.unique(torch.abs(W)))[0]
    T = float("inf")
    for t in abs_vals:
        num_pos = (W >= t).sum()
        if num_pos.item() == 0:
            continue
        num_neg = (W <= -t).sum()
        fdr_hat = (1 + num_neg.item()) / max(1, num_pos.item())
        if fdr_hat <= q:
            T = float(t.item())
            break
    sig_mask = W >= T if math.isfinite(T) else torch.zeros_like(W, dtype=torch.bool)
    return T, sig_mask


def _prepare_inputs_for_generation_cd(self: GenerationMixin, input_ids, **model_kwargs):
    """Swap 'images' with 'images_cd' for the VCD pass."""
    mk = copy.deepcopy(model_kwargs)
    if "images_cd" in mk:
        mk["images"] = mk.pop("images_cd")
    # Remove other CD-related params to avoid confusion
    mk.pop("images_cd2", None)
    return self.prepare_inputs_for_generation(input_ids, **mk)


def _prepare_inputs_for_generation_cd2(self: GenerationMixin, input_ids, **model_kwargs):
    """Swap 'images' with 'images_cd2' for the second fission view (hypothesis test)."""
    mk = copy.deepcopy(model_kwargs)
    if "images_cd2" in mk:
        mk["images"] = mk.pop("images_cd2")
    # Remove other CD-related params
    mk.pop("images_cd", None)
    return self.prepare_inputs_for_generation(input_ids, **mk)


def _sample_with_vcd_htest(
    self: GenerationMixin,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    # --- init ---
    logits_processor = logits_processor or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated; use stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(...)).",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper or LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False

    # --- VCD flags & auto fission (optional) ---
    use_cd = model_kwargs.get("images_cd") is not None
    cd_sigma = model_kwargs.get("cd_sigma", None)
    if (not use_cd) and (cd_sigma is not None) and _HAS_FISSION:
        x = model_kwargs["images"].clamp(0, 1)
        v1_np, v2_np = fission_gaussian(x.detach().cpu().numpy(), blur_std=float(cd_sigma))
        model_kwargs["images_cd"] = torch.from_numpy(v1_np).to(x.device, x.dtype).clamp(0, 1)
        model_kwargs["images_cd2"] = torch.from_numpy(v2_np).to(x.device, x.dtype).clamp(0, 1)
        use_cd = True

    # Keep a separate copy for CD operations
    model_kwargs_cd = copy.deepcopy(model_kwargs)

    # --- hypothesis testing setup ---
    htest = bool(model_kwargs.get("htest", False))
    htest_q = float(model_kwargs.get("htest_q", 0.10))
    if htest and (model_kwargs.get("images_cd2") is None):
        warnings.warn("htest=True but images_cd2 is missing; hypothesis testing disabled.")
        htest = False
    W_buffer: List[torch.Tensor] = []

    # --- decoding loop ---
    while True:
        if synced_gpus:
            flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=input_ids.device)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            if flag.item() == 0.0:
                break

        # Standard forward pass
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_cd:
            # Contrastive forward (view 1)
            model_inputs_cd = _prepare_inputs_for_generation_cd(self, input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]

            cd_alpha = model_kwargs.get("cd_alpha", 0.5)
            cd_beta = model_kwargs.get("cd_beta", 0.1)

            cutoff = torch.log(
                torch.tensor(cd_beta, device=next_token_logits.device, dtype=next_token_logits.dtype)
            ) + next_token_logits.max(dim=-1, keepdim=True).values

            diffs = (1.0 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)
            next_token_scores = cd_logits
            probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Hypothesis testing: second exchangeable view
            # if htest:
            #     model_inputs_cd2 = _prepare_inputs_for_generation_cd2(self, input_ids, **model_kwargs_cd)
            #     outputs_cd2 = self(
            #         **model_inputs_cd2,
            #         return_dict=True,
            #         output_attentions=output_attentions,
            #         output_hidden_states=output_hidden_states,
            #     )
            #     next_token_logits_cd2 = outputs_cd2.logits[:, -1, :]

            #     bidx = torch.arange(next_token_logits.size(0), device=next_token_logits.device)
            #     s1 = next_token_logits[bidx, next_tokens] - next_token_logits_cd[bidx, next_tokens]
            #     s2 = next_token_logits[bidx, next_tokens] - next_token_logits_cd2[bidx, next_tokens]
            #     W_step = s1 - s2  # (B,)
            #     W_buffer.append(W_step.detach())
            if htest:
                # ------------------- CORRECTED LOGIC START -------------------
                # Get the logits for the second contrastive view
                model_inputs_cd2 = _prepare_inputs_for_generation_cd2(self, input_ids, **model_kwargs_cd)
                outputs_cd2 = self(
                    **model_inputs_cd2,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_cd2 = outputs_cd2.logits[:, -1, :]

                # Compute the second set of VCD logits
                diffs2 = (1.0 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd2
                # Apply cutoff
                cutoff2 = torch.log(
                    torch.tensor(cd_beta, device=next_token_logits.device, dtype=next_token_logits.dtype)
                ) + next_token_logits.max(dim=-1, keepdim=True).values
                cd_logits2 = diffs2.masked_fill(next_token_logits < cutoff2, -float("inf"))

                # Now, compute the W-statistic for the *entire vocabulary*
                # The W-statistic is a vector of scores for all possible tokens.
                W_scores = cd_logits - cd_logits2
                
                # Now, sample the next token using the first set of VCD logits (original logic)
                cd_logits = logits_processor(input_ids, cd_logits)
                cd_logits = logits_warper(input_ids, cd_logits)
                next_token_scores = cd_logits
                probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Extract the W-statistic value for the chosen token
                bidx = torch.arange(next_token_logits.size(0), device=next_token_logits.device)
                W_step = W_scores[bidx, next_tokens]

                W_buffer.append(W_step.detach())

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # book-keeping
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                if self.config.is_encoder_decoder:
                    decoder_attentions += (outputs.decoder_attentions,)
                    cross_attentions += (outputs.cross_attentions,)
                else:
                    decoder_attentions += (outputs.attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, `pad_token_id` must also be defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    # outputs
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            out = SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            out = SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )

        if htest and len(W_buffer) > 0:
            W_all = torch.stack(W_buffer, dim=0).reshape(-1)
            T, sig_mask = _knockoff_threshold(W_all, htest_q)
            out.htest = {
                "W_vis": W_all,
                "T": T,
                "sig_mask": sig_mask,
                "FDR_q": htest_q,
                "desc": "Visual dependence via Data Fission (v1,v2) using knockoff-style FDR.",
            }
        return out
    else:
        return input_ids


def patch_vcd_htest():
    """Monkey-patch HF GenerationMixin.sample/_sample to our VCD+HT version."""
    
    # First, save original methods
    _original_prepare_inputs = GenerationMixin.prepare_inputs_for_generation
    _original_validate = GenerationMixin._validate_model_kwargs
    
    # Patch prepare_inputs_for_generation to preserve VCD parameters
    def _prepare_inputs_for_generation_vcd(self, input_ids, **kwargs):
        """Preserve VCD/HTest custom parameters through generation pipeline."""
        # Call original method first
        model_inputs = _original_prepare_inputs(self, input_ids, **kwargs)
        
        # Preserve our custom parameters
        custom_keys = [
            'images_cd', 'images_cd2', 'cd_alpha', 'cd_beta', 
            'cd_sigma', 'htest', 'htest_q'
        ]
        for key in custom_keys:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        
        return model_inputs
    
    # Patch _validate_model_kwargs to allow our custom parameters
    def _validate_model_kwargs_vcd(self, model_kwargs):
        """Allow VCD/HTest parameters to pass validation."""
        # Remove our custom parameters before validation
        vcd_keys = ['images_cd', 'images_cd2', 'cd_alpha', 'cd_beta', 
                    'cd_sigma', 'htest', 'htest_q']
        model_kwargs_filtered = {k: v for k, v in model_kwargs.items() 
                                 if k not in vcd_keys}
        
        # Call original validation with filtered kwargs
        return _original_validate(self, model_kwargs_filtered)
    
    # Apply patches
    GenerationMixin.prepare_inputs_for_generation = _prepare_inputs_for_generation_vcd
    GenerationMixin._validate_model_kwargs = _validate_model_kwargs_vcd
    
    # Patch the sampling method
    GenerationMixin.sample = _sample_with_vcd_htest
    GenerationMixin._sample = _sample_with_vcd_htest
    
    # Attach helper methods
    GenerationMixin.prepare_inputs_for_generation_cd = _prepare_inputs_for_generation_cd
    GenerationMixin.prepare_inputs_for_generation_cd2 = _prepare_inputs_for_generation_cd2
    
    print("[VCD+HTest] Patched GenerationMixin: sample, prepare_inputs_for_generation, _validate_model_kwargs")