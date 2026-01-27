from transformers import (LogitsProcessor)
import torch.nn.functional as F

class GuidanceLogits(LogitsProcessor):

    def __init__(self, guidance_strength, guidance, images, attention_mask, model, tokenizer=None):
        """
        Args:
            guidance_strength (float): The guidance strength for the logits.
            guidance (torch.Tensor): The guidance input tensor.
            images (torch.Tensor): The images tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            model (torch.nn.Module): The model to use.
        """
        self.guidance_strength = guidance_strength
        self.guidance = guidance.cuda()
        self.images = images
        self.attention_mask = attention_mask
        self.model = model
        self.out = None
        self.tokenizer = tokenizer

    def __call__(self, input_ids, logits):
        logits = F.log_softmax(logits, dim=-1)
        if self.out is None:
            self.out = self.model(input_ids=self.guidance, 
                                  pixel_values=self.images, 
                                  attention_mask=self.attention_mask,
                                  use_cache=True)
        else:
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  past_key_values=self.out.past_key_values)

        if len(self.out.logits) == 1:
            guidance_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        else:
            guidance_logits = F.log_softmax(self.out.logits[:,-1:], dim=-1).to(logits.device)
            guidance_logits = guidance_logits.squeeze(1)

        out = self.guidance_strength * (guidance_logits - logits) + logits
        out = F.log_softmax(out, dim=-1)

        return out