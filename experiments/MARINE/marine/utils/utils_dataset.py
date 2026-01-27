import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


class COCOEvalDataset(Dataset):
    """
    Dataset class for COCO-style evaluation of LVLMs.
    Each item includes the image, prompt, negative prompt, and tokenized inputs.
    """
    def __init__(
        self,
        questions: List[dict],
        image_dir: str,
        processor,
        tokenizer,
        conv_mode: str,
        mm_use_im_start_end: bool
    ):
        self.questions = questions
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        data = self.questions[idx]
        question_id = data["id"]
        img_id = data.get("image")

        if img_id is None:
            raise ValueError(f"Missing image in question {question_id}")

        image_path = os.path.join(self.image_dir, img_id)
        image = Image.open(image_path)

        qs = data["conversations"][0]["value"].replace("<image>", "").strip()
        qs_neg = data["conversations"][-1]["value"]

        # Add image tokens
        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        prompt = image_token + "\n" + qs
        guidance_prompt = image_token + "\n" + qs_neg
        cur_prompt = "<image>\n" + qs

        # Build conversation prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        conv_neg = conv_templates[self.conv_mode].copy()
        conv_neg.append_message(conv_neg.roles[0], guidance_prompt)
        conv_neg.append_message(conv_neg.roles[1], None)
        full_prompt_neg = conv_neg.get_prompt()

        # Tokenize
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        guidance_inputs = self.processor(text=full_prompt_neg, images=image, return_tensors="pt")

        return (
            cur_prompt,
            question_id,
            img_id,
            inputs["input_ids"],
            guidance_inputs["input_ids"],
            inputs["pixel_values"],
            guidance_inputs["pixel_values"],
            inputs["attention_mask"],
            guidance_inputs["attention_mask"]
        )


def custom_collate_fn(batch: List[Tuple[
    str,          # cur_prompt
    str,          # question_id
    str,          # img_id
    torch.Tensor, # input_ids
    torch.Tensor, # guidance_input_ids
    torch.Tensor, # image_tensor
    torch.Tensor, # guidance_image_tensor
    torch.Tensor, # attention_mask
    torch.Tensor  # guidance_attention_mask
]]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function to pad input/guidance_ids and attention masks,
    and stack image tensors. All outputs are moved to CUDA.
    """
    (
        prompts,
        question_ids,
        img_ids,
        input_ids_list,
        guidance_ids_list,
        image_tensors,
        guidance_image_tensors,
        attention_masks_list,
        guidance_attention_masks_list
    ) = zip(*batch)

    def process_sequence(seq_list):
        seq_list = [seq.squeeze(0).flip(dims=[0]) for seq in seq_list]
        return pad_sequence(seq_list, batch_first=True, padding_value=0).flip(dims=[1])

    input_ids_batch = process_sequence(input_ids_list).cuda()
    guidance_ids_batch = process_sequence(guidance_ids_list).cuda()
    
    image_tensor_batch = torch.stack(image_tensors).squeeze(1).cuda()
    guidance_image_tensor_batch = torch.stack(guidance_image_tensors).squeeze(1).cuda()

    attn_mask_batch = process_sequence(attention_masks_list).cuda()
    guidance_attn_mask_batch = process_sequence(guidance_attention_masks_list).cuda()

    return (
        list(prompts),
        list(question_ids),
        list(img_ids),
        input_ids_batch,
        guidance_ids_batch,
        image_tensor_batch,
        guidance_image_tensor_batch,
        attn_mask_batch,
        guidance_attn_mask_batch
    )
