from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  

import torch, json, os
from PIL import Image
from torchvision import transforms
from vcd_utils.vcd_add_noise import add_diffusion_noise  


def load_pope(path_json):
    with open(path_json, 'r') as f:
        return json.load(f)

def preprocess_image(pil_img, image_processor):
    return image_processor(pil_img, return_tensors='pt')['pixel_values'][0]  # [C,H,W]

def get_yes_token_id(tokenizer):
    return tokenizer.encode("Yes", add_special_tokens=False)[0]

@torch.no_grad()
def yes_logit_for(image_tensor, question_text, model, tokenizer, tau=0.1, noise_step=50):
    input_ids = tokenizer(question_text, return_tensors='pt').input_ids.to(model.device)

    v = image_tensor.to(model.device)
    v_pos = add_diffusion_noise(v, noise_step=noise_step)             
    v_neg = add_diffusion_noise(v, noise_step=noise_step) * (-1) + v  

    out_pos = model(input_ids=input_ids, images=v_pos.unsqueeze(0).half(), return_dict=True)
    out_neg = model(input_ids=input_ids, images=v_neg.unsqueeze(0).half(), return_dict=True)
    last_pos = out_pos.logits[:, -1, :]   # [1, V]
    last_neg = out_neg.logits[:, -1, :]

    yes_id = get_yes_token_id(tokenizer)
    return last_pos[0, yes_id].item(), last_neg[0, yes_id].item()


import numpy as np

def fdp_hat(deltas, t):
    neg = np.sum(deltas <= -t)
    pos = np.sum(deltas >=  t)
    return neg / max(1, pos)

def pick_threshold(deltas, q=0.1):
    grid = np.quantile(deltas, np.linspace(0.5, 0.99, 40))
    for t in sorted(set(grid)):
        if fdp_hat(deltas, t) <= q:
            return float(t)
    return float('inf')



from collections import Counter

def evaluate_coco_pope_random(json_path, images_root, model, tokenizer, q=0.1, noise_step=50):
    data = load_pope(json_path)
    deltas, y_true, y_pred = [], [], []

    for ex in data:
        img_path = os.path.join(images_root, ex['image'])
        pil = Image.open(img_path).convert('RGB')
        v = preprocess_image(pil, image_processor=model.image_processor)  
        qtext = ex['question']

        yes_pos, yes_neg = yes_logit_for(v, qtext, model, tokenizer, tau=0.1, noise_step=noise_step)
        y_pred.append('yes' if yes_pos > 0 else 'no')  

        delta = abs(yes_pos) - abs(yes_neg)
        deltas.append(delta)

        y_true.append(ex['answer'].lower())

    deltas = np.array(deltas)
    T = pick_threshold(deltas, q=q)
    sig_yes_mask = (deltas >= T)

    cnt = Counter()
    for yt, yp in zip(y_true, y_pred):
        cnt[('TP' if yt=='yes' and yp=='yes' else
             'TN' if yt=='no'  and yp=='no'  else
             'FP' if yt=='no'  and yp=='yes' else
             'FN')] += 1
    TP, TN, FP, FN = cnt['TP'], cnt['TN'], cnt['FP'], cnt['FN']
    acc = (TP+TN)/max(1,TP+TN+FP+FN)
    prec = TP/max(1,TP+FP)
    rec  = TP/max(1,TP+FN)
    f1   = 2*prec*rec/max(1e-9,prec+rec)

    yes_indices = [i for i, yp in enumerate(y_pred) if yp=='yes']
    vis_yes = np.mean(sig_yes_mask[yes_indices]) if yes_indices else 0.0
    halluc_yes = 1 - vis_yes if yes_indices else 0.0

    return dict(T=T, acc=acc, prec=prec, rec=rec, f1=f1,
                vis_supported_yes=vis_yes, hallucinated_yes=halluc_yes,
                n=len(data))
