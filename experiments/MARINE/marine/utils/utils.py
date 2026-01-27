import math


def get_answers_file_name(args, model_name, pretrain_mm_mlp_adapter=None, vm_pretrain_mm_mlp_adapter=None):
    if pretrain_mm_mlp_adapter is not None and vm_pretrain_mm_mlp_adapter is not None:
        #     --pretrain_mm_mlp_adapter "./llava/backbone/checkpoints_tune/llava-llama-2-7b-chat-DETR-pretrain-1000-tune-1/mm_projector.bin" \
        # special_name = "DETR-pretrain-1000-tune-1"
        try:
            special_name = pretrain_mm_mlp_adapter.split('/')[-2].split('chat-')[-1]
        except:
            special_name = pretrain_mm_mlp_adapter.split('/')[-2]
        try:
            special_name += f"-{vm_pretrain_mm_mlp_adapter.split('/')[-2].split('chat-')[-1]}"
        except:
            special_name += f"-{vm_pretrain_mm_mlp_adapter.split('/')[-2]}"
    else:
        special_name = ""

    file_name = ""
    
    if '7b' in model_name:
        file_name += '-7b'
    elif '13b' in model_name:
        file_name += '-13b'
    
    if 'lora' in model_name:
        file_name += '_lora'

    file_name += f'-bs{args.batch_size}'        
    file_name += f'-s{args.seed}'
    
    if special_name:
        file_name += f'-{special_name}'
    
    if args.guidance_strength is not None:
        answers_file = args.question_file.replace('.json', f'{file_name}-guidance_strength{args.guidance_strength}.jsonl')
    else:
        answers_file = args.question_file.replace('.json', f'{file_name}-guidance_strength.jsonl')
    
    return answers_file


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    # convert dataframes to lists
    lst = list(lst)
    chunks = split_list(lst, n)
    return chunks[k]


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
   
