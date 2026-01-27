import os
import json
import numpy as np
from typing import List, Dict
from prompt_template import (
    PromptTemplate,
    grounding_dict,
    pope_grounding_dict,
)
from nltk.corpus import wordnet as wn
from find_intersection import parse_synonyms, synonyms_txt, normalize_object     
from eval.utils import load_config

def combine_detected_objects(list1: List[Dict], list2: List[Dict]) -> List[Dict]:
    combined = []
    dict2 = {item['image']: item['objects'] for item in list2}
    for item1 in list1:
        img = item1['image']
        if img in dict2:
            combined.append({
                'image': img,
                'objects': find_intersection_nouns_original(item1['objects'], dict2[img])
            })
    return combined

def get_object_synonyms(word):
    """
    Returns a set of synonyms for a given word using WordNet, 
    but only includes nouns representing objects.
    """
    synonyms = set()
    for syn in wn.synsets(word, pos=wn.NOUN):  # Only consider noun synsets
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Replace underscores with spaces
    return synonyms

def find_intersection_nouns_original(obj_list1, obj_list2):
    """
    Finds the intersection of obj_list1 and obj_list2 using object-specific synonyms (nouns only),
    but the result includes only the original objects from the input lists.
    """
    synonyms_map = parse_synonyms(synonyms_txt)

    # Normalize obj_list1 and obj_list2
    normalized_obj_list1 = [normalize_object(obj, synonyms_map) for obj in obj_list1]
    normalized_obj_list2 = [normalize_object(obj, synonyms_map) for obj in obj_list2]

    # Create a mapping from original words to their expanded synonym sets
    synonym_map1 = {obj: get_object_synonyms(obj).union({obj}) for obj in normalized_obj_list1}
    synonym_map2 = {obj: get_object_synonyms(obj).union({obj}) for obj in normalized_obj_list2}

    # Find the intersection by checking if any synonym set in list 1 intersects with a synonym set in list 2
    intersection_set = set()
    for obj1, syn_set1 in synonym_map1.items():
        for obj2, syn_set2 in synonym_map2.items():
            if syn_set1.intersection(syn_set2):
                intersection_set.add(obj1)  # Only add the original object from obj_list1
    return list(intersection_set)



def generate_qa_guidance(question_path, guidance_path, metric, save_dir, save_name, grounding_template):

    # Load questions
    try:
        with open(question_path, 'r') as f:
            questions = json.load(f)
    except:
        with open(question_path, 'r') as f:
            questions = [json.loads(line) for line in f.readlines()]

    # Load guidance
    if isinstance(guidance_path, list):
        all_guidance = []
        for path in guidance_path:
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    data = [json.loads(line) for line in f]

            all_guidance.append(data)
        guidance = combine_detected_objects(all_guidance[0], all_guidance[1])
    else:
        with open(guidance_path, 'r') as f:
            guidance = json.load(f)

    print("Loaded", len(questions), "questions")
    print("Loaded", len(guidance), "guidance entries")

    np.random.seed(42)
    random_list = np.random.randint(0, 4, len(questions))

    questions_out = []
    labels_out = []

    for i, q in enumerate(questions):
        # match image
        matched = next((g for g in guidance if g['image'] == q['image']), None)
        if matched is None:
            raise ValueError(f"Image {q['image']} not found in guidance")

        # choose prompt and format
        if metric == "chair":
            prompt = q['conversations'][0]['value']
            q['question_id'] = q['id']
        elif metric == "qa90":
            prompt = q['instruction']
            q['question_id'] = i
        else:
            prompt = q['text']

        grounded_prompt = grounding_template.generate_prompt(random_list[i], prompt, object_dict=matched)

        questions_out.append({
            "id": q['question_id'],
            "image": q['image'],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""},
                {"from": "guidance", "value": grounded_prompt}
            ]
        })

        if 'label' in q:
            labels_out.append({
                "id": q['question_id'],
                "image": q['image'],
                "label": q['label']
            })

    # Save
    os.makedirs(os.path.join(save_dir, "question"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "label"), exist_ok=True)

    with open(os.path.join(save_dir, "question", save_name), 'w') as f:
        json.dump(questions_out, f, indent=4)
        print(f"Saved {len(questions_out)} questions to {os.path.join(save_dir, 'question', save_name)}")

    if labels_out:
        with open(os.path.join(save_dir, "label", save_name.replace(".json", "_label.json")), 'w') as f:
            json.dump(labels_out, f, indent=4)


TEMPLATE_MAP = {
    "chair": grounding_dict,
    "pope": pope_grounding_dict,
    "qa90": grounding_dict,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data/marine_qa/question")
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--metric", type=str, default="pope")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--th_detr", type=float, default=0.95)
    parser.add_argument("--th_ram", type=float, default=0.68)
    parser.add_argument("--guidance_dir", type=str, default="./data/marine_qa/guidance")
    args = parser.parse_args()

    question_path,_ = load_config(args.metric, args.dataset)
    grounding_template = PromptTemplate(TEMPLATE_MAP[args.metric], obj_token="<OBJECT_LIST>")

    guidance_path = [
        os.path.join(args.guidance_dir, f"{args.dataset}_detr_th{args.th_detr}.json"),
        os.path.join(args.guidance_dir, f"{args.dataset}_ram_th{args.th_ram}.json")
    ]

    generate_qa_guidance(question_path, guidance_path, args.metric, args.save_dir, args.save_name, grounding_template)
