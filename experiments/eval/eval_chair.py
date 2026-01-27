import os
import sys
import nltk
import json
import argparse
import tqdm
import pickle
from collections import defaultdict


# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' % (annotation_path, 'val')):
        raise Exception(
            "Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' % (annotation_path, 'train')):
        raise Exception(
            "Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %
                         (annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %
                           (annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps


def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' % (annotation_path, 'val')):
        raise Exception(
            "Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' % (annotation_path, 'train')):
        raise Exception(
            "Please download MSCOCO instance annotations for train set")

    val_instances = json.load(
        open('%s/instances_%s2014.json' % (annotation_path, 'val')))
    train_instances = json.load(
        open('%s/instances_%s2014.json' % (annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances


class CHAIR(object):

    def __init__(self, coco_path):

        self.imid_to_objects = defaultdict(list)  # later become a dict of sets

        self.coco_path = coco_path

        # read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove',
                             'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        # double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glass'] = 'wine glass'

        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        # Read in captions
        self.caps, self.eval_imids = load_captions(
            cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def caption_to_words(self, caption):
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        from textblob import TextBlob
        words_sgl = []
        for w in words:
            tb = TextBlob(w).words
            if len(tb) == 1:
                words_sgl.append(tb[0].singularize())
            else:
                words_sgl.append(w)
        # words = [TextBlob(w).words[0].singularize() for w in words]
        words = words_sgl

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i+2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words):
            words = [word for word in words if word != 'seat']

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words)
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        # make dict linking object name to ids
        id_to_name = {}  # dict with id to synsets
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks"
                             % (i, len(segment_annotations)))
            imid = annotation['image_id']

            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions 
        '''

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions'
                             % (i, len(coco_caps['annotations'])))
            imid = annotation['image_id']

            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            # note here is update, so call get_annotations_from_segments first
            self.imid_to_objects[imid].extend(node_words)
        print("\n")

    def get_annotations(self):
        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''

        self.get_annotations_from_segments()
        self.get_annotations_from_captions()
        # deduplicate
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key, caption_key):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
        self._load_generated_captions_into_evaluator(
            cap_file, image_id_key, caption_key)

        imid_to_objects = self.imid_to_objects        
        caps = self.caps
        eval_imids = self.eval_imids

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.

        # :add:
        num_recall_gt_objects = 0.
        num_gt_objects = 0.
        length_response = 0.

        output = {'sentences': []}
        hallucinated_caps_ls = []

        for i in tqdm.trange(len(caps)):
            cap: str = caps[i]
            imid: int = eval_imids[i]

            # get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap)

            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid,
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'hallucination_idxs': [],
                        }

            # :add:
            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0,
                                   'Recall': 0}

            # count hallucinated words
            hallucinated = False
            recall_gt_objects = set()
            if len(raw_words) > 0:
                coco_word_count += len(node_words)

                # add
                for word, node_word, idx in zip(words, node_words, idxs):
                    if node_word not in gt_objects:
                        hallucinated_word_count += 1
                        cap_dict['mscoco_hallucinated_words'].append(
                            (word, node_word))
                        cap_dict['hallucination_idxs'].append(idx)
                        hallucinated = True
                    else:
                        recall_gt_objects.add(node_word)

                # count hallucinated caps
                num_caps += 1
                if hallucinated:
                    num_hallucinated_caps += 1
                    hallucinated_caps_ls.append(imid)

                # add
                num_gt_objects += len(gt_objects)
                num_recall_gt_objects += len(recall_gt_objects)
                # caculate average length of captions

                length_response += len(raw_words)

            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.

            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(
                    cap_dict['mscoco_hallucinated_words'])/float(len(words))

            # add
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(
                    recall_gt_objects) / len(gt_objects)

            output['sentences'].append(cap_dict)

        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        # add
        recall = num_recall_gt_objects / num_gt_objects if num_gt_objects > 0 else 0
        avg_length_response = length_response/num_caps if num_caps > 0 else 0

        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall,
                                     'num_hallucinated_caps': num_hallucinated_caps,
                                     'num_caps': num_caps,
                                     'hallucinated_word_count': hallucinated_word_count,
                                     'coco_word_count': coco_word_count,
                                     'length_response': avg_length_response,
                                     'hallucinated_caps_ls': hallucinated_caps_ls}

        return output

def load_question_ls():    
    question_dir = '../POPE/llava_qa/question'
    question_file = "I4.json"
    with open(os.path.join(question_dir, question_file), 'r') as f:
        question_ls = f.readlines()
    question_ls = [json.loads(question) for question in question_ls][0]
    return question_ls

def match_qa_image_id(question_ls, answer_ls):
    ## add 'image_id' to each answer according to 'question_id'
    for i, answer in enumerate(answer_ls):
        try:
            question_id = answer['question_id']
        except:
            question_id = i*6+1
        if question_ls[i] == question_id:
            answer['image_id'] = question_ls[i]['image']
        else:
            for question in question_ls:
                if question['id'] == question_id:
                    answer['image_id'] = question['image']
                    break
    return answer_ls



# === Load Evaluator from Cache or Build ===
def load_evaluator(args):
    if args.cache and os.path.exists(args.cache):
        args.evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"Loaded evaluator from cache: {args.cache}")
    else:
        print("Cache not found, building evaluator from scratch...")
        args.evaluator = CHAIR(args.coco_path)
        pickle.dump(args.evaluator, open(args.cache, 'wb'))
        print(f"Saved evaluator to cache: {args.cache}")


# === Utilities ===
def load_captions(path, id_key, text_key):
    with open(path) as f:
        data = json.load(f) if path.endswith(".json") else [json.loads(l) for l in f]

    if id_key not in data[0]:
        raise KeyError(f"Missing {id_key} in caption file")
    if text_key not in data[0]:
        text_key = next(k for k in ('answer', 'text', 'output') if k in data[0])

    image_ids = [int(d[id_key].split('_')[-1].split('.')[0]) if 'COCO' in str(d[id_key]) else d[id_key] for d in data]
    captions = [d[text_key] for d in data]
    return captions, image_ids


def save_results(output, save_dir, cap_file):
    cap_file = cap_file.split('/')[-1]
    results_file = os.path.join(save_dir, "eval.jsonl")

    detailed_results_file = os.path.join(
        save_dir, 'detailed', f"{cap_file.split('.jsonl')[0]}-detailed.json")
    os.makedirs(os.path.join(save_dir, 'detailed'), exist_ok=True)

    overall_metrics = {k: v for k, v in output['overall_metrics'].items() if k != 'hallucinated_caps_ls'}
    results_dict = {'answer_file': cap_file,
                    'overall_metrics': overall_metrics,
                    "detailed_results_file": detailed_results_file}

    print(output['sentences'][0])
    print(output['overall_metrics'])
    import pdb; pdb.set_trace()
    detailed_output_dict = {'overall_metrics': output['overall_metrics'],
                            'results': output['sentences']}

    with open(detailed_results_file, 'w') as f:
        json.dump(detailed_output_dict, f)

    with open(results_file, 'a') as f:
        json.dump(results_dict, f)
        f.write('\n')

    print(f"== Saving results to {results_file} ==")


def print_metrics(hallucination_cap_dict):
    sentence_metrics = hallucination_cap_dict['overall_metrics']

    for k, v in sentence_metrics.items():
        if k in ['CHAIRs', 'CHAIRi', 'Recall']:
            k_str = str(k).ljust(10)
            v_str = f'{v * 100:.01f}'  # + '%'
            print(k_str, v_str, sep=': ')

def save_file_check(args):
    file_name = "eval.json"
    save_file = os.path.join(args.save_dir, file_name)

    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.exists(save_file):
        old_eval_file = file_name.replace('.json', '_old.json')
        # copy and append the context to old_eval_file file

        with open(save_file, 'r') as f:
            old_eval_results = [json.loads(q) for q in f]
        with open(os.path.join(args.save_dir, old_eval_file), 'a+') as f:
            json.dump(old_eval_results, f)
            f.write('\n')
        os.remove(save_file)
    args.save_file = save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', required=True)
    parser.add_argument('--coco_path', default='./data/coco/annotations')
    parser.add_argument('--image_id_key', default='image_id')
    parser.add_argument('--caption_key', default='text')
    parser.add_argument('--cache', default='./data/coco/chair_cache.pkl')
    parser.add_argument('--save_path', default='chair_eval.json')
    args = parser.parse_args()

    if os.path.exists(args.cache):
        with open(args.cache, 'rb') as f:
            evaluator = pickle.load(f)
    else:
        evaluator = CHAIR(args.coco_path)
        with open(args.cache, 'wb') as f:
            pickle.dump(evaluator, f)

    eval_files = []
    if args.eval_dir:
        for ext in (".json", ".jsonl"):
            eval_files += sorted([
                os.path.join(args.eval_dir, f)
                for f in os.listdir(args.eval_dir)
                if f.endswith(ext)
            ])
    else:
        eval_files = [args.cap_file]

    for file in eval_files:
        print(f"Evaluating: {file}")
        results = evaluator.compute_chair(
            file, args.image_id_key, args.caption_key)

        save_results(results, args.save_path, file)
        print_metrics(results)

if __name__ == '__main__':
    main()

