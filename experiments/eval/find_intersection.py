
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

def parse_synonyms(synonyms_txt):
    """
    Parses the synonyms text into a dictionary where each word is a key, and the value is the set of its synonyms.
    """
    synonyms_map = {}
    lines = synonyms_txt.strip().splitlines()

    for line in lines:
        synonyms = [word.strip() for word in line.split(",") if word.strip()]
        for synonym in synonyms:
            synonyms_map[synonym] = synonyms[0] #set(synonyms)

    return synonyms_map

def normalize_object(obj, synonyms_map):
    """
    Normalizes an object name to its canonical synonym set.
    """
    # print(obj)
    if isinstance(obj, list):
        obj = obj[0]
    if obj in synonyms_map:
        return synonyms_map[obj]
    return obj  # If not found, return the object itself as a set.

def find_intersection(obj_list1, obj_list2, synonyms_txt=synonyms_txt):
    """
    Finds the intersection of obj_list1 and obj_list2, considering the synonyms in synonyms_txt.
    """
    # Parse the synonyms text
    synonyms_map = parse_synonyms(synonyms_txt)

    # Normalize obj_list1 and obj_list2
    normalized_obj_list1 = [normalize_object(obj, synonyms_map) for obj in obj_list1]
    normalized_obj_list2 = [normalize_object(obj, synonyms_map) for obj in obj_list2]

    # Find the intersection
    # import pdb; pdb.set_trace()
    intersection_set = set()
    # for obj_set1 in normalized_obj_list1:
        # for obj_set2 in normalized_obj_list2:
    intersection_set.update(set(normalized_obj_list1) & set(normalized_obj_list2))
    intersection_lst = list(intersection_set)
    return intersection_lst


if __name__ == '__main__':
    
    obj_list1 = ['man', 'bicycle', 'car', 'cell phone']
    obj_list2 = ['girl', 'bike', 'bus', 'telephone']

    result = find_intersection(obj_list1, obj_list2, synonyms_txt)
    print(result)
