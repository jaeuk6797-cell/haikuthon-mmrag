"""Sentence-level prompt splitter (regex only, no LLM, no external API).

Splits a prompt into parts and labels each sentence:
  scene     : mentions no object (style / view / lighting / location only)
  existence : mentions objects but has no spatial/action relation, and none of its
              objects is used by a relation elsewhere in the prompt
              ("One cat is also in the living room.", "One spoon is black.")
  linked    : like existence, but its objects appear in a relation in another
              sentence, so it must stay together with that relation
  relation  : links objects by position or action
Only whole sentences are separated. Objects listed inside a relation sentence
(", and one dog") are left in place on purpose.

Usage: python src/split_prompt.py path/to/prompts.jsonl   (prints coverage stats)
"""
import json
import re
import sys
import collections as C

NOUNS = set(
    "person box cabinet bowl bench table desk scarf shelf ball helmet motorcycle suitcase tree "
    "train horse flower pot parrot stool cube pen bear violin frame bed car cylinder shirt "
    "pitcher tomato armchair cookie screwdriver pizza sofa skateboard glass chair tray rose pig "
    "tape slice egg bicycle bow scooter pumpkin mug strawberry eraser vase shovel apple apron "
    "corn rabbit trowel cupcake basketball glove umbrella bonsai robot cat snowboard kite "
    "saucepan knife lemon ear cheese handbag clock backpack ski boot guitar sneaker surfboard "
    "hat donut book notebook measure dog owl bat camera headphones cone paintbrush tent bird "
    "wedge colander sandwich cushion smartphone sunflower can saw pliers racket pear wristwatch "
    "coat tablet spoon envelope bread kettle dress fork broccoli wrench stapler turtle rake "
    "chicken loaf laptop donkey hamster scissors clipboard fern computer rack sunglasses "
    "dumbbell plate jacket penguin wheelbarrow pyramid tulip sheep cow ruler bag fox bottle "
    "banana bus goat carrot truck duck rug cactus deer cucumber pan croissant sock airplane "
    "elephant lamp boat helicopter hammer cart zebra teapot giraffe food container block skate".split()
)
IRR = {"scarves": "scarf", "shelves": "shelf", "cacti": "cactus", "loaves": "loaf",
       "strawberries": "strawberry", "tomatoes": "tomato", "people": "person", "knives": "knife",
       "foxes": "fox", "buses": "bus", "boxes": "box", "glasses": "glass", "sandwiches": "sandwich",
       "dresses": "dress", "skis": "ski"}
SPATIAL = (r"next to|beside|alongside|\bnear\b|left of|right of|on the left|on the right|\babove\b|"
           r"\bbelow\b|beneath|under (?:one|the|two|three)|on top of|resting on|rests on|sits on|"
           r"\bon one\b|\bon the\b|inside|in front of|behind|\bover\b|among|along\b|"
           r"to (?:its|their|the) (?:left|right)|supports|\bon which\b|\bon top\b")
ACTION = r"\bholds?\b|holding|wearing|wears|\brides?\b|riding"
SCENE = (r"(?:in|on|at|inside) (?:a|an|one|the) (?:(?:[a-z\-]+ ){0,2})?(?:studio with (?:a|one) plain backdrop|"
         r"studio|farm|park|living room|garden|bedroom|kitchen|office|patio|meadow|street|"
         r"airport(?: apron)?|dining room|pasture|parking lot|classroom|forest|cafe|playground|"
         r"beach|garage|orchard|lakeside|store|library|schoolyard|harbor|hall|workshop|campsite|field)")


def norm(w):
    if w in IRR:
        return IRR[w]
    if w in NOUNS:
        return w
    if w.endswith("s") and w[:-1] in NOUNS:
        return w[:-1]
    if w.endswith("es") and w[:-2] in NOUNS:
        return w[:-2]
    return None


def clean(s):
    s = re.sub(SCENE, "", s.lower())
    s = re.sub(r"studio with (?:a|one) plain backdrop", "", s)
    return re.sub(r"(?:under|in|with) (?:soft|warm|direct|side)[a-z ]*(?:lighting|sunlight)", "", s)


def introduced_objects(sentence):
    """Nouns introduced as 'one/two/three/single ... NOUN' within one sentence."""
    tk = re.findall(r"[a-z]+(?:-[a-z]+)*|[,.]", sentence.lower())
    out = []
    for i, w in enumerate(tk):
        if w not in ("one", "two", "three", "single"):
            continue
        for k in range(i + 1, min(i + 7, len(tk))):
            if tk[k] == ".":
                break
            if k > i + 1 and tk[k] in ("one", "two", "three", "is", "are", "with"):
                break
            nn = norm(tk[k])
            if nn:
                out.append(nn)
                break
    return out


def mentioned_nouns(sentence):
    """All nouns mentioned in a sentence, with or without a quantity word."""
    out = set()
    for w in re.findall(r"[a-z]+(?:-[a-z]+)*", sentence.lower()):
        nn = norm(w)
        if nn:
            out.add(nn)
    return out


def split_prompt(prompt):
    """Return [(label, sentence)] for each sentence of the prompt.

    A sentence is 'existence' only if it has no relation AND none of its objects
    is used by a relation in another sentence (otherwise it stays 'linked',
    i.e. it must be kept together with the relation that uses its objects).
    """
    sents = [s for s in re.split(r"(?<=[.])\s+", prompt.strip()) if s]
    info = []
    for s in sents:
        c = clean(s)
        nouns = mentioned_nouns(s)
        has_rel = bool(re.search(SPATIAL, c) or re.search(ACTION, c))
        if not nouns:
            label = "scene"
        elif has_rel:
            label = "relation"
        else:
            label = "existence"
        info.append([label, s, nouns])
    rel_nouns = set()
    for label, _, nouns in info:
        if label == "relation":
            rel_nouns |= nouns
    parts = []
    for label, s, nouns in info:
        if label == "existence" and nouns & rel_nouns:
            label = "linked"
        parts.append((label, s))
    return parts


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
    T = [json.loads(l)["prompts"] for l in open(path, encoding="utf-8")]
    lab = C.Counter()
    with_exist = 0
    exist_objs = 0
    total_objs = 0
    for t in T:
        parts = split_prompt(t)
        lab.update(p[0] for p in parts)
        has = False
        for label, s in parts:
            n = len(introduced_objects(s))
            total_objs += n
            if label == "existence":
                exist_objs += n
                has = True
        with_exist += has
    n_sent = sum(lab.values())
    print("sentences", n_sent, dict(lab))
    print("prompts with >=1 separable existence sentence: %d (%.1f%%)" % (with_exist, 100 * with_exist / len(T)))
    print("objects in existence sentences: %d of %d (%.1f%%)" % (exist_objs, total_objs, 100 * exist_objs / total_objs))
    import random
    random.seed(1)
    ex = [t for t in T if any(l == "existence" for l, _ in split_prompt(t))]
    for t in random.sample(ex, 6):
        print("---")
        for l, s in split_prompt(t):
            print("  [%s] %s" % (l, s))
