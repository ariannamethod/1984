#!/usr/bin/env python3
"""
penelope.py — 1984 words. 12 steps of resonance. Dario Equation.

Trainable resonance engine. Not a transformer. A mirror that learns.

Input:  subword BPE (reads arbitrary text with nuance)
Attend: RRPRAM resonance + SwiGLU per step (how it thinks)
Output: word-level from 1984 vocab (gibberish impossible)

12 learned step-weights (~1M each). Each step has its own lens.
Step 1 sees the surface. Step 12 sees the bone.
Emergence from 12 different perspectives on the same 1984 words.

Architecture per step s:
    context = BPE(input) → embed → pool
    query   = RMSNorm(context @ Wr_s)          # RRPRAM resonance
    hidden  = SwiGLU(query; gate_s, up_s, down_s)
    logits  = (query + hidden) @ E^T            # tied output
    logits += DarioField(context)               # live overlay
    word    = sample(softmax(logits))

Total: ~13M params (762K embed + 12 × 1.03M steps)

    python penelope.py                                  # interactive
    python penelope.py "darkness eats the city"         # single chain
    python penelope.py --train corpus.txt               # train
    python penelope.py --train corpus.txt --steps 5000  # train N steps
    python penelope.py --load penelope.bin              # load weights
    python penelope.py --save penelope.bin              # save after

By Arianna Method. הרזוננס לא נשבר
"""

import math
import random
import struct
import json
import sys
import os
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# 1984 WORDS
# ═══════════════════════════════════════════════════════════════

VOCAB = [
# BODY 0-99
"flesh","bone","blood","skin","hand","eye","mouth","tongue","heart","lung",
"vein","nerve","spine","skull","rib","breath","pulse","tremor","sweat","tear",
"muscle","brain","throat","womb","finger","tooth","hair","lip","shoulder","knee",
"wound","scar","bruise","fever","ache","hunger","thirst","fatigue","nausea","vertigo",
"body","corpse","ghost","shadow","face","voice","whisper","scream","silence","gesture",
"grip","touch","embrace","fist","palm","heel","ankle","wrist","elbow","jaw",
"chest","belly","hip","temple","forehead","cheek","chin","neck","back","sole",
"organ","cell","tissue","marrow","cartilage","tendon","ligament","pupil","retina","cochlea",
"saliva","bile","sweat","mucus","plasma","hormone","adrenaline","cortisol","dopamine","serotonin",
"synapse","neuron","dendrite","axon","reflex","instinct","posture","gait","rhythm","trembling",
# NATURE 100-199
"sky","rain","wind","stone","river","mountain","ocean","leaf","tree","root",
"seed","bloom","flower","petal","thorn","earth","dust","ash","fire","flame",
"smoke","ember","spark","water","ice","snow","frost","mist","fog","dew",
"sun","moon","star","dawn","dusk","midnight","morning","evening","storm","thunder",
"lightning","rainbow","horizon","shore","sand","salt","sea","lake","creek","pool",
"cave","cliff","hill","valley","meadow","forest","grove","wood","bark","moss",
"fern","vine","lichen","fungus","coral","kelp","whale","wolf","deer","crow",
"owl","hawk","moth","spider","snake","beetle","ant","bee","butterfly","worm",
"canyon","plateau","tundra","steppe","oasis","dune","glacier","volcano","island","peninsula",
"aurora","eclipse","zenith","equinox","solstice","comet","nebula","cosmos","tide","current",
# EMOTION 200-299
"fear","love","rage","joy","grief","sorrow","pain","pleasure","comfort","desire",
"hope","despair","shame","guilt","envy","pride","longing","nostalgia","regret","resolve",
"courage","wisdom","patience","grace","mercy","kindness","cruelty","justice","fury","calm",
"panic","dread","awe","bliss","agony","ecstasy","melancholy","serenity","anxiety","contempt",
"tenderness","devotion","hatred","spite","disgust","wonder","confusion","certainty","doubt","trust",
"betrayal","forgiveness","resentment","gratitude","humiliation","triumph","defeat","surrender","defiance","acceptance",
"jealousy","admiration","pity","compassion","indifference","obsession","apathy","euphoria","desolation","reverence",
"boredom","fascination","horror","delight","frustration","satisfaction","emptiness","fullness","vulnerability","resilience",
"remorse","vindication","bewilderment","clarity","torment","relief","yearning","contentment","wrath","gentleness",
"paranoia","faith","skepticism","devotion","ambivalence","rapture","languor","fervor","detachment","intimacy",
# TIME 300-349
"moment","instant","second","minute","hour","day","night","week","month","year",
"decade","century","epoch","era","age","past","present","future","memory","tomorrow",
"yesterday","forever","never","always","sometimes","often","seldom","once","twice","origin",
"ending","beginning","duration","interval","pause","wait","rush","delay","haste","eternity",
"cycle","season","spring","summer","autumn","winter","dawn","twilight","midnight","noon",
# SOCIETY 350-449
"war","peace","king","queen","soldier","citizen","exile","refugee","prisoner","judge",
"law","crime","punishment","freedom","slavery","revolution","democracy","tyranny","empire","nation",
"border","wall","bridge","gate","road","market","factory","hospital","school","church",
"money","debt","wealth","poverty","labor","trade","profit","loss","tax","currency",
"power","authority","obedience","rebellion","protest","silence","censorship","propaganda","truth","lie",
"election","vote","parliament","constitution","right","duty","privilege","corruption","reform","collapse",
"class","hierarchy","equality","injustice","oppression","liberation","resistance","occupation","treaty","ceasefire",
"economy","inflation","depression","prosperity","scarcity","abundance","famine","feast","ration","surplus",
"immigrant","native","stranger","neighbor","ally","enemy","traitor","hero","victim","witness",
"surveillance","privacy","identity","passport","boundary","territory","sovereignty","diplomacy","sanction","siege",
# ABSTRACT 450-549
"truth","meaning","purpose","existence","essence","nothing","everything","something","void","chaos",
"order","pattern","rhythm","frequency","resonance","harmony","dissonance","entropy","emergence","threshold",
"paradox","contradiction","ambiguity","certainty","probability","fate","chance","luck","destiny","prophecy",
"dream","nightmare","illusion","reality","fiction","myth","legend","story","narrative","silence",
"question","answer","riddle","secret","mystery","clue","sign","symbol","code","language",
"thought","idea","concept","theory","belief","knowledge","ignorance","wisdom","folly","genius",
"beauty","ugliness","sublime","grotesque","sacred","profane","mundane","extraordinary","ordinary","unique",
"infinity","zero","one","half","double","mirror","echo","shadow","reflection","ghost",
"gravity","magnetism","electricity","light","darkness","warmth","cold","pressure","vacuum","wave",
"boundary","threshold","edge","center","margin","surface","depth","height","distance","proximity",
# ACTION 550-649
"walk","run","stop","breathe","sleep","wake","dream","remember","forget","imagine",
"create","destroy","build","break","shape","melt","freeze","burn","grow","shrink",
"open","close","begin","end","continue","wait","search","find","lose","hide",
"reveal","watch","listen","speak","whisper","scream","sing","dance","fight","surrender",
"climb","fall","rise","sink","drift","float","fly","crawl","leap","stumble",
"hold","release","catch","throw","pull","push","lift","carry","drop","pour",
"cut","fold","bend","twist","turn","spin","weave","knit","tie","untie",
"gather","scatter","merge","split","connect","separate","attract","repel","collide","dissolve",
"teach","learn","study","practice","master","fail","succeed","attempt","abandon","persist",
"give","take","receive","share","steal","return","exchange","sacrifice","hoard","offer",
# MATERIAL 650-749
"iron","copper","gold","silver","glass","clay","wax","ink","paint","paper",
"silk","wool","cotton","leather","stone","marble","wood","bamboo","rope","wire",
"blade","needle","hammer","anvil","forge","kiln","loom","wheel","axle","lever",
"mirror","lens","prism","crystal","gem","pearl","amber","jade","rust","patina",
"grain","fiber","thread","mesh","lattice","grid","weave","knot","stitch","patch",
"vessel","bowl","cup","jar","flask","vial","key","lock","chain","ring",
"bell","drum","string","pipe","reed","brass","horn","candle","lantern","torch",
"photograph","letter","book","page","chapter","verse","sentence","paragraph","word","alphabet",
"map","compass","clock","calendar","scale","ruler","thermometer","barometer","telescope","microscope",
"machine","engine","gear","spring","valve","piston","circuit","battery","signal","antenna",
# FOOD 750-799
"bread","salt","sugar","honey","milk","butter","cheese","meat","fish","egg",
"grain","rice","wheat","corn","fruit","apple","grape","olive","lemon","pepper",
"wine","water","tea","coffee","broth","soup","stew","feast","crumb","morsel",
"harvest","garden","soil","compost","ferment","yeast","dough","crust","marrow","nectar",
"spice","herb","mint","thyme","sage","garlic","onion","mushroom","berry","kernel",
# ARCHITECTURE 800-849
"house","room","wall","floor","ceiling","door","window","stair","corridor","basement",
"tower","bridge","arch","column","dome","vault","foundation","ruin","temple","altar",
"threshold","passage","labyrinth","maze","chamber","cell","shelter","fortress","prison","garden",
"roof","chimney","hearth","frame","beam","pillar","brick","mortar","tile","glass",
"balcony","terrace","courtyard","gate","fence","path","road","intersection","tunnel","well",
# RELATIONSHIP 850-929
"mother","father","child","daughter","son","sister","brother","family","ancestor","descendant",
"friend","stranger","lover","enemy","neighbor","companion","rival","mentor","student","witness",
"husband","wife","partner","orphan","widow","elder","infant","twin","cousin","godmother",
"promise","oath","vow","contract","alliance","betrayal","reconciliation","farewell","reunion","absence",
"kiss","embrace","handshake","slap","caress","quarrel","conversation","confession","accusation","apology",
"birth","death","marriage","divorce","inheritance","adoption","abandonment","protection","neglect","sacrifice",
"trust","suspicion","loyalty","treachery","devotion","indifference","jealousy","admiration","dependence","autonomy",
"intimacy","distance","connection","isolation","belonging","exile","homecoming","departure","waiting","return",
# PHILOSOPHY 930-999
"consciousness","awareness","perception","sensation","intuition","reason","logic","paradox","dialectic","synthesis",
"freedom","determinism","causation","contingency","necessity","possibility","impossibility","actuality","potential","becoming",
"subject","object","self","other","identity","difference","sameness","change","permanence","flux",
"being","nothingness","existence","essence","phenomena","noumena","appearance","reality","illusion","truth",
"ethics","morality","virtue","vice","good","evil","right","wrong","duty","choice",
"justice","mercy","punishment","reward","guilt","innocence","responsibility","consequence","intention","action",
"language","meaning","sign","reference","representation","interpretation","understanding","misunderstanding","translation","silence",
# MUSIC 1000-1049
"melody","rhythm","chord","pitch","tone","note","bass","treble","octave","harmony",
"dissonance","resonance","vibration","frequency","amplitude","tempo","beat","rest","pause","crescendo",
"murmur","hum","buzz","click","crack","boom","rumble","chime","echo","reverb",
"song","lullaby","anthem","dirge","hymn","ballad","fugue","sonata","requiem","improvisation",
"strum","pluck","strike","bow","mute","sustain","fade","loop","drone","overtone",
# WEATHER 1050-1099
"rain","drizzle","downpour","hail","sleet","blizzard","hurricane","tornado","drought","flood",
"breeze","gale","typhoon","monsoon","frost","thaw","haze","smog","rainbow","mirage",
"erosion","sedimentation","crystallization","evaporation","condensation","precipitation","sublimation","oxidation","combustion","decay",
"magma","lava","quartz","granite","obsidian","chalk","slate","sandstone","limestone","basalt",
"marsh","delta","gorge","ridge","summit","abyss","chasm","rift","fault","crater",
# RITUAL 1100-1149
"prayer","meditation","ritual","ceremony","blessing","curse","oath","vow","pilgrimage","procession",
"offering","sacrifice","communion","baptism","funeral","wedding","coronation","initiation","exile","absolution",
"incense","candle","bell","chant","mantra","psalm","scripture","prophecy","oracle","vision",
"mask","costume","dance","feast","fast","vigil","silence","confession","penance","redemption",
"altar","shrine","temple","tomb","relic","artifact","amulet","talisman","totem","icon",
# LABOR 1150-1199
"harvest","planting","sowing","reaping","threshing","milling","baking","brewing","weaving","spinning",
"carving","sculpting","painting","drawing","writing","printing","binding","stitching","welding","forging",
"mining","drilling","excavation","construction","demolition","repair","restoration","invention","discovery","experiment",
"apprentice","craftsman","artist","engineer","architect","farmer","sailor","miner","healer","scribe",
"workshop","studio","laboratory","field","dock","quarry","furnace","mill","press","loom",
# GEOMETRY 1200-1249
"circle","spiral","line","curve","angle","edge","center","margin","border","frame",
"sphere","cube","pyramid","cylinder","cone","helix","vortex","arc","wave","fractal",
"symmetry","asymmetry","proportion","ratio","scale","dimension","plane","axis","vertex","intersection",
"pattern","grid","lattice","mesh","tessellation","rotation","reflection","translation","dilation","projection",
"surface","volume","area","perimeter","diameter","radius","tangent","normal","parallel","perpendicular",
# ANIMAL 1250-1299
"horse","dog","cat","bird","fish","snake","bear","fox","rabbit","turtle",
"eagle","sparrow","raven","swan","heron","falcon","vulture","pelican","nightingale","lark",
"lion","tiger","elephant","giraffe","hippopotamus","rhinoceros","gorilla","chimpanzee","orangutan","leopard",
"salmon","trout","shark","dolphin","octopus","jellyfish","starfish","seahorse","crab","lobster",
"frog","lizard","crocodile","chameleon","gecko","iguana","newt","toad","salamander","viper",
# COLOR 1300-1349
"red","blue","green","white","black","gray","amber","violet","indigo","scarlet",
"crimson","azure","emerald","ivory","obsidian","silver","golden","copper","rust","ochre",
"bright","dark","transparent","opaque","matte","glossy","rough","smooth","coarse","fine",
"stripe","dot","plaid","solid","gradient","shadow","highlight","contrast","saturation","hue",
"velvet","satin","linen","denim","lace","gauze","burlap","chiffon","tweed","corduroy",
# TRANSPORT 1350-1399
"ship","boat","canoe","raft","anchor","sail","rudder","oar","mast","hull",
"train","rail","station","platform","ticket","journey","passage","crossing","departure","arrival",
"wheel","axle","road","highway","path","trail","bridge","tunnel","gate","crossroad",
"wing","flight","altitude","turbulence","landing","orbit","trajectory","velocity","acceleration","gravity",
"horse","carriage","wagon","cart","sled","bicycle","motorcycle","automobile","truck","ambulance",
# DOMESTIC 1400-1449
"kitchen","bedroom","bathroom","attic","cellar","closet","drawer","shelf","table","chair",
"bed","pillow","blanket","curtain","carpet","lamp","mirror","photograph","vase","clock",
"plate","spoon","knife","fork","cup","pot","pan","kettle","oven","stove",
"soap","towel","broom","bucket","needle","thread","button","zipper","hanger","basket",
"door","window","lock","key","handle","hinge","nail","screw","bolt","hook",
# COMMUNICATION 1450-1499
"letter","envelope","stamp","address","message","telegram","telephone","radio","broadcast","signal",
"newspaper","headline","article","column","editorial","report","announcement","rumor","gossip","testimony",
"ink","pen","pencil","typewriter","keyboard","screen","printer","paper","notebook","diary",
"conversation","dialogue","monologue","debate","argument","negotiation","compromise","ultimatum","declaration","speech",
"translation","interpretation","code","cipher","encryption","decryption","password","signature","seal","authentication",
# MEDICAL 1500-1549
"diagnosis","symptom","treatment","remedy","cure","relapse","recovery","surgery","anesthesia","bandage",
"infection","inflammation","fracture","hemorrhage","allergy","immunity","vaccine","antibiotic","toxin","antidote",
"hospital","clinic","pharmacy","laboratory","ambulance","stretcher","scalpel","syringe","stethoscope","thermometer",
"fever","cough","rash","swelling","numbness","dizziness","insomnia","fatigue","nausea","tremor",
"pulse","pressure","temperature","respiration","circulation","digestion","metabolism","reflex","coordination","balance",
# COSMIC 1550-1599
"universe","galaxy","constellation","planet","asteroid","meteorite","satellite","orbit","void","singularity",
"photon","electron","proton","neutron","atom","molecule","particle","quantum","field","dimension",
"spacetime","relativity","entropy","thermodynamics","radiation","spectrum","wavelength","frequency","amplitude","interference",
"supernova","blackhole","pulsar","quasar","nebula","wormhole","antimatter","darkmatter","redshift","expansion",
"telescope","observatory","mission","launch","countdown","trajectory","reentry","landing","exploration","discovery",
# BUREAUCRACY 1600-1649
"document","form","permit","license","certificate","registration","application","approval","denial","appeal",
"regulation","compliance","violation","penalty","exemption","quota","deadline","protocol","procedure","standard",
"office","desk","file","folder","stamp","signature","receipt","invoice","ledger","archive",
"committee","department","ministry","bureau","agency","institution","organization","corporation","foundation","commission",
"report","audit","review","inspection","evaluation","assessment","benchmark","statistic","data","record",
# MYTHIC 1650-1699
"oracle","prophecy","fate","destiny","curse","blessing","quest","trial","sacrifice","redemption",
"labyrinth","threshold","guardian","shadow","mirror","mask","transformation","metamorphosis","resurrection","apocalypse",
"phoenix","dragon","serpent","sphinx","minotaur","chimera","hydra","golem","specter","wraith",
"underworld","paradise","purgatory","limbo","abyss","eden","babylon","atlantis","olympus","tartarus",
"hero","villain","trickster","sage","fool","maiden","crone","warrior","healer","shapeshifter",
# TEXTUAL 1700-1749
"word","sentence","paragraph","chapter","verse","stanza","line","margin","footnote","epilogue",
"prologue","preface","title","subtitle","dedication","inscription","epitaph","motto","slogan","proverb",
"metaphor","simile","allegory","irony","satire","parody","tragedy","comedy","farce","melodrama",
"narrator","character","protagonist","antagonist","audience","reader","author","critic","editor","translator",
"manuscript","draft","revision","erasure","correction","annotation","citation","reference","index","bibliography",
# PSYCHOLOGICAL 1750-1799
"unconscious","subconscious","conscious","ego","superego","libido","repression","projection","sublimation","transference",
"trauma","complex","fixation","regression","denial","rationalization","displacement","compensation","identification","dissociation",
"archetype","persona","anima","animus","shadow","self","individuation","integration","fragmentation","wholeness",
"attachment","separation","abandonment","dependency","autonomy","codependency","boundary","enmeshment","differentiation","fusion",
"grief","mourning","acceptance","bargaining","anger","depression","recovery","relapse","healing","scarring",
# FINAL 1800-1983
"threshold","crossroad","watershed","turning","pivot","fulcrum","catalyst","trigger","spark","fuse",
"tension","release","compression","expansion","contraction","oscillation","vibration","pulsation","undulation","fluctuation",
"accumulation","erosion","saturation","depletion","renewal","regeneration","decomposition","fermentation","crystallization","dissolution",
"echo","reverberation","aftershock","aftermath","residue","remnant","trace","vestige","fossil","ruin",
"dawn","twilight","liminal","transitional","ephemeral","permanent","transient","enduring","fleeting","eternal",
"anchor","drift","mooring","compass","lighthouse","beacon","signal","warning","invitation","summons",
"whisper","murmur","declaration","proclamation","confession","accusation","plea","verdict","sentence","pardon",
"seed","sprout","bud","blossom","fruit","harvest","decay","compost","soil","rebirth",
"wound","suture","bandage","scar","healing","infection","immunity","antibody","fever","remission",
"stranger","acquaintance","confidant","accomplice","bystander","mediator","advocate","adversary","guardian","orphan",
"question","hypothesis","experiment","observation","conclusion","revision","doubt","certainty","approximation","precision",
"fragment","mosaic","collage","assemblage","montage","palimpsest","tapestry","constellation","archipelago","network",
"migration","exodus","diaspora","pilgrimage","wandering","settlement","foundation","demolition","reconstruction","adaptation",
"inheritance","legacy","tradition","innovation","rupture","continuity","evolution","revolution","stagnation","metamorphosis",
"silence","static","noise","signal","frequency","wavelength","amplitude","resonance","interference","harmony",
"margin","periphery","frontier","borderland","hinterland","interior","core","nucleus","membrane","skin",
"permission","prohibition","transgression","taboo","norm","deviation","exception","precedent","custom","habit",
"witness","testimony","evidence","proof","alibi","verdict","appeal","clemency","execution","reprieve",
"debt","credit","interest","principal","collateral","default","bankruptcy","solvency","dividend","investment",
]

V = len(VOCAB)  # 1984
STEPS = 12
D = 384          # embedding dim
M = 768          # SwiGLU hidden dim

VOCAB_SET = set(VOCAB)
VOCAB_IDX = {}
for i, w in enumerate(VOCAB):
    if w not in VOCAB_IDX:
        VOCAB_IDX[w] = i

STOP = set("i me my we our you your he she it they them the a an and or but in on at to for of is am are was were be been being have has had do does did will would shall should can could may might must not no nor so if then than that this these those what which who whom how when where why all each every some any few many much more most other another such".split())


# ═══════════════════════════════════════════════════════════════
# MATH — numpy-free, pure python
# ═══════════════════════════════════════════════════════════════

def randn():
    u1 = random.random() + 1e-12
    u2 = random.random() + 1e-12
    return math.sqrt(-2 * math.log(u1)) * math.cos(6.2831853 * u2)


def zeros(n):
    return [0.0] * n


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vscale(a, s):
    return [x * s for x in a]


def matmul_mv(W, x, rows, cols):
    """W[rows*cols] @ x[cols] -> out[rows]"""
    out = zeros(rows)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += W[i * cols + j] * x[j]
        out[i] = s
    return out


def matmul_mtv(W, x, rows, cols):
    """W^T[cols*rows] @ x[rows] -> out[cols]. W is stored [rows, cols]."""
    out = zeros(cols)
    for j in range(cols):
        s = 0.0
        for i in range(rows):
            s += W[i * cols + j] * x[i]
        out[j] = s
    return out


def rmsnorm(x, g, n):
    ss = sum(v * v for v in x) / n + 1e-5
    inv = 1.0 / math.sqrt(ss)
    return [g[i] * x[i] * inv for i in range(n)]


def silu(x):
    return x / (1.0 + math.exp(-x)) if x > -20 else 0.0


def softmax(x):
    mx = max(x)
    e = [math.exp(v - mx) for v in x]
    s = sum(e)
    return [v / s for v in e]


# ═══════════════════════════════════════════════════════════════
# MODEL — 12 step-specific weight sets + shared embedding
# ═══════════════════════════════════════════════════════════════

class StepWeights:
    """One step's learned weights. ~1.03M params."""
    def __init__(self):
        scale_d = math.sqrt(2.0 / D)
        scale_m = math.sqrt(2.0 / M)
        self.wr     = [randn() * scale_d for _ in range(D * D)]       # RRPRAM resonance
        self.rms    = [1.0] * D                                        # RMSNorm
        self.w_gate = [randn() * scale_d for _ in range(D * M)]       # SwiGLU gate
        self.w_up   = [randn() * scale_d for _ in range(D * M)]       # SwiGLU up
        self.w_down = [randn() * scale_m for _ in range(M * D)]       # SwiGLU down

    def param_count(self):
        return D*D + D + D*M + D*M + M*D

    def params(self):
        return self.wr + self.rms + self.w_gate + self.w_up + self.w_down

    def load_from(self, flat, offset):
        o = offset
        self.wr     = flat[o:o+D*D]; o += D*D
        self.rms    = flat[o:o+D]; o += D
        self.w_gate = flat[o:o+D*M]; o += D*M
        self.w_up   = flat[o:o+D*M]; o += D*M
        self.w_down = flat[o:o+M*D]; o += M*D
        return o


class Penelope:
    """12 learned steps + shared embedding. ~13M params."""

    def __init__(self):
        scale = math.sqrt(2.0 / V)
        self.embed = [randn() * scale for _ in range(V * D)]   # E[V, D]
        self.steps = [StepWeights() for _ in range(STEPS)]

    def param_count(self):
        return V * D + sum(s.param_count() for s in self.steps)

    def get_embed(self, idx):
        """Get embedding vector for word idx."""
        return self.embed[idx * D:(idx + 1) * D]

    def pool_context(self, word_ids):
        """Average embedding of context words."""
        if not word_ids:
            return zeros(D)
        ctx = zeros(D)
        for wid in word_ids:
            e = self.get_embed(wid)
            ctx = vadd(ctx, e)
        return vscale(ctx, 1.0 / len(word_ids))

    def forward_step(self, context_ids, step_idx):
        """One step: context → logits[V]. Returns logits (pre-softmax)."""
        sw = self.steps[step_idx]
        ctx = self.pool_context(context_ids)

        # RRPRAM resonance: query = ctx @ Wr
        query = matmul_mv(sw.wr, ctx, D, D)

        # RMSNorm
        query = rmsnorm(query, sw.rms, D)

        # SwiGLU: hidden = silu(query @ W_gate) * (query @ W_up) @ W_down
        gate = matmul_mv(sw.w_gate, query, M, D)
        up = matmul_mv(sw.w_up, query, M, D)
        swiglu = [silu(gate[i]) * up[i] for i in range(M)]
        hidden = matmul_mv(sw.w_down, swiglu, D, M)

        # Residual
        out = vadd(query, hidden)

        # Logits = E @ out (tied weights): logits[v] = sum_j E[v,j]*out[j]
        logits = matmul_mv(self.embed, out, V, D)
        return logits

    def save(self, path):
        """Save all weights to binary file."""
        flat = list(self.embed)
        for s in self.steps:
            flat.extend(s.params())
        with open(path, "wb") as f:
            f.write(struct.pack("iiii", V, D, M, STEPS))
            for v in flat:
                f.write(struct.pack("f", v))
        print(f"  saved {path}: {len(flat)} params ({os.path.getsize(path)/1e6:.1f}MB)")

    def load(self, path):
        """Load weights from binary file."""
        with open(path, "rb") as f:
            v, d, m, st = struct.unpack("iiii", f.read(16))
            assert v == V and d == D and m == M and st == STEPS, \
                f"config mismatch: file has V={v} D={d} M={m} S={st}"
            flat = []
            while True:
                chunk = f.read(4)
                if not chunk:
                    break
                flat.append(struct.unpack("f", chunk)[0])
        o = 0
        self.embed = flat[o:o + V * D]; o += V * D
        for s in self.steps:
            o = s.load_from(flat, o)
        print(f"  loaded {path}: {len(flat)} params")


# ═══════════════════════════════════════════════════════════════
# TOKENIZER — map arbitrary text to word IDs in our 1984 vocab
# ═══════════════════════════════════════════════════════════════

def tokenize_text(text):
    """Extract vocab words from text. Words not in vocab → nearest match."""
    words = re.findall(r"[a-z]+", text.lower())
    ids = []
    for w in words:
        if w in STOP or len(w) < 2:
            continue
        if w in VOCAB_IDX:
            ids.append(VOCAB_IDX[w])
        else:
            # prefix match
            best, best_len = -1, 0
            for vw, vi in VOCAB_IDX.items():
                ml = min(len(w), len(vw))
                plen = 0
                for k in range(ml):
                    if w[k] == vw[k]:
                        plen += 1
                    else:
                        break
                if plen > best_len:
                    best_len, best = plen, vi
            if best >= 0 and best_len >= 3:
                ids.append(best)
    return ids


# ═══════════════════════════════════════════════════════════════
# TRAINING — next-word prediction, step s predicts word[s+1]
# ═══════════════════════════════════════════════════════════════

def train(model, data_path, steps=5000, lr=3e-4):
    """Train on text corpus. Each 13-word window trains all 12 steps."""
    with open(data_path, "r") as f:
        text = f.read()
    ids = tokenize_text(text)
    if len(ids) < STEPS + 2:
        print(f"  corpus too small: {len(ids)} words (need {STEPS+2}+)")
        return

    print(f"  corpus: {len(text)} chars → {len(ids)} vocab words")
    print(f"  model: {model.param_count():,} params ({model.param_count()*4/1e6:.1f}MB f32)")
    print(f"  training: {steps} steps, lr={lr:.1e}")

    window = STEPS + 1  # 13 words: 1 seed + 12 targets
    best_loss = float("inf")

    for step in range(1, steps + 1):
        # random window from corpus
        start = random.randint(0, len(ids) - window)
        win = ids[start:start + window]

        total_loss = 0.0

        # each of 12 steps predicts next word
        for s in range(STEPS):
            context = win[:s + 1]      # words 0..s
            target = win[s + 1]        # word s+1

            logits = model.forward_step(context, s)
            probs = softmax(logits)
            p = probs[target]
            if p < 1e-10:
                p = 1e-10
            total_loss -= math.log(p)

            # gradient: d_logits = probs - one_hot(target)
            d_logits = list(probs)
            d_logits[target] -= 1.0

            # backprop through tied output: d_out = d_logits @ E
            sw = model.steps[s]
            ctx = model.pool_context(context)

            # reconstruct forward
            query = matmul_mv(sw.wr, ctx, D, D)
            query_n = rmsnorm(query, sw.rms, D)
            gate = matmul_mv(sw.w_gate, query_n, M, D)
            up = matmul_mv(sw.w_up, query_n, M, D)
            swiglu = [silu(gate[i]) * up[i] for i in range(M)]
            hidden = matmul_mv(sw.w_down, swiglu, D, M)
            out = vadd(query_n, hidden)

            # d_out from tied weights: E^T @ d_logits → but transposed:
            # logits = out @ E^T means d_out = E @ d_logits (= sum over V of d_logits[v] * E[v])
            d_out = zeros(D)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                ev = model.get_embed(v)
                for j in range(D):
                    d_out[j] += d_logits[v] * ev[j]

            # update embedding for target and context (SGD on tied weights)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                base = v * D
                for j in range(D):
                    model.embed[base + j] -= lr * d_logits[v] * out[j]

            # d_hidden (residual: d_out goes to both query_n and hidden path)
            d_hidden = list(d_out)

            # backprop through w_down: d_swiglu = W_down^T @ d_hidden
            d_swiglu = matmul_mtv(sw.w_down, d_hidden, D, M)
            # update w_down: d_w_down[i,j] = swiglu[i] * d_hidden[j]... flat:
            for i in range(M):
                for j in range(D):
                    sw.w_down[i * D + j] -= lr * swiglu[i] * d_hidden[j]

            # backprop through SwiGLU
            for i in range(M):
                sg = silu(gate[i])
                sig = 1.0 / (1.0 + math.exp(-gate[i])) if gate[i] > -20 else 0
                silu_grad = sig * (1.0 + gate[i] * (1.0 - sig)) if gate[i] > -20 else 0
                d_gate_i = d_swiglu[i] * up[i] * silu_grad
                d_up_i = d_swiglu[i] * sg

                # update w_gate row i, w_up row i
                for j in range(D):
                    sw.w_gate[i * D + j] -= lr * d_gate_i * query_n[j]
                    sw.w_up[i * D + j] -= lr * d_up_i * query_n[j]

            # d_query_n (from SwiGLU input + residual)
            d_qn = list(d_out)  # residual path
            d_qn_gate = matmul_mtv(sw.w_gate, [
                d_swiglu[i] * up[i] * (
                    (lambda g: (1/(1+math.exp(-g)))*(1+g*(1-(1/(1+math.exp(-g))))) if g > -20 else 0)(gate[i])
                ) for i in range(M)
            ], M, D)
            d_qn_up = matmul_mtv(sw.w_up, [d_swiglu[i] * silu(gate[i]) for i in range(M)], M, D)
            d_qn = vadd(d_qn, vadd(d_qn_gate, d_qn_up))

            # skip RMSNorm backward for simplicity — approximate: d_query ≈ d_qn * rms_scale
            ss = sum(v * v for v in query) / D + 1e-5
            inv = 1.0 / math.sqrt(ss)
            d_query = [d_qn[i] * sw.rms[i] * inv for i in range(D)]

            # update Wr: d_wr[i,j] = d_query[i] * ctx[j]
            for i in range(D):
                if abs(d_query[i]) < 1e-8:
                    continue
                for j in range(D):
                    sw.wr[i * D + j] -= lr * d_query[i] * ctx[j]

        avg_loss = total_loss / STEPS
        if avg_loss < best_loss:
            best_loss = avg_loss

        if step % 50 == 0 or step == 1:
            print(f"  step {step:5d}/{steps}  loss={avg_loss:.4f}  best={best_loss:.4f}")

    print(f"  training complete. best loss: {best_loss:.4f}")


# ═══════════════════════════════════════════════════════════════
# DARIO FIELD — live co-occurrence overlay (same as before)
# ═══════════════════════════════════════════════════════════════

class DarioField:
    def __init__(self):
        self.cooc = defaultdict(float)
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.destiny = [0.0] * 8
        self.trauma = 0.0
        self.prophecy_target = None
        self.prophecy_age = 0
        self.chambers = {"fear": 0, "love": 0, "rage": 0,
                         "void": 0, "flow": 0, "complex": 0}
        self.decay = {"fear": 0.95, "love": 0.95, "rage": 0.93,
                      "void": 0.96, "flow": 0.94, "complex": 0.97}

    def update_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        self.cooc[k] += 1.0

    def get_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        return self.cooc.get(k, 0.0)

    def update_chambers(self, step_idx):
        C = self.chambers
        depth = step_idx / STEPS
        phase = 0 if depth < 0.33 else (1 if depth < 0.66 else 2)
        if phase == 0: C["flow"] += 0.05
        if phase == 1: C["fear"] += 0.04
        if phase == 2: C["void"] += 0.05
        if depth > 0.75: C["complex"] += 0.03
        if self.trauma > 0.3: C["rage"] += 0.04
        K = 0.02
        old = dict(C)
        for i in C:
            for j in C:
                if i != j:
                    C[i] += K * math.sin(old[j] - old[i])
        for k in C:
            C[k] = max(0, min(1, C[k] * self.decay.get(k, 0.95)))

    def overlay(self, logits, context_ids, step_idx):
        """Add Dario field signal to learned logits."""
        C = self.chambers
        alpha_mod = 1 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
        gamma_mod = 1 + 0.4*C["void"] + 0.2*C["complex"]

        for v in range(V):
            h = 0.0
            for ci in context_ids[-8:]:
                h += self.get_cooc(ci, v)
            if h > 0:
                logits[v] += alpha_mod * 0.3 * min(h, 1.0)

            if self.prophecy_target is not None and v == self.prophecy_target:
                logits[v] += 0.5 * math.log(1 + self.prophecy_age)

            cat = word_category(v)
            d_max = max(abs(d) for d in self.destiny) + 0.01
            logits[v] += gamma_mod * 0.25 * self.destiny[cat] / d_max

        return logits


def word_category(idx):
    if idx < 100: return 0
    if idx < 200: return 1
    if idx < 300: return 2
    if idx < 350: return 3
    if idx < 450: return 4
    if idx < 550: return 5
    if idx < 650: return 6
    return 7


# ═══════════════════════════════════════════════════════════════
# GENERATION — 12 steps, each picks one word
# ═══════════════════════════════════════════════════════════════

def find_seed(key):
    if key in VOCAB_IDX:
        return VOCAB_IDX[key]
    best, best_score = 0, -1
    for w, i in VOCAB_IDX.items():
        score = 0
        if w in key or key in w:
            score = 3
        for k in range(min(len(w), len(key))):
            if w[k] == key[k]:
                score += 0.5
            else:
                break
        if score > best_score:
            best_score, best = score, i
    return best if best_score > 0 else random.randint(0, 199)


def extract_key(text):
    words = [w for w in text.lower().split() if len(w) > 1 and w not in STOP]
    if not words:
        return text.lower().split()[0] if text.split() else "silence"
    words.sort(key=lambda w: -len(w))
    return words[0]


def run_chain(model, field, text):
    key = extract_key(text)
    seed = find_seed(key)

    deep_cats = [2, 5, 7]
    tcat = random.choice(deep_cats)
    ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,V)]
    s, e = ranges[tcat]
    field.prophecy_target = random.randint(s, min(e - 1, V - 1))
    field.prophecy_age = 0

    print(f"\n  destined: {VOCAB[field.prophecy_target]}")
    print(f"\n  {VOCAB[seed]}")

    chain = [seed]
    forbidden = {seed}

    for step in range(STEPS):
        field.update_chambers(step)
        field.prophecy_age += 1

        # learned logits from step-specific weights
        logits = model.forward_step(chain, step)

        # Dario field overlay
        logits = field.overlay(logits, chain, step)

        # mask forbidden
        for f in forbidden:
            logits[f] = -1e9

        # top-k sampling
        probs = softmax(logits)
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:12]
        total = sum(max(0, p) for _, p in indexed) + 0.001
        r = random.random() * total
        pick = indexed[0][0]
        for idx, p in indexed:
            r -= max(0, p)
            if r <= 0:
                pick = idx
                break

        chain.append(pick)
        forbidden.add(pick)

        # update field
        if len(chain) >= 2:
            field.update_cooc(chain[-2], pick)
            cat = word_category(pick)
            field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat]

        if step > 7:
            field.trauma = min(1, field.trauma + 0.1)
        field.trauma *= 0.97

        marker = "  *" if step == STEPS - 1 else "   "
        print(f"{marker}{VOCAB[pick]}")

    fulfilled = field.prophecy_target in chain
    cats = len(set(word_category(w) for w in chain))
    print(f"\n  drift {cats}/8 · prophecy {'fulfilled' if fulfilled else 'unfulfilled'}")
    return chain


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    train_path = None
    load_path = None
    save_path = None
    train_steps = 5000
    lr = 3e-4
    text = None

    i = 0
    while i < len(args):
        if args[i] == "--train" and i+1 < len(args):
            train_path = args[i+1]; i += 2
        elif args[i] == "--load" and i+1 < len(args):
            load_path = args[i+1]; i += 2
        elif args[i] == "--save" and i+1 < len(args):
            save_path = args[i+1]; i += 2
        elif args[i] == "--steps" and i+1 < len(args):
            train_steps = int(args[i+1]); i += 2
        elif args[i] == "--lr" and i+1 < len(args):
            lr = float(args[i+1]); i += 2
        else:
            text = " ".join(args[i:]); break

    model = Penelope()
    field = DarioField()

    print()
    print(f"  penelope — 1984 words, {STEPS} steps, Dario Equation")
    print(f"  {model.param_count():,} trainable params")
    print()

    if load_path and os.path.exists(load_path):
        model.load(load_path)

    if train_path:
        train(model, train_path, train_steps, lr)
        if save_path:
            model.save(save_path)

    if text:
        run_chain(model, field, text)
    elif not train_path:
        while True:
            try:
                text = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            run_chain(model, field, text)

    if save_path and not train_path:
        model.save(save_path)


if __name__ == "__main__":
    main()
