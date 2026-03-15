#!/usr/bin/env python3
"""
penelope.py — 1984 words. 12 steps of resonance. Dario Equation.

Trainable resonance engine. Not a transformer. A mirror that learns.

Input:  text → BPE subword tokens (2048-token byte-pair encoding)
Attend: RRPRAM resonance + SwiGLU per step (how it thinks)
Output: word-level from 1984 vocab (gibberish impossible)

12 learned step-weights (~1.03M each). Each step has its own lens.
Step 1 sees the surface. Step 12 sees the bone.

Architecture per step s:
    context = pool(embed_in(BPE tokens))
    query   = RMSNorm(context @ Wr_s)          # RRPRAM resonance
    hidden  = SwiGLU(query; gate_s, up_s, down_s)
    logits  = (query + hidden) @ E_out^T        # separate output embed
    logits += DarioField(context)               # live overlay
    word    = sample(softmax(logits))

Total: ~14M params (786K embed_in + 762K embed_out + 12 x 1.03M steps)

  score(w) = B + alpha*H + beta*F + gamma*A + T      (Dario Equation)

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
"debt","credit","interest","principal",
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
# BPE — 2048-token byte-pair encoding (256 bytes + 1792 merges)
# ═══════════════════════════════════════════════════════════════

BPE_VOCAB = 2048
BPE_MERGES = 1792

BPE_TABLE = [
    (115, 32),(101, 32),(46, 32),(105, 110),(105, 256),(101, 114),(111, 110),(116, 104),
    (116, 32),(101, 110),(97, 110),(116, 105),(104, 260),(101, 115),(121, 32),(258, 268),
    (100, 32),(111, 114),(259, 103),(97, 114),(97, 108),(274, 32),(267, 262),(111, 117),
    (101, 256),(114, 101),(111, 32),(105, 116),(97, 116),(58, 32),(111, 109),(115, 116),
    (100, 105),(101, 108),(104, 97),(114, 269),(112, 261),(261, 32),(263, 257),(266, 272),
    (278, 32),(111, 119),(97, 99),(105, 115),(266, 32),(44, 32),(39, 256),(276, 32),
    (108, 105),(265, 99),(114, 97),(116, 282),(117, 114),(101, 272),(105, 109),(102, 102),
    (101, 120),(101, 99),(101, 109),(102, 32),(102, 273),(114, 111),(101, 116),(10, 10),
    (97, 285),(113, 285),(10, 320),(46, 319),(323, 321),(117, 110),(97, 32),(117, 108),
    (101, 118),(265, 264),(290, 264),(119, 330),(105, 99),(265, 116),(275, 257),(284, 116),
    (97, 115),(103, 104),(97, 296),(63, 322),(288, 311),(105, 32),(115, 105),(99, 262),
    (110, 111),(112, 291),(305, 257),(101, 271),(100, 101),(111, 108),(105, 108),(286, 101),
    (283, 270),(263, 101),(97, 98),(101, 100),(115, 351),(97, 103),(99, 286),(275, 105),
    (115, 117),(262, 32),(340, 261),(114, 117),(269, 115),(111, 315),(97, 112),(119, 104),
    (262, 257),(105, 114),(108, 270),(98, 101),(115, 99),(109, 101),(98, 257),(265, 32),
    (259, 32),(115, 289),(267, 118),(114, 105),(111, 99),(115, 104),(267, 109),(97, 109),
    (112, 317),(344, 264),(113, 117),(105, 263),(263, 300),(335, 261),(109, 273),(266, 110),
    (119, 273),(112, 111),(101, 264),(279, 108),(121, 258),(395, 272),(97, 278),(267, 99),
    (353, 270),(119, 101),(359, 391),(402, 326),(45, 32),(109, 32),(273, 32),(266, 99),
    (97, 100),(324, 331),(257, 260),(121, 279),(121, 271),(111, 263),(263, 277),(119, 387),
    (112, 108),(276, 352),(290, 112),(102, 101),(101, 258),(105, 100),(100, 97),(279, 264),
    (117, 109),(100, 117),(104, 32),(337, 264),(292, 105),(115, 271),(116, 114),(100, 256),
    (100, 277),(99, 104),(109, 270),(107, 32),(276, 108),(100, 111),(116, 256),(109, 389),
    (103, 117),(118, 261),(115, 112),(105, 264),(99, 111),(108, 257),(294, 115),(258, 403),
    (119, 397),(422, 270),(411, 32),(98, 117),(258, 341),(99, 300),(121, 302),(112, 275),
    (116, 111),(114, 297),(116, 306),(269, 256),(110, 282),(275, 32),(105, 427),(258, 294),
    (328, 261),(98, 318),(316, 32),(102, 259),(115, 262),(114, 286),(97, 264),(277, 295),
    (115, 107),(99, 108),(304, 102),(348, 112),(117, 115),(265, 115),(110, 364),(100, 282),
    (101, 301),(312, 428),(108, 32),(356, 414),(292, 468),(116, 117),(281, 99),(415, 423),
    (99, 275),(116, 108),(112, 32),(342, 361),(105, 287),(103, 103),(111, 111),(308, 257),
    (327, 116),(259, 116),(265, 267),(261, 257),(297, 110),(263, 293),(297, 32),(335, 265),
    (115, 258),(112, 273),(97, 107),(98, 317),(390, 257),(263, 261),(97, 117),(430, 270),
    (377, 102),(103, 110),(377, 315),(451, 264),(109, 111),(362, 329),(279, 115),(355, 271),
    (275, 272),(118, 472),(425, 507),(522, 521),(109, 462),(287, 363),(101, 103),(306, 501),
    (527, 388),(278, 303),(269, 271),(278, 256),(524, 374),(112, 304),(116, 297),(534, 520),
    (356, 382),(102, 105),(263, 259),(536, 280),(511, 307),(273, 105),(367, 499),(292, 418),
    (325, 100),(110, 100),(477, 338),(543, 256),(39, 264),(298, 426),(437, 280),(263, 269),
    (401, 375),(446, 546),(465, 552),(314, 111),(99, 101),(390, 110),(508, 388),(100, 269),
    (362, 346),(274, 271),(98, 413),(439, 256),(551, 257),(263, 470),(400, 334),(121, 394),
    (339, 294),(448, 540),(474, 257),(458, 424),(381, 105),(266, 100),(314, 32),(115, 380),
    (575, 105),(562, 32),(261, 103),(484, 417),(339, 523),(118, 105),(104, 349),(266, 103),
    (409, 260),(357, 257),(386, 269),(112, 114),(358, 316),(105, 122),(104, 502),(111, 112),
    (111, 441),(99, 296),(555, 529),(383, 257),(101, 119),(116, 354),(262, 103),(557, 277),
    (98, 105),(116, 261),(281, 408),(102, 327),(99, 366),(101, 549),(316, 109),(277, 115),
    (475, 265),(101, 302),(419, 289),(99, 114),(512, 45),(345, 329),(119, 97),(102, 469),
    (580, 454),(263, 260),(270, 260),(371, 277),(615, 32),(116, 259),(259, 264),(279, 114),
    (109, 266),(290, 256),(284, 257),(378, 257),(115, 257),(98, 108),(116, 289),(287, 114),
    (291, 112),(111, 100),(284, 309),(261, 118),(103, 259),(34, 32),(101, 275),(349, 117),
    (115, 595),(312, 116),(103, 306),(407, 257),(479, 450),(112, 97),(104, 289),(632, 262),
    (109, 329),(110, 297),(265, 578),(516, 378),(550, 385),(382, 257),(109, 262),(467, 431),
    (392, 435),(282, 260),(102, 306),(115, 121),(324, 590),(456, 282),(283, 256),(259, 459),
    (328, 265),(312, 492),(342, 262),(102, 298),(398, 256),(447, 655),(263, 574),(345, 333),
    (611, 299),(99, 281),(107, 257),(104, 293),(266, 264),(292, 102),(505, 116),(343, 102),
    (288, 115),(369, 32),(283, 514),(481, 305),(333, 271),(457, 256),(313, 116),(584, 294),
    (108, 266),(292, 606),(260, 385),(660, 644),(121, 263),(105, 513),(115, 308),(688, 440),
    (538, 107),(677, 313),(112, 104),(293, 263),(340, 332),(279, 337),(373, 394),(440, 350),
    (488, 114),(99, 334),(115, 418),(415, 32),(349, 111),(280, 307),(116, 265),(116, 370),
    (260, 104),(332, 303),(287, 259),(304, 674),(500, 32),(110, 313),(646, 112),(97, 259),
    (99, 97),(481, 346),(373, 288),(327, 461),(120, 105),(299, 301),(119, 259),(537, 289),
    (581, 596),(99, 379),(353, 681),(361, 331),(108, 598),(706, 280),(266, 724),(650, 270),
    (281, 108),(278, 258),(556, 112),(104, 310),(280, 334),(651, 338),(360, 99),(115, 101),
    (287, 284),(476, 264),(734, 318),(630, 482),(111, 311),(328, 293),(392, 107),(99, 266),
    (358, 416),(102, 97),(299, 405),(436, 256),(413, 293),(623, 99),(586, 531),(105, 315),
    (308, 277),(291, 262),(263, 32),(345, 346),(485, 281),(452, 569),(708, 103),(372, 105),
    (610, 32),(571, 97),(279, 545),(298, 278),(455, 399),(116, 271),(559, 729),(116, 641),
    (525, 99),(381, 397),(283, 117),(103, 266),(98, 336),(107, 649),(109, 259),(100, 760),
    (273, 779),(309, 376),(109, 314),(589, 280),(631, 284),(265, 117),(333, 370),(727, 272),
    (489, 396),(118, 257),(288, 486),(280, 102),(108, 101),(772, 723),(274, 301),(115, 313),
    (291, 757),(328, 375),(356, 368),(119, 283),(425, 99),(639, 278),(774, 374),(104, 111),
    (266, 101),(717, 364),(366, 533),(588, 597),(115, 264),(419, 461),(775, 495),(809, 275),
    (109, 275),(310, 496),(817, 808),(104, 257),(274, 258),(695, 585),(310, 678),(510, 263),
    (662, 716),(664, 277),(358, 112),(343, 767),(376, 283),(818, 518),(324, 806),(803, 478),
    (582, 432),(259, 284),(325, 811),(98, 770),(732, 293),(525, 493),(98, 273),(460, 836),
    (109, 308),(280, 436),(333, 338),(509, 410),(544, 293),(822, 676),(837, 108),(100, 500),
    (272, 365),(355, 258),(362, 790),(371, 636),(463, 791),(766, 713),(834, 445),(274, 322),
    (498, 116),(97, 256),(642, 442),(105, 102),(288, 714),(710, 491),(635, 332),(778, 338),
    (99, 369),(784, 787),(99, 755),(102, 363),(298, 485),(393, 287),(420, 460),(604, 764),
    (694, 667),(700, 496),(744, 480),(258, 539),(269, 438),(101, 107),(331, 690),(363, 621),
    (372, 879),(39, 32),(267, 337),(277, 661),(301, 300),(309, 620),(541, 842),(814, 404),
    (860, 593),(886, 535),(45, 570),(284, 280),(295, 815),(380, 634),(602, 663),(625, 797),
    (792, 843),(878, 567),(107, 259),(406, 839),(443, 577),(483, 487),(528, 771),(535, 894),
    (553, 365),(553, 895),(613, 899),(617, 874),(682, 850),(715, 832),(761, 407),(783, 907),
    (800, 841),(828, 884),(830, 904),(835, 359),(854, 892),(858, 883),(861, 913),(865, 908),
    (882, 896),(887, 909),(889, 897),(893, 903),(900, 916),(901, 917),(905, 921),(906, 671),
    (911, 912),(918, 922),(919, 928),(920, 929),(923, 902),(925, 931),(926, 933),(930, 932),
    (934, 927),(392, 431),(109, 97),(393, 622),(115, 805),(263, 258),(370, 404),(384, 118),
    (489, 121),(691, 721),(852, 935),(360, 493),(386, 417),(102, 336),(560, 554),(851, 110),
    (99, 308),(898, 848),(936, 946),(367, 657),(424, 300),(687, 950),(704, 270),(924, 121),
    (107, 270),(409, 448),(583, 108),(867, 788),(103, 685),(99, 833),(114, 104),(269, 669),
    (324, 453),(406, 547),(961, 450),(295, 547),(307, 483),(439, 301),(463, 560),(292, 624),
    (517, 962),(608, 432),(840, 960),(949, 965),(396, 368),(480, 756),(563, 558),(564, 758),
    (607, 829),(728, 885),(844, 880),(846, 709),(942, 400),(974, 563),(977, 731),(979, 471),
    (115, 325),(116, 363),(310, 697),(368, 810),(373, 656),(414, 985),(532, 475),(532, 872),
    (565, 821),(566, 640),(652, 973),(654, 449),(658, 996),(845, 1000),(871, 989),(888, 964),
    (939, 972),(963, 984),(967, 983),(969, 1001),(971, 1002),(976, 1010),(978, 986),(980, 999),
    (981, 998),(987, 1006),(988, 1008),(990, 1004),(991, 1009),(995, 269),(997, 1013),(1005, 1017),
    (1007, 1014),(1011, 1022),(1012, 1019),(1015, 1016),(1018, 1023),(1020, 1028),(1024, 1027),(1025, 1029),
    (1026, 1021),(1031, 1032),(400, 777),(736, 398),(824, 953),(970, 747),(504, 539),(702, 670),
    (748, 699),(855, 954),(873, 618),(966, 692),(336, 576),(446, 863),(464, 478),(466, 705),
    (473, 1046),(528, 542),(542, 566),(558, 1048),(619, 831),(725, 994),(763, 982),(785, 1042),
    (802, 955),(866, 1047),(940, 1038),(1030, 941),(1034, 371),(1036, 718),(1037, 1056),(1039, 1050),
    (1040, 1053),(1045, 1057),(1052, 1055),(1054, 1058),(1063, 1049),(1065, 1051),(1066, 1061),(1067, 1070),
    (343, 776),(672, 260),(1035, 572),(1059, 1033),(310, 533),(753, 350),(339, 1069),(947, 876),
    (875, 1071),(600, 853),(659, 545),(544, 261),(1043, 405),(1060, 1080),(1064, 944),(102, 494),
    (568, 1075),(827, 518),(421, 856),(794, 711),(503, 737),(742, 99),(294, 937),(428, 633),
    (1044, 668),(110, 101),(307, 605),(712, 956),(280, 801),(288, 300),(291, 426),(401, 877),
    (653, 365),(720, 1101),(864, 1105),(432, 847),(449, 1099),(453, 768),(726, 1107),(1072, 264),
    (1091, 683),(1104, 1108),(1113, 1111),(106, 745),(115, 267),(258, 599),(281, 383),(404, 517),
    (487, 730),(564, 910),(567, 1094),(675, 735),(733, 1123),(780, 299),(795, 1102),(798, 825),
    (870, 1106),(948, 1098),(951, 1127),(958, 1096),(1076, 1126),(1079, 1110),(1081, 1125),(1084, 1124),
    (1095, 1117),(1100, 1120),(1109, 1119),(1112, 1128),(1121, 1137),(1122, 1131),(1129, 1136),(1130, 1133),
    (1132, 1143),(1135, 406),(1138, 1142),(1139, 1145),(1140, 1134),(1146, 1144),(281, 276),(992, 449),
    (105, 262),(339, 1114),(1147, 1092),(1154, 1141),(346, 260),(637, 268),(121, 256),(265, 399),
    (759, 434),(99, 273),(509, 366),(576, 303),(112, 101),(97, 627),(679, 421),(121, 115),
    (345, 491),(751, 548),(275, 270),(868, 303),(119, 275),(278, 271),(384, 804),(823, 1159),
    (592, 696),(103, 789),(108, 97),(698, 368),(1177, 259),(337, 116),(498, 303),(579, 260),
    (276, 103),(647, 628),(503, 296),(112, 336),(479, 385),(746, 270),(108, 111),(115, 97),
    (110, 459),(769, 302),(409, 1160),(281, 386),(968, 434),(103, 111),(358, 109),(108, 259),
    (354, 423),(447, 569),(1116, 108),(538, 435),(571, 326),(283, 303),(701, 32),(1087, 116),
    (273, 270),(261, 271),(952, 114),(341, 1188),(494, 272),(1207, 587),(256, 334),(109, 333),
    (299, 109),(665, 1182),(813, 365),(119, 266),(112, 389),(276, 271),(1220, 110),(299, 264),
    (285, 34),(116, 302),(279, 110),(357, 103),(341, 1203),(378, 352),(281, 118),(289, 270),
    (1068, 1228),(332, 32),(1153, 1211),(325, 99),(341, 1149),(109, 506),(588, 264),(269, 258),
    (1232, 1085),(304, 103),(1074, 490),(1082, 469),(98, 313),(1155, 1236),(316, 464),(799, 308),
    (693, 273),(103, 114),(572, 102),(360, 98),(273, 100),(281, 417),(283, 454),(269, 116),
    (283, 412),(1210, 329),(98, 114),(98, 270),(526, 282),(360, 112),(116, 293),(419, 275),
    (101, 112),(117, 287),(110, 548),(121, 277),(261, 116),(112, 117),(116, 379),(265, 272),
    (354, 108),(467, 272),(1093, 364),(259, 1247),(288, 103),(1276, 1205),(116, 506),(121, 262),
    (433, 275),(103, 318),(276, 370),(114, 1206),(305, 101),(312, 112),(398, 271),(46, 1157),
    (101, 336),(317, 108),(118, 276),(299, 360),(104, 101),(116, 309),(261, 256),(433, 350),
    (442, 369),(826, 318),(1219, 438),(100, 265),(104, 609),(261, 258),(279, 427),(289, 108),
    (452, 1273),(474, 410),(108, 412),(263, 1283),(269, 264),(277, 109),(457, 32),(614, 256),
    (327, 264),(265, 100),(265, 103),(325, 105),(310, 943),(313, 104),(453, 374),(102, 286),
    (816, 107),(109, 117),(556, 355),(110, 749),(345, 666),(277, 260),(310, 869),(348, 408),
    (1304, 959),(110, 526),(286, 32),(345, 115),(510, 456),(703, 264),(1161, 486),(1299, 105),
    (411, 114),(673, 891),(343, 116),(383, 600),(283, 396),(298, 738),(401, 275),(98, 1227),
    (115, 862),(304, 99),(1195, 369),(452, 838),(890, 1073),(1078, 765),(1295, 100),(1310, 1148),
    (1347, 1351),(433, 583),(444, 112),(765, 1086),(1041, 1328),(367, 375),(371, 1279),(497, 261),
    (1358, 272),(505, 264),(100, 279),(287, 266),(1362, 98),(269, 881),(314, 329),(1103, 1271),
    (1180, 1231),(531, 334),(393, 115),(1217, 100),(1317, 266),(1234, 1245),(354, 573),(488, 98),
    (408, 288),(1336, 32),(100, 313),(464, 121),(1174, 1229),(1375, 361),(116, 258),(308, 110),
    (1381, 1213),(739, 32),(380, 107),(384, 102),(1327, 1199),(1374, 262),(348, 1168),(1274, 603),
    (354, 445),(645, 267),(1176, 277),(1350, 104),(115, 301),(594, 1343),(915, 740),(104, 573),
    (295, 434),(344, 399),(101, 311),(782, 100),(298, 104),(1115, 434),(1243, 257),(1372, 1216),
    (287, 1118),(97, 118),(110, 293),(430, 1267),(648, 1291),(292, 738),(1395, 1212),(117, 281),
    (1399, 445),(105, 304),(273, 387),(343, 621),(541, 636),(1184, 1418),(1341, 116),(1419, 117),
    (101, 490),(102, 332),(342, 513),(97, 272),(103, 101),(276, 754),(1179, 1376),(100, 271),
    (259, 1410),(1332, 45),(112, 281),(1433, 1334),(316, 749),(492, 506),(1202, 482),(106, 111),
    (455, 116),(741, 260),(1238, 122),(299, 104),(689, 643),(1354, 1309),(114, 257),(283, 271),
    (1090, 270),(1187, 264),(32, 260),(106, 286),(109, 1437),(416, 266),(1089, 1192),(307, 374),
    (1198, 283),(1290, 117),(1339, 296),(288, 118),(304, 287),(1285, 686),(1421, 405),(119, 350),
    (259, 338),(444, 513),(1235, 1268),(381, 297),(1261, 1361),(299, 1152),(1166, 1156),(384, 99),
    (786, 1208),(1089, 478),(1346, 280),(1445, 1407),(298, 257),(437, 269),(1289, 108),(384, 629),
    (607, 862),(110, 502),(115, 796),(401, 369),(407, 347),(1408, 1480),(119, 114),(1296, 303),
    (372, 379),(373, 266),(407, 410),(418, 112),(782, 310),(100, 257),(633, 270),(39, 1446),
    (276, 614),(444, 1252),(647, 115),(673, 1165),(98, 276),(115, 881),(497, 289),(752, 318),
    (112, 105),(289, 105),(399, 32),(752, 312),(781, 256),(1502, 1241),(267, 115),(350, 352),
    (510, 116),(1281, 256),(332, 426),(357, 410),(612, 1364),(111, 115),(379, 118),(441, 115),
    (263, 314),(1272, 347),(384, 116),(1222, 256),(100, 1118),(280, 295),(348, 102),(1240, 1355),
    (109, 421),(258, 460),(312, 313),(1320, 318),(1530, 117),(111, 267),(1335, 303),(1342, 277),
    (100, 314),(119, 262),(739, 271),(1251, 1488),(1321, 372),(1442, 368),(1496, 1158),(121, 301),
    (1003, 599),(110, 261),(372, 1478),(594, 1509),(975, 684),(1540, 445),(98, 281),(1394, 1487),
    (279, 256),(316, 405),(456, 1428),(669, 959),(672, 299),(1163, 722),(1329, 1533),(100, 1167),
    (1559, 102),(108, 494),(115, 111),(266, 116),(281, 106),(367, 1514),(102, 108),(102, 350),
    (110, 318),(393, 497),(111, 332),(348, 97),(1041, 1555),(100, 318),(372, 104),(1078, 1201),
    (1344, 257),(116, 276),(278, 302),(608, 100),(1201, 1086),(1477, 1266),(110, 262),(1549, 1472),
    (99, 336),(281, 284),(283, 302),(357, 281),(437, 1330),(680, 366),(1275, 352),(1463, 482),
    (99, 107),(109, 257),(465, 1262),(416, 1288),(586, 296),(1263, 302),(1482, 1424),(101, 263),
    (108, 396),(109, 332),(115, 635),(260, 259),(269, 666),(99, 349),(103, 366),(276, 693),
    (1430, 593),(98, 389),(111, 98),(263, 1302),(298, 99),(1257, 1497),(1314, 357),(1588, 1546),
    (270, 295),(316, 99),(1492, 1429),(291, 639),(1589, 1569),(447, 838),(685, 1148),(1554, 509),
    (1621, 1622),(115, 463),(298, 101),(975, 329),(1539, 112),(327, 275),(103, 457),(110, 462),
    (116, 1308),(313, 296),(750, 890),(1244, 286),(1380, 1333),(1422, 643),(1459, 273),(1557, 326),
    (366, 112),(703, 1225),(1197, 276),(269, 301),(816, 379),(1162, 1223),(327, 438),(360, 311),
    (281, 1427),(290, 793),(353, 121),(355, 327),(1571, 762),(1574, 1651),(102, 379),(263, 271),
    (443, 260),(1466, 719),(1634, 1500),(108, 526),(287, 1386),(291, 308),(582, 405),(1660, 1662),
    (1661, 486),(386, 275),(1606, 32),(259, 118),(298, 378),(393, 342),(396, 294),(469, 299),
    (1373, 1352),(1638, 99),(311, 101),(342, 1493),(696, 256),(807, 264),(1650, 1495),(109, 121),
    (273, 271),(1378, 1469),(314, 112),(1249, 279),(1598, 1653),(287, 1184),(298, 264),(344, 1685),
    (1467, 699),(1677, 1278),(98, 1383),(263, 375),(305, 347),(938, 376),(1090, 454),(1613, 1464),
    (1687, 105),(473, 336),(786, 541),(1187, 115),(1237, 280),(110, 596),(291, 1646),(301, 294),
    (310, 722),(392, 272),(1440, 1545),(1483, 272),(1665, 601),(98, 457),(109, 299),(117, 100),
    (333, 256),(378, 347),(451, 1592),(1709, 115),(312, 290),(409, 550),(490, 260),(781, 277),
    (1250, 116),(258, 605),(310, 1168),(372, 359),(438, 307),(1331, 495),(98, 379),(354, 115),
    (612, 705),(1701, 32),(104, 1265),(115, 394),(807, 287),(1151, 1723),(1387, 1604),(119, 859),
    (259, 1297),(261, 105),(298, 107),(313, 438),(367, 289),(442, 1303),(592, 1740),(1083, 287),
    (1379, 368),(1735, 341),(102, 369),(372, 281),(608, 431),(1246, 271),(1284, 1088),(1370, 342),
    (259, 307),(488, 630),(597, 256),(1435, 264),(1449, 1452),(281, 103),(1179, 1609),(1465, 105),
    (1714, 394),(373, 300),(41, 271),(281, 109),(298, 435),(355, 103),(1326, 406),(1479, 574),
    (99, 121),(260, 577),(336, 262),(1461, 668),(1657, 258),(1696, 326),(1715, 293),(350, 108),
    (579, 1632),(700, 1312),(1202, 108),(1528, 1348),(312, 1322),(325, 593),(381, 283),(413, 1301),
    (570, 444),(601, 271),(1250, 264),(1269, 381),(1532, 627),(1776, 1702),(261, 110),(283, 121),
    (308, 410),(336, 1504),(436, 32),(476, 257),(1384, 622),(1793, 306),(111, 302),(116, 495),
    (118, 380),(358, 1393),(433, 313),(591, 259),(680, 440),(1800, 354),(103, 336),(105, 112),
    (276, 1167),(472, 1264),(799, 262),(1666, 554),(1780, 256),(1809, 399),(115, 109),(263, 701),
    (624, 859),(1420, 417),(1524, 256),(1607, 648),(1745, 1699),(1788, 1560),(1805, 1629),(98, 306),
    (118, 293),(515, 276),(689, 1165),(1083, 437),(1097, 355),(1690, 423),(102, 117),(118, 349),
    (382, 626),(1175, 1340),(1582, 45),(121, 638),(288, 372),(439, 115),(781, 108),(993, 812),
    (1178, 119),(1293, 1577),(1516, 264),(1664, 296),(116, 277),(118, 289),(295, 279),(328, 421),
    (331, 368),(442, 1626),(687, 1242),(703, 116),(1176, 471),(1803, 1152),(1850, 554),(281, 1164),
    (393, 259),(443, 1761),(617, 515),(915, 280),(1093, 459),(1371, 1648),(1468, 1683),(1717, 1857),
    (1804, 299),(259, 99),(259, 1801),(266, 310),(298, 296),(342, 287),(1162, 270),(1186, 442),
    (1226, 272),(1240, 1580),(1260, 1652),(1520, 271),(1725, 307),(1777, 404),(1806, 304),(97, 1172),
    (98, 1505),(105, 118),(325, 116),(629, 347),(1541, 260),(1789, 334),(1802, 107),(98, 261),
    (99, 383),(258, 1300),(280, 1360),(344, 263),(436, 412),(523, 270),(682, 260),(1640, 638),
    (1742, 405),(357, 719),(381, 275),(453, 1896),(464, 692),(568, 1596),(702, 1771),(1315, 1519),
    (1411, 1837),(1432, 1846),(1819, 554),(1838, 1765),(1848, 1368),(1856, 1724),(1863, 1455),(1876, 1902),
    (1911, 1899),(99, 279),(267, 373),(331, 1318),(331, 1368),(358, 280),(466, 1858),(601, 1529),
    (659, 287),(725, 1565),(1171, 1457),(1360, 1916),(1365, 1753),(1397, 585),(1444, 1923),(1451, 282),
    (1474, 1719),(1485, 1924),(1656, 1877),(1668, 754),(1703, 1904),(1704, 626),(1728, 1151),(1772, 1778),
    (1820, 1705),(1826, 1931),(1833, 619),(1894, 1935),(1905, 1919),(1906, 1940),(1908, 1921),(1909, 1941),
    (1912, 1938),(1918, 1930),(1926, 299),(1928, 1942),(1939, 1932),(1943, 1946),(1945, 1944),(1947, 1948),
    (306, 103),(455, 711),(587, 310),(1230, 101),(1242, 1603),(1810, 100),(1832, 295),(1956, 1958),
    (265, 256),(314, 684),(473, 117),(695, 357),(731, 1318),(733, 294),(1326, 293),(1456, 1412),
    (1507, 1721),(1562, 301),(1601, 457),(1658, 1490),(1748, 1953),(1767, 295),(1811, 670),(1972, 1964),
    (45, 1237),(99, 1265),(107, 101),(612, 1413),(1257, 1822),(1292, 276),(1371, 602),(1475, 256),
    (1537, 548),(1670, 1974),(1681, 1976),(1769, 1973),(1787, 1890),(1825, 1969),(1959, 1968),(1965, 1783),
    (1980, 1985),(1987, 1499),(1988, 1992),(1990, 1991),(1994, 1993),(309, 1259),(343, 99),(380, 114),
    (408, 1312),(1486, 283),(1512, 286),(1747, 375),(1901, 1949),(1995, 1915),(455, 1808),(1097, 309),
    (1258, 295),(1388, 609),(1498, 420),(1879, 265),(1996, 1849),(383, 32),(568, 2005),(638, 110),
    (1185, 307),(1708, 1348),(101, 603),(105, 348),(109, 684),(116, 119),(121, 45),(317, 441),
    (1277, 1618),(1367, 1453),(1619, 1369),(1784, 549),(1841, 435),(1954, 1170),(98, 1494),(455, 267),
    (587, 298),(1402, 686),(97, 1506),(498, 281),(1630, 762),(1716, 476),(1982, 302),(103, 394),
    (104, 638),(108, 354),(276, 105),(304, 109),(312, 1324),(613, 1986),(742, 1322),(1074, 112),
]


def bpe_encode(text):
    """BPE encoder: lowercase text to subword token IDs."""
    seq = [ord(c.lower()) if c.upper() != c else ord(c) for c in text]
    # lowercase ASCII
    seq2 = []
    for ch in text:
        b = ord(ch)
        if 65 <= b <= 90:  # A-Z
            b = b - 65 + 97
        seq2.append(b)
    seq = seq2

    for m_idx in range(BPE_MERGES):
        left, right = BPE_TABLE[m_idx]
        new_id = 256 + m_idx
        j = 0
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
    return seq


# Precomputed BPE encoding of each vocab word (for generation)
vocab_bpe = {}     # word_index -> list of BPE token IDs
vocab_bpe_len = {} # word_index -> length

def init_vocab_bpe():
    for i in range(V):
        ids = bpe_encode(VOCAB[i])
        vocab_bpe[i] = ids
        vocab_bpe_len[i] = len(ids)


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

    def load_from_bytes(self, data, offset):
        o = offset
        n = D*D
        self.wr = list(struct.unpack_from(f"{n}f", data, o)); o += n*4
        n = D
        self.rms = list(struct.unpack_from(f"{n}f", data, o)); o += n*4
        n = D*M
        self.w_gate = list(struct.unpack_from(f"{n}f", data, o)); o += n*4
        self.w_up = list(struct.unpack_from(f"{n}f", data, o)); o += n*4
        n = M*D
        self.w_down = list(struct.unpack_from(f"{n}f", data, o)); o += n*4
        return o


class Penelope:
    """12 learned steps + split embeddings (embed_in for BPE, embed_out for vocab). ~14M params."""

    def __init__(self):
        scale_bpe = math.sqrt(2.0 / BPE_VOCAB)
        scale_v = math.sqrt(2.0 / V)
        self.embed_in  = [randn() * scale_bpe for _ in range(BPE_VOCAB * D)]   # E_in[BPE_VOCAB, D]
        self.embed_out = [randn() * scale_v for _ in range(V * D)]             # E_out[V, D]
        self.steps = [StepWeights() for _ in range(STEPS)]

    def param_count(self):
        return BPE_VOCAB * D + V * D + sum(s.param_count() for s in self.steps)

    def get_embed_in(self, idx):
        """Get input embedding vector for BPE token idx."""
        return self.embed_in[idx * D:(idx + 1) * D]

    def get_embed_out(self, idx):
        """Get output embedding vector for word idx."""
        return self.embed_out[idx * D:(idx + 1) * D]

    def pool_context(self, bpe_ids):
        """Average input embedding of BPE token IDs."""
        if not bpe_ids:
            return zeros(D)
        ctx = zeros(D)
        for bid in bpe_ids:
            e = self.get_embed_in(bid)
            ctx = vadd(ctx, e)
        return vscale(ctx, 1.0 / len(bpe_ids))

    def forward_step(self, bpe_ids, step_idx):
        """One step: BPE context → logits[V]. Returns logits (pre-softmax)."""
        sw = self.steps[step_idx]
        ctx = self.pool_context(bpe_ids)

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

        # Logits = E_out @ out (separate output embed)
        logits = matmul_mv(self.embed_out, out, V, D)
        return logits

    def save(self, path):
        """Save all weights to binary file (v2 format)."""
        with open(path, "wb") as f:
            # v2 header: magic, BPE_VOCAB, V, D, M, STEPS
            f.write(struct.pack("iiiiii", 0x50454E32, BPE_VOCAB, V, D, M, STEPS))
            for v in self.embed_in:
                f.write(struct.pack("f", v))
            for v in self.embed_out:
                f.write(struct.pack("f", v))
            for s in self.steps:
                for v in s.params():
                    f.write(struct.pack("f", v))
        total = self.param_count()
        expected = 24 + total * 4
        actual = os.path.getsize(path)
        ok = "OK" if actual == expected else "SIZE MISMATCH!"
        print(f"  saved {path}: {total} params ({actual/1e6:.1f}MB) [{ok}]")

    def load(self, path):
        """Load weights from binary file. Supports v2 (magic 0x50454E32) and v1 migration."""
        with open(path, "rb") as f:
            data = f.read()

        o = 0
        magic = struct.unpack_from("i", data, o)[0]; o += 4

        if magic == 0x50454E32:
            # v2 format
            hdr = struct.unpack_from("iiiii", data, o); o += 20
            bv, nw, dd, mm, ss = hdr
            assert bv == BPE_VOCAB and nw == V and dd == D and mm == M and ss == STEPS, \
                f"v2 config mismatch: BV={bv} V={nw} D={dd} M={mm} S={ss}"
            ein_sz = BPE_VOCAB * D
            eout_sz = V * D
            self.embed_in = list(struct.unpack_from(f"{ein_sz}f", data, o)); o += ein_sz * 4
            self.embed_out = list(struct.unpack_from(f"{eout_sz}f", data, o)); o += eout_sz * 4
            for s in self.steps:
                o = s.load_from_bytes(data, o)
            print(f"  loaded v2 {path}: {self.param_count()} params")
        else:
            # v1 format: magic was actually header[0]=V
            hdr = struct.unpack_from("iii", data, o); o += 12
            assert magic == V and hdr[0] == D and hdr[1] == M and hdr[2] == STEPS, \
                f"v1 config mismatch: V={magic} D={hdr[0]} M={hdr[1]} S={hdr[2]}"
            # read old embed -> embed_out, init embed_in randomly
            eout_sz = V * D
            self.embed_out = list(struct.unpack_from(f"{eout_sz}f", data, o)); o += eout_sz * 4
            scale_bpe = math.sqrt(2.0 / BPE_VOCAB)
            self.embed_in = [randn() * scale_bpe for _ in range(BPE_VOCAB * D)]
            for s in self.steps:
                o = s.load_from_bytes(data, o)
            print(f"  WARNING: migrated v1 weights -> v2 (embed_in initialized randomly)")
            print(f"  loaded v1 {path}: {self.param_count()} params (embed_out from v1, embed_in new)")


# ═══════════════════════════════════════════════════════════════
# BPE INPUT — stem + greedy longest vocab match
#
# Three-stage tokenizer for arbitrary text:
#   1. Exact vocab match     ("fire" → fire)
#   2. Suffix stripping       ("burning" → burn, "created" → create)
#   3. Greedy decomposition   ("heartbreak" → heart + break)
#
# The 1984 vocab words ARE the BPE token vocabulary.
# Greedy longest-match IS BPE encoding.
# ═══════════════════════════════════════════════════════════════

SUFFIXES = [
    "ting","ning","ring","ling","ding","ping","bing","ging","ming","king",
    "sing","zing",
    "ing","ment","ness","tion","sion","able","ible","ence","ance",
    "eous","ious","ful","less","ize","ise","ous","ive","ity",
    "ly","er","ed","est","al","en","es","s",
]

VOCAB_LENS = [len(w) for w in VOCAB]


def try_stem(word):
    """Strip suffix, try exact match, stem+'e', doubled consonant removal."""
    wlen = len(word)
    for sfx in SUFFIXES:
        slen = len(sfx)
        if wlen <= slen + 2:
            continue
        if not word.endswith(sfx):
            continue
        stem = word[:wlen - slen]
        # exact stem
        if stem in VOCAB_IDX:
            return VOCAB_IDX[stem]
        # stem + 'e' (creat→create, danc→dance)
        stem_e = stem + "e"
        if stem_e in VOCAB_IDX:
            return VOCAB_IDX[stem_e]
        # doubled consonant (runn→run, swimm→swim)
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            stem_short = stem[:-1]
            if stem_short in VOCAB_IDX:
                return VOCAB_IDX[stem_short]
    return -1


def greedy_vocab_match(word):
    """Greedy longest vocab match within a word. Returns list of IDs."""
    ids = []
    pos = 0
    wlen = len(word)
    while pos < wlen:
        best, best_len = -1, 0
        for v in range(V):
            vl = VOCAB_LENS[v]
            if vl <= best_len or vl > wlen - pos:
                continue
            if word[pos:pos + vl] == VOCAB[v]:
                best, best_len = v, vl
        if best >= 0 and best_len >= 3:
            ids.append(best)
            pos += best_len
        else:
            pos += 1
    return ids


def tokenize_vocab(text):
    """Three-stage BPE: exact → stem → greedy vocab decomposition."""
    words = re.findall(r"[a-z]+", text.lower())
    ids = []
    for w in words:
        if len(w) < 2 or w in STOP:
            continue
        # 1. exact vocab match
        if w in VOCAB_IDX:
            ids.append(VOCAB_IDX[w])
            continue
        # 2. stem + match
        idx = try_stem(w)
        if idx >= 0:
            ids.append(idx)
            continue
        # 3. greedy longest vocab match (BPE decomposition)
        for sub_id in greedy_vocab_match(w):
            if not ids or ids[-1] != sub_id:
                ids.append(sub_id)
    return ids


def tokenize_for_training(text):
    """Dual tokenization: returns (word_ids, word_bpe) where
    word_ids[i] = vocab ID, word_bpe[i] = list of BPE token IDs for the word."""
    words = re.findall(r"[a-z]+", text.lower())
    word_ids = []
    word_bpe = []
    for w in words:
        if len(w) < 2 or w in STOP:
            continue
        # get vocab ID
        vid = -1
        if w in VOCAB_IDX:
            vid = VOCAB_IDX[w]
        else:
            vid = try_stem(w)
            if vid < 0:
                subs = greedy_vocab_match(w)
                if subs:
                    vid = subs[0]
        if vid < 0:
            continue
        # get BPE encoding
        bpe_ids = bpe_encode(w)
        word_ids.append(vid)
        word_bpe.append(bpe_ids)
    return word_ids, word_bpe


# ═══════════════════════════════════════════════════════════════
# TRAINING — next-word prediction, step s predicts word[s+1]
#
# Dual tokenization: BPE for input context, 1984-vocab for targets.
# ═══════════════════════════════════════════════════════════════

def train(model, data_path, steps=5000, lr=3e-4):
    """Train on text corpus. Dual tokenization: BPE for input, vocab IDs for targets."""
    with open(data_path, "r") as f:
        text = f.read()

    # Dual tokenization: get both vocab IDs and BPE tokens per word
    word_ids, word_bpe = tokenize_for_training(text)
    if len(word_ids) < STEPS + 2:
        print(f"  corpus too small: {len(word_ids)} words (need {STEPS+2}+)")
        return

    print(f"  corpus: {len(text)} chars -> {len(word_ids)} vocab words")
    print(f"  model: {model.param_count():,} params ({model.param_count()*4/1e6:.1f}MB f32)")
    print(f"  training: {steps} steps, lr={lr:.1e}")

    window = STEPS + 1  # 13 words: 1 seed + 12 targets
    best_loss = float("inf")

    for step in range(1, steps + 1):
        # random window from corpus
        start = random.randint(0, len(word_ids) - window)

        total_loss = 0.0

        # each of 12 steps predicts next word
        for s in range(STEPS):
            target = word_ids[start + s + 1]

            # collect BPE tokens from context words
            ctx_bpe = []
            for w in range(s + 1):
                ctx_bpe.extend(word_bpe[start + w])

            logits = model.forward_step(ctx_bpe, s)
            probs = softmax(logits)
            p = probs[target]
            if p < 1e-10:
                p = 1e-10
            total_loss -= math.log(p)

            # gradient: d_logits = probs - one_hot(target)
            d_logits = list(probs)
            d_logits[target] -= 1.0

            sw = model.steps[s]
            ctx = model.pool_context(ctx_bpe)

            # reconstruct forward
            query = matmul_mv(sw.wr, ctx, D, D)
            query_n = rmsnorm(query, sw.rms, D)
            gate = matmul_mv(sw.w_gate, query_n, M, D)
            up = matmul_mv(sw.w_up, query_n, M, D)
            swiglu = [silu(gate[i]) * up[i] for i in range(M)]
            hidden = matmul_mv(sw.w_down, swiglu, D, M)
            out = vadd(query_n, hidden)

            # d_out from embed_out (separate output embed)
            d_out = zeros(D)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                ev = model.get_embed_out(v)
                for j in range(D):
                    d_out[j] += d_logits[v] * ev[j]

            # update embed_out (SGD)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                base = v * D
                for j in range(D):
                    model.embed_out[base + j] -= lr * d_logits[v] * out[j]

            # d_hidden (residual: d_out goes to both query_n and hidden path)
            d_hidden = list(d_out)

            # backprop through w_down: d_swiglu = W_down^T @ d_hidden
            d_swiglu = matmul_mtv(sw.w_down, d_hidden, D, M)
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

            # approx RMSNorm backward
            ss = sum(v * v for v in query) / D + 1e-5
            inv = 1.0 / math.sqrt(ss)
            d_query = [d_qn[i] * sw.rms[i] * inv for i in range(D)]

            # update Wr and get d_ctx
            d_ctx = zeros(D)
            for i in range(D):
                if abs(d_query[i]) < 1e-8:
                    continue
                for j in range(D):
                    sw.wr[i * D + j] -= lr * d_query[i] * ctx[j]
                    d_ctx[j] += d_query[i] * sw.wr[i * D + j]

            # backprop d_ctx through pool_context to embed_in
            inv_n = 1.0 / max(len(ctx_bpe), 1)
            for bid in ctx_bpe:
                base = bid * D
                for j in range(D):
                    model.embed_in[base + j] -= lr * d_ctx[j] * inv_n

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

    # BPE token buffer for context — grows as words are added
    bpe_buf = list(vocab_bpe.get(seed, bpe_encode(VOCAB[seed])))

    for step in range(STEPS):
        field.update_chambers(step)
        field.prophecy_age += 1

        # learned logits — uses BPE context
        logits = model.forward_step(bpe_buf, step)

        # Dario field overlay (still uses word-level chain for co-occurrence)
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

        # append picked word's BPE tokens to buffer
        pick_bpe = vocab_bpe.get(pick, bpe_encode(VOCAB[pick]))
        bpe_buf.extend(pick_bpe)

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

    init_vocab_bpe()
    model = Penelope()
    field = DarioField()

    print()
    print(f"  penelope — 1984 words, {STEPS} steps, Dario Equation")
    print(f"  {model.param_count():,} trainable params ({model.param_count()*4/1e6:.1f}MB f32)")
    print(f"  BPE input: {BPE_VOCAB} subword tokens")
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
