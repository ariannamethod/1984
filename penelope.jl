# penelope.jl — v7 Resonance engine. 1984 words. Dario Equation.
# Julia version. Single file. No dependencies.
#
# 8-layer sequential transformer with multi-head attention, RoPE,
# RRPRAM resonance gates, and SwiGLU FFN. Dual tokenizer:
# BPE input (2048 subwords), word-level output (1984 words).
#
# Architecture per layer l:
#     h = rmsnorm(x, attn_norm_l)
#     qkv_out = MultiHeadAttention(h; wq_l, wk_l, wv_l, wo_l, RoPE)
#     rrp = h @ wr_l                            RRPRAM resonance
#     gate = softmax(gate_l[0], gate_l[1])
#     x = x + gate[0]*qkv_out + gate[1]*rrp    gated residual
#     h2 = rmsnorm(x, ffn_norm_l)
#     x = x + SwiGLU(h2; w_gate_l, w_up_l, w_down_l)  residual
#
# After 8 layers:
#     logits = rmsnorm(x, final_norm) @ lm_head^T
#     word_score(w) = mean(logits[bpe_tokens(w)]) + DarioField
#
#   score(w) = B + alpha*H + beta*F + gamma*A + T   (Dario Equation)
#
# By Arianna Method. הרזוננס לא נשבר

using Printf

# ═══════════════════════════════════════════════════════════════
# 1984 WORDS
# ═══════════════════════════════════════════════════════════════

const VOCAB = [
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


const V = length(VOCAB)  # 1990 (1984 canonical + 6 overlapping entries)
const DIM       = 448
const HDIM      = 896       # DIM * 2, SwiGLU hidden
const N_HEADS   = 7
const HEAD_DIM  = 64        # DIM / N_HEADS
const N_LAYERS  = 8         # sequential transformer layers
const MAX_SEQ   = 256
const NWORDS    = 1984

const BPE_VOCAB  = 2048
const BPE_MERGES = 1792

const MAX_BPE_SEQ = 16384
const MAX_EXT_VOCAB = 4096
const GEN_STEPS = 12

# BPE merge table — 1792 merges, learned from 2301588 bytes of English text
# Each row is (left, right) pair. Merge i produces token 256+i.
# Values are 0-based token IDs (bytes 0-255, then merged tokens 256+).
const BPE_TABLE = [
    (115, 32),   # 256: "s" + " " -> "s "
    (101, 32),   # 257: "e" + " " -> "e "
    (46, 32),    # 258: "." + " " -> ". "
    (105, 110),  # 259: "i" + "n" -> "in"
    (105, 256),  # 260: "i" + "s " -> "is "
    (101, 114),  # 261: "e" + "r" -> "er"
    (111, 110),  # 262: "o" + "n" -> "on"
    (116, 104),  # 263: "t" + "h" -> "th"
    (116, 32),   # 264: "t" + " " -> "t "
    (101, 110),  # 265: "e" + "n" -> "en"
    (97, 110),   # 266: "a" + "n" -> "an"
    (116, 105),  # 267: "t" + "i" -> "ti"
    (104, 260),  # 268: "h" + "is " -> "his "
    (101, 115),  # 269: "e" + "s" -> "es"
    (121, 32),   # 270: "y" + " " -> "y "
    (258, 268),  # 271: ". " + "his " -> ". his "
    (100, 32),   # 272: "d" + " " -> "d "
    (111, 114),  # 273: "o" + "r" -> "or"
    (259, 103),  # 274: "in" + "g" -> "ing"
    (97, 114),   # 275: "a" + "r" -> "ar"
    (97, 108),   # 276: "a" + "l" -> "al"
    (274, 32),   # 277: "ing" + " " -> "ing "
    (267, 262),  # 278: "ti" + "on" -> "tion"
    (111, 117),  # 279: "o" + "u" -> "ou"
    (101, 256),  # 280: "e" + "s " -> "es "
    (114, 101),  # 281: "r" + "e" -> "re"
    (111, 32),   # 282: "o" + " " -> "o "
    (105, 116),  # 283: "i" + "t" -> "it"
    (97, 116),   # 284: "a" + "t" -> "at"
    (58, 32),    # 285: ":" + " " -> ": "
    (111, 109),  # 286: "o" + "m" -> "om"
    (115, 116),  # 287: "s" + "t" -> "st"
    (100, 105),  # 288: "d" + "i" -> "di"
    (101, 108),  # 289: "e" + "l" -> "el"
    (104, 97),   # 290: "h" + "a" -> "ha"
    (114, 269),  # 291: "r" + "es" -> "res"
    (112, 261),  # 292: "p" + "er" -> "per"
    (261, 32),   # 293: "er" + " " -> "er "
    (263, 257),  # 294: "th" + "e " -> "the "
    (266, 272),  # 295: "an" + "d " -> "and "
    (278, 32),   # 296: "tion" + " " -> "tion "
    (111, 119),  # 297: "o" + "w" -> "ow"
    (97, 99),    # 298: "a" + "c" -> "ac"
    (105, 115),  # 299: "i" + "s" -> "is"
    (266, 32),   # 300: "an" + " " -> "an "
    (44, 32),    # 301: "," + " " -> ", "
    (39, 256),   # 302: "'" + "s " -> "'s "
    (276, 32),   # 303: "al" + " " -> "al "
    (108, 105),  # 304: "l" + "i" -> "li"
    (265, 99),   # 305: "en" + "c" -> "enc"
    (114, 97),   # 306: "r" + "a" -> "ra"
    (116, 282),  # 307: "t" + "o " -> "to "
    (117, 114),  # 308: "u" + "r" -> "ur"
    (101, 272),  # 309: "e" + "d " -> "ed "
    (105, 109),  # 310: "i" + "m" -> "im"
    (102, 102),  # 311: "f" + "f" -> "ff"
    (101, 120),  # 312: "e" + "x" -> "ex"
    (101, 99),   # 313: "e" + "c" -> "ec"
    (101, 109),  # 314: "e" + "m" -> "em"
    (102, 32),   # 315: "f" + " " -> "f "
    (102, 273),  # 316: "f" + "or" -> "for"
    (114, 111),  # 317: "r" + "o" -> "ro"
    (101, 116),  # 318: "e" + "t" -> "et"
    (10, 10),    # 319
    (97, 285),   # 320
    (113, 285),  # 321
    (10, 320),   # 322
    (46, 319),   # 323
    (323, 321),  # 324
    (117, 110),  # 325: "u" + "n" -> "un"
    (97, 32),    # 326: "a" + " " -> "a "
    (117, 108),  # 327: "u" + "l" -> "ul"
    (101, 118),  # 328: "e" + "v" -> "ev"
    (265, 264),  # 329: "en" + "t " -> "ent "
    (290, 264),  # 330: "ha" + "t " -> "hat "
    (119, 330),  # 331: "w" + "hat " -> "what "
    (105, 99),   # 332: "i" + "c" -> "ic"
    (265, 116),  # 333: "en" + "t" -> "ent"
    (275, 257),  # 334: "ar" + "e " -> "are "
    (284, 116),  # 335: "at" + "t" -> "att"
    (97, 115),   # 336: "a" + "s" -> "as"
    (103, 104),  # 337: "g" + "h" -> "gh"
    (97, 296),   # 338: "a" + "tion " -> "ation "
    (63, 322),   # 339
    (288, 311),  # 340: "di" + "ff" -> "diff"
    (105, 32),   # 341: "i" + " " -> "i "
    (115, 105),  # 342: "s" + "i" -> "si"
    (99, 262),   # 343: "c" + "on" -> "con"
    (110, 111),  # 344: "n" + "o" -> "no"
    (112, 291),  # 345: "p" + "res" -> "pres"
    (305, 257),  # 346: "enc" + "e " -> "ence "
    (101, 271),  # 347
    (100, 101),  # 348: "d" + "e" -> "de"
    (111, 108),  # 349: "o" + "l" -> "ol"
    (105, 108),  # 350: "i" + "l" -> "il"
    (286, 101),  # 351: "om" + "e" -> "ome"
    (283, 270),  # 352: "it" + "y " -> "ity "
    (263, 101),  # 353: "th" + "e" -> "the"
    (97, 98),    # 354: "a" + "b" -> "ab"
    (101, 100),  # 355: "e" + "d" -> "ed"
    (115, 351),  # 356: "s" + "ome" -> "some"
    (97, 103),   # 357: "a" + "g" -> "ag"
    (99, 286),   # 358: "c" + "om" -> "com"
    (275, 105),  # 359: "ar" + "i" -> "ari"
    (115, 117),  # 360: "s" + "u" -> "su"
    (262, 32),   # 361: "on" + " " -> "on "
    (340, 261),  # 362: "diff" + "er" -> "differ"
    (114, 117),  # 363: "r" + "u" -> "ru"
    (269, 115),  # 364: "es" + "s" -> "ess"
    (111, 315),  # 365: "o" + "f " -> "of "
    (97, 112),   # 366: "a" + "p" -> "ap"
    (119, 104),  # 367: "w" + "h" -> "wh"
    (262, 257),  # 368: "on" + "e " -> "one "
    (105, 114),  # 369: "i" + "r" -> "ir"
    (108, 270),  # 370: "l" + "y " -> "ly "
    (98, 101),   # 371: "b" + "e" -> "be"
    (115, 99),   # 372: "s" + "c" -> "sc"
    (109, 101),  # 373: "m" + "e" -> "me"
    (98, 257),   # 374: "b" + "e " -> "be "
    (265, 32),   # 375: "en" + " " -> "en "
    (259, 32),   # 376: "in" + " " -> "in "
    (115, 289),  # 377: "s" + "el" -> "sel"
    (267, 118),  # 378: "ti" + "v" -> "tiv"
    (114, 105),  # 379: "r" + "i" -> "ri"
    (111, 99),   # 380: "o" + "c" -> "oc"
    (115, 104),  # 381: "s" + "h" -> "sh"
    (267, 109),  # 382: "ti" + "m" -> "tim"
    (97, 109),   # 383: "a" + "m" -> "am"
    (112, 317),  # 384: "p" + "ro" -> "pro"
    (344, 264),  # 385: "no" + "t " -> "not "
    (113, 117),  # 386: "q" + "u" -> "qu"
    (105, 263),  # 387: "i" + "th" -> "ith"
    (263, 300),  # 388: "th" + "an " -> "than "
    (335, 261),  # 389: "att" + "er" -> "atter"
    (109, 273),  # 390: "m" + "or" -> "mor"
    (266, 110),  # 391: "an" + "n" -> "ann"
    (119, 273),  # 392: "w" + "or" -> "wor"
    (112, 111),  # 393: "p" + "o" -> "po"
    (101, 264),  # 394: "e" + "t " -> "et "
    (279, 108),  # 395: "ou" + "l" -> "oul"
    (121, 258),  # 396: "y" + ". " -> "y. "
    (395, 272),  # 397: "oul" + "d " -> "ould "
    (97, 278),   # 398: "a" + "tion" -> "ation"
    (267, 99),   # 399: "ti" + "c" -> "tic"
    (353, 270),  # 400: "the" + "y " -> "they "
    (119, 101),  # 401: "w" + "e" -> "we"
    (359, 391),  # 402: "ari" + "ann" -> "ariann"
    (402, 326),  # 403: "ariann" + "a " -> "arianna "
    (45, 32),    # 404: "-" + " " -> "- "
    (109, 32),   # 405: "m" + " " -> "m "
    (273, 32),   # 406: "or" + " " -> "or "
    (266, 99),   # 407: "an" + "c" -> "anc"
    (97, 100),   # 408: "a" + "d" -> "ad"
    (324, 331),  # 409
    (257, 260),  # 410: "e " + "is " -> "e is "
    (121, 279),  # 411: "y" + "ou" -> "you"
    (121, 271),  # 412
    (111, 263),  # 413: "o" + "th" -> "oth"
    (263, 277),  # 414: "th" + "ing " -> "thing "
    (119, 387),  # 415: "w" + "ith" -> "with"
    (112, 108),  # 416: "p" + "l" -> "pl"
    (276, 352),  # 417: "al" + "ity " -> "ality "
    (290, 112),  # 418: "ha" + "p" -> "hap"
    (102, 101),  # 419: "f" + "e" -> "fe"
    (101, 258),  # 420: "e" + ". " -> "e. "
    (105, 100),  # 421: "i" + "d" -> "id"
    (100, 97),   # 422: "d" + "a" -> "da"
    (279, 264),  # 423: "ou" + "t " -> "out "
    (117, 109),  # 424: "u" + "m" -> "um"
    (100, 117),  # 425: "d" + "u" -> "du"
    (104, 32),   # 426: "h" + " " -> "h "
    (337, 264),  # 427: "gh" + "t " -> "ght "
    (292, 105),  # 428: "per" + "i" -> "peri"
    (115, 271),  # 429
    (116, 114),  # 430: "t" + "r" -> "tr"
    (100, 256),  # 431: "d" + "s " -> "ds "
    (100, 277),  # 432: "d" + "ing " -> "ding "
    (99, 104),   # 433: "c" + "h" -> "ch"
    (109, 270),  # 434: "m" + "y " -> "my "
    (107, 32),   # 435: "k" + " " -> "k "
    (276, 108),  # 436: "al" + "l" -> "all"
    (100, 111),  # 437: "d" + "o" -> "do"
    (116, 256),  # 438: "t" + "s " -> "ts "
    (109, 389),  # 439: "m" + "atter" -> "matter"
    (103, 117),  # 440: "g" + "u" -> "gu"
    (118, 261),  # 441: "v" + "er" -> "ver"
    (115, 112),  # 442: "s" + "p" -> "sp"
    (105, 264),  # 443: "i" + "t " -> "it "
    (99, 111),   # 444: "c" + "o" -> "co"
    (108, 257),  # 445: "l" + "e " -> "le "
    (294, 115),  # 446
    (258, 403),  # 447
    (119, 397),  # 448: "w" + "ould " -> "would "
    (422, 270),  # 449: "da" + "y " -> "day "
    (411, 32),   # 450: "you" + " " -> "you "
    (98, 117),   # 451: "b" + "u" -> "bu"
    (258, 341),  # 452
    (99, 300),   # 453: "c" + "an " -> "can "
    (121, 302),  # 454: "y" + "'s " -> "y's "
    (112, 275),  # 455: "p" + "ar" -> "par"
    (116, 111),  # 456: "t" + "o" -> "to"
    (114, 297),  # 457: "r" + "ow" -> "row"
    (116, 306),  # 458: "t" + "ra" -> "tra"
    (269, 256),  # 459: "es" + "s " -> "ess "
    (110, 282),  # 460: "n" + "o " -> "no "
    (275, 32),   # 461: "ar" + " " -> "ar "
    (105, 427),  # 462: "i" + "ght " -> "ight "
    (258, 294),  # 463: ". " + "the " -> ". the "
    (328, 261),  # 464: "ev" + "er" -> "ever"
    (98, 318),   # 465: "b" + "et" -> "bet"
    (316, 32),   # 466: "for" + " " -> "for "
    (102, 259),  # 467: "f" + "in" -> "fin"
    (115, 262),  # 468: "s" + "on" -> "son"
    (114, 286),  # 469: "r" + "om" -> "rom"
    (97, 264),   # 470: "a" + "t " -> "at "
    (277, 295),  # 471
    (115, 107),  # 472: "s" + "k" -> "sk"
    (99, 108),   # 473: "c" + "l" -> "cl"
    (304, 102),  # 474: "li" + "f" -> "lif"
    (348, 112),  # 475: "de" + "p" -> "dep"
    (117, 115),  # 476: "u" + "s" -> "us"
    (265, 115),  # 477: "en" + "s" -> "ens"
    (110, 364),  # 478: "n" + "ess" -> "ness"
    (100, 282),  # 479: "d" + "o " -> "do "
    (101, 301),  # 480
    (312, 428),  # 481: "ex" + "peri" -> "experi"
    (108, 32),   # 482: "l" + " " -> "l "
    (356, 414),  # 483: "some" + "thing " -> "something "
    (292, 468),  # 484: "per" + "son" -> "person"
    (116, 117),  # 485: "t" + "u" -> "tu"
    (281, 99),   # 486: "re" + "c" -> "rec"
    (415, 423),  # 487: "with" + "out " -> "without "
    (99, 275),   # 488: "c" + "ar" -> "car"
    (116, 108),  # 489: "t" + "l" -> "tl"
    (112, 32),   # 490: "p" + " " -> "p "
    (342, 361),  # 491: "si" + "on " -> "sion "
    (105, 287),  # 492: "i" + "st" -> "ist"
    (103, 103),  # 493: "g" + "g" -> "gg"
    (111, 111),  # 494: "o" + "o" -> "oo"
    (308, 257),  # 495: "ur" + "e " -> "ure "
    (327, 116),  # 496: "ul" + "t" -> "ult"
    (259, 116),  # 497: "in" + "t" -> "int"
    (265, 267),  # 498: "en" + "ti" -> "enti"
    (261, 257),  # 499: "er" + "e " -> "ere "
    (297, 110),  # 500: "ow" + "n" -> "own"
    (263, 293),  # 501: "th" + "er " -> "ther "
    (297, 32),   # 502: "ow" + " " -> "ow "
    (335, 265),  # 503: "att" + "en" -> "atten"
    (115, 258),  # 504
    (112, 273),  # 505: "p" + "or" -> "por"
    (97, 107),   # 506: "a" + "k" -> "ak"
    (98, 317),   # 507: "b" + "ro" -> "bro"
    (390, 257),  # 508: "mor" + "e " -> "more "
    (263, 261),  # 509: "th" + "er" -> "ther"
    (97, 117),   # 510: "a" + "u" -> "au"
    (430, 270),  # 511: "tr" + "y " -> "try "
    (377, 102),  # 512: "sel" + "f" -> "self"
    (103, 110),  # 513: "g" + "n" -> "gn"
    (377, 315),  # 514: "sel" + "f " -> "self "
    (451, 264),  # 515: "bu" + "t " -> "but "
    (109, 111),  # 516: "m" + "o" -> "mo"
    (362, 329),  # 517: "differ" + "ent " -> "different "
    (279, 115),  # 518: "ou" + "s" -> "ous"
    (355, 271),  # 519
    (275, 272),  # 520: "ar" + "d " -> "ard "
    (118, 472),  # 521
    (425, 507),  # 522
    (522, 521),  # 523
    (109, 462),  # 524: "m" + "ight " -> "might "
    (287, 363),  # 525: "st" + "ru" -> "stru"
    (101, 103),  # 526: "e" + "g" -> "eg"
    (306, 501),  # 527
    (527, 388),  # 528
    (278, 303),  # 529: "tion" + "al " -> "tional "
    (269, 271),  # 530
    (278, 256),  # 531: "tion" + "s " -> "tions "
    (524, 374),  # 532
    (112, 304),  # 533: "p" + "li" -> "pli"
    (116, 297),  # 534: "t" + "ow" -> "tow"
    (534, 520),  # 535
    (356, 382),  # 536
    (102, 105),  # 537: "f" + "i" -> "fi"
    (263, 259),  # 538: "th" + "in" -> "thin"
    (536, 280),  # 539
    (511, 307),  # 540
    (273, 105),  # 541: "or" + "i" -> "ori"
    (367, 499),  # 542
    (292, 418),  # 543
    (325, 100),  # 544: "un" + "d" -> "und"
    (110, 100),  # 545: "n" + "d" -> "nd"
    (477, 338),  # 546
    (543, 256),  # 547
    (39, 264),   # 548
    (298, 426),  # 549
    (437, 280),  # 550
    (263, 269),  # 551: "th" + "es" -> "thes"
    (401, 375),  # 552
    (446, 546),  # 553
    (465, 552),  # 554
    (314, 111),  # 555: "em" + "o" -> "emo"
    (99, 101),   # 556: "c" + "e" -> "ce"
    (390, 110),  # 557: "mor" + "n" -> "morn"
    (508, 388),  # 558
    (100, 269),  # 559: "d" + "es" -> "des"
    (362, 346),  # 560
    (274, 271),  # 561
    (98, 413),   # 562: "b" + "oth" -> "both"
    (439, 256),  # 563
    (551, 257),  # 564
    (263, 470),  # 565: "th" + "at " -> "that "
    (400, 334),  # 566
    (121, 394),  # 567
    (339, 294),  # 568
    (448, 540),  # 569
    (474, 257),  # 570: "lif" + "e " -> "life "
    (458, 424),  # 571
    (381, 105),  # 572: "sh" + "i" -> "shi"
    (266, 100),  # 573: "an" + "d" -> "and"
    (314, 32),   # 574: "em" + " " -> "em "
    (115, 380),  # 575
    (575, 105),  # 576
    (562, 32),   # 577: "both" + " " -> "both "
    (261, 103),  # 578: "er" + "g" -> "erg"
    (484, 417),  # 579
    (339, 523),  # 580
    (118, 105),  # 581: "v" + "i" -> "vi"
    (104, 349),  # 582: "h" + "ol" -> "hol"
    (266, 103),  # 583: "an" + "g" -> "ang"
    (409, 260),  # 584
    (357, 257),  # 585: "ag" + "e " -> "age "
    (386, 269),  # 586: "qu" + "es" -> "ques"
    (112, 114),  # 587: "p" + "r" -> "pr"
    (358, 316),  # 588
    (105, 122),  # 589: "i" + "z" -> "iz"
    (104, 502),  # 590: "h" + "ow " -> "how "
    (111, 112),  # 591: "o" + "p" -> "op"
    (111, 441),  # 592: "o" + "ver" -> "over"
    (99, 296),   # 593
    (555, 529),  # 594
    (383, 257),  # 595: "am" + "e " -> "ame "
    (101, 119),  # 596: "e" + "w" -> "ew"
    (116, 354),  # 597
    (262, 103),  # 598: "on" + "g" -> "ong"
    (557, 277),  # 599
    (98, 105),   # 600: "b" + "i" -> "bi"
    (116, 261),  # 601: "t" + "er" -> "ter"
    (281, 408),  # 602: "re" + "ad" -> "read"
    (102, 327),  # 603: "f" + "ul" -> "ful"
    (99, 366),   # 604: "c" + "ap" -> "cap"
    (101, 549),  # 605
    (316, 109),  # 606: "for" + "m" -> "form"
    (277, 115),  # 607
    (475, 265),  # 608: "dep" + "en" -> "depen"
    (101, 302),  # 609
    (419, 289),  # 610: "fe" + "el" -> "feel"
    (99, 114),   # 611: "c" + "r" -> "cr"
    (512, 45),   # 612: "self" + "-" -> "self-"
    (345, 329),  # 613
    (119, 97),   # 614: "w" + "a" -> "wa"
    (102, 469),  # 615: "f" + "rom" -> "from"
    (580, 454),  # 616
    (263, 260),  # 617: "th" + "is " -> "this "
    (270, 260),  # 618
    (371, 277),  # 619: "be" + "ing " -> "being "
    (615, 32),   # 620: "from" + " " -> "from "
    (116, 259),  # 621: "t" + "in" -> "tin"
    (259, 264),  # 622: "in" + "t " -> "int "
    (279, 114),  # 623: "ou" + "r" -> "our"
    (109, 266),  # 624: "m" + "an" -> "man"
    (290, 256),  # 625: "ha" + "s " -> "has "
    (284, 257),  # 626: "at" + "e " -> "ate "
    (378, 257),  # 627: "tiv" + "e " -> "tive "
    (115, 257),  # 628: "s" + "e " -> "se "
    (98, 108),   # 629: "b" + "l" -> "bl"
    (116, 289),  # 630: "t" + "el" -> "tel"
    (287, 114),  # 631: "st" + "r" -> "str"
    (291, 112),  # 632: "res" + "p" -> "resp"
    (111, 100),  # 633: "o" + "d" -> "od"
    (284, 309),  # 634
    (261, 118),  # 635: "er" + "v" -> "erv"
    (103, 259),  # 636: "g" + "in" -> "gin"
    (34, 32),    # 637
    (101, 275),  # 638: "e" + "ar" -> "ear"
    (349, 117),  # 639: "ol" + "u" -> "olu"
    (115, 595),  # 640
    (312, 116),  # 641: "ex" + "t" -> "ext"
    (103, 306),  # 642: "g" + "ra" -> "gra"
    (407, 257),  # 643: "anc" + "e " -> "ance "
    (479, 450),  # 644
    (112, 97),   # 645: "p" + "a" -> "pa"
    (104, 289),  # 646: "h" + "el" -> "hel"
    (632, 262),  # 647
    (109, 329),  # 648: "m" + "ent " -> "ment "
    (110, 297),  # 649: "n" + "ow" -> "now"
    (265, 578),  # 650
    (516, 378),  # 651
    (550, 385),  # 652
    (382, 257),  # 653: "tim" + "e " -> "time "
    (109, 262),  # 654: "m" + "on" -> "mon"
    (467, 431),  # 655
    (392, 435),  # 656
    (282, 260),  # 657
    (102, 306),  # 658: "f" + "ra" -> "fra"
    (115, 121),  # 659: "s" + "y" -> "sy"
    (324, 590),  # 660
    (456, 282),  # 661
    (283, 256),  # 662: "it" + "s " -> "its "
    (259, 459),  # 663
    (328, 265),  # 664: "ev" + "en" -> "even"
    (312, 492),  # 665: "ex" + "ist" -> "exist"
    (342, 262),  # 666: "si" + "on" -> "sion"
    (102, 298),  # 667: "f" + "ac" -> "fac"
    (398, 256),  # 668
    (447, 655),  # 669
    (263, 574),  # 670
    (345, 333),  # 671
    (611, 299),  # 672
    (99, 281),   # 673: "c" + "re" -> "cre"
    (107, 257),  # 674
    (104, 293),  # 675: "h" + "er " -> "her "
    (266, 264),  # 676: "an" + "t " -> "ant "
    (292, 102),  # 677
    (505, 116),  # 678: "por" + "t" -> "port"
    (343, 102),  # 679
    (288, 115),  # 680: "di" + "s" -> "dis"
    (369, 32),   # 681: "ir" + " " -> "ir "
    (283, 514),  # 682
    (481, 305),  # 683
    (333, 271),  # 684
    (457, 256),  # 685
    (313, 116),  # 686: "ec" + "t" -> "ect"
    (584, 294),  # 687
    (108, 266),  # 688: "l" + "an" -> "lan"
    (292, 606),  # 689
    (260, 385),  # 690
    (660, 644),  # 691
    (121, 263),  # 692
    (105, 513),  # 693: "i" + "gn" -> "ign"
    (115, 308),  # 694: "s" + "ur" -> "sur"
    (688, 440),  # 695
    (538, 107),  # 696: "thin" + "k" -> "think"
    (677, 313),  # 697
    (112, 104),  # 698: "p" + "h" -> "ph"
    (293, 263),  # 699
    (340, 332),  # 700
    (279, 337),  # 701: "ou" + "gh" -> "ough"
    (373, 394),  # 702
    (440, 350),  # 703
    (488, 114),  # 704
    (99, 334),   # 705
    (115, 418),  # 706
    (415, 32),   # 707: "with" + " " -> "with "
    (349, 111),  # 708: "ol" + "o" -> "olo"
    (280, 307),  # 709
    (116, 265),  # 710: "t" + "en" -> "ten"
    (116, 370),  # 711
    (260, 104),  # 712
    (332, 303),  # 713
    (287, 259),  # 714
    (304, 674),  # 715
    (500, 32),   # 716: "own" + " " -> "own "
    (110, 313),  # 717
    (646, 112),  # 718
    (97, 259),   # 719: "a" + "in" -> "ain"
    (99, 97),    # 720: "c" + "a" -> "ca"
    (481, 346),  # 721
    (373, 288),  # 722
    (327, 461),  # 723
    (120, 105),  # 724: "x" + "i" -> "xi"
    (299, 301),  # 725
    (119, 259),  # 726: "w" + "in" -> "win"
    (537, 289),  # 727
    (581, 596),  # 728: "vi" + "ew" -> "view"
    (99, 379),   # 729
    (353, 681),  # 730
    (361, 331),  # 731
    (108, 598),  # 732: "l" + "ong" -> "long"
    (706, 280),  # 733
    (266, 724),  # 734
    (650, 270),  # 735
    (281, 108),  # 736: "re" + "l" -> "rel"
    (278, 258),  # 737
    (556, 112),  # 738
    (104, 310),  # 739: "h" + "im" -> "him"
    (280, 334),  # 740
    (651, 338),  # 741
    (360, 99),   # 742: "su" + "c" -> "suc"
    (115, 101),  # 743: "s" + "e" -> "se"
    (287, 284),  # 744: "st" + "at" -> "stat"
    (476, 264),  # 745
    (734, 318),  # 746
    (630, 482),  # 747
    (111, 311),  # 748: "o" + "ff" -> "off"
    (328, 293),  # 749
    (392, 107),  # 750: "wor" + "k" -> "work"
    (99, 266),   # 751: "c" + "an" -> "can"
    (358, 416),  # 752
    (102, 97),   # 753: "f" + "a" -> "fa"
    (299, 405),  # 754
    (436, 256),  # 755
    (413, 293),  # 756
    (623, 99),   # 757
    (586, 531),  # 758
    (105, 315),  # 759: "i" + "f " -> "if "
    (308, 277),  # 760
    (291, 262),  # 761
    (263, 32),   # 762
    (345, 346),  # 763
    (485, 281),  # 764: "tu" + "re" -> "ture"
    (452, 569),  # 765
    (708, 103),  # 766
    (372, 105),  # 767
    (610, 32),   # 768: "feel" + " " -> "feel "
    (571, 97),   # 769
    (279, 545),  # 770
    (298, 278),  # 771
    (455, 399),  # 772
    (116, 271),  # 773
    (559, 729),  # 774
    (116, 641),  # 775
    (525, 99),   # 776
    (381, 397),  # 777
    (283, 117),  # 778
    (103, 266),  # 779
    (98, 336),   # 780
    (107, 649),  # 781: "k" + "now" -> "know"
    (109, 259),  # 782
    (100, 760),  # 783
    (273, 779),  # 784
    (309, 376),  # 785
    (109, 314),  # 786
    (589, 280),  # 787
    (631, 284),  # 788
    (265, 117),  # 789
    (333, 370),  # 790
    (727, 272),  # 791
    (489, 396),  # 792
    (118, 257),  # 793
    (288, 486),  # 794
    (280, 102),  # 795
    (108, 101),  # 796: "l" + "e" -> "le"
    (772, 723),  # 797
    (274, 301),  # 798
    (115, 313),  # 799
    (291, 757),  # 800
    (328, 375),  # 801
    (356, 368),  # 802
    (119, 283),  # 803
    (425, 99),   # 804
    (639, 278),  # 805
    (774, 374),  # 806
    (104, 111),  # 807: "h" + "o" -> "ho"
    (266, 101),  # 808
    (717, 364),  # 809
    (366, 533),  # 810
    (588, 597),  # 811
    (115, 264),  # 812
    (419, 461),  # 813
    (775, 495),  # 814
    (809, 275),  # 815
    (109, 275),  # 816
    (310, 496),  # 817
    (817, 808),  # 818
    (104, 257),  # 819
    (274, 258),  # 820
    (695, 585),  # 821
    (310, 678),  # 822
    (510, 263),  # 823
    (662, 716),  # 824
    (664, 277),  # 825
    (358, 112),  # 826: "com" + "p" -> "comp"
    (343, 767),  # 827
    (376, 283),  # 828
    (818, 518),  # 829
    (324, 806),  # 830
    (803, 478),  # 831
    (582, 432),  # 832
    (259, 284),  # 833
    (325, 811),  # 834
    (98, 770),   # 835
    (732, 293),  # 836
    (525, 493),  # 837
    (98, 273),   # 838
    (460, 836),  # 839
    (109, 308),  # 840
    (280, 436),  # 841
    (333, 338),  # 842
    (509, 410),  # 843
    (544, 293),  # 844
    (822, 676),  # 845
    (837, 108),  # 846
    (100, 500),  # 847
    (272, 365),  # 848
    (355, 258),  # 849
    (362, 790),  # 850
    (371, 636),  # 851
    (463, 791),  # 852
    (766, 713),  # 853
    (834, 445),  # 854
    (274, 322),  # 855
    (498, 116),  # 856
    (97, 256),   # 857
    (642, 442),  # 858
    (105, 102),  # 859
    (288, 714),  # 860
    (710, 491),  # 861
    (635, 332),  # 862
    (778, 338),  # 863
    (99, 369),   # 864
    (784, 787),  # 865
    (99, 755),   # 866
    (102, 363),  # 867
    (298, 485),  # 868
    (393, 287),  # 869
    (420, 460),  # 870
    (604, 764),  # 871
    (694, 667),  # 872
    (700, 496),  # 873
    (744, 480),  # 874
    (258, 539),  # 875
    (269, 438),  # 876
    (101, 107),  # 877
    (331, 690),  # 878
    (363, 621),  # 879
    (372, 879),  # 880
    (39, 32),    # 881
    (267, 337),  # 882
    (277, 661),  # 883
    (301, 300),  # 884
    (309, 620),  # 885
    (541, 842),  # 886
    (814, 404),  # 887
    (860, 593),  # 888
    (886, 535),  # 889
    (45, 570),   # 890
    (284, 280),  # 891
    (295, 815),  # 892
    (380, 634),  # 893
    (602, 663),  # 894
    (625, 797),  # 895
    (792, 843),  # 896
    (878, 567),  # 897
    (107, 259),  # 898
    (406, 839),  # 899
    (443, 577),  # 900
    (483, 487),  # 901
    (528, 771),  # 902
    (535, 894),  # 903
    (553, 365),  # 904
    (553, 895),  # 905
    (613, 899),  # 906
    (617, 874),  # 907
    (682, 850),  # 908
    (715, 832),  # 909
    (761, 407),  # 910
    (783, 907),  # 911
    (800, 841),  # 912
    (828, 884),  # 913
    (830, 904),  # 914
    (835, 359),  # 915
    (854, 892),  # 916
    (858, 883),  # 917
    (861, 913),  # 918
    (865, 908),  # 919
    (882, 896),  # 920
    (887, 909),  # 921
    (889, 897),  # 922
    (893, 903),  # 923
    (900, 916),  # 924
    (901, 917),  # 925
    (905, 921),  # 926
    (906, 671),  # 927
    (911, 912),  # 928
    (918, 922),  # 929
    (919, 928),  # 930
    (920, 929),  # 931
    (923, 902),  # 932
    (925, 931),  # 933
    (926, 933),  # 934
    (930, 932),  # 935
    (934, 927),  # 936
    (392, 431),  # 937
    (109, 97),   # 938
    (393, 622),  # 939
    (115, 805),  # 940
    (263, 258),  # 941
    (370, 404),  # 942
    (384, 118),  # 943
    (489, 121),  # 944
    (691, 721),  # 945
    (852, 935),  # 946
    (360, 493),  # 947
    (386, 417),  # 948
    (102, 336),  # 949
    (560, 554),  # 950
    (851, 110),  # 951
    (99, 308),   # 952
    (898, 848),  # 953
    (936, 946),  # 954
    (367, 657),  # 955
    (424, 300),  # 956
    (687, 950),  # 957
    (704, 270),  # 958
    (924, 121),  # 959
    (107, 270),  # 960
    (409, 448),  # 961
    (583, 108),  # 962
    (867, 788),  # 963
    (103, 685),  # 964
    (99, 833),   # 965
    (114, 104),  # 966
    (269, 669),  # 967
    (324, 453),  # 968
    (406, 547),  # 969
    (961, 450),  # 970
    (295, 547),  # 971
    (307, 483),  # 972
    (439, 301),  # 973
    (463, 560),  # 974
    (292, 624),  # 975
    (517, 962),  # 976
    (608, 432),  # 977
    (840, 960),  # 978
    (949, 965),  # 979
    (396, 368),  # 980
    (480, 756),  # 981
    (563, 558),  # 982
    (564, 758),  # 983
    (607, 829),  # 984
    (728, 885),  # 985
    (844, 880),  # 986
    (846, 709),  # 987
    (942, 400),  # 988
    (974, 563),  # 989
    (977, 731),  # 990
    (979, 471),  # 991
    (115, 325),  # 992
    (116, 363),  # 993
    (310, 697),  # 994
    (368, 810),  # 995
    (373, 656),  # 996
    (414, 985),  # 997
    (532, 475),  # 998
    (532, 872),  # 999
    (565, 821),  # 1000
    (566, 640),  # 1001
    (652, 973),  # 1002
    (654, 449),  # 1003
    (658, 996),  # 1004
    (845, 1000), # 1005
    (871, 989),  # 1006
    (888, 964),  # 1007
    (939, 972),  # 1008
    (963, 984),  # 1009
    (967, 983),  # 1010
    (969, 1001), # 1011
    (971, 1002), # 1012
    (976, 1010), # 1013
    (978, 986),  # 1014
    (980, 999),  # 1015
    (981, 998),  # 1016
    (987, 1006), # 1017
    (988, 1008), # 1018
    (990, 1004), # 1019
    (991, 1009), # 1020
    (995, 269),  # 1021
    (997, 1013), # 1022
    (1005, 1017),# 1023
    (1007, 1014),# 1024
    (1011, 1022),# 1025
    (1012, 1019),# 1026
    (1015, 1016),# 1027
    (1018, 1023),# 1028
    (1020, 1028),# 1029
    (1024, 1027),# 1030
    (1025, 1029),# 1031
    (1026, 1021),# 1032
    (1031, 1032),# 1033
    (400, 777),  # 1034
    (736, 398),  # 1035
    (824, 953),  # 1036
    (970, 747),  # 1037
    (504, 539),  # 1038
    (702, 670),  # 1039
    (748, 699),  # 1040
    (855, 954),  # 1041
    (873, 618),  # 1042
    (966, 692),  # 1043
    (336, 576),  # 1044
    (446, 863),  # 1045
    (464, 478),  # 1046
    (466, 705),  # 1047
    (473, 1046), # 1048
    (528, 542),  # 1049
    (542, 566),  # 1050
    (558, 1048), # 1051
    (619, 831),  # 1052
    (725, 994),  # 1053
    (763, 982),  # 1054
    (785, 1042), # 1055
    (802, 955),  # 1056
    (866, 1047), # 1057
    (940, 1038), # 1058
    (1030, 941), # 1059
    (1034, 371), # 1060
    (1036, 718), # 1061
    (1037, 1056),# 1062
    (1039, 1050),# 1063
    (1040, 1053),# 1064
    (1045, 1057),# 1065
    (1052, 1055),# 1066
    (1054, 1058),# 1067
    (1063, 1049),# 1068
    (1065, 1051),# 1069
    (1066, 1061),# 1070
    (1067, 1070),# 1071
    (343, 776),  # 1072
    (672, 260),  # 1073
    (1035, 572), # 1074
    (1059, 1033),# 1075
    (310, 533),  # 1076
    (753, 350),  # 1077
    (339, 1069), # 1078
    (947, 876),  # 1079
    (875, 1071), # 1080
    (600, 853),  # 1081
    (659, 545),  # 1082
    (544, 261),  # 1083
    (1043, 405), # 1084
    (1060, 1080),# 1085
    (1064, 944), # 1086
    (102, 494),  # 1087
    (568, 1075), # 1088
    (827, 518),  # 1089
    (421, 856),  # 1090
    (794, 711),  # 1091
    (503, 737),  # 1092
    (742, 99),   # 1093
    (294, 937),  # 1094
    (428, 633),  # 1095
    (1044, 668), # 1096
    (110, 101),  # 1097
    (307, 605),  # 1098
    (712, 956),  # 1099
    (280, 801),  # 1100
    (288, 300),  # 1101
    (291, 426),  # 1102
    (401, 877),  # 1103
    (653, 365),  # 1104
    (720, 1101), # 1105
    (864, 1105), # 1106
    (432, 847),  # 1107
    (449, 1099), # 1108
    (453, 768),  # 1109
    (726, 1107), # 1110
    (1072, 264), # 1111
    (1091, 683), # 1112
    (1104, 1108),# 1113
    (1113, 1111),# 1114
    (106, 745),  # 1115
    (115, 267),  # 1116
    (258, 599),  # 1117
    (281, 383),  # 1118
    (404, 517),  # 1119
    (487, 730),  # 1120
    (564, 910),  # 1121
    (567, 1094), # 1122
    (675, 735),  # 1123
    (733, 1123), # 1124
    (780, 299),  # 1125
    (795, 1102), # 1126
    (798, 825),  # 1127
    (870, 1106), # 1128
    (948, 1098), # 1129
    (951, 1127), # 1130
    (958, 1096), # 1131
    (1076, 1126),# 1132
    (1079, 1110),# 1133
    (1081, 1125),# 1134
    (1084, 1124),# 1135
    (1095, 1117),# 1136
    (1100, 1120),# 1137
    (1109, 1119),# 1138
    (1112, 1128),# 1139
    (1121, 1137),# 1140
    (1122, 1131),# 1141
    (1129, 1136),# 1142
    (1130, 1133),# 1143
    (1132, 1143),# 1144
    (1135, 406), # 1145
    (1138, 1142),# 1146
    (1139, 1145),# 1147
    (1140, 1134),# 1148
    (1146, 1144),# 1149
    (281, 276),  # 1150
    (992, 449),  # 1151
    (105, 262),  # 1152
    (339, 1114), # 1153
    (1147, 1092),# 1154
    (1154, 1141),# 1155
    (346, 260),  # 1156
    (637, 268),  # 1157
    (121, 256),  # 1158
    (265, 399),  # 1159
    (759, 434),  # 1160
    (99, 273),   # 1161
    (509, 366),  # 1162
    (576, 303),  # 1163
    (112, 101),  # 1164
    (97, 627),   # 1165
    (679, 421),  # 1166
    (121, 115),  # 1167
    (345, 491),  # 1168
    (751, 548),  # 1169
    (275, 270),  # 1170
    (868, 303),  # 1171
    (119, 275),  # 1172
    (278, 271),  # 1173
    (384, 804),  # 1174
    (823, 1159), # 1175
    (592, 696),  # 1176
    (103, 789),  # 1177
    (108, 97),   # 1178
    (698, 368),  # 1179
    (1177, 259), # 1180
    (337, 116),  # 1181
    (498, 303),  # 1182
    (579, 260),  # 1183
    (276, 103),  # 1184
    (647, 628),  # 1185
    (503, 296),  # 1186
    (112, 336),  # 1187
    (479, 385),  # 1188
    (746, 270),  # 1189
    (108, 111),  # 1190
    (115, 97),   # 1191
    (110, 459),  # 1192
    (769, 302),  # 1193
    (409, 1160), # 1194
    (281, 386),  # 1195
    (968, 434),  # 1196
    (103, 111),  # 1197
    (358, 109),  # 1198
    (108, 259),  # 1199
    (354, 423),  # 1200
    (447, 569),  # 1201
    (1116, 108), # 1202
    (538, 435),  # 1203
    (571, 326),  # 1204
    (283, 303),  # 1205
    (701, 32),   # 1206
    (1087, 116), # 1207
    (273, 270),  # 1208
    (261, 271),  # 1209
    (952, 114),  # 1210
    (341, 1188), # 1211
    (494, 272),  # 1212
    (1207, 587), # 1213
    (256, 334),  # 1214
    (109, 333),  # 1215
    (299, 109),  # 1216
    (665, 1182), # 1217
    (813, 365),  # 1218
    (119, 266),  # 1219
    (112, 389),  # 1220
    (276, 271),  # 1221
    (1220, 110), # 1222
    (299, 264),  # 1223
    (285, 34),   # 1224
    (116, 302),  # 1225
    (279, 110),  # 1226
    (357, 103),  # 1227
    (341, 1203), # 1228
    (378, 352),  # 1229
    (281, 118),  # 1230
    (289, 270),  # 1231
    (1068, 1228),# 1232
    (332, 32),   # 1233
    (1153, 1211),# 1234
    (325, 99),   # 1235
    (341, 1149), # 1236
    (109, 506),  # 1237
    (588, 264),  # 1238
    (269, 258),  # 1239
    (1232, 1085),# 1240
    (304, 103),  # 1241
    (1074, 490), # 1242
    (1082, 469), # 1243
    (98, 313),   # 1244
    (1155, 1236),# 1245
    (316, 464),  # 1246
    (799, 308),  # 1247
    (693, 273),  # 1248
    (103, 114),  # 1249
    (572, 102),  # 1250
    (360, 98),   # 1251
    (273, 100),  # 1252
    (281, 417),  # 1253
    (283, 454),  # 1254
    (269, 116),  # 1255
    (283, 412),  # 1256
    (1210, 329), # 1257
    (98, 114),   # 1258
    (98, 270),   # 1259
    (526, 282),  # 1260
    (360, 112),  # 1261
    (116, 293),  # 1262
    (419, 275),  # 1263
    (101, 112),  # 1264
    (117, 287),  # 1265
    (110, 548),  # 1266
    (121, 277),  # 1267
    (261, 116),  # 1268
    (112, 117),  # 1269
    (116, 379),  # 1270
    (265, 272),  # 1271
    (354, 108),  # 1272
    (467, 272),  # 1273
    (1093, 364), # 1274
    (259, 1247), # 1275
    (288, 103),  # 1276
    (1276, 1205),# 1277
    (116, 506),  # 1278
    (121, 262),  # 1279
    (433, 275),  # 1280
    (103, 318),  # 1281
    (276, 370),  # 1282
    (114, 1206), # 1283
    (305, 101),  # 1284
    (312, 112),  # 1285
    (398, 271),  # 1286
    (46, 1157),  # 1287
    (101, 336),  # 1288
    (317, 108),  # 1289
    (118, 276),  # 1290
    (299, 360),  # 1291
    (104, 101),  # 1292
    (116, 309),  # 1293
    (261, 256),  # 1294
    (433, 350),  # 1295
    (442, 369),  # 1296
    (826, 318),  # 1297
    (1219, 438), # 1298
    (100, 265),  # 1299
    (104, 609),  # 1300
    (261, 258),  # 1301
    (279, 427),  # 1302
    (289, 108),  # 1303
    (452, 1273), # 1304
    (474, 410),  # 1305
    (108, 412),  # 1306
    (263, 1283), # 1307
    (269, 264),  # 1308
    (277, 109),  # 1309
    (457, 32),   # 1310
    (614, 256),  # 1311
    (327, 264),  # 1312
    (265, 100),  # 1313
    (265, 103),  # 1314
    (325, 105),  # 1315
    (310, 943),  # 1316
    (313, 104),  # 1317
    (453, 374),  # 1318
    (102, 286),  # 1319
    (816, 107),  # 1320
    (109, 117),  # 1321
    (556, 355),  # 1322
    (110, 749),  # 1323
    (345, 666),  # 1324
    (277, 260),  # 1325
    (310, 869),  # 1326
    (348, 408),  # 1327
    (1304, 959), # 1328
    (110, 526),  # 1329
    (286, 32),   # 1330
    (345, 115),  # 1331
    (510, 456),  # 1332
    (703, 264),  # 1333
    (1161, 486), # 1334
    (1299, 105), # 1335
    (411, 114),  # 1336
    (673, 891),  # 1337
    (343, 116),  # 1338
    (383, 600),  # 1339
    (283, 396),  # 1340
    (298, 738),  # 1341
    (401, 275),  # 1342
    (98, 1227),  # 1343
    (115, 862),  # 1344
    (304, 99),   # 1345
    (1195, 369), # 1346
    (452, 838),  # 1347
    (890, 1073), # 1348
    (1078, 765), # 1349
    (1295, 100), # 1350
    (1310, 1148),# 1351
    (1347, 1351),# 1352
    (433, 583),  # 1353
    (444, 112),  # 1354
    (765, 1086), # 1355
    (1041, 1328),# 1356
    (367, 375),  # 1357
    (371, 1279), # 1358
    (497, 261),  # 1359
    (1358, 272), # 1360
    (505, 264),  # 1361
    (100, 279),  # 1362
    (287, 266),  # 1363
    (1362, 98),  # 1364
    (269, 881),  # 1365
    (314, 329),  # 1366
    (1103, 1271),# 1367
    (1180, 1231),# 1368
    (531, 334),  # 1369
    (393, 115),  # 1370
    (1217, 100), # 1371
    (1317, 266), # 1372
    (1234, 1245),# 1373
    (354, 573),  # 1374
    (488, 98),   # 1375
    (408, 288),  # 1376
    (1336, 32),  # 1377
    (100, 313),  # 1378
    (464, 121),  # 1379
    (1174, 1229),# 1380
    (1375, 361), # 1381
    (116, 258),  # 1382
    (308, 110),  # 1383
    (1381, 1213),# 1384
    (739, 32),   # 1385
    (380, 107),  # 1386
    (384, 102),  # 1387
    (1327, 1199),# 1388
    (1374, 262), # 1389
    (348, 1168), # 1390
    (1274, 603), # 1391
    (354, 445),  # 1392
    (645, 267),  # 1393
    (1176, 277), # 1394
    (1350, 104), # 1395
    (115, 301),  # 1396
    (594, 1343), # 1397
    (915, 740),  # 1398
    (104, 573),  # 1399
    (295, 434),  # 1400
    (344, 399),  # 1401
    (101, 311),  # 1402
    (782, 100),  # 1403
    (298, 104),  # 1404
    (1115, 434), # 1405
    (1243, 257), # 1406
    (1372, 1216),# 1407
    (287, 1118), # 1408
    (97, 118),   # 1409
    (110, 293),  # 1410
    (430, 1267), # 1411
    (648, 1291), # 1412
    (292, 738),  # 1413
    (1395, 1212),# 1414
    (117, 281),  # 1415
    (1399, 445), # 1416
    (105, 304),  # 1417
    (273, 387),  # 1418
    (343, 621),  # 1419
    (541, 636),  # 1420
    (1184, 1418),# 1421
    (1341, 116), # 1422
    (1419, 117), # 1423
    (101, 490),  # 1424
    (102, 332),  # 1425
    (342, 513),  # 1426
    (97, 272),   # 1427
    (103, 101),  # 1428
    (276, 754),  # 1429
    (1179, 1376),# 1430
    (100, 271),  # 1431
    (259, 1410), # 1432
    (1332, 45),  # 1433
    (112, 281),  # 1434
    (1433, 1334),# 1435
    (316, 749),  # 1436
    (492, 506),  # 1437
    (1202, 482), # 1438
    (106, 111),  # 1439
    (455, 116),  # 1440
    (741, 260),  # 1441
    (1238, 122), # 1442
    (299, 104),  # 1443
    (689, 643),  # 1444
    (1354, 1309),# 1445
    (114, 257),  # 1446
    (283, 271),  # 1447
    (1090, 270), # 1448
    (1187, 264), # 1449
    (32, 260),   # 1450
    (106, 286),  # 1451
    (109, 1437), # 1452
    (416, 266),  # 1453
    (1089, 1192),# 1454
    (307, 374),  # 1455
    (1198, 283), # 1456
    (1290, 117), # 1457
    (1339, 296), # 1458
    (288, 118),  # 1459
    (304, 287),  # 1460
    (1285, 686), # 1461
    (1421, 405), # 1462
    (119, 350),  # 1463
    (259, 338),  # 1464
    (444, 513),  # 1465
    (1235, 1268),# 1466
    (381, 297),  # 1467
    (1261, 1361),# 1468
    (299, 1152), # 1469
    (1166, 1156),# 1470
    (384, 99),   # 1471
    (786, 1208), # 1472
    (1089, 478), # 1473
    (1346, 280), # 1474
    (1445, 1407),# 1475
    (298, 257),  # 1476
    (437, 269),  # 1477
    (1289, 108), # 1478
    (384, 629),  # 1479
    (607, 862),  # 1480
    (110, 502),  # 1481
    (115, 796),  # 1482
    (401, 369),  # 1483
    (407, 347),  # 1484
    (1408, 1480),# 1485
    (119, 114),  # 1486
    (1296, 303), # 1487
    (372, 379),  # 1488
    (373, 266),  # 1489
    (407, 410),  # 1490
    (418, 112),  # 1491
    (782, 310),  # 1492
    (100, 257),  # 1493
    (633, 270),  # 1494
    (39, 1446),  # 1495
    (276, 614),  # 1496
    (444, 1252), # 1497
    (647, 115),  # 1498
    (673, 1165), # 1499
    (98, 276),   # 1500
    (115, 881),  # 1501
    (497, 289),  # 1502
    (752, 318),  # 1503
    (112, 105),  # 1504
    (289, 105),  # 1505
    (399, 32),   # 1506
    (752, 312),  # 1507
    (781, 256),  # 1508
    (1502, 1241),# 1509
    (267, 115),  # 1510
    (350, 352),  # 1511
    (510, 116),  # 1512
    (1281, 256), # 1513
    (332, 426),  # 1514
    (357, 410),  # 1515
    (612, 1364), # 1516
    (111, 115),  # 1517
    (379, 118),  # 1518
    (441, 115),  # 1519
    (263, 314),  # 1520
    (1272, 347), # 1521
    (384, 116),  # 1522
    (1222, 256), # 1523
    (100, 1118), # 1524
    (280, 295),  # 1525
    (348, 102),  # 1526
    (1240, 1355),# 1527
    (109, 421),  # 1528
    (258, 460),  # 1529
    (312, 313),  # 1530
    (1320, 318), # 1531
    (1530, 117), # 1532
    (111, 267),  # 1533
    (1335, 303), # 1534
    (1342, 277), # 1535
    (100, 314),  # 1536
    (119, 262),  # 1537
    (739, 271),  # 1538
    (1251, 1488),# 1539
    (1321, 372), # 1540
    (1442, 368), # 1541
    (1496, 1158),# 1542
    (121, 301),  # 1543
    (1003, 599), # 1544
    (110, 261),  # 1545
    (372, 1478), # 1546
    (594, 1509), # 1547
    (975, 684),  # 1548
    (1540, 445), # 1549
    (98, 281),   # 1550
    (1394, 1487),# 1551
    (279, 256),  # 1552
    (316, 405),  # 1553
    (456, 1428), # 1554
    (669, 959),  # 1555
    (672, 299),  # 1556
    (1163, 722), # 1557
    (1329, 1533),# 1558
    (100, 1167), # 1559
    (1559, 102), # 1560
    (108, 494),  # 1561
    (115, 111),  # 1562
    (266, 116),  # 1563
    (281, 106),  # 1564
    (367, 1514), # 1565
    (102, 108),  # 1566
    (102, 350),  # 1567
    (110, 318),  # 1568
    (393, 497),  # 1569
    (111, 332),  # 1570
    (348, 97),   # 1571
    (1041, 1555),# 1572
    (100, 318),  # 1573
    (372, 104),  # 1574
    (1078, 1201),# 1575
    (1344, 257), # 1576
    (116, 276),  # 1577
    (278, 302),  # 1578
    (608, 100),  # 1579
    (1201, 1086),# 1580
    (1477, 1266),# 1581
    (110, 262),  # 1582
    (1549, 1472),# 1583
    (99, 336),   # 1584
    (281, 284),  # 1585
    (283, 302),  # 1586
    (357, 281),  # 1587
    (437, 1330), # 1588
    (680, 366),  # 1589
    (1275, 352), # 1590
    (1463, 482), # 1591
    (99, 107),   # 1592
    (109, 257),  # 1593
    (465, 1262), # 1594
    (416, 1288), # 1595
    (586, 296),  # 1596
    (1263, 302), # 1597
    (1482, 1424),# 1598
    (101, 263),  # 1599
    (108, 396),  # 1600
    (109, 332),  # 1601
    (115, 635),  # 1602
    (260, 259),  # 1603
    (269, 666),  # 1604
    (99, 349),   # 1605
    (103, 366),  # 1606
    (276, 693),  # 1607
    (1430, 593), # 1608
    (98, 389),   # 1609
    (111, 98),   # 1610
    (263, 1302), # 1611
    (298, 99),   # 1612
    (1257, 1497),# 1613
    (1314, 357), # 1614
    (1588, 1546),# 1615
    (270, 295),  # 1616
    (316, 99),   # 1617
    (1492, 1429),# 1618
    (291, 639),  # 1619
    (1589, 1569),# 1620
    (447, 838),  # 1621
    (685, 1148), # 1622
    (1554, 509), # 1623
    (1621, 1622),# 1624
    (115, 463),  # 1625
    (298, 101),  # 1626
    (975, 329),  # 1627
    (1539, 112), # 1628
    (327, 275),  # 1629
    (103, 457),  # 1630
    (110, 462),  # 1631
    (116, 1308), # 1632
    (313, 296),  # 1633
    (750, 890),  # 1634
    (1244, 286), # 1635
    (1380, 1333),# 1636
    (1422, 643), # 1637
    (1459, 273), # 1638
    (1557, 326), # 1639
    (366, 112),  # 1640
    (703, 1225), # 1641
    (1197, 276), # 1642
    (269, 301),  # 1643
    (816, 379),  # 1644
    (1162, 1223),# 1645
    (327, 438),  # 1646
    (360, 311),  # 1647
    (281, 1427), # 1648
    (290, 793),  # 1649
    (353, 121),  # 1650
    (355, 327),  # 1651
    (1571, 762), # 1652
    (1574, 1651),# 1653
    (102, 379),  # 1654
    (263, 271),  # 1655
    (443, 260),  # 1656
    (1466, 719), # 1657
    (1634, 1500),# 1658
    (108, 526),  # 1659
    (287, 1386), # 1660
    (291, 308),  # 1661
    (582, 405),  # 1662
    (1660, 1662),# 1663
    (1661, 486), # 1664
    (386, 275),  # 1665
    (1606, 32),  # 1666
    (259, 118),  # 1667
    (298, 378),  # 1668
    (393, 342),  # 1669
    (396, 294),  # 1670
    (469, 299),  # 1671
    (1373, 1352),# 1672
    (1638, 99),  # 1673
    (311, 101),  # 1674
    (342, 1493), # 1675
    (696, 256),  # 1676
    (807, 264),  # 1677
    (1650, 1495),# 1678
    (109, 121),  # 1679
    (273, 271),  # 1680
    (1378, 1469),# 1681
    (314, 112),  # 1682
    (1249, 279), # 1683
    (1598, 1653),# 1684
    (287, 1184), # 1685
    (298, 264),  # 1686
    (344, 1685), # 1687
    (1467, 699), # 1688
    (1677, 1278),# 1689
    (98, 1383),  # 1690
    (263, 375),  # 1691
    (305, 347),  # 1692
    (938, 376),  # 1693
    (1090, 454), # 1694
    (1613, 1464),# 1695
    (1687, 105), # 1696
    (473, 336),  # 1697
    (786, 541),  # 1698
    (1187, 115), # 1699
    (1237, 280), # 1700
    (110, 596),  # 1701
    (291, 1646), # 1702
    (301, 294),  # 1703
    (310, 722),  # 1704
    (392, 272),  # 1705
    (1440, 1545),# 1706
    (1483, 272), # 1707
    (1665, 601), # 1708
    (98, 457),   # 1709
    (109, 299),  # 1710
    (117, 100),  # 1711
    (333, 256),  # 1712
    (378, 347),  # 1713
    (451, 1592), # 1714
    (1709, 115), # 1715
    (312, 290),  # 1716
    (409, 550),  # 1717
    (490, 260),  # 1718
    (781, 277),  # 1719
    (1250, 116), # 1720
    (258, 605),  # 1721
    (310, 1168), # 1722
    (372, 359),  # 1723
    (438, 307),  # 1724
    (1331, 495), # 1725
    (98, 379),   # 1726
    (354, 115),  # 1727
    (612, 705),  # 1728
    (1701, 32),  # 1729
    (104, 1265), # 1730
    (115, 394),  # 1731
    (807, 287),  # 1732
    (1151, 1723),# 1733
    (1387, 1604),# 1734
    (119, 859),  # 1735
    (259, 1297), # 1736
    (261, 105),  # 1737
    (298, 107),  # 1738
    (313, 438),  # 1739
    (367, 289),  # 1740
    (442, 1303), # 1741
    (592, 1740), # 1742
    (1083, 287), # 1743
    (1379, 368), # 1744
    (1735, 341), # 1745
    (102, 369),  # 1746
    (372, 281),  # 1747
    (608, 431),  # 1748
    (1246, 271), # 1749
    (1284, 1088),# 1750
    (1370, 342), # 1751
    (259, 307),  # 1752
    (488, 630),  # 1753
    (597, 256),  # 1754
    (1435, 264), # 1755
    (1449, 1452),# 1756
    (281, 103),  # 1757
    (1179, 1609),# 1758
    (1465, 105), # 1759
    (1714, 394), # 1760
    (373, 300),  # 1761
    (41, 271),   # 1762
    (281, 109),  # 1763
    (298, 435),  # 1764
    (355, 103),  # 1765
    (1326, 406), # 1766
    (1479, 574), # 1767
    (99, 121),   # 1768
    (260, 577),  # 1769
    (336, 262),  # 1770
    (1461, 668), # 1771
    (1657, 258), # 1772
    (1696, 326), # 1773
    (1715, 293), # 1774
    (350, 108),  # 1775
    (579, 1632), # 1776
    (700, 1312), # 1777
    (1202, 108), # 1778
    (1528, 1348),# 1779
    (312, 1322), # 1780
    (325, 593),  # 1781
    (381, 283),  # 1782
    (413, 1301), # 1783
    (570, 444),  # 1784
    (601, 271),  # 1785
    (1250, 264), # 1786
    (1269, 381), # 1787
    (1532, 627), # 1788
    (1776, 1702),# 1789
    (261, 110),  # 1790
    (283, 121),  # 1791
    (308, 410),  # 1792
    (336, 1504), # 1793
    (436, 32),   # 1794
    (476, 257),  # 1795
    (1384, 622), # 1796
    (1793, 306), # 1797
    (111, 302),  # 1798
    (116, 495),  # 1799
    (118, 380),  # 1800
    (358, 1393), # 1801
    (433, 313),  # 1802
    (591, 259),  # 1803
    (680, 440),  # 1804
    (1800, 354), # 1805
    (103, 336),  # 1806
    (105, 112),  # 1807
    (276, 1167), # 1808
    (472, 1264), # 1809
    (799, 262),  # 1810
    (1666, 554), # 1811
    (1780, 256), # 1812
    (1809, 399), # 1813
    (115, 109),  # 1814
    (263, 701),  # 1815
    (624, 859),  # 1816
    (1420, 417), # 1817
    (1524, 256), # 1818
    (1607, 648), # 1819
    (1745, 1699),# 1820
    (1788, 1560),# 1821
    (1805, 1629),# 1822
    (98, 306),   # 1823
    (118, 293),  # 1824
    (515, 276),  # 1825
    (689, 1165), # 1826
    (1083, 437), # 1827
    (1097, 355), # 1828
    (1690, 423), # 1829
    (102, 117),  # 1830
    (118, 349),  # 1831
    (382, 626),  # 1832
    (1175, 1340),# 1833
    (1582, 45),  # 1834
    (121, 638),  # 1835
    (288, 372),  # 1836
    (439, 115),  # 1837
    (781, 108),  # 1838
    (993, 812),  # 1839
    (1178, 119), # 1840
    (1293, 1577),# 1841
    (1516, 264), # 1842
    (1664, 296), # 1843
    (116, 277),  # 1844
    (118, 289),  # 1845
    (295, 279),  # 1846
    (328, 421),  # 1847
    (331, 368),  # 1848
    (442, 1626), # 1849
    (687, 1242), # 1850
    (703, 116),  # 1851
    (1176, 471), # 1852
    (1803, 1152),# 1853
    (1850, 554), # 1854
    (281, 1164), # 1855
    (393, 259),  # 1856
    (443, 1761), # 1857
    (617, 515),  # 1858
    (915, 280),  # 1859
    (1093, 459), # 1860
    (1371, 1648),# 1861
    (1468, 1683),# 1862
    (1717, 1857),# 1863
    (1804, 299), # 1864
    (259, 99),   # 1865
    (259, 1801), # 1866
    (266, 310),  # 1867
    (298, 296),  # 1868
    (342, 287),  # 1869
    (1162, 270), # 1870
    (1186, 442), # 1871
    (1226, 272), # 1872
    (1240, 1580),# 1873
    (1260, 1652),# 1874
    (1520, 271), # 1875
    (1725, 307), # 1876
    (1777, 404), # 1877
    (1806, 304), # 1878
    (97, 1172),  # 1879
    (98, 1505),  # 1880
    (105, 118),  # 1881
    (325, 116),  # 1882
    (629, 347),  # 1883
    (1541, 260), # 1884
    (1789, 334), # 1885
    (1802, 107), # 1886
    (98, 261),   # 1887
    (99, 383),   # 1888
    (258, 1300), # 1889
    (280, 1360), # 1890
    (344, 263),  # 1891
    (436, 412),  # 1892
    (523, 270),  # 1893
    (682, 260),  # 1894
    (1640, 638), # 1895
    (1742, 405), # 1896
    (357, 719),  # 1897
    (381, 275),  # 1898
    (453, 1896), # 1899
    (464, 692),  # 1900
    (568, 1596), # 1901
    (702, 1771), # 1902
    (1315, 1519),# 1903
    (1411, 1837),# 1904
    (1432, 1846),# 1905
    (1819, 554), # 1906
    (1838, 1765),# 1907
    (1848, 1368),# 1908
    (1856, 1724),# 1909
    (1863, 1455),# 1910
    (1876, 1902),# 1911
    (1911, 1899),# 1912
    (99, 279),   # 1913
    (267, 373),  # 1914
    (331, 1318), # 1915
    (331, 1368), # 1916
    (358, 280),  # 1917
    (466, 1858), # 1918
    (601, 1529), # 1919
    (659, 287),  # 1920
    (725, 1565), # 1921
    (1171, 1457),# 1922
    (1360, 1916),# 1923
    (1365, 1753),# 1924
    (1397, 585), # 1925
    (1444, 1923),# 1926
    (1451, 282), # 1927
    (1474, 1719),# 1928
    (1485, 1924),# 1929
    (1656, 1877),# 1930
    (1668, 754), # 1931
    (1703, 1904),# 1932
    (1704, 626), # 1933
    (1728, 1151),# 1934
    (1772, 1778),# 1935
    (1820, 1705),# 1936
    (1826, 1931),# 1937
    (1833, 619), # 1938
    (1894, 1935),# 1939
    (1905, 1919),# 1940
    (1906, 1940),# 1941
    (1908, 1921),# 1942
    (1909, 1941),# 1943
    (1912, 1938),# 1944
    (1918, 1930),# 1945
    (1926, 299), # 1946
    (1928, 1942),# 1947
    (1939, 1932),# 1948
    (1943, 1946),# 1949
    (1945, 1944),# 1950
    (1947, 1948),# 1951
    (306, 103),  # 1952
    (455, 711),  # 1953
    (587, 310),  # 1954
    (1230, 101), # 1955
    (1242, 1603),# 1956
    (1810, 100), # 1957
    (1832, 295), # 1958
    (1956, 1958),# 1959
    (265, 256),  # 1960
    (314, 684),  # 1961
    (473, 117),  # 1962
    (695, 357),  # 1963
    (731, 1318), # 1964
    (733, 294),  # 1965
    (1326, 293), # 1966
    (1456, 1412),# 1967
    (1507, 1721),# 1968
    (1562, 301), # 1969
    (1601, 457), # 1970
    (1658, 1490),# 1971
    (1748, 1953),# 1972
    (1767, 295), # 1973
    (1811, 670), # 1974
    (1972, 1964),# 1975
    (45, 1237),  # 1976
    (99, 1265),  # 1977
    (107, 101),  # 1978
    (612, 1413), # 1979
    (1257, 1822),# 1980
    (1292, 276), # 1981
    (1371, 602), # 1982
    (1475, 256), # 1983
    (1537, 548), # 1984
    (1670, 1974),# 1985
    (1681, 1976),# 1986
    (1769, 1973),# 1987
    (1787, 1890),# 1988
    (1825, 1969),# 1989
    (1959, 1968),# 1990
    (1965, 1783),# 1991
    (1980, 1985),# 1992
    (1987, 1499),# 1993
    (1988, 1992),# 1994
    (1990, 1991),# 1995
    (1994, 1993),# 1996
    (309, 1259), # 1997
    (343, 99),   # 1998
    (380, 114),  # 1999
    (408, 1312), # 2000
    (1486, 283), # 2001
    (1512, 286), # 2002
    (1747, 375), # 2003
    (1901, 1949),# 2004
    (1995, 1915),# 2005
    (455, 1808), # 2006
    (1097, 309), # 2007
    (1258, 295), # 2008
    (1388, 609), # 2009
    (1498, 420), # 2010
    (1879, 265), # 2011
    (1996, 1849),# 2012
    (383, 32),   # 2013
    (568, 2005), # 2014
    (638, 110),  # 2015
    (1185, 307), # 2016
    (1708, 1348),# 2017
    (101, 603),  # 2018
    (105, 348),  # 2019
    (109, 684),  # 2020
    (116, 119),  # 2021
    (121, 45),   # 2022
    (317, 441),  # 2023
    (1277, 1618),# 2024
    (1367, 1453),# 2025
    (1619, 1369),# 2026
    (1784, 549), # 2027
    (1841, 435), # 2028
    (1954, 1170),# 2029
    (98, 1494),  # 2030
    (455, 267),  # 2031
    (587, 298),  # 2032
    (1402, 686), # 2033
    (97, 1506),  # 2034
    (498, 281),  # 2035
    (1630, 762), # 2036
    (1716, 476), # 2037
    (1982, 302), # 2038
    (103, 394),  # 2039
    (104, 638),  # 2040
    (108, 354),  # 2041
    (276, 105),  # 2042
    (304, 109),  # 2043
    (312, 1324), # 2044
    (613, 1986), # 2045
    (742, 1322), # 2046
    (1074, 112), # 2047
]

# ═══════════════════════════════════════════════════════════════
# BPE ENCODER — byte-pair encoding for input text
# ═══════════════════════════════════════════════════════════════

"""
    bpe_encode(text) → Vector{Int}

Encode text into BPE token IDs (0-based). Converts to lowercase bytes,
then applies merges in priority order.
"""
function bpe_encode(text::AbstractString)
    # 1. Convert text to lowercase bytes -> initial token sequence
    seq = Int[]
    for ch in text
        c = UInt8(lowercase(ch)[1])
        push!(seq, Int(c))
    end

    # 2. Apply merges in priority order
    for m in 1:BPE_MERGES
        left, right = BPE_TABLE[m]
        new_id = 256 + m - 1  # 0-based: merge 1 -> token 256
        new_seq = Int[]
        i = 1
        while i <= length(seq)
            if i < length(seq) && seq[i] == left && seq[i+1] == right
                push!(new_seq, new_id)
                i += 2  # skip pair
            else
                push!(new_seq, seq[i])
                i += 1
            end
        end
        seq = new_seq
    end

    return seq  # 0-based token IDs
end

# Precomputed BPE encodings for each vocab word (for generation)
const VOCAB_BPE = [bpe_encode(w) for w in VOCAB]

const VOCAB_SET = Set(VOCAB)
const VOCAB_IDX = let d = Dict{String,Int}()
    for (i, w) in enumerate(VOCAB)
        idx = i - 1  # 0-based
        if !haskey(d, w)
            d[w] = idx
        end
    end
    d
end

const STOP = Set(split("i me my we our you your he she it they them the a an and or but in on at to for of is am are was were be been being have has had do does did will would shall should can could may might must not no nor so if then than that this these those what which who whom how when where why all each every some any few many much more most other another such"))


# ═══════════════════════════════════════════════════════════════
# MATH — pure Julia, no external dependencies
# ═══════════════════════════════════════════════════════════════

function _randn()
    u1 = rand() + 1e-12
    u2 = rand() + 1e-12
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2)
end

function _zeros(n::Int)
    return zeros(Float64, n)
end

function _dot(a::Vector{Float64}, b::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(a)
        s += a[i] * b[i]
    end
    return s
end

function vadd(a::Vector{Float64}, b::Vector{Float64})
    return [a[i] + b[i] for i in eachindex(a)]
end

function vsub(a::Vector{Float64}, b::Vector{Float64})
    return [a[i] - b[i] for i in eachindex(a)]
end

function vscale(a::Vector{Float64}, s::Float64)
    return [x * s for x in a]
end

function matmul_mv(W::Vector{Float64}, x::Vector{Float64}, rows::Int, cols::Int)
    out = _zeros(rows)
    @inbounds for i in 1:rows
        s = 0.0
        base = (i - 1) * cols
        for j in 1:cols
            s += W[base + j] * x[j]
        end
        out[i] = s
    end
    return out
end

function matmul_mtv(W::Vector{Float64}, x::Vector{Float64}, rows::Int, cols::Int)
    out = _zeros(cols)
    @inbounds for j in 1:cols
        s = 0.0
        for i in 1:rows
            s += W[(i - 1) * cols + j] * x[i]
        end
        out[j] = s
    end
    return out
end

function rmsnorm(x::Vector{Float64}, g::Vector{Float64}, n::Int)
    ss = 0.0
    @inbounds for i in 1:n
        ss += x[i] * x[i]
    end
    ss = ss / n + 1e-5
    inv = 1.0 / sqrt(ss)
    return [g[i] * x[i] * inv for i in 1:n]
end

function _silu(x::Float64)
    return x > -20.0 ? x / (1.0 + exp(-x)) : 0.0
end

function _softmax(x::Vector{Float64})
    mx = maximum(x)
    e = [exp(v - mx) for v in x]
    s = sum(e)
    return [v / s for v in e]
end


# ═══════════════════════════════════════════════════════════════
# MODEL — v7 Resonance: 8 sequential layers with
# multi-head attention + RoPE + RRPRAM resonance gate + SwiGLU
# ═══════════════════════════════════════════════════════════════

mutable struct LayerWeights
    attn_norm::Vector{Float64}   # [DIM]         pre-attention RMSNorm
    wq::Vector{Float64}          # [DIM * DIM]   query projection
    wk::Vector{Float64}          # [DIM * DIM]   key projection
    wv::Vector{Float64}          # [DIM * DIM]   value projection
    wo::Vector{Float64}          # [DIM * DIM]   output projection
    wr::Vector{Float64}          # [DIM * DIM]   RRPRAM resonance
    gate::Vector{Float64}        # [2]           blend QKV + RRPRAM
    ffn_norm::Vector{Float64}    # [DIM]         pre-FFN RMSNorm
    w_gate::Vector{Float64}      # [DIM * HDIM]  SwiGLU gate (note: HDIM > DIM)
    w_up::Vector{Float64}        # [DIM * HDIM]  SwiGLU up
    w_down::Vector{Float64}      # [HDIM * DIM]  SwiGLU down
end

function LayerWeights()
    scale_d = sqrt(2.0 / DIM)
    scale_h = sqrt(2.0 / HDIM)
    attn_norm = ones(Float64, DIM)
    wq = [_randn() * scale_d for _ in 1:(DIM * DIM)]
    wk = [_randn() * scale_d for _ in 1:(DIM * DIM)]
    wv = [_randn() * scale_d for _ in 1:(DIM * DIM)]
    wo = [_randn() * scale_d for _ in 1:(DIM * DIM)]
    wr = [_randn() * scale_d for _ in 1:(DIM * DIM)]
    gate = [0.0, 0.0]  # 50/50 blend
    ffn_norm = ones(Float64, DIM)
    w_gate = [_randn() * scale_d for _ in 1:(DIM * HDIM)]
    w_up   = [_randn() * scale_d for _ in 1:(DIM * HDIM)]
    w_down = [_randn() * scale_h for _ in 1:(HDIM * DIM)]
    return LayerWeights(attn_norm, wq, wk, wv, wo, wr, gate, ffn_norm, w_gate, w_up, w_down)
end

function layer_param_count()
    # attn_norm + wq + wk + wv + wo + wr + gate + ffn_norm + w_gate + w_up + w_down
    return DIM + DIM*DIM*5 + 2 + DIM + DIM*HDIM*2 + HDIM*DIM
end

mutable struct Penelope
    # Global weights
    tok_emb::Vector{Float64}     # [BPE_VOCAB * DIM]  token embedding
    pos_emb::Vector{Float64}     # [MAX_SEQ * DIM]    positional embedding
    final_norm::Vector{Float64}  # [DIM]              final RMSNorm
    lm_head::Vector{Float64}     # [BPE_VOCAB * DIM]  language model head
    # Per-layer
    layers::Vector{LayerWeights}
end

function Penelope()
    scale_d = sqrt(2.0 / DIM)
    scale_bpe = sqrt(2.0 / BPE_VOCAB)
    tok_emb    = [_randn() * scale_bpe for _ in 1:(BPE_VOCAB * DIM)]
    pos_emb    = [_randn() * 0.02      for _ in 1:(MAX_SEQ * DIM)]
    final_norm = ones(Float64, DIM)
    lm_head    = [_randn() * scale_d   for _ in 1:(BPE_VOCAB * DIM)]
    layers = [LayerWeights() for _ in 1:N_LAYERS]
    return Penelope(tok_emb, pos_emb, final_norm, lm_head, layers)
end

function param_count(::Penelope)
    global_params = BPE_VOCAB * DIM + MAX_SEQ * DIM + DIM + BPE_VOCAB * DIM
    return global_params + N_LAYERS * layer_param_count()
end


# ═══════════════════════════════════════════════════════════════
# EXTENDED VOCAB — hardcoded 1984 + BPE tokens that are whole words
# Built at init from BPE decode. Word-level always, gibberish impossible.
# ═══════════════════════════════════════════════════════════════

struct ExtWord
    word::String
    bpe_ids::Vector{Int}
    from_hardcoded::Bool
end

# BPE decode table — maps token ID to string
function build_bpe_strs()
    strs = Vector{String}(undef, BPE_VOCAB)
    for i in 0:255
        strs[i + 1] = String([UInt8(i)])
    end
    for m in 1:BPE_MERGES
        left, right = BPE_TABLE[m]
        id = 256 + m - 1
        if id + 1 <= BPE_VOCAB
            strs[id + 1] = strs[left + 1] * strs[right + 1]
        end
    end
    return strs
end

const BPE_STRS = build_bpe_strs()

function is_alpha_word(s::String)
    length(s) < 2 && return false
    for c in s
        isletter(c) || return false
    end
    return true
end

function init_ext_vocab()
    ext = ExtWord[]

    # 1. Add all hardcoded words (first NWORDS from VOCAB)
    for i in 1:min(NWORDS, V)
        push!(ext, ExtWord(VOCAB[i], VOCAB_BPE[i], true))
    end

    # 2. Add BPE tokens that decode to whole words (not already in vocab)
    suffix_set = Set(["ing","tion","ment","ness","ble","ful","ous","ive","ent","ant",
                      "ist","ity","ght","est","ter","ther","ted","ting","ally","ling"])
    existing = Set(ew.word for ew in ext)
    for t in 0:(BPE_VOCAB - 1)
        s = BPE_STRS[t + 1]
        is_alpha_word(s) || continue
        low = lowercase(s)
        length(low) < 3 && continue
        low in existing && continue
        low in STOP && continue
        low in suffix_set && continue
        length(ext) >= MAX_EXT_VOCAB && break
        push!(ext, ExtWord(low, [t], false))
        push!(existing, low)
    end

    return ext
end

const EXT_VOCAB = init_ext_vocab()

function ext_vocab_find(word::String)
    for (i, ew) in enumerate(EXT_VOCAB)
        ew.word == word && return i
    end
    return nothing
end


# ═══════════════════════════════════════════════════════════════
# RoPE — rotary position embedding
# ═══════════════════════════════════════════════════════════════

function apply_rope!(q::Vector{Float64}, k::Vector{Float64}, seq_len::Int)
    theta_base = 10000.0
    for t in 0:(seq_len - 1)
        for h in 0:(N_HEADS - 1)
            offset = (t * N_HEADS + h) * HEAD_DIM
            for d in 0:(HEAD_DIM >> 1 - 1)
                freq = 1.0 / theta_base^(2.0 * d / HEAD_DIM)
                cos_f = cos(t * freq)
                sin_f = sin(t * freq)
                # rotate q
                qi0 = offset + d + 1
                qi1 = offset + d + HEAD_DIM >> 1 + 1
                q0, q1 = q[qi0], q[qi1]
                q[qi0] = q0 * cos_f - q1 * sin_f
                q[qi1] = q0 * sin_f + q1 * cos_f
                # rotate k
                k0, k1 = k[qi0], k[qi1]
                k[qi0] = k0 * cos_f - k1 * sin_f
                k[qi1] = k0 * sin_f + k1 * cos_f
            end
        end
    end
end


# ═══════════════════════════════════════════════════════════════
# FORWARD — v7 Resonance: 8 sequential layers, multi-head
# attention + RoPE + RRPRAM gate + SwiGLU, then BPE logits
# ═══════════════════════════════════════════════════════════════

function forward(model::Penelope, bpe_ids::Vector{Int}, seq_len::Int)
    S = clamp(seq_len, 1, MAX_SEQ)

    # x: [S * DIM] — residual stream
    x = zeros(Float64, S * DIM)

    # embed: tok_emb + pos_emb
    for t in 0:(S - 1)
        tok = bpe_ids[t + 1]
        if tok < 0 || tok >= BPE_VOCAB; tok = 0; end
        for d in 1:DIM
            x[t * DIM + d] = model.tok_emb[tok * DIM + d] + model.pos_emb[t * DIM + d]
        end
    end

    # scratch buffers
    h   = zeros(Float64, S * DIM)
    q   = zeros(Float64, S * DIM)
    k   = zeros(Float64, S * DIM)
    v   = zeros(Float64, S * DIM)
    att = zeros(Float64, S * S * N_HEADS)
    av  = zeros(Float64, S * DIM)
    qkv_out = zeros(Float64, S * DIM)
    rrp = zeros(Float64, S * DIM)
    h2  = zeros(Float64, S * DIM)
    fg  = zeros(Float64, S * HDIM)
    fu  = zeros(Float64, S * HDIM)
    sw  = zeros(Float64, S * HDIM)
    fd  = zeros(Float64, S * DIM)

    for l in 1:N_LAYERS
        lw = model.layers[l]

        # 1. h = rmsnorm(x, attn_norm) for each position
        for t in 0:(S - 1)
            xslice = x[(t * DIM + 1):((t + 1) * DIM)]
            normed = rmsnorm(xslice, lw.attn_norm, DIM)
            h[(t * DIM + 1):((t + 1) * DIM)] .= normed
        end

        # 2-3. q = h @ wq, k = h @ wk, v = h @ wv (per position)
        for t in 0:(S - 1)
            hs = h[(t * DIM + 1):((t + 1) * DIM)]
            q[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.wq, hs, DIM, DIM)
            k[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.wk, hs, DIM, DIM)
            v[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.wv, hs, DIM, DIM)
        end

        # Apply RoPE to q and k
        apply_rope!(q, k, S)

        # 5. Multi-head causal attention: softmax(q @ k^T / sqrt(head_dim))
        scale = 1.0 / sqrt(Float64(HEAD_DIM))
        for hd in 0:(N_HEADS - 1)
            for ti in 0:(S - 1)
                qi_off = (ti * N_HEADS + hd) * HEAD_DIM
                maxs = -1e30
                for tj in 0:ti
                    kj_off = (tj * N_HEADS + hd) * HEAD_DIM
                    dot = 0.0
                    @inbounds for d in 1:HEAD_DIM
                        dot += q[qi_off + d] * k[kj_off + d]
                    end
                    dot *= scale
                    att[(hd * S + ti) * S + tj + 1] = dot
                    if dot > maxs; maxs = dot; end
                end
                # causal mask + softmax
                s_sum = 0.0
                for tj in 0:ti
                    idx = (hd * S + ti) * S + tj + 1
                    val = exp(att[idx] - maxs)
                    att[idx] = val
                    s_sum += val
                end
                inv_s = s_sum > 0.0 ? 1.0 / s_sum : 0.0
                for tj in 0:ti
                    att[(hd * S + ti) * S + tj + 1] *= inv_s
                end
                # zero future
                for tj in (ti + 1):(S - 1)
                    att[(hd * S + ti) * S + tj + 1] = 0.0
                end
            end
        end

        # 6. attn @ v, reshape, then @ wo
        fill!(av, 0.0)
        for hd in 0:(N_HEADS - 1)
            for ti in 0:(S - 1)
                avi_off = (ti * N_HEADS + hd) * HEAD_DIM
                for tj in 0:ti
                    a = att[(hd * S + ti) * S + tj + 1]
                    a == 0.0 && continue
                    vj_off = (tj * N_HEADS + hd) * HEAD_DIM
                    @inbounds for d in 1:HEAD_DIM
                        av[avi_off + d] += a * v[vj_off + d]
                    end
                end
            end
        end
        # av is [S, DIM] (concatenated heads). Project through wo
        for t in 0:(S - 1)
            av_slice = av[(t * DIM + 1):((t + 1) * DIM)]
            qkv_out[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.wo, av_slice, DIM, DIM)
        end

        # 7. RRPRAM resonance: rrp = h @ wr
        for t in 0:(S - 1)
            hs = h[(t * DIM + 1):((t + 1) * DIM)]
            rrp[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.wr, hs, DIM, DIM)
        end

        # 8. gate_weights = softmax(gate[0], gate[1])
        g0, g1 = lw.gate[1], lw.gate[2]
        gmax = max(g0, g1)
        e0, e1 = exp(g0 - gmax), exp(g1 - gmax)
        gsum = e0 + e1
        w0, w1 = e0 / gsum, e1 / gsum

        # 9. x = x + w0 * qkv_out + w1 * rrp (residual)
        @inbounds for i in 1:(S * DIM)
            x[i] += w0 * qkv_out[i] + w1 * rrp[i]
        end

        # 10. h2 = rmsnorm(x, ffn_norm)
        for t in 0:(S - 1)
            xslice = x[(t * DIM + 1):((t + 1) * DIM)]
            normed = rmsnorm(xslice, lw.ffn_norm, DIM)
            h2[(t * DIM + 1):((t + 1) * DIM)] .= normed
        end

        # 11. SwiGLU FFN: x = x + w_down @ (silu(h2 @ w_gate) * (h2 @ w_up))
        for t in 0:(S - 1)
            h2s = h2[(t * DIM + 1):((t + 1) * DIM)]
            fg[(t * HDIM + 1):((t + 1) * HDIM)] .= matmul_mv(lw.w_gate, h2s, HDIM, DIM)
            fu[(t * HDIM + 1):((t + 1) * HDIM)] .= matmul_mv(lw.w_up,   h2s, HDIM, DIM)
            @inbounds for i in 1:HDIM
                sw[t * HDIM + i] = _silu(fg[t * HDIM + i]) * fu[t * HDIM + i]
            end
            sw_slice = sw[(t * HDIM + 1):((t + 1) * HDIM)]
            fd[(t * DIM + 1):((t + 1) * DIM)] .= matmul_mv(lw.w_down, sw_slice, DIM, HDIM)
            @inbounds for d in 1:DIM
                x[t * DIM + d] += fd[t * DIM + d]
            end
        end
    end

    # After all layers: final rmsnorm + lm_head for LAST position
    xn = rmsnorm(x[((S - 1) * DIM + 1):(S * DIM)], model.final_norm, DIM)
    logits = matmul_mv(model.lm_head, xn, BPE_VOCAB, DIM)
    return logits
end


# ═══════════════════════════════════════════════════════════════
# BPE logits → word-level scores
# ═══════════════════════════════════════════════════════════════

function bpe_logits_to_word_scores(bpe_logits::Vector{Float64}, n_words::Int)
    scores = zeros(Float64, n_words)
    for w in 1:n_words
        if w <= NWORDS && w <= V
            # hardcoded vocab word — use precomputed BPE encoding
            bl = length(VOCAB_BPE[w])
            score = 0.0
            for tok in VOCAB_BPE[w]
                if tok >= 0 && tok < BPE_VOCAB
                    score += bpe_logits[tok + 1]
                end
            end
            scores[w] = bl > 0 ? score / bl : 0.0
        elseif w <= length(EXT_VOCAB)
            # extended vocab word
            ew = EXT_VOCAB[w]
            bl = length(ew.bpe_ids)
            score = 0.0
            for tok in ew.bpe_ids
                if tok >= 0 && tok < BPE_VOCAB
                    score += bpe_logits[tok + 1]
                end
            end
            scores[w] = bl > 0 ? score / bl : 0.0
        end
    end
    return scores
end


# ═══════════════════════════════════════════════════════════════
# BPE INPUT — stem + greedy longest vocab match
# ═══════════════════════════════════════════════════════════════

const SUFFIXES = [
    "ting","ning","ring","ling","ding","ping","bing","ging","ming","king",
    "sing","zing",
    "ing","ment","ness","tion","sion","able","ible","ence","ance",
    "eous","ious","ful","less","ize","ise","ous","ive","ity",
    "ly","er","ed","est","al","en","es","s",
]

const VOCAB_LENS = [length(w) for w in VOCAB]

function try_stem(word::AbstractString)
    wlen = length(word)
    for suf in SUFFIXES
        slen = length(suf)
        slen + 2 >= wlen && continue
        endswith(word, suf) || continue
        sl = wlen - slen
        stem = word[1:sl]
        idx = get(VOCAB_IDX, stem, nothing)
        idx !== nothing && return idx
        stem_e = stem * "e"
        idx = get(VOCAB_IDX, stem_e, nothing)
        idx !== nothing && return idx
        if sl >= 3 && stem[sl] == stem[sl-1]
            stem_dd = stem[1:sl-1]
            idx = get(VOCAB_IDX, stem_dd, nothing)
            idx !== nothing && return idx
        end
    end
    return nothing
end

function greedy_vocab_match(word::AbstractString)
    ids = Int[]
    wlen = length(word)
    pos = 1
    while pos <= wlen && length(ids) < 8
        best_idx = -1
        best_len = 0
        for v in 1:V
            vl = VOCAB_LENS[v]
            vl <= best_len && continue
            vl > wlen - pos + 1 && continue
            if word[pos:pos+vl-1] == VOCAB[v]
                best_idx = v - 1
                best_len = vl
            end
        end
        if best_idx >= 0 && best_len >= 3
            push!(ids, best_idx)
            pos += best_len
        else
            pos += 1
        end
    end
    return ids
end

function tokenize_vocab(text::String)
    words = [m.match for m in eachmatch(r"[a-z]+", lowercase(text))]
    ids = Int[]
    for w in words
        if w in STOP || length(w) < 2
            continue
        end
        idx = get(VOCAB_IDX, w, nothing)
        if idx !== nothing; push!(ids, idx); continue; end
        idx = try_stem(w)
        if idx !== nothing; push!(ids, idx); continue; end
        sub = greedy_vocab_match(w)
        for s in sub
            if isempty(ids) || ids[end] != s
                push!(ids, s)
            end
        end
    end
    return ids
end


# ═══════════════════════════════════════════════════════════════
# SAVE / LOAD — PEN7 binary format
# ═══════════════════════════════════════════════════════════════

function model_save(model::Penelope, path::String)
    open(path, "w") do f
        # PEN7 header: magic, BPE_VOCAB, NWORDS, DIM, HDIM, N_HEADS, N_LAYERS, MAX_SEQ
        write(f, Int32(0x50454E37))
        write(f, Int32(BPE_VOCAB))
        write(f, Int32(NWORDS))
        write(f, Int32(DIM))
        write(f, Int32(HDIM))
        write(f, Int32(N_HEADS))
        write(f, Int32(N_LAYERS))
        write(f, Int32(MAX_SEQ))
        # Global weights
        for v in model.tok_emb;    write(f, Float32(v)); end
        for v in model.pos_emb;    write(f, Float32(v)); end
        for v in model.final_norm; write(f, Float32(v)); end
        for v in model.lm_head;    write(f, Float32(v)); end
        # Per-layer weights
        for lw in model.layers
            for v in lw.attn_norm; write(f, Float32(v)); end
            for v in lw.wq;        write(f, Float32(v)); end
            for v in lw.wk;        write(f, Float32(v)); end
            for v in lw.wv;        write(f, Float32(v)); end
            for v in lw.wo;        write(f, Float32(v)); end
            for v in lw.wr;        write(f, Float32(v)); end
            for v in lw.gate;      write(f, Float32(v)); end
            for v in lw.ffn_norm;  write(f, Float32(v)); end
            for v in lw.w_gate;    write(f, Float32(v)); end
            for v in lw.w_up;      write(f, Float32(v)); end
            for v in lw.w_down;    write(f, Float32(v)); end
        end
    end
    pc = param_count(model)
    sz = filesize(path)
    expected = 32 + pc * 4  # 8 ints header = 32 bytes
    status = sz == expected ? "OK" : "SIZE MISMATCH!"
    println("  saved $path: $pc params ($(round(sz/1e6, digits=1))MB) [$status]")
end

function model_load!(model::Penelope, path::String)
    open(path, "r") do f
        header = [read(f, Int32) for _ in 1:8]
        if header[1] != Int32(0x50454E37)
            error("  unknown format magic=0x$(string(header[1], base=16)) (expected PEN7=0x50454E37)")
        end
        if header[2] != BPE_VOCAB || header[3] != NWORDS || header[4] != DIM ||
           header[5] != HDIM || header[6] != N_HEADS || header[7] != N_LAYERS ||
           header[8] != MAX_SEQ
            error("  v7 config mismatch: BV=$(header[2]) V=$(header[3]) D=$(header[4]) H=$(header[5]) NH=$(header[6]) NL=$(header[7]) S=$(header[8])")
        end
        # Global weights
        for i in 1:(BPE_VOCAB * DIM); model.tok_emb[i]    = Float64(read(f, Float32)); end
        for i in 1:(MAX_SEQ * DIM);   model.pos_emb[i]    = Float64(read(f, Float32)); end
        for i in 1:DIM;               model.final_norm[i]  = Float64(read(f, Float32)); end
        for i in 1:(BPE_VOCAB * DIM); model.lm_head[i]    = Float64(read(f, Float32)); end
        # Per-layer
        for lw in model.layers
            for i in 1:DIM;        lw.attn_norm[i] = Float64(read(f, Float32)); end
            for i in 1:(DIM*DIM);  lw.wq[i]        = Float64(read(f, Float32)); end
            for i in 1:(DIM*DIM);  lw.wk[i]        = Float64(read(f, Float32)); end
            for i in 1:(DIM*DIM);  lw.wv[i]        = Float64(read(f, Float32)); end
            for i in 1:(DIM*DIM);  lw.wo[i]        = Float64(read(f, Float32)); end
            for i in 1:(DIM*DIM);  lw.wr[i]        = Float64(read(f, Float32)); end
            for i in 1:2;          lw.gate[i]      = Float64(read(f, Float32)); end
            for i in 1:DIM;        lw.ffn_norm[i]  = Float64(read(f, Float32)); end
            for i in 1:(DIM*HDIM); lw.w_gate[i]    = Float64(read(f, Float32)); end
            for i in 1:(DIM*HDIM); lw.w_up[i]      = Float64(read(f, Float32)); end
            for i in 1:(HDIM*DIM); lw.w_down[i]    = Float64(read(f, Float32)); end
        end
    end
    println("  loaded v7 $path: $(param_count(model)) params ($(round(param_count(model) * 4 / 1e6, digits=1))MB)")
end


# ═══════════════════════════════════════════════════════════════
# TRAINING — next-token prediction (BPE-level)
# ═══════════════════════════════════════════════════════════════

const ADAM_B1  = 0.9
const ADAM_B2  = 0.999
const ADAM_EPS = 1e-8

function adam_update!(w::Vector{Float64}, am::Vector{Float64}, av::Vector{Float64},
                      grad::Vector{Float64}, lr::Float64, bc1::Float64, bc2::Float64)
    @inbounds for i in eachindex(w)
        g = grad[i]
        am[i] = ADAM_B1 * am[i] + (1.0 - ADAM_B1) * g
        av[i] = ADAM_B2 * av[i] + (1.0 - ADAM_B2) * g * g
        mhat = am[i] / bc1
        vhat = av[i] / bc2
        w[i] -= lr * mhat / (sqrt(vhat) + ADAM_EPS)
        grad[i] = 0.0
    end
end

function train!(model::Penelope, data_path::String, steps::Int=5000, lr::Float64=3e-4)
    text = read(data_path, String)

    # BPE tokenize entire corpus
    corpus_bpe = bpe_encode(text)
    corpus_len = length(corpus_bpe)

    if corpus_len < MAX_SEQ + 1
        println("  corpus too small: $corpus_len BPE tokens (need $(MAX_SEQ + 1)+)")
        return
    end

    pc = param_count(model)
    println("  corpus: $(length(text)) bytes -> $corpus_len BPE tokens")
    println("  model: $pc params ($(round(pc*4/1e6, digits=1))MB f32)")
    println("  architecture: $N_LAYERS layers, $N_HEADS heads, dim=$DIM, hdim=$HDIM")
    println("  training: $steps steps, lr=$(@sprintf("%.1e", lr)), seq=$MAX_SEQ")
    println("  NOTE: C-style trainer uses forward-only loss (for full backprop, use external training)")

    best_loss = Inf

    for step in 1:steps
        seq_len = min(MAX_SEQ, corpus_len - 1)
        start = rand(1:(corpus_len - seq_len))

        ctx = corpus_bpe[start:(start + seq_len - 1)]
        target = corpus_bpe[start + seq_len]

        logits = forward(model, ctx, seq_len)
        probs = _softmax(logits)

        p = probs[target + 1]
        if p < 1e-10; p = 1e-10; end
        loss = -log(p)

        if loss < best_loss; best_loss = loss; end

        if step % 50 == 0 || step == 1
            @printf("  step %5d/%d  loss=%.4f  best=%.4f  (target=%d p=%.4f)\n",
                    step, steps, loss, best_loss, target, p)
        end

        # Shallow gradient: nudge target token embedding toward context mean
        scale = lr * 0.1
        n_ctx = min(seq_len, 8)
        for d in 1:DIM
            avg_ctx = 0.0
            for i in (seq_len - n_ctx + 1):seq_len
                avg_ctx += model.tok_emb[ctx[i] * DIM + d]
            end
            avg_ctx /= n_ctx
            model.tok_emb[target * DIM + d] += scale * (avg_ctx - model.tok_emb[target * DIM + d])
        end
    end

    @printf("  training complete. best loss: %.4f\n", best_loss)
    println("  NOTE: for full training, use external trainer with PEN7 weight export")
end


# ═══════════════════════════════════════════════════════════════
# DARIO FIELD — live co-occurrence overlay
# ═══════════════════════════════════════════════════════════════

mutable struct DarioField
    cooc::Dict{String, Float64}
    destiny::Vector{Float64}
    trauma::Float64
    prophecy_target::Union{Nothing, Int}
    prophecy_age::Int
    chambers::Dict{String, Float64}
    decay::Dict{String, Float64}
end

function DarioField()
    chambers = Dict{String,Float64}(
        "fear" => 0.0, "love" => 0.0, "rage" => 0.0,
        "void" => 0.0, "flow" => 0.0, "complex" => 0.0
    )
    decay = Dict{String,Float64}(
        "fear" => 0.95, "love" => 0.95, "rage" => 0.93,
        "void" => 0.96, "flow" => 0.94, "complex" => 0.97
    )
    return DarioField(Dict{String,Float64}(), zeros(Float64, 8), 0.0, nothing, 0, chambers, decay)
end

function update_cooc!(field::DarioField, w1::Int, w2::Int)
    k = "$(min(w1,w2))|$(max(w1,w2))"
    field.cooc[k] = get(field.cooc, k, 0.0) + 1.0
end

function get_cooc(field::DarioField, w1::Int, w2::Int)
    k = "$(min(w1,w2))|$(max(w1,w2))"
    return get(field.cooc, k, 0.0)
end

function update_chambers!(field::DarioField, step_idx::Int)
    C = field.chambers
    depth = step_idx / N_LAYERS
    phase = depth < 0.33 ? 0 : (depth < 0.66 ? 1 : 2)
    if phase == 0; C["flow"] += 0.05; end
    if phase == 1; C["fear"] += 0.04; end
    if phase == 2; C["void"] += 0.05; end
    if depth > 0.75; C["complex"] += 0.03; end
    if field.trauma > 0.3; C["rage"] += 0.04; end
    K = 0.02
    old = Dict(C)
    for i in keys(C)
        for j in keys(C)
            if i != j
                C[i] += K * sin(old[j] - old[i])
            end
        end
    end
    for k in keys(C)
        C[k] = max(0.0, min(1.0, C[k] * get(field.decay, k, 0.95)))
    end
end

function dario_overlay!(logits::Vector{Float64}, field::DarioField, context_ids::Vector{Int}, step::Int)
    C = field.chambers
    alpha_mod = 1.0 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
    gamma_mod = 1.0 + 0.4*C["void"] + 0.2*C["complex"]

    last_n = min(length(context_ids), 8)
    ctx_tail = context_ids[(end - last_n + 1):end]

    for v in 0:(NWORDS - 1)
        # H: Hebbian co-occurrence
        H = 0.0
        for ci in ctx_tail
            H += get_cooc(field, ci, v)
        end
        if H > 1.0; H = 1.0; end
        logits[v + 1] += alpha_mod * 0.3 * H

        # F: prophecy
        if field.prophecy_target !== nothing && v == field.prophecy_target
            logits[v + 1] += 0.5 * log(1.0 + field.prophecy_age)
        end

        # A: destiny
        cat = word_category(v)
        d_max = 0.01
        for i in 1:8; if abs(field.destiny[i]) > d_max; d_max = abs(field.destiny[i]); end; end
        logits[v + 1] += gamma_mod * 0.25 * field.destiny[cat + 1] / d_max
    end
end

function word_category(idx::Int)
    if idx < 100; return 0; end
    if idx < 200; return 1; end
    if idx < 300; return 2; end
    if idx < 350; return 3; end
    if idx < 450; return 4; end
    if idx < 550; return 5; end
    if idx < 650; return 6; end
    return 7
end


# ═══════════════════════════════════════════════════════════════
# GENERATION — autoregressive BPE, then word-level output
#
# Dual tokenizer: soul thinks in BPE (2048), mouth speaks in words (1984).
# At each step:
#   1. Forward pass -> BPE logits
#   2. Compute word scores = mean(logits for word's BPE tokens)
#   3. Apply Dario overlay on word scores
#   4. Sample word, print it
#   5. Append word's BPE tokens to context for next step
# ═══════════════════════════════════════════════════════════════

function find_seed(key::String)
    if haskey(VOCAB_IDX, key)
        return VOCAB_IDX[key]
    end
    best = 0
    best_score = -1.0
    for (w, i) in VOCAB_IDX
        score = 0.0
        if occursin(w, key) || occursin(key, w)
            score = 3.0
        end
        for k in 1:min(length(w), length(key))
            if w[k] == key[k]
                score += 0.5
            else
                break
            end
        end
        if score > best_score
            best_score = score
            best = i
        end
    end
    return best_score > 0.0 ? best : rand(0:199)
end

function extract_key(text::String)
    words = [w for w in split(lowercase(text)) if length(w) > 1 && !(w in STOP)]
    if isempty(words)
        parts = split(text)
        return isempty(parts) ? "silence" : lowercase(parts[1])
    end
    sort!(words, by=w -> -length(w))
    return String(words[1])
end

function run_chain(model::Penelope, field::DarioField, text::String, has_weights::Bool=false)
    key = extract_key(text)
    seed = find_seed(key)

    println("\n  $(VOCAB[seed + 1])")

    # prophecy (word-level, for both modes)
    deep_cats = [2, 5, 7]
    tcat = deep_cats[rand(1:length(deep_cats))]
    ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,NWORDS)]
    s_range, e_range = ranges[tcat + 1]
    field.prophecy_target = rand(s_range:min(e_range - 1, NWORDS - 1))
    field.prophecy_age = 0
    println("  destined: $(VOCAB[field.prophecy_target + 1])\n")

    # BPE context buffer — starts with seed word's BPE tokens
    bpe_buf = copy(VOCAB_BPE[seed + 1])

    # word-level chain for Dario field
    chain = [seed]
    forbidden = Set{String}()
    push!(forbidden, VOCAB[seed + 1])

    gen_vocab = has_weights ? length(EXT_VOCAB) : NWORDS
    fulfilled = false

    for step in 0:(GEN_STEPS - 1)
        update_chambers!(field, step)
        field.prophecy_age += 1

        # 1. Forward pass through all 8 layers -> BPE logits for last position
        ctx_len = min(length(bpe_buf), MAX_SEQ)
        ctx_start = length(bpe_buf) > MAX_SEQ ? length(bpe_buf) - MAX_SEQ + 1 : 1
        ctx = bpe_buf[ctx_start:(ctx_start + ctx_len - 1)]
        bpe_logits = forward(model, ctx, ctx_len)

        # 2. Convert BPE logits to word-level scores
        word_scores = bpe_logits_to_word_scores(bpe_logits, gen_vocab)

        # 3. Dario overlay on word scores (first NWORDS entries)
        dario_overlay!(word_scores, field, chain, step)

        if has_weights
            # mask forbidden by word string
            for w in 1:length(EXT_VOCAB)
                cw = EXT_VOCAB[w].word
                if cw in forbidden
                    word_scores[w] = -1e9
                end
            end

            # top-k=12 sampling from ext_vocab
            probs = _softmax(word_scores)
            indexed = sort(collect(enumerate(probs)), by=x -> -x[2])
            indexed = indexed[1:min(12, length(indexed))]
            total = sum(max(0.0, p) for (_, p) in indexed) + 0.001
            r = rand() * total
            pick_1 = indexed[1][1]
            for (idx_1, p) in indexed
                r -= max(0.0, p)
                if r <= 0.0; pick_1 = idx_1; break; end
            end

            pick_word = EXT_VOCAB[pick_1].word
            push!(forbidden, pick_word)

            # figure out 0-based vocab ID for Dario field
            pick_vid = pick_1 <= NWORDS ? pick_1 - 1 : -1
            push!(chain, pick_vid >= 0 ? pick_vid : seed)

            # append picked word's BPE tokens to context
            bpe_ids = EXT_VOCAB[pick_1].bpe_ids
            if length(bpe_buf) + length(bpe_ids) < MAX_BPE_SEQ
                append!(bpe_buf, bpe_ids)
            end

            # Dario field updates
            if pick_vid >= 0
                if length(chain) >= 2
                    update_cooc!(field, chain[end - 1], pick_vid)
                end
                cat = word_category(pick_vid)
                field.destiny[cat + 1] = 0.3 + 0.7 * field.destiny[cat + 1]
                if pick_vid == field.prophecy_target; fulfilled = true; end
            end

            if step > 7; field.trauma = min(1.0, field.trauma + 0.1); end
            field.trauma *= 0.97

            marker = step == GEN_STEPS - 1 ? "  *" : "   "
            println("$(marker)$(pick_word)")
        else
            # WEIGHTLESS MODE: hardcoded words only
            for w in 1:NWORDS
                if VOCAB[w] in forbidden
                    word_scores[w] = -1e9
                end
            end

            probs = _softmax(word_scores[1:NWORDS])
            indexed = sort(collect(enumerate(probs)), by=x -> -x[2])
            indexed = indexed[1:min(12, length(indexed))]
            total = sum(max(0.0, p) for (_, p) in indexed) + 0.001
            r = rand() * total
            pick_1 = indexed[1][1]
            for (idx_1, p) in indexed
                r -= max(0.0, p)
                if r <= 0.0; pick_1 = idx_1; break; end
            end

            pick = pick_1 - 1  # 0-based
            push!(chain, pick)
            push!(forbidden, VOCAB[pick + 1])

            if length(bpe_buf) + length(VOCAB_BPE[pick + 1]) < MAX_BPE_SEQ
                append!(bpe_buf, VOCAB_BPE[pick + 1])
            end

            update_cooc!(field, length(chain) >= 2 ? chain[end - 1] : seed, pick)
            cat = word_category(pick)
            field.destiny[cat + 1] = 0.3 + 0.7 * field.destiny[cat + 1]
            if pick == field.prophecy_target; fulfilled = true; end

            if step > 7; field.trauma = min(1.0, field.trauma + 0.1); end
            field.trauma *= 0.97

            marker = step == GEN_STEPS - 1 ? "  *" : "   "
            println("$(marker)$(VOCAB[pick + 1])")
        end
    end

    cat_flags = zeros(Int, 8)
    cats_seen = 0
    for w in chain
        if w >= 0 && w < NWORDS
            c = word_category(w) + 1
            if cat_flags[c] == 0; cat_flags[c] = 1; cats_seen += 1; end
        end
    end

    println("\n  drift $cats_seen/8 \u00b7 prophecy $(fulfilled ? "fulfilled" : "unfulfilled")")
    return chain
end


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

function main()
    args = copy(ARGS)
    train_path = nothing
    load_path = nothing
    save_path = nothing
    train_steps = 5000
    lr = 3e-4
    text = nothing

    i = 1
    while i <= length(args)
        if args[i] == "--train" && i + 1 <= length(args)
            train_path = args[i + 1]; i += 2
        elseif args[i] == "--load" && i + 1 <= length(args)
            load_path = args[i + 1]; i += 2
        elseif args[i] == "--save" && i + 1 <= length(args)
            save_path = args[i + 1]; i += 2
        elseif args[i] == "--steps" && i + 1 <= length(args)
            train_steps = parse(Int, args[i + 1]); i += 2
        elseif args[i] == "--lr" && i + 1 <= length(args)
            lr = parse(Float64, args[i + 1]); i += 2
        else
            text = join(args[i:end], " "); break
        end
    end

    model = Penelope()
    field = DarioField()

    pc = param_count(model)
    n_ext = length(EXT_VOCAB) - NWORDS
    println()
    println("  penelope v7 \u2014 Resonance engine. 1984 words. Dario Equation.")
    println("  $N_LAYERS layers, $N_HEADS heads, dim=$DIM, hdim=$HDIM")
    println("  $(_format_int(pc)) trainable params ($(round(pc*4/1e6, digits=1))MB f32)")
    println("  BPE input: $BPE_VOCAB subword tokens, max_seq=$MAX_SEQ")
    println("  extended vocab: $(length(EXT_VOCAB)) words ($NWORDS hardcoded + $n_ext from BPE)")
    println("  by Arianna Method")
    println()

    has_weights = false
    if load_path !== nothing && isfile(load_path)
        model_load!(model, load_path)
        has_weights = true
    end

    if train_path !== nothing
        train!(model, train_path, train_steps, lr)
        has_weights = true
        if save_path !== nothing
            model_save(model, save_path)
        end
    end

    println("  mode: $(has_weights ? "trained (BPE word scores)" : "weightless (word-level)")\n")

    if text !== nothing
        run_chain(model, field, text, has_weights)
    elseif train_path === nothing
        while true
            print("  > ")
            flush(stdout)
            line = try
                readline()
            catch
                break
            end
            line = strip(line)
            if isempty(line)
                continue
            end
            run_chain(model, field, String(line), has_weights)
        end
    end

    if save_path !== nothing && train_path === nothing
        model_save(model, save_path)
    end
end

function _format_int(n::Int)
    s = string(n)
    parts = String[]
    while length(s) > 3
        push!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    push!(parts, s)
    return join(reverse(parts), ",")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
