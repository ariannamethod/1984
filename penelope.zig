// penelope.zig — 1984 words. 12 steps of resonance. Dario Equation.
//
// Faithful port of penelope.c to Zig 0.15.2.
// By Arianna Method. הרזוננס לא נשבר

const std = @import("std");

const NWORDS: usize = 1984;
const NSTEPS: usize = 12;
const DIM: usize = 384;
const HDIM: usize = 768;
const MAX_COOC: usize = 32768;
const MAX_BIG: usize = 16384;
const MAX_CTX: usize = 64;

// ═══════════════════════════════════════════════════════════════
// 1984 WORDS
// ═══════════════════════════════════════════════════════════════

const VOCAB = [NWORDS][]const u8{
    // BODY 0-99
    "flesh",     "bone",      "blood",     "skin",      "hand",
    "eye",       "mouth",     "tongue",    "heart",     "lung",
    "vein",      "nerve",     "spine",     "skull",     "rib",
    "breath",    "pulse",     "tremor",    "sweat",     "tear",
    "muscle",    "brain",     "throat",    "womb",      "finger",
    "tooth",     "hair",      "lip",       "shoulder",  "knee",
    "wound",     "scar",      "bruise",    "fever",     "ache",
    "hunger",    "thirst",    "fatigue",   "nausea",    "vertigo",
    "body",      "corpse",    "ghost",     "shadow",    "face",
    "voice",     "whisper",   "scream",    "silence",   "gesture",
    "grip",      "touch",     "embrace",   "fist",      "palm",
    "heel",      "ankle",     "wrist",     "elbow",     "jaw",
    "chest",     "belly",     "hip",       "temple",    "forehead",
    "cheek",     "chin",      "neck",      "back",      "sole",
    "organ",     "cell",      "tissue",    "marrow",    "cartilage",
    "tendon",    "ligament",  "pupil",     "retina",    "cochlea",
    "saliva",    "bile",      "sweat",     "mucus",     "plasma",
    "hormone",   "adrenaline","cortisol",  "dopamine",  "serotonin",
    "synapse",   "neuron",    "dendrite",  "axon",      "reflex",
    "instinct",  "posture",   "gait",      "rhythm",    "trembling",
    // NATURE 100-199
    "sky",       "rain",      "wind",      "stone",     "river",
    "mountain",  "ocean",     "leaf",      "tree",      "root",
    "seed",      "bloom",     "flower",    "petal",     "thorn",
    "earth",     "dust",      "ash",       "fire",      "flame",
    "smoke",     "ember",     "spark",     "water",     "ice",
    "snow",      "frost",     "mist",      "fog",       "dew",
    "sun",       "moon",      "star",      "dawn",      "dusk",
    "midnight",  "morning",   "evening",   "storm",     "thunder",
    "lightning", "rainbow",   "horizon",   "shore",     "sand",
    "salt",      "sea",       "lake",      "creek",     "pool",
    "cave",      "cliff",     "hill",      "valley",    "meadow",
    "forest",    "grove",     "wood",      "bark",      "moss",
    "fern",      "vine",      "lichen",    "fungus",    "coral",
    "kelp",      "whale",     "wolf",      "deer",      "crow",
    "owl",       "hawk",      "moth",      "spider",    "snake",
    "beetle",    "ant",       "bee",       "butterfly", "worm",
    "canyon",    "plateau",   "tundra",    "steppe",    "oasis",
    "dune",      "glacier",   "volcano",   "island",    "peninsula",
    "aurora",    "eclipse",   "zenith",    "equinox",   "solstice",
    "comet",     "nebula",    "cosmos",    "tide",      "current",
    // EMOTION 200-299
    "fear",      "love",      "rage",      "joy",       "grief",
    "sorrow",    "pain",      "pleasure",  "comfort",   "desire",
    "hope",      "despair",   "shame",     "guilt",     "envy",
    "pride",     "longing",   "nostalgia", "regret",    "resolve",
    "courage",   "wisdom",    "patience",  "grace",     "mercy",
    "kindness",  "cruelty",   "justice",   "fury",      "calm",
    "panic",     "dread",     "awe",       "bliss",     "agony",
    "ecstasy",   "melancholy","serenity",  "anxiety",   "contempt",
    "tenderness","devotion",  "hatred",    "spite",     "disgust",
    "wonder",    "confusion", "certainty", "doubt",     "trust",
    "betrayal",  "forgiveness","resentment","gratitude", "humiliation",
    "triumph",   "defeat",    "surrender", "defiance",  "acceptance",
    "jealousy",  "admiration","pity",      "compassion","indifference",
    "obsession", "apathy",    "euphoria",  "desolation","reverence",
    "boredom",   "fascination","horror",   "delight",   "frustration",
    "satisfaction","emptiness","fullness",  "vulnerability","resilience",
    "remorse",   "vindication","bewilderment","clarity","torment",
    "relief",    "yearning",  "contentment","wrath",    "gentleness",
    "paranoia",  "faith",     "skepticism","devotion",  "ambivalence",
    "rapture",   "languor",   "fervor",    "detachment","intimacy",
    // TIME 300-349
    "moment",    "instant",   "second",    "minute",    "hour",
    "day",       "night",     "week",      "month",     "year",
    "decade",    "century",   "epoch",     "era",       "age",
    "past",      "present",   "future",    "memory",    "tomorrow",
    "yesterday", "forever",   "never",     "always",    "sometimes",
    "often",     "seldom",    "once",      "twice",     "origin",
    "ending",    "beginning", "duration",  "interval",  "pause",
    "wait",      "rush",      "delay",     "haste",     "eternity",
    "cycle",     "season",    "spring",    "summer",    "autumn",
    "winter",    "dawn",      "twilight",  "midnight",  "noon",
    // SOCIETY 350-449
    "war",       "peace",     "king",      "queen",     "soldier",
    "citizen",   "exile",     "refugee",   "prisoner",  "judge",
    "law",       "crime",     "punishment","freedom",   "slavery",
    "revolution","democracy", "tyranny",   "empire",    "nation",
    "border",    "wall",      "bridge",    "gate",      "road",
    "market",    "factory",   "hospital",  "school",    "church",
    "money",     "debt",      "wealth",    "poverty",   "labor",
    "trade",     "profit",    "loss",      "tax",       "currency",
    "power",     "authority", "obedience", "rebellion", "protest",
    "silence",   "censorship","propaganda","truth",     "lie",
    "election",  "vote",      "parliament","constitution","right",
    "duty",      "privilege", "corruption","reform",    "collapse",
    "class",     "hierarchy", "equality",  "injustice", "oppression",
    "liberation","resistance","occupation","treaty",    "ceasefire",
    "economy",   "inflation", "depression","prosperity","scarcity",
    "abundance", "famine",    "feast",     "ration",    "surplus",
    "immigrant", "native",    "stranger",  "neighbor",  "ally",
    "enemy",     "traitor",   "hero",      "victim",    "witness",
    "surveillance","privacy", "identity",  "passport",  "boundary",
    "territory", "sovereignty","diplomacy","sanction",  "siege",
    // ABSTRACT 450-549
    "truth",     "meaning",   "purpose",   "existence", "essence",
    "nothing",   "everything","something", "void",      "chaos",
    "order",     "pattern",   "rhythm",    "frequency", "resonance",
    "harmony",   "dissonance","entropy",   "emergence", "threshold",
    "paradox",   "contradiction","ambiguity","certainty","probability",
    "fate",      "chance",    "luck",      "destiny",   "prophecy",
    "dream",     "nightmare", "illusion",  "reality",   "fiction",
    "myth",      "legend",    "story",     "narrative", "silence",
    "question",  "answer",    "riddle",    "secret",    "mystery",
    "clue",      "sign",      "symbol",    "code",      "language",
    "thought",   "idea",      "concept",   "theory",    "belief",
    "knowledge", "ignorance", "wisdom",    "folly",     "genius",
    "beauty",    "ugliness",  "sublime",   "grotesque", "sacred",
    "profane",   "mundane",   "extraordinary","ordinary","unique",
    "infinity",  "zero",      "one",       "half",      "double",
    "mirror",    "echo",      "shadow",    "reflection","ghost",
    "gravity",   "magnetism", "electricity","light",    "darkness",
    "warmth",    "cold",      "pressure",  "vacuum",    "wave",
    "boundary",  "threshold", "edge",      "center",    "margin",
    "surface",   "depth",     "height",    "distance",  "proximity",
    // ACTION 550-649
    "walk",      "run",       "stop",      "breathe",   "sleep",
    "wake",      "dream",     "remember",  "forget",    "imagine",
    "create",    "destroy",   "build",     "break",     "shape",
    "melt",      "freeze",    "burn",      "grow",      "shrink",
    "open",      "close",     "begin",     "end",       "continue",
    "wait",      "search",    "find",      "lose",      "hide",
    "reveal",    "watch",     "listen",    "speak",     "whisper",
    "scream",    "sing",      "dance",     "fight",     "surrender",
    "climb",     "fall",      "rise",      "sink",      "drift",
    "float",     "fly",       "crawl",     "leap",      "stumble",
    "hold",      "release",   "catch",     "throw",     "pull",
    "push",      "lift",      "carry",     "drop",      "pour",
    "cut",       "fold",      "bend",      "twist",     "turn",
    "spin",      "weave",     "knit",      "tie",       "untie",
    "gather",    "scatter",   "merge",     "split",     "connect",
    "separate",  "attract",   "repel",     "collide",   "dissolve",
    "teach",     "learn",     "study",     "practice",  "master",
    "fail",      "succeed",   "attempt",   "abandon",   "persist",
    "give",      "take",      "receive",   "share",     "steal",
    "return",    "exchange",  "sacrifice", "hoard",     "offer",
    // MATERIAL 650-749
    "iron",      "copper",    "gold",      "silver",    "glass",
    "clay",      "wax",       "ink",       "paint",     "paper",
    "silk",      "wool",      "cotton",    "leather",   "stone",
    "marble",    "wood",      "bamboo",    "rope",      "wire",
    "blade",     "needle",    "hammer",    "anvil",     "forge",
    "kiln",      "loom",      "wheel",     "axle",      "lever",
    "mirror",    "lens",      "prism",     "crystal",   "gem",
    "pearl",     "amber",     "jade",      "rust",      "patina",
    "grain",     "fiber",     "thread",    "mesh",      "lattice",
    "grid",      "weave",     "knot",      "stitch",    "patch",
    "vessel",    "bowl",      "cup",       "jar",       "flask",
    "vial",      "key",       "lock",      "chain",     "ring",
    "bell",      "drum",      "string",    "pipe",      "reed",
    "brass",     "horn",      "candle",    "lantern",   "torch",
    "photograph","letter",    "book",      "page",      "chapter",
    "verse",     "sentence",  "paragraph", "word",      "alphabet",
    "map",       "compass",   "clock",     "calendar",  "scale",
    "ruler",     "thermometer","barometer","telescope",  "microscope",
    "machine",   "engine",    "gear",      "spring",    "valve",
    "piston",    "circuit",   "battery",   "signal",    "antenna",
    // FOOD 750-799
    "bread",     "salt",      "sugar",     "honey",     "milk",
    "butter",    "cheese",    "meat",      "fish",      "egg",
    "grain",     "rice",      "wheat",     "corn",      "fruit",
    "apple",     "grape",     "olive",     "lemon",     "pepper",
    "wine",      "water",     "tea",       "coffee",    "broth",
    "soup",      "stew",      "feast",     "crumb",     "morsel",
    "harvest",   "garden",    "soil",      "compost",   "ferment",
    "yeast",     "dough",     "crust",     "marrow",    "nectar",
    "spice",     "herb",      "mint",      "thyme",     "sage",
    "garlic",    "onion",     "mushroom",  "berry",     "kernel",
    // ARCHITECTURE 800-849
    "house",     "room",      "wall",      "floor",     "ceiling",
    "door",      "window",    "stair",     "corridor",  "basement",
    "tower",     "bridge",    "arch",      "column",    "dome",
    "vault",     "foundation","ruin",      "temple",    "altar",
    "threshold", "passage",   "labyrinth", "maze",      "chamber",
    "cell",      "shelter",   "fortress",  "prison",    "garden",
    "roof",      "chimney",   "hearth",    "frame",     "beam",
    "pillar",    "brick",     "mortar",    "tile",      "glass",
    "balcony",   "terrace",   "courtyard", "gate",      "fence",
    "path",      "road",      "intersection","tunnel",  "well",
    // RELATIONSHIP 850-929
    "mother",    "father",    "child",     "daughter",  "son",
    "sister",    "brother",   "family",    "ancestor",  "descendant",
    "friend",    "stranger",  "lover",     "enemy",     "neighbor",
    "companion", "rival",     "mentor",    "student",   "witness",
    "husband",   "wife",      "partner",   "orphan",    "widow",
    "elder",     "infant",    "twin",      "cousin",    "godmother",
    "promise",   "oath",      "vow",       "contract",  "alliance",
    "betrayal",  "reconciliation","farewell","reunion",  "absence",
    "kiss",      "embrace",   "handshake", "slap",      "caress",
    "quarrel",   "conversation","confession","accusation","apology",
    "birth",     "death",     "marriage",  "divorce",   "inheritance",
    "adoption",  "abandonment","protection","neglect",   "sacrifice",
    "trust",     "suspicion", "loyalty",   "treachery", "devotion",
    "indifference","jealousy","admiration","dependence","autonomy",
    "intimacy",  "distance",  "connection","isolation",  "belonging",
    "exile",     "homecoming","departure", "waiting",   "return",
    // PHILOSOPHY 930-999
    "consciousness","awareness","perception","sensation","intuition",
    "reason",    "logic",     "paradox",   "dialectic", "synthesis",
    "freedom",   "determinism","causation","contingency","necessity",
    "possibility","impossibility","actuality","potential","becoming",
    "subject",   "object",    "self",      "other",     "identity",
    "difference","sameness",  "change",    "permanence","flux",
    "being",     "nothingness","existence","essence",   "phenomena",
    "noumena",   "appearance","reality",   "illusion",  "truth",
    "ethics",    "morality",  "virtue",    "vice",      "good",
    "evil",      "right",     "wrong",     "duty",      "choice",
    "justice",   "mercy",     "punishment","reward",    "guilt",
    "innocence", "responsibility","consequence","intention","action",
    "language",  "meaning",   "sign",      "reference", "representation",
    "interpretation","understanding","misunderstanding","translation","silence",
    // MUSIC 1000-1049
    "melody",    "rhythm",    "chord",     "pitch",     "tone",
    "note",      "bass",      "treble",    "octave",    "harmony",
    "dissonance","resonance", "vibration", "frequency", "amplitude",
    "tempo",     "beat",      "rest",      "pause",     "crescendo",
    "murmur",    "hum",       "buzz",      "click",     "crack",
    "boom",      "rumble",    "chime",     "echo",      "reverb",
    "song",      "lullaby",   "anthem",    "dirge",     "hymn",
    "ballad",    "fugue",     "sonata",    "requiem",   "improvisation",
    "strum",     "pluck",     "strike",    "bow",       "mute",
    "sustain",   "fade",      "loop",      "drone",     "overtone",
    // WEATHER 1050-1099
    "rain",      "drizzle",   "downpour",  "hail",      "sleet",
    "blizzard",  "hurricane", "tornado",   "drought",   "flood",
    "breeze",    "gale",      "typhoon",   "monsoon",   "frost",
    "thaw",      "haze",      "smog",      "rainbow",   "mirage",
    "erosion",   "sedimentation","crystallization","evaporation","condensation",
    "precipitation","sublimation","oxidation","combustion","decay",
    "magma",     "lava",      "quartz",    "granite",   "obsidian",
    "chalk",     "slate",     "sandstone", "limestone", "basalt",
    "marsh",     "delta",     "gorge",     "ridge",     "summit",
    "abyss",     "chasm",     "rift",      "fault",     "crater",
    // RITUAL 1100-1149
    "prayer",    "meditation","ritual",    "ceremony",  "blessing",
    "curse",     "oath",      "vow",       "pilgrimage","procession",
    "offering",  "sacrifice", "communion", "baptism",   "funeral",
    "wedding",   "coronation","initiation","exile",     "absolution",
    "incense",   "candle",    "bell",      "chant",     "mantra",
    "psalm",     "scripture", "prophecy",  "oracle",    "vision",
    "mask",      "costume",   "dance",     "feast",     "fast",
    "vigil",     "silence",   "confession","penance",   "redemption",
    "altar",     "shrine",    "temple",    "tomb",      "relic",
    "artifact",  "amulet",    "talisman",  "totem",     "icon",
    // LABOR 1150-1199
    "harvest",   "planting",  "sowing",    "reaping",   "threshing",
    "milling",   "baking",    "brewing",   "weaving",   "spinning",
    "carving",   "sculpting", "painting",  "drawing",   "writing",
    "printing",  "binding",   "stitching", "welding",   "forging",
    "mining",    "drilling",  "excavation","construction","demolition",
    "repair",    "restoration","invention","discovery",  "experiment",
    "apprentice","craftsman", "artist",    "engineer",  "architect",
    "farmer",    "sailor",    "miner",     "healer",    "scribe",
    "workshop",  "studio",    "laboratory","field",     "dock",
    "quarry",    "furnace",   "mill",      "press",     "loom",
    // GEOMETRY 1200-1249
    "circle",    "spiral",    "line",      "curve",     "angle",
    "edge",      "center",    "margin",    "border",    "frame",
    "sphere",    "cube",      "pyramid",   "cylinder",  "cone",
    "helix",     "vortex",    "arc",       "wave",      "fractal",
    "symmetry",  "asymmetry", "proportion","ratio",     "scale",
    "dimension", "plane",     "axis",      "vertex",    "intersection",
    "pattern",   "grid",      "lattice",   "mesh",      "tessellation",
    "rotation",  "reflection","translation","dilation",  "projection",
    "surface",   "volume",    "area",      "perimeter", "diameter",
    "radius",    "tangent",   "normal",    "parallel",  "perpendicular",
    // ANIMAL 1250-1299
    "horse",     "dog",       "cat",       "bird",      "fish",
    "snake",     "bear",      "fox",       "rabbit",    "turtle",
    "eagle",     "sparrow",   "raven",     "swan",      "heron",
    "falcon",    "vulture",   "pelican",   "nightingale","lark",
    "lion",      "tiger",     "elephant",  "giraffe",   "hippopotamus",
    "rhinoceros","gorilla",   "chimpanzee","orangutan", "leopard",
    "salmon",    "trout",     "shark",     "dolphin",   "octopus",
    "jellyfish", "starfish",  "seahorse",  "crab",      "lobster",
    "frog",      "lizard",    "crocodile", "chameleon", "gecko",
    "iguana",    "newt",      "toad",      "salamander","viper",
    // COLOR 1300-1349
    "red",       "blue",      "green",     "white",     "black",
    "gray",      "amber",     "violet",    "indigo",    "scarlet",
    "crimson",   "azure",     "emerald",   "ivory",     "obsidian",
    "silver",    "golden",    "copper",    "rust",      "ochre",
    "bright",    "dark",      "transparent","opaque",   "matte",
    "glossy",    "rough",     "smooth",    "coarse",    "fine",
    "stripe",    "dot",       "plaid",     "solid",     "gradient",
    "shadow",    "highlight", "contrast",  "saturation","hue",
    "velvet",    "satin",     "linen",     "denim",     "lace",
    "gauze",     "burlap",    "chiffon",   "tweed",     "corduroy",
    // TRANSPORT 1350-1399
    "ship",      "boat",      "canoe",     "raft",      "anchor",
    "sail",      "rudder",    "oar",       "mast",      "hull",
    "train",     "rail",      "station",   "platform",  "ticket",
    "journey",   "passage",   "crossing",  "departure", "arrival",
    "wheel",     "axle",      "road",      "highway",   "path",
    "trail",     "bridge",    "tunnel",    "gate",      "crossroad",
    "wing",      "flight",    "altitude",  "turbulence","landing",
    "orbit",     "trajectory","velocity",  "acceleration","gravity",
    "horse",     "carriage",  "wagon",     "cart",      "sled",
    "bicycle",   "motorcycle","automobile","truck",      "ambulance",
    // DOMESTIC 1400-1449
    "kitchen",   "bedroom",   "bathroom",  "attic",     "cellar",
    "closet",    "drawer",    "shelf",     "table",     "chair",
    "bed",       "pillow",    "blanket",   "curtain",   "carpet",
    "lamp",      "mirror",    "photograph","vase",       "clock",
    "plate",     "spoon",     "knife",     "fork",      "cup",
    "pot",       "pan",       "kettle",    "oven",      "stove",
    "soap",      "towel",     "broom",     "bucket",    "needle",
    "thread",    "button",    "zipper",    "hanger",    "basket",
    "door",      "window",    "lock",      "key",       "handle",
    "hinge",     "nail",      "screw",     "bolt",      "hook",
    // COMMUNICATION 1450-1499
    "letter",    "envelope",  "stamp",     "address",   "message",
    "telegram",  "telephone", "radio",     "broadcast", "signal",
    "newspaper", "headline",  "article",   "column",    "editorial",
    "report",    "announcement","rumor",   "gossip",    "testimony",
    "ink",       "pen",       "pencil",    "typewriter","keyboard",
    "screen",    "printer",   "paper",     "notebook",  "diary",
    "conversation","dialogue","monologue", "debate",    "argument",
    "negotiation","compromise","ultimatum","declaration","speech",
    "translation","interpretation","code", "cipher",    "encryption",
    "decryption","password",  "signature", "seal",      "authentication",
    // MEDICAL 1500-1549
    "diagnosis", "symptom",   "treatment", "remedy",    "cure",
    "relapse",   "recovery",  "surgery",   "anesthesia","bandage",
    "infection", "inflammation","fracture","hemorrhage","allergy",
    "immunity",  "vaccine",   "antibiotic","toxin",     "antidote",
    "hospital",  "clinic",    "pharmacy",  "laboratory","ambulance",
    "stretcher", "scalpel",   "syringe",   "stethoscope","thermometer",
    "fever",     "cough",     "rash",      "swelling",  "numbness",
    "dizziness", "insomnia",  "fatigue",   "nausea",    "tremor",
    "pulse",     "pressure",  "temperature","respiration","circulation",
    "digestion", "metabolism","reflex",    "coordination","balance",
    // COSMIC 1550-1599
    "universe",  "galaxy",    "constellation","planet", "asteroid",
    "meteorite", "satellite", "orbit",     "void",      "singularity",
    "photon",    "electron",  "proton",    "neutron",   "atom",
    "molecule",  "particle",  "quantum",   "field",     "dimension",
    "spacetime", "relativity","entropy",   "thermodynamics","radiation",
    "spectrum",  "wavelength","frequency", "amplitude", "interference",
    "supernova", "blackhole", "pulsar",    "quasar",    "nebula",
    "wormhole",  "antimatter","darkmatter","redshift",  "expansion",
    "telescope", "observatory","mission",  "launch",    "countdown",
    "trajectory","reentry",   "landing",   "exploration","discovery",
    // BUREAUCRACY 1600-1649
    "document",  "form",      "permit",    "license",   "certificate",
    "registration","application","approval","denial",    "appeal",
    "regulation","compliance","violation", "penalty",   "exemption",
    "quota",     "deadline",  "protocol",  "procedure", "standard",
    "office",    "desk",      "file",      "folder",    "stamp",
    "signature", "receipt",   "invoice",   "ledger",    "archive",
    "committee", "department","ministry",  "bureau",    "agency",
    "institution","organization","corporation","foundation","commission",
    "report",    "audit",     "review",    "inspection","evaluation",
    "assessment","benchmark", "statistic", "data",      "record",
    // MYTHIC 1650-1699
    "oracle",    "prophecy",  "fate",      "destiny",   "curse",
    "blessing",  "quest",     "trial",     "sacrifice", "redemption",
    "labyrinth", "threshold", "guardian",  "shadow",    "mirror",
    "mask",      "transformation","metamorphosis","resurrection","apocalypse",
    "phoenix",   "dragon",    "serpent",   "sphinx",    "minotaur",
    "chimera",   "hydra",     "golem",     "specter",   "wraith",
    "underworld","paradise",  "purgatory", "limbo",     "abyss",
    "eden",      "babylon",   "atlantis",  "olympus",   "tartarus",
    "hero",      "villain",   "trickster", "sage",      "fool",
    "maiden",    "crone",     "warrior",   "healer",    "shapeshifter",
    // TEXTUAL 1700-1749
    "word",      "sentence",  "paragraph", "chapter",   "verse",
    "stanza",    "line",      "margin",    "footnote",  "epilogue",
    "prologue",  "preface",   "title",     "subtitle",  "dedication",
    "inscription","epitaph",  "motto",     "slogan",    "proverb",
    "metaphor",  "simile",    "allegory",  "irony",     "satire",
    "parody",    "tragedy",   "comedy",    "farce",     "melodrama",
    "narrator",  "character", "protagonist","antagonist","audience",
    "reader",    "author",    "critic",    "editor",    "translator",
    "manuscript","draft",     "revision",  "erasure",   "correction",
    "annotation","citation",  "reference", "index",     "bibliography",
    // PSYCHOLOGICAL 1750-1799
    "unconscious","subconscious","conscious","ego",     "superego",
    "libido",    "repression","projection","sublimation","transference",
    "trauma",    "complex",   "fixation",  "regression","denial",
    "rationalization","displacement","compensation","identification","dissociation",
    "archetype", "persona",   "anima",     "animus",    "shadow",
    "self",      "individuation","integration","fragmentation","wholeness",
    "attachment","separation","abandonment","dependency","autonomy",
    "codependency","boundary","enmeshment","differentiation","fusion",
    "grief",     "mourning",  "acceptance","bargaining","anger",
    "depression","recovery",  "relapse",   "healing",   "scarring",
    // FINAL 1800-1983
    "threshold", "crossroad", "watershed", "turning",   "pivot",
    "fulcrum",   "catalyst",  "trigger",   "spark",     "fuse",
    "tension",   "release",   "compression","expansion","contraction",
    "oscillation","vibration","pulsation", "undulation","fluctuation",
    "accumulation","erosion", "saturation","depletion",  "renewal",
    "regeneration","decomposition","fermentation","crystallization","dissolution",
    "echo",      "reverberation","aftershock","aftermath","residue",
    "remnant",   "trace",     "vestige",   "fossil",    "ruin",
    "dawn",      "twilight",  "liminal",   "transitional","ephemeral",
    "permanent", "transient", "enduring",  "fleeting",  "eternal",
    "anchor",    "drift",     "mooring",   "compass",   "lighthouse",
    "beacon",    "signal",    "warning",   "invitation","summons",
    "whisper",   "murmur",    "declaration","proclamation","confession",
    "accusation","plea",      "verdict",   "sentence",  "pardon",
    "seed",      "sprout",    "bud",       "blossom",   "fruit",
    "harvest",   "decay",     "compost",   "soil",      "rebirth",
    "wound",     "suture",    "bandage",   "scar",      "healing",
    "infection", "immunity",  "antibody",  "fever",     "remission",
    "stranger",  "acquaintance","confidant","accomplice","bystander",
    "mediator",  "advocate",  "adversary", "guardian",  "orphan",
    "question",  "hypothesis","experiment","observation","conclusion",
    "revision",  "doubt",     "certainty", "approximation","precision",
    "fragment",  "mosaic",    "collage",   "assemblage","montage",
    "palimpsest","tapestry",  "constellation","archipelago","network",
    "migration", "exodus",    "diaspora",  "pilgrimage","wandering",
    "settlement","foundation","demolition","reconstruction","adaptation",
    "inheritance","legacy",   "tradition", "innovation","rupture",
    "continuity","evolution", "revolution","stagnation","metamorphosis",
    "silence",   "static",    "noise",     "signal",    "frequency",
    "wavelength","amplitude", "resonance", "interference","harmony",
    "margin",    "periphery", "frontier",  "borderland","hinterland",
    "interior",  "core",      "nucleus",   "membrane",  "skin",
    "permission","prohibition","transgression","taboo", "norm",
    "deviation", "exception", "precedent", "custom",    "habit",
    "witness",   "testimony", "evidence",  "proof",     "alibi",
    "verdict",   "appeal",    "clemency",  "execution", "reprieve",
    "debt",      "credit",    "interest",  "principal",
};

// ═══════════════════════════════════════════════════════════════
// STOPWORDS
// ═══════════════════════════════════════════════════════════════

const STOPS = [_][]const u8{
    "i",    "me",    "my",    "we",    "our",   "you",   "your",  "he",
    "she",  "it",    "they",  "them",  "the",   "a",     "an",    "and",
    "or",   "but",   "in",    "on",    "at",    "to",    "for",   "of",
    "is",   "am",    "are",   "was",   "were",  "be",    "been",  "being",
    "have", "has",   "had",   "do",    "does",  "did",   "will",  "would",
    "shall","should","can",   "could", "may",   "might", "must",  "not",
    "no",   "nor",   "so",    "if",    "then",  "than",  "that",  "this",
    "these","those", "what",  "which", "who",   "whom",  "how",   "when",
    "where","why",   "all",   "each",  "every", "some",  "any",   "few",
    "many", "much",  "more",  "most",  "other", "another","such",
};

fn isStop(w: []const u8) bool {
    for (STOPS) |s| {
        if (std.mem.eql(u8, w, s)) return true;
    }
    return false;
}

fn findWord(w: []const u8) i32 {
    for (0..NWORDS) |i| {
        if (std.mem.eql(u8, VOCAB[i], w)) return @intCast(i);
    }
    return -1;
}

fn wordCategory(idx: usize) usize {
    if (idx < 100) return 0;
    if (idx < 200) return 1;
    if (idx < 300) return 2;
    if (idx < 350) return 3;
    if (idx < 450) return 4;
    if (idx < 550) return 5;
    if (idx < 650) return 6;
    return 7;
}

// ═══════════════════════════════════════════════════════════════
// MATH UTILS
// ═══════════════════════════════════════════════════════════════

var g_prng: std.Random.DefaultPrng = undefined;

fn randf() f32 {
    return g_prng.random().float(f32);
}

fn clampf(x: f32, lo: f32, hi: f32) f32 {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

fn randn() f32 {
    const r1 = randf() + 1e-12;
    const r2 = randf() + 1e-12;
    return @sqrt(-2.0 * @log(r1)) * @cos(6.2831853 * r2);
}

fn siluf(x: f32) f32 {
    if (x > -20.0) {
        return x / (1.0 + @exp(-x));
    }
    return 0.0;
}

// ═══════════════════════════════════════════════════════════════
// TRAINABLE MODEL — 12 step-specific weight sets + shared embed
// ═══════════════════════════════════════════════════════════════

const StepWeights = struct {
    wr: []f32, // [DIM * DIM]
    rms: []f32, // [DIM]
    w_gate: []f32, // [DIM * HDIM]
    w_up: []f32, // [DIM * HDIM]
    w_down: []f32, // [HDIM * DIM]
};

const StepAdam = struct {
    wr_m: []f32,
    wr_v: []f32,
    rms_m: []f32,
    rms_v: []f32,
    gate_m: []f32,
    gate_v: []f32,
    up_m: []f32,
    up_v: []f32,
    down_m: []f32,
    down_v: []f32,
};

const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

const Model = struct {
    embed: []f32, // [NWORDS * DIM]
    embed_m: []f32,
    embed_v: []f32,
    steps: [NSTEPS]StepWeights,
    adam: [NSTEPS]StepAdam,
    adam_t: i32,
};

fn stepParamCount() usize {
    return DIM * DIM + DIM + DIM * HDIM + DIM * HDIM + HDIM * DIM;
}

fn totalParamCount() usize {
    return NWORDS * DIM + NSTEPS * stepParamCount();
}

fn allocZeroed(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const slice = try allocator.alloc(f32, n);
    @memset(slice, 0);
    return slice;
}

fn modelInit(m: *Model, allocator: std.mem.Allocator) !void {
    const embed_sz = NWORDS * DIM;
    const scale_v: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(NWORDS)));
    const scale_d: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(DIM)));
    const scale_m: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(HDIM)));

    m.embed = try allocator.alloc(f32, embed_sz);
    m.embed_m = try allocZeroed(allocator, embed_sz);
    m.embed_v = try allocZeroed(allocator, embed_sz);
    m.adam_t = 0;

    for (0..embed_sz) |i| {
        m.embed[i] = randn() * scale_v;
    }

    for (0..NSTEPS) |s| {
        m.steps[s].wr = try allocator.alloc(f32, DIM * DIM);
        m.steps[s].rms = try allocator.alloc(f32, DIM);
        m.steps[s].w_gate = try allocator.alloc(f32, DIM * HDIM);
        m.steps[s].w_up = try allocator.alloc(f32, DIM * HDIM);
        m.steps[s].w_down = try allocator.alloc(f32, HDIM * DIM);

        m.adam[s].wr_m = try allocZeroed(allocator, DIM * DIM);
        m.adam[s].wr_v = try allocZeroed(allocator, DIM * DIM);
        m.adam[s].rms_m = try allocZeroed(allocator, DIM);
        m.adam[s].rms_v = try allocZeroed(allocator, DIM);
        m.adam[s].gate_m = try allocZeroed(allocator, DIM * HDIM);
        m.adam[s].gate_v = try allocZeroed(allocator, DIM * HDIM);
        m.adam[s].up_m = try allocZeroed(allocator, DIM * HDIM);
        m.adam[s].up_v = try allocZeroed(allocator, DIM * HDIM);
        m.adam[s].down_m = try allocZeroed(allocator, HDIM * DIM);
        m.adam[s].down_v = try allocZeroed(allocator, HDIM * DIM);

        for (0..DIM * DIM) |i| m.steps[s].wr[i] = randn() * scale_d;
        for (0..DIM) |i| m.steps[s].rms[i] = 1.0;
        for (0..DIM * HDIM) |i| m.steps[s].w_gate[i] = randn() * scale_d;
        for (0..DIM * HDIM) |i| m.steps[s].w_up[i] = randn() * scale_d;
        for (0..HDIM * DIM) |i| m.steps[s].w_down[i] = randn() * scale_m;
    }
}

fn modelFree(m: *Model, allocator: std.mem.Allocator) void {
    allocator.free(m.embed);
    allocator.free(m.embed_m);
    allocator.free(m.embed_v);
    for (0..NSTEPS) |s| {
        allocator.free(m.steps[s].wr);
        allocator.free(m.steps[s].rms);
        allocator.free(m.steps[s].w_gate);
        allocator.free(m.steps[s].w_up);
        allocator.free(m.steps[s].w_down);
        allocator.free(m.adam[s].wr_m);
        allocator.free(m.adam[s].wr_v);
        allocator.free(m.adam[s].rms_m);
        allocator.free(m.adam[s].rms_v);
        allocator.free(m.adam[s].gate_m);
        allocator.free(m.adam[s].gate_v);
        allocator.free(m.adam[s].up_m);
        allocator.free(m.adam[s].up_v);
        allocator.free(m.adam[s].down_m);
        allocator.free(m.adam[s].down_v);
    }
}

// ═══════════════════════════════════════════════════════════════
// ADAM OPTIMIZER
// ═══════════════════════════════════════════════════════════════

fn adamUpdate(w: []f32, am: []f32, av: []f32, grad: []f32, lr: f32, bc1: f32, bc2: f32) void {
    for (0..w.len) |i| {
        const g = grad[i];
        am[i] = ADAM_B1 * am[i] + (1 - ADAM_B1) * g;
        av[i] = ADAM_B2 * av[i] + (1 - ADAM_B2) * g * g;
        const mhat = am[i] / bc1;
        const vhat = av[i] / bc2;
        w[i] -= lr * mhat / (@sqrt(vhat) + ADAM_EPS);
        grad[i] = 0;
    }
}

fn modelSave(m: *Model, path: []const u8) !void {
    const stderr = (std.fs.File{ .handle = std.posix.STDERR_FILENO }).deprecatedWriter();
    const stdout_w = (std.fs.File{ .handle = std.posix.STDOUT_FILENO }).deprecatedWriter();

    const file = std.fs.cwd().createFile(path, .{}) catch {
        try stderr.print("  cannot open {s} for writing\n", .{path});
        return;
    };
    defer file.close();
    const writer = file.deprecatedWriter();

    // Write header
    const header = [4]i32{ @intCast(NWORDS), @intCast(DIM), @intCast(HDIM), @intCast(NSTEPS) };
    try writer.writeAll(std.mem.sliceAsBytes(&header));

    // Write embed
    try writer.writeAll(std.mem.sliceAsBytes(m.embed));

    // Write step weights
    for (0..NSTEPS) |s| {
        try writer.writeAll(std.mem.sliceAsBytes(m.steps[s].wr));
        try writer.writeAll(std.mem.sliceAsBytes(m.steps[s].rms));
        try writer.writeAll(std.mem.sliceAsBytes(m.steps[s].w_gate));
        try writer.writeAll(std.mem.sliceAsBytes(m.steps[s].w_up));
        try writer.writeAll(std.mem.sliceAsBytes(m.steps[s].w_down));
    }

    // Verify
    const check = try std.fs.cwd().openFile(path, .{});
    defer check.close();
    const sz = try check.getEndPos();
    const expected: u64 = 16 + @as(u64, totalParamCount()) * 4;
    const ok_str = if (sz == expected) "OK" else "SIZE MISMATCH!";
    try stdout_w.print("  saved {s}: {d} params ({d:.1}MB) [{s}]\n", .{
        path,
        totalParamCount(),
        @as(f64, @floatFromInt(sz)) / 1e6,
        ok_str,
    });
}

fn modelLoad(m: *Model, path: []const u8) !bool {
    const stderr = (std.fs.File{ .handle = std.posix.STDERR_FILENO }).deprecatedWriter();
    const stdout_w = (std.fs.File{ .handle = std.posix.STDOUT_FILENO }).deprecatedWriter();

    const file = std.fs.cwd().openFile(path, .{}) catch {
        try stderr.print("  cannot open {s}\n", .{path});
        return false;
    };
    defer file.close();
    const reader = file.deprecatedReader();

    // Read header
    var header: [4]i32 = undefined;
    const header_bytes = std.mem.sliceAsBytes(&header);
    const n_read = try reader.readAll(header_bytes);
    if (n_read != header_bytes.len) {
        try stderr.print("  cannot read header from {s}\n", .{path});
        return false;
    }

    if (header[0] != @as(i32, NWORDS) or header[1] != @as(i32, DIM) or
        header[2] != @as(i32, HDIM) or header[3] != @as(i32, NSTEPS))
    {
        try stderr.print("  config mismatch: V={d} D={d} M={d} S={d}\n", .{
            header[0], header[1], header[2], header[3],
        });
        return false;
    }

    // Read embed
    _ = try reader.readAll(std.mem.sliceAsBytes(m.embed));

    // Read step weights
    for (0..NSTEPS) |s| {
        _ = try reader.readAll(std.mem.sliceAsBytes(m.steps[s].wr));
        _ = try reader.readAll(std.mem.sliceAsBytes(m.steps[s].rms));
        _ = try reader.readAll(std.mem.sliceAsBytes(m.steps[s].w_gate));
        _ = try reader.readAll(std.mem.sliceAsBytes(m.steps[s].w_up));
        _ = try reader.readAll(std.mem.sliceAsBytes(m.steps[s].w_down));
    }

    try stdout_w.print("  loaded {s}: {d} params\n", .{ path, totalParamCount() });
    return true;
}

// ═══════════════════════════════════════════════════════════════
// FORWARD — one step produces logits[NWORDS]
// ═══════════════════════════════════════════════════════════════

fn poolContext(m: *Model, ids: []const i32, ctx: []f32) void {
    const n = ids.len;
    @memset(ctx[0..DIM], 0);
    for (0..n) |i| {
        const id: usize = @intCast(ids[i]);
        for (0..DIM) |j| {
            ctx[j] += m.embed[id * DIM + j];
        }
    }
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(if (n > 0) n else 1));
    for (0..DIM) |j| {
        ctx[j] *= inv;
    }
}

fn matmulMv(W: []const f32, x: []const f32, out: []f32, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        var s: f32 = 0;
        for (0..cols) |j| {
            s += W[i * cols + j] * x[j];
        }
        out[i] = s;
    }
}

fn matmulMtv(W: []const f32, x: []const f32, out: []f32, rows: usize, cols: usize) void {
    for (0..cols) |j| {
        var s: f32 = 0;
        for (0..rows) |i| {
            s += W[i * cols + j] * x[i];
        }
        out[j] = s;
    }
}

fn rmsnorm(x: []const f32, g: []const f32, out: []f32, n: usize) void {
    var ss: f32 = 0;
    for (0..n) |i| ss += x[i] * x[i];
    ss = ss / @as(f32, @floatFromInt(n)) + 1e-5;
    const inv = 1.0 / @sqrt(ss);
    for (0..n) |i| out[i] = g[i] * x[i] * inv;
}

fn forwardStep(
    m: *Model,
    ctx_ids: []const i32,
    step_idx: usize,
    logits: []f32,
    query: []f32,
    query_n: []f32,
    gate_buf: []f32,
    up_buf: []f32,
    swiglu_buf: []f32,
    hidden: []f32,
    out: []f32,
) void {
    const sw = &m.steps[step_idx];
    var ctx: [DIM]f32 = undefined;
    poolContext(m, ctx_ids, &ctx);

    // RRPRAM: query = ctx @ Wr
    matmulMv(sw.wr, &ctx, query, DIM, DIM);
    // RMSNorm
    rmsnorm(query, sw.rms, query_n, DIM);
    // SwiGLU
    matmulMv(sw.w_gate, query_n, gate_buf, HDIM, DIM);
    matmulMv(sw.w_up, query_n, up_buf, HDIM, DIM);
    for (0..HDIM) |i| {
        swiglu_buf[i] = siluf(gate_buf[i]) * up_buf[i];
    }
    matmulMv(sw.w_down, swiglu_buf, hidden, DIM, HDIM);
    // Residual
    for (0..DIM) |i| {
        out[i] = query_n[i] + hidden[i];
    }
    // Logits = E @ out (tied weights)
    matmulMv(m.embed, out, logits, NWORDS, DIM);
}

// ═══════════════════════════════════════════════════════════════
// SOFTMAX
// ═══════════════════════════════════════════════════════════════

fn softmaxV(x: []const f32, out: []f32, n: usize) void {
    var mx = x[0];
    for (1..n) |i| {
        if (x[i] > mx) mx = x[i];
    }
    var s: f32 = 0;
    for (0..n) |i| {
        out[i] = @exp(x[i] - mx);
        s += out[i];
    }
    for (0..n) |i| {
        out[i] /= s;
    }
}

// ═══════════════════════════════════════════════════════════════
// DARIO FIELD — live heuristic overlay
// ═══════════════════════════════════════════════════════════════

const CoocEntry = struct { a: i32, b: i32, val: f32 };
const BigramEntry = struct { prev: i32, next: i32, val: f32 };

var cooc: [MAX_COOC]CoocEntry = undefined;
var cooc_n: usize = 0;
var bigs: [MAX_BIG]BigramEntry = undefined;
var big_n: usize = 0;
var destiny = [8]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
var trauma: f32 = 0;
var prophecy_target: i32 = -1;
var prophecy_age: i32 = 0;

// Kuramoto chambers
const CH_FEAR: usize = 0;
const CH_LOVE: usize = 1;
const CH_RAGE: usize = 2;
const CH_VOID: usize = 3;
const CH_FLOW: usize = 4;
const CH_COMPLEX: usize = 5;
const NCH: usize = 6;

var chambers = [NCH]f32{ 0, 0, 0, 0, 0, 0 };
const ch_decay = [NCH]f32{ 0.95, 0.95, 0.93, 0.96, 0.94, 0.97 };

fn coocUpdate(a_in: i32, b_in: i32) void {
    var a = a_in;
    var b = b_in;
    if (a > b) {
        const t = a;
        a = b;
        b = t;
    }
    for (0..cooc_n) |i| {
        if (cooc[i].a == a and cooc[i].b == b) {
            cooc[i].val += 1.0;
            return;
        }
    }
    if (cooc_n < MAX_COOC) {
        cooc[cooc_n] = CoocEntry{ .a = a, .b = b, .val = 1.0 };
        cooc_n += 1;
    }
}

fn coocGet(a_in: i32, b_in: i32) f32 {
    var a = a_in;
    var b = b_in;
    if (a > b) {
        const t = a;
        a = b;
        b = t;
    }
    for (0..cooc_n) |i| {
        if (cooc[i].a == a and cooc[i].b == b) return cooc[i].val;
    }
    return 0;
}

fn updateChambers(step_idx: usize) void {
    const depth = @as(f32, @floatFromInt(step_idx)) / @as(f32, @floatFromInt(NSTEPS));
    const phase: usize = if (depth < 0.33) 0 else if (depth < 0.66) 1 else 2;
    if (phase == 0) chambers[CH_FLOW] += 0.05;
    if (phase == 1) chambers[CH_FEAR] += 0.04;
    if (phase == 2) chambers[CH_VOID] += 0.05;
    if (depth > 0.75) chambers[CH_COMPLEX] += 0.03;
    if (trauma > 0.3) chambers[CH_RAGE] += 0.04;

    const K: f32 = 0.02;
    var old: [NCH]f32 = undefined;
    @memcpy(&old, &chambers);
    for (0..NCH) |i| {
        for (0..NCH) |j| {
            if (i != j) chambers[i] += K * @sin(old[j] - old[i]);
        }
    }
    for (0..NCH) |i| {
        chambers[i] = clampf(chambers[i] * ch_decay[i], 0, 1);
    }
}

fn darioOverlay(logits: []f32, ctx: []const i32, step: usize) void {
    _ = step;
    const alpha_mod = 1 + 0.3 * chambers[CH_LOVE] - 0.2 * chambers[CH_RAGE] + 0.1 * chambers[CH_FLOW];
    const gamma_mod = 1 + 0.4 * chambers[CH_VOID] + 0.2 * chambers[CH_COMPLEX];

    const ctx_n = ctx.len;
    const last_n: usize = if (ctx_n > 8) 8 else ctx_n;
    const start = ctx_n - last_n;

    for (0..NWORDS) |v| {
        // H: Hebbian co-occurrence
        var H: f32 = 0;
        for (start..ctx_n) |i| {
            H += coocGet(ctx[i], @intCast(v));
        }
        if (H > 1) H = 1;
        logits[v] += alpha_mod * 0.3 * H;

        // F: prophecy
        if (prophecy_target >= 0 and v == @as(usize, @intCast(prophecy_target))) {
            logits[v] += 0.5 * @log(1.0 + @as(f32, @floatFromInt(prophecy_age)));
        }

        // A: destiny
        const cat = wordCategory(v);
        var d_max: f32 = 0.01;
        for (0..8) |i| {
            if (@abs(destiny[i]) > d_max) d_max = @abs(destiny[i]);
        }
        logits[v] += gamma_mod * 0.25 * destiny[cat] / d_max;
    }
}

// ═══════════════════════════════════════════════════════════════
// TOKENIZE — extract vocab word IDs from text
// ═══════════════════════════════════════════════════════════════

fn isDelimiter(c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', ',', '.', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']', '{', '}' => true,
        else => false,
    };
}

fn tokenizeText(text: []const u8, ids: []i32) usize {
    var buf: [4096]u8 = undefined;
    var bi: usize = 0;
    for (text) |c| {
        if (bi >= 4094) break;
        buf[bi] = std.ascii.toLower(c);
        bi += 1;
    }
    const input = buf[0..bi];

    var n: usize = 0;
    var i: usize = 0;
    while (i < input.len and n < ids.len) {
        // skip delimiters
        while (i < input.len and isDelimiter(input[i])) : (i += 1) {}
        if (i >= input.len) break;
        // find end of token
        const tok_start = i;
        while (i < input.len and !isDelimiter(input[i])) : (i += 1) {}
        const tok = input[tok_start..i];

        if (tok.len < 2 or isStop(tok)) continue;

        const idx = findWord(tok);
        if (idx >= 0) {
            ids[n] = idx;
            n += 1;
        } else {
            // prefix match
            var best: i32 = -1;
            var best_len: usize = 0;
            for (0..NWORDS) |vi| {
                const vocab_word = VOCAB[vi];
                const mn = @min(vocab_word.len, tok.len);
                var plen: usize = 0;
                for (0..mn) |j| {
                    if (vocab_word[j] == tok[j]) {
                        plen += 1;
                    } else break;
                }
                if (plen > best_len) {
                    best_len = plen;
                    best = @intCast(vi);
                }
            }
            if (best >= 0 and best_len >= 3) {
                ids[n] = best;
                n += 1;
            }
        }
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════
// TRAINING — next-word prediction
// ═══════════════════════════════════════════════════════════════

fn train(m: *Model, data_path: []const u8, train_steps_count: i32, lr: f32, allocator: std.mem.Allocator) !void {
    const stderr = (std.fs.File{ .handle = std.posix.STDERR_FILENO }).deprecatedWriter();
    const stdout_w = (std.fs.File{ .handle = std.posix.STDOUT_FILENO }).deprecatedWriter();

    const file = std.fs.cwd().openFile(data_path, .{}) catch {
        try stderr.print("  cannot open {s}\n", .{data_path});
        return;
    };
    defer file.close();

    const fsz = try file.getEndPos();
    const text = try allocator.alloc(u8, fsz);
    defer allocator.free(text);
    _ = try file.deprecatedReader().readAll(text);

    const max_ids = fsz / 2 + 1;
    const ids = try allocator.alloc(i32, max_ids);
    defer allocator.free(ids);
    const n_ids = tokenizeText(text, ids);

    const window: usize = NSTEPS + 1;
    if (n_ids < window + 1) {
        try stderr.print("  corpus too small: {d} words (need {d}+)\n", .{ n_ids, window + 1 });
        return;
    }

    try stdout_w.print("  corpus: {d} bytes -> {d} vocab words\n", .{ fsz, n_ids });
    try stdout_w.print("  model: {d} params ({d:.1}MB f32)\n", .{
        totalParamCount(),
        @as(f64, @floatFromInt(totalParamCount())) * 4.0 / 1e6,
    });
    try stdout_w.print("  optimizer: Adam (Chuck lineage) b1={d:.1} b2={d:.3}\n", .{ ADAM_B1, ADAM_B2 });
    try stdout_w.print("  training: {d} steps, lr={e}\n", .{ train_steps_count, lr });

    // alloc scratch buffers
    const logits = try allocator.alloc(f32, NWORDS);
    defer allocator.free(logits);
    const probs = try allocator.alloc(f32, NWORDS);
    defer allocator.free(probs);
    const d_logits = try allocator.alloc(f32, NWORDS);
    defer allocator.free(d_logits);
    const d_out = try allocator.alloc(f32, DIM);
    defer allocator.free(d_out);
    const query = try allocator.alloc(f32, DIM);
    defer allocator.free(query);
    const query_n = try allocator.alloc(f32, DIM);
    defer allocator.free(query_n);
    const gate_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(up_buf);
    const swiglu_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(swiglu_buf);
    const hidden = try allocator.alloc(f32, DIM);
    defer allocator.free(hidden);
    const out = try allocator.alloc(f32, DIM);
    defer allocator.free(out);
    const d_swiglu = try allocator.alloc(f32, HDIM);
    defer allocator.free(d_swiglu);
    const ctx = try allocator.alloc(f32, DIM);
    defer allocator.free(ctx);

    // gradient accumulators
    const g_embed = try allocZeroed(allocator, NWORDS * DIM);
    defer allocator.free(g_embed);

    var g_wr: [NSTEPS][]f32 = undefined;
    var g_gate: [NSTEPS][]f32 = undefined;
    var g_up: [NSTEPS][]f32 = undefined;
    var g_down: [NSTEPS][]f32 = undefined;
    for (0..NSTEPS) |s| {
        g_wr[s] = try allocZeroed(allocator, DIM * DIM);
        g_gate[s] = try allocZeroed(allocator, DIM * HDIM);
        g_up[s] = try allocZeroed(allocator, DIM * HDIM);
        g_down[s] = try allocZeroed(allocator, HDIM * DIM);
    }
    defer {
        for (0..NSTEPS) |s| {
            allocator.free(g_wr[s]);
            allocator.free(g_gate[s]);
            allocator.free(g_up[s]);
            allocator.free(g_down[s]);
        }
    }

    var best_loss: f32 = 1e9;

    var step: i32 = 1;
    while (step <= train_steps_count) : (step += 1) {
        const start_pos = g_prng.random().uintLessThan(usize, n_ids - window);
        const win = ids[start_pos..];

        var total_loss: f32 = 0;

        // zero grad accumulators
        @memset(g_embed, 0);
        for (0..NSTEPS) |s| {
            @memset(g_wr[s], 0);
            @memset(g_gate[s], 0);
            @memset(g_up[s], 0);
            @memset(g_down[s], 0);
        }

        for (0..NSTEPS) |s| {
            const ctx_n = s + 1;
            const target: usize = @intCast(win[s + 1]);
            const sw = &m.steps[s];

            // forward
            forwardStep(m, win[0..ctx_n], s, logits, query, query_n, gate_buf, up_buf, swiglu_buf, hidden, out);
            softmaxV(logits, probs, NWORDS);

            var p = probs[target];
            if (p < 1e-10) p = 1e-10;
            total_loss -= @log(p);

            // d_logits = probs - one_hot(target)
            @memcpy(d_logits, probs);
            d_logits[target] -= 1.0;

            // d_out = E^T @ d_logits (from tied output)
            for (0..DIM) |j| {
                var s_val: f32 = 0;
                for (0..NWORDS) |v| {
                    s_val += d_logits[v] * m.embed[v * DIM + j];
                }
                d_out[j] = s_val;
            }

            // accumulate embed gradient
            for (0..NWORDS) |v| {
                if (@abs(d_logits[v]) < 1e-8) continue;
                for (0..DIM) |j| {
                    g_embed[v * DIM + j] += d_logits[v] * out[j];
                }
            }

            // backprop through w_down
            matmulMtv(sw.w_down, d_out, d_swiglu, DIM, HDIM);
            for (0..HDIM) |ii| {
                for (0..DIM) |j| {
                    g_down[s][ii * DIM + j] += swiglu_buf[ii] * d_out[j];
                }
            }

            // backprop through SwiGLU
            for (0..HDIM) |ii| {
                const sg = siluf(gate_buf[ii]);
                const sig: f32 = if (gate_buf[ii] > -20) 1.0 / (1.0 + @exp(-gate_buf[ii])) else 0;
                const silu_grad: f32 = if (gate_buf[ii] > -20) sig * (1.0 + gate_buf[ii] * (1.0 - sig)) else 0;
                const d_gate_i = d_swiglu[ii] * up_buf[ii] * silu_grad;
                const d_up_i = d_swiglu[ii] * sg;

                for (0..DIM) |j| {
                    g_gate[s][ii * DIM + j] += d_gate_i * query_n[j];
                    g_up[s][ii * DIM + j] += d_up_i * query_n[j];
                }
            }

            // d_query (approx RMSNorm backward)
            var ss: f32 = 0;
            for (0..DIM) |ii| ss += query[ii] * query[ii];
            ss = ss / @as(f32, @floatFromInt(DIM)) + 1e-5;
            const inv = 1.0 / @sqrt(ss);
            var d_query: [DIM]f32 = undefined;
            for (0..DIM) |ii| {
                d_query[ii] = d_out[ii] * sw.rms[ii] * inv;
            }

            // accumulate Wr gradient
            poolContext(m, win[0..ctx_n], ctx);
            for (0..DIM) |ii| {
                if (@abs(d_query[ii]) < 1e-8) continue;
                for (0..DIM) |j| {
                    g_wr[s][ii * DIM + j] += d_query[ii] * ctx[j];
                }
            }
        }

        // Adam step
        m.adam_t += 1;
        const bc1 = 1.0 - std.math.pow(f32, ADAM_B1, @floatFromInt(m.adam_t));
        const bc2 = 1.0 - std.math.pow(f32, ADAM_B2, @floatFromInt(m.adam_t));

        adamUpdate(m.embed, m.embed_m, m.embed_v, g_embed, lr, bc1, bc2);

        for (0..NSTEPS) |s| {
            adamUpdate(m.steps[s].wr, m.adam[s].wr_m, m.adam[s].wr_v, g_wr[s], lr, bc1, bc2);
            adamUpdate(m.steps[s].w_gate, m.adam[s].gate_m, m.adam[s].gate_v, g_gate[s], lr, bc1, bc2);
            adamUpdate(m.steps[s].w_up, m.adam[s].up_m, m.adam[s].up_v, g_up[s], lr, bc1, bc2);
            adamUpdate(m.steps[s].w_down, m.adam[s].down_m, m.adam[s].down_v, g_down[s], lr, bc1, bc2);
        }

        const avg_loss = total_loss / @as(f32, @floatFromInt(NSTEPS));
        if (avg_loss < best_loss) best_loss = avg_loss;

        if (@mod(step, 50) == 0 or step == 1) {
            try stdout_w.print("  step {d:>5}/{d}  loss={d:.4}  best={d:.4}\n", .{
                step, train_steps_count, avg_loss, best_loss,
            });
        }
    }

    try stdout_w.print("  training complete. best loss: {d:.4}\n", .{best_loss});
}

// ═══════════════════════════════════════════════════════════════
// GENERATION — 12 steps, each picks one word
// ═══════════════════════════════════════════════════════════════

fn findSeed(key: []const u8) i32 {
    const idx = findWord(key);
    if (idx >= 0) return idx;

    var best: i32 = -1;
    var best_score: f32 = -1;
    for (0..NWORDS) |i| {
        var score: f32 = 0;
        // Check if key contains vocab word or vice versa
        if (std.mem.indexOf(u8, VOCAB[i], key) != null or std.mem.indexOf(u8, key, VOCAB[i]) != null) {
            score = 3;
        }
        const ml = VOCAB[i].len;
        const kl = key.len;
        const mn = @min(ml, kl);
        for (0..mn) |j| {
            if (VOCAB[i][j] == key[j]) {
                score += 0.5;
            } else break;
        }
        if (score > best_score) {
            best_score = score;
            best = @intCast(i);
        }
    }
    if (best >= 0 and best_score > 0) return best;
    return @intCast(g_prng.random().uintLessThan(u32, 200));
}

fn extractKey(text: []const u8, out_buf: []u8) []const u8 {
    var buf: [1024]u8 = undefined;
    var bi: usize = 0;
    for (text) |c| {
        if (bi >= 1022) break;
        buf[bi] = std.ascii.toLower(c);
        bi += 1;
    }
    const input = buf[0..bi];

    var best_start: usize = 0;
    var best_end: usize = 0;
    var best_len: usize = 0;

    var i: usize = 0;
    while (i < input.len) {
        // skip whitespace
        while (i < input.len and (input[i] == ' ' or input[i] == '\t' or input[i] == '\n')) : (i += 1) {}
        if (i >= input.len) break;
        const tok_start = i;
        while (i < input.len and input[i] != ' ' and input[i] != '\t' and input[i] != '\n') : (i += 1) {}
        const tok = input[tok_start..i];

        if (tok.len > 1 and !isStop(tok)) {
            if (tok.len > best_len) {
                best_len = tok.len;
                best_start = tok_start;
                best_end = i;
            }
        }
    }

    if (best_len > 0) {
        const len = @min(best_len, out_buf.len - 1);
        @memcpy(out_buf[0..len], input[best_start..][0..len]);
        return out_buf[0..len];
    } else {
        const fallback = "silence";
        @memcpy(out_buf[0..fallback.len], fallback);
        return out_buf[0..fallback.len];
    }
}

fn runChain(m: *Model, text: []const u8, allocator: std.mem.Allocator) !void {
    const stdout_w = (std.fs.File{ .handle = std.posix.STDOUT_FILENO }).deprecatedWriter();

    var key_buf: [64]u8 = undefined;
    const key = extractKey(text, &key_buf);
    const seed = findSeed(key);

    // prophecy
    const deep_cats = [_]usize{ 2, 5, 7 };
    const tcat = deep_cats[g_prng.random().uintLessThan(usize, 3)];
    const ranges = [8][2]usize{
        .{ 0, 100 },
        .{ 100, 200 },
        .{ 200, 300 },
        .{ 300, 350 },
        .{ 350, 450 },
        .{ 450, 550 },
        .{ 550, 650 },
        .{ 650, NWORDS },
    };
    const range_size = ranges[tcat][1] - ranges[tcat][0];
    var pt = @as(i32, @intCast(ranges[tcat][0])) + @as(i32, @intCast(g_prng.random().uintLessThan(usize, range_size)));
    if (pt >= @as(i32, NWORDS)) pt = @as(i32, NWORDS) - 1;
    prophecy_target = pt;
    prophecy_age = 0;

    try stdout_w.print("\n  destined: {s}\n", .{VOCAB[@intCast(prophecy_target)]});
    try stdout_w.print("\n  {s}\n", .{VOCAB[@intCast(seed)]});

    var chain: [NSTEPS + 1]i32 = undefined;
    var chain_n: usize = 0;
    var forbidden: [NSTEPS + 100]i32 = undefined;
    var nforbid: usize = 0;
    chain[chain_n] = seed;
    chain_n += 1;
    forbidden[nforbid] = seed;
    nforbid += 1;

    var prev = seed;

    // scratch
    const logits = try allocator.alloc(f32, NWORDS);
    defer allocator.free(logits);
    const probs = try allocator.alloc(f32, NWORDS);
    defer allocator.free(probs);
    const query = try allocator.alloc(f32, DIM);
    defer allocator.free(query);
    const query_n = try allocator.alloc(f32, DIM);
    defer allocator.free(query_n);
    const gate_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(up_buf);
    const swiglu_buf = try allocator.alloc(f32, HDIM);
    defer allocator.free(swiglu_buf);
    const hidden = try allocator.alloc(f32, DIM);
    defer allocator.free(hidden);
    const out = try allocator.alloc(f32, DIM);
    defer allocator.free(out);

    var fulfilled = false;

    for (0..NSTEPS) |step| {
        updateChambers(step);
        prophecy_age += 1;

        // learned logits
        forwardStep(m, chain[0..chain_n], step, logits, query, query_n, gate_buf, up_buf, swiglu_buf, hidden, out);

        // Dario field overlay
        darioOverlay(logits, chain[0..chain_n], step);

        // mask forbidden
        for (0..nforbid) |f_i| {
            logits[@intCast(forbidden[f_i])] = -1e9;
        }

        // top-k=12 sampling
        softmaxV(logits, probs, NWORDS);

        const Sc = struct { idx: i32, p: f32 };
        var top: [12]Sc = undefined;
        for (0..12) |ii| top[ii] = Sc{ .idx = 0, .p = -1 };

        for (0..NWORDS) |w| {
            for (0..12) |k| {
                if (probs[w] > top[k].p) {
                    var j: usize = 11;
                    while (j > k) : (j -= 1) {
                        top[j] = top[j - 1];
                    }
                    top[k] = Sc{ .idx = @intCast(w), .p = probs[w] };
                    break;
                }
            }
        }

        var total: f32 = 0.001;
        for (0..12) |ii| total += if (top[ii].p > 0) top[ii].p else 0;
        var r = randf() * total;
        var pick = top[0].idx;
        for (0..12) |ii| {
            const pp = if (top[ii].p > 0) top[ii].p else 0;
            r -= pp;
            if (r <= 0) {
                pick = top[ii].idx;
                break;
            }
        }

        chain[chain_n] = pick;
        chain_n += 1;
        forbidden[nforbid] = pick;
        nforbid += 1;

        coocUpdate(prev, pick);
        const cat = wordCategory(@intCast(pick));
        destiny[cat] = 0.3 + 0.7 * destiny[cat];

        if (pick == prophecy_target) fulfilled = true;

        if (step > 7) {
            trauma = if (trauma + 0.1 < 1) trauma + 0.1 else 1;
        }
        trauma *= 0.97;

        if (step == NSTEPS - 1) {
            try stdout_w.print("  *{s}\n", .{VOCAB[@intCast(pick)]});
        } else {
            try stdout_w.print("   {s}\n", .{VOCAB[@intCast(pick)]});
        }
        prev = pick;
    }

    var cats_seen: usize = 0;
    var cat_flags = [8]bool{ false, false, false, false, false, false, false, false };
    for (0..chain_n) |ci| {
        const c = wordCategory(@intCast(chain[ci]));
        if (!cat_flags[c]) {
            cat_flags[c] = true;
            cats_seen += 1;
        }
    }

    try stdout_w.print("\n  drift {d}/8 \xc2\xb7 prophecy {s}\n", .{
        cats_seen,
        if (fulfilled) "fulfilled" else "unfulfilled",
    });
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const stdout_w = (std.fs.File{ .handle = std.posix.STDOUT_FILENO }).deprecatedWriter();

    // Seed PRNG from timestamp
    const ts: u64 = @intCast(std.time.timestamp());
    g_prng = std.Random.DefaultPrng.init(ts);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var train_path: ?[]const u8 = null;
    var load_path: ?[]const u8 = null;
    var save_path: ?[]const u8 = null;
    var train_steps_count: i32 = 5000;
    var lr: f32 = 3e-4;
    var text_input: ?[]const u8 = null;

    var i: usize = 1;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (std.mem.eql(u8, arg, "--train") and i + 1 < args.len) {
            i += 1;
            train_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--load") and i + 1 < args.len) {
            i += 1;
            load_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--save") and i + 1 < args.len) {
            i += 1;
            save_path = std.mem.sliceTo(args[i], 0);
        } else if (std.mem.eql(u8, arg, "--steps") and i + 1 < args.len) {
            i += 1;
            train_steps_count = std.fmt.parseInt(i32, std.mem.sliceTo(args[i], 0), 10) catch 5000;
        } else if (std.mem.eql(u8, arg, "--lr") and i + 1 < args.len) {
            i += 1;
            lr = std.fmt.parseFloat(f32, std.mem.sliceTo(args[i], 0)) catch 3e-4;
        } else {
            // collect remaining as text
            var textbuf: [2048]u8 = undefined;
            var pos: usize = 0;
            var j = i;
            while (j < args.len) : (j += 1) {
                const a = std.mem.sliceTo(args[j], 0);
                if (j > i and pos < textbuf.len - 1) {
                    textbuf[pos] = ' ';
                    pos += 1;
                }
                const copy_len = @min(a.len, textbuf.len - pos - 1);
                @memcpy(textbuf[pos..][0..copy_len], a[0..copy_len]);
                pos += copy_len;
            }
            // We need to keep the text alive; copy to heap
            const heap_text = try allocator.alloc(u8, pos);
            @memcpy(heap_text, textbuf[0..pos]);
            text_input = heap_text;
            break;
        }
        i += 1;
    }

    var model: Model = undefined;
    try modelInit(&model, allocator);

    try stdout_w.print("\n  penelope \xe2\x80\x94 1984 words, {d} steps, Dario Equation\n", .{NSTEPS});
    try stdout_w.print("  {d} trainable params ({d:.1}MB f32)\n", .{
        totalParamCount(),
        @as(f64, @floatFromInt(totalParamCount())) * 4.0 / 1e6,
    });
    try stdout_w.print("  by Arianna Method\n\n", .{});

    if (load_path) |lp| {
        _ = try modelLoad(&model, lp);
    }
    if (train_path) |tp| {
        try train(&model, tp, train_steps_count, lr, allocator);
        if (save_path) |sp| {
            try modelSave(&model, sp);
        }
    }

    if (text_input) |txt| {
        try runChain(&model, txt, allocator);
    } else if (train_path == null) {
        const stdin = (std.fs.File{ .handle = std.posix.STDIN_FILENO }).deprecatedReader();
        var line_buf: [1024]u8 = undefined;
        while (true) {
            try stdout_w.print("  > ", .{});
            const line = stdin.readUntilDelimiter(&line_buf, '\n') catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            // trim trailing \r
            var trimmed = line;
            while (trimmed.len > 0 and (trimmed[trimmed.len - 1] == '\r' or trimmed[trimmed.len - 1] == '\n')) {
                trimmed = trimmed[0 .. trimmed.len - 1];
            }
            if (trimmed.len == 0) continue;
            try runChain(&model, trimmed, allocator);
        }
    }

    if (save_path != null and train_path == null) {
        try modelSave(&model, save_path.?);
    }

    modelFree(&model, allocator);
}
