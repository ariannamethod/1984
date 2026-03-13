// penelope.ts — 1984 words. 12 steps of resonance. Dario Equation.
//
// Trainable resonance engine. Not a transformer. A mirror that learns.
//
// Input:  text → vocab word IDs (prefix matching)
// Attend: RRPRAM resonance + SwiGLU per step (how it thinks)
// Output: word-level from 1984 vocab (gibberish impossible)
//
// 12 learned step-weights (~1M each). Each step has its own lens.
// Step 1 sees the surface. Step 12 sees the bone.
//
// Architecture per step s:
//     context = pool(embed(words))
//     query   = RMSNorm(context @ Wr_s)          RRPRAM resonance
//     hidden  = SwiGLU(query; gate_s, up_s, down_s)
//     logits  = (query + hidden) @ E^T            tied output
//     logits += DarioField(context)               live overlay
//     word    = sample(softmax(logits))
//
// Total: ~13M params (762K embed + 12 × 1.03M steps)
//
//   npx tsx penelope.ts                                  # interactive
//   npx tsx penelope.ts "darkness eats the city"         # single chain
//   npx tsx penelope.ts --train corpus.txt               # train
//   npx tsx penelope.ts --train corpus.txt --steps 5000  # train N steps
//   npx tsx penelope.ts --load penelope.bin              # load weights
//   npx tsx penelope.ts --save penelope.bin              # save after
//
// By Arianna Method. הרזוננס לא נשבר

import * as fs from "fs";
import * as readline from "readline";

// ═══════════════════════════════════════════════════════════════
// 1984 WORDS
// ═══════════════════════════════════════════════════════════════

const VOCAB: string[] = [
  // BODY 0-99
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
  // NATURE 100-199
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
  // EMOTION 200-299
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
  // TIME 300-349
  "moment","instant","second","minute","hour","day","night","week","month","year",
  "decade","century","epoch","era","age","past","present","future","memory","tomorrow",
  "yesterday","forever","never","always","sometimes","often","seldom","once","twice","origin",
  "ending","beginning","duration","interval","pause","wait","rush","delay","haste","eternity",
  "cycle","season","spring","summer","autumn","winter","dawn","twilight","midnight","noon",
  // SOCIETY 350-449
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
  // ABSTRACT 450-549
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
  // ACTION 550-649
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
  // MATERIAL 650-749
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
  // FOOD 750-799
  "bread","salt","sugar","honey","milk","butter","cheese","meat","fish","egg",
  "grain","rice","wheat","corn","fruit","apple","grape","olive","lemon","pepper",
  "wine","water","tea","coffee","broth","soup","stew","feast","crumb","morsel",
  "harvest","garden","soil","compost","ferment","yeast","dough","crust","marrow","nectar",
  "spice","herb","mint","thyme","sage","garlic","onion","mushroom","berry","kernel",
  // ARCHITECTURE 800-849
  "house","room","wall","floor","ceiling","door","window","stair","corridor","basement",
  "tower","bridge","arch","column","dome","vault","foundation","ruin","temple","altar",
  "threshold","passage","labyrinth","maze","chamber","cell","shelter","fortress","prison","garden",
  "roof","chimney","hearth","frame","beam","pillar","brick","mortar","tile","glass",
  "balcony","terrace","courtyard","gate","fence","path","road","intersection","tunnel","well",
  // RELATIONSHIP 850-929
  "mother","father","child","daughter","son","sister","brother","family","ancestor","descendant",
  "friend","stranger","lover","enemy","neighbor","companion","rival","mentor","student","witness",
  "husband","wife","partner","orphan","widow","elder","infant","twin","cousin","godmother",
  "promise","oath","vow","contract","alliance","betrayal","reconciliation","farewell","reunion","absence",
  "kiss","embrace","handshake","slap","caress","quarrel","conversation","confession","accusation","apology",
  "birth","death","marriage","divorce","inheritance","adoption","abandonment","protection","neglect","sacrifice",
  "trust","suspicion","loyalty","treachery","devotion","indifference","jealousy","admiration","dependence","autonomy",
  "intimacy","distance","connection","isolation","belonging","exile","homecoming","departure","waiting","return",
  // PHILOSOPHY 930-999
  "consciousness","awareness","perception","sensation","intuition","reason","logic","paradox","dialectic","synthesis",
  "freedom","determinism","causation","contingency","necessity","possibility","impossibility","actuality","potential","becoming",
  "subject","object","self","other","identity","difference","sameness","change","permanence","flux",
  "being","nothingness","existence","essence","phenomena","noumena","appearance","reality","illusion","truth",
  "ethics","morality","virtue","vice","good","evil","right","wrong","duty","choice",
  "justice","mercy","punishment","reward","guilt","innocence","responsibility","consequence","intention","action",
  "language","meaning","sign","reference","representation","interpretation","understanding","misunderstanding","translation","silence",
  // MUSIC 1000-1049
  "melody","rhythm","chord","pitch","tone","note","bass","treble","octave","harmony",
  "dissonance","resonance","vibration","frequency","amplitude","tempo","beat","rest","pause","crescendo",
  "murmur","hum","buzz","click","crack","boom","rumble","chime","echo","reverb",
  "song","lullaby","anthem","dirge","hymn","ballad","fugue","sonata","requiem","improvisation",
  "strum","pluck","strike","bow","mute","sustain","fade","loop","drone","overtone",
  // WEATHER 1050-1099
  "rain","drizzle","downpour","hail","sleet","blizzard","hurricane","tornado","drought","flood",
  "breeze","gale","typhoon","monsoon","frost","thaw","haze","smog","rainbow","mirage",
  "erosion","sedimentation","crystallization","evaporation","condensation","precipitation","sublimation","oxidation","combustion","decay",
  "magma","lava","quartz","granite","obsidian","chalk","slate","sandstone","limestone","basalt",
  "marsh","delta","gorge","ridge","summit","abyss","chasm","rift","fault","crater",
  // RITUAL 1100-1149
  "prayer","meditation","ritual","ceremony","blessing","curse","oath","vow","pilgrimage","procession",
  "offering","sacrifice","communion","baptism","funeral","wedding","coronation","initiation","exile","absolution",
  "incense","candle","bell","chant","mantra","psalm","scripture","prophecy","oracle","vision",
  "mask","costume","dance","feast","fast","vigil","silence","confession","penance","redemption",
  "altar","shrine","temple","tomb","relic","artifact","amulet","talisman","totem","icon",
  // LABOR 1150-1199
  "harvest","planting","sowing","reaping","threshing","milling","baking","brewing","weaving","spinning",
  "carving","sculpting","painting","drawing","writing","printing","binding","stitching","welding","forging",
  "mining","drilling","excavation","construction","demolition","repair","restoration","invention","discovery","experiment",
  "apprentice","craftsman","artist","engineer","architect","farmer","sailor","miner","healer","scribe",
  "workshop","studio","laboratory","field","dock","quarry","furnace","mill","press","loom",
  // GEOMETRY 1200-1249
  "circle","spiral","line","curve","angle","edge","center","margin","border","frame",
  "sphere","cube","pyramid","cylinder","cone","helix","vortex","arc","wave","fractal",
  "symmetry","asymmetry","proportion","ratio","scale","dimension","plane","axis","vertex","intersection",
  "pattern","grid","lattice","mesh","tessellation","rotation","reflection","translation","dilation","projection",
  "surface","volume","area","perimeter","diameter","radius","tangent","normal","parallel","perpendicular",
  // ANIMAL 1250-1299
  "horse","dog","cat","bird","fish","snake","bear","fox","rabbit","turtle",
  "eagle","sparrow","raven","swan","heron","falcon","vulture","pelican","nightingale","lark",
  "lion","tiger","elephant","giraffe","hippopotamus","rhinoceros","gorilla","chimpanzee","orangutan","leopard",
  "salmon","trout","shark","dolphin","octopus","jellyfish","starfish","seahorse","crab","lobster",
  "frog","lizard","crocodile","chameleon","gecko","iguana","newt","toad","salamander","viper",
  // COLOR 1300-1349
  "red","blue","green","white","black","gray","amber","violet","indigo","scarlet",
  "crimson","azure","emerald","ivory","obsidian","silver","golden","copper","rust","ochre",
  "bright","dark","transparent","opaque","matte","glossy","rough","smooth","coarse","fine",
  "stripe","dot","plaid","solid","gradient","shadow","highlight","contrast","saturation","hue",
  "velvet","satin","linen","denim","lace","gauze","burlap","chiffon","tweed","corduroy",
  // TRANSPORT 1350-1399
  "ship","boat","canoe","raft","anchor","sail","rudder","oar","mast","hull",
  "train","rail","station","platform","ticket","journey","passage","crossing","departure","arrival",
  "wheel","axle","road","highway","path","trail","bridge","tunnel","gate","crossroad",
  "wing","flight","altitude","turbulence","landing","orbit","trajectory","velocity","acceleration","gravity",
  "horse","carriage","wagon","cart","sled","bicycle","motorcycle","automobile","truck","ambulance",
  // DOMESTIC 1400-1449
  "kitchen","bedroom","bathroom","attic","cellar","closet","drawer","shelf","table","chair",
  "bed","pillow","blanket","curtain","carpet","lamp","mirror","photograph","vase","clock",
  "plate","spoon","knife","fork","cup","pot","pan","kettle","oven","stove",
  "soap","towel","broom","bucket","needle","thread","button","zipper","hanger","basket",
  "door","window","lock","key","handle","hinge","nail","screw","bolt","hook",
  // COMMUNICATION 1450-1499
  "letter","envelope","stamp","address","message","telegram","telephone","radio","broadcast","signal",
  "newspaper","headline","article","column","editorial","report","announcement","rumor","gossip","testimony",
  "ink","pen","pencil","typewriter","keyboard","screen","printer","paper","notebook","diary",
  "conversation","dialogue","monologue","debate","argument","negotiation","compromise","ultimatum","declaration","speech",
  "translation","interpretation","code","cipher","encryption","decryption","password","signature","seal","authentication",
  // MEDICAL 1500-1549
  "diagnosis","symptom","treatment","remedy","cure","relapse","recovery","surgery","anesthesia","bandage",
  "infection","inflammation","fracture","hemorrhage","allergy","immunity","vaccine","antibiotic","toxin","antidote",
  "hospital","clinic","pharmacy","laboratory","ambulance","stretcher","scalpel","syringe","stethoscope","thermometer",
  "fever","cough","rash","swelling","numbness","dizziness","insomnia","fatigue","nausea","tremor",
  "pulse","pressure","temperature","respiration","circulation","digestion","metabolism","reflex","coordination","balance",
  // COSMIC 1550-1599
  "universe","galaxy","constellation","planet","asteroid","meteorite","satellite","orbit","void","singularity",
  "photon","electron","proton","neutron","atom","molecule","particle","quantum","field","dimension",
  "spacetime","relativity","entropy","thermodynamics","radiation","spectrum","wavelength","frequency","amplitude","interference",
  "supernova","blackhole","pulsar","quasar","nebula","wormhole","antimatter","darkmatter","redshift","expansion",
  "telescope","observatory","mission","launch","countdown","trajectory","reentry","landing","exploration","discovery",
  // BUREAUCRACY 1600-1649
  "document","form","permit","license","certificate","registration","application","approval","denial","appeal",
  "regulation","compliance","violation","penalty","exemption","quota","deadline","protocol","procedure","standard",
  "office","desk","file","folder","stamp","signature","receipt","invoice","ledger","archive",
  "committee","department","ministry","bureau","agency","institution","organization","corporation","foundation","commission",
  "report","audit","review","inspection","evaluation","assessment","benchmark","statistic","data","record",
  // MYTHIC 1650-1699
  "oracle","prophecy","fate","destiny","curse","blessing","quest","trial","sacrifice","redemption",
  "labyrinth","threshold","guardian","shadow","mirror","mask","transformation","metamorphosis","resurrection","apocalypse",
  "phoenix","dragon","serpent","sphinx","minotaur","chimera","hydra","golem","specter","wraith",
  "underworld","paradise","purgatory","limbo","abyss","eden","babylon","atlantis","olympus","tartarus",
  "hero","villain","trickster","sage","fool","maiden","crone","warrior","healer","shapeshifter",
  // TEXTUAL 1700-1749
  "word","sentence","paragraph","chapter","verse","stanza","line","margin","footnote","epilogue",
  "prologue","preface","title","subtitle","dedication","inscription","epitaph","motto","slogan","proverb",
  "metaphor","simile","allegory","irony","satire","parody","tragedy","comedy","farce","melodrama",
  "narrator","character","protagonist","antagonist","audience","reader","author","critic","editor","translator",
  "manuscript","draft","revision","erasure","correction","annotation","citation","reference","index","bibliography",
  // PSYCHOLOGICAL 1750-1799
  "unconscious","subconscious","conscious","ego","superego","libido","repression","projection","sublimation","transference",
  "trauma","complex","fixation","regression","denial","rationalization","displacement","compensation","identification","dissociation",
  "archetype","persona","anima","animus","shadow","self","individuation","integration","fragmentation","wholeness",
  "attachment","separation","abandonment","dependency","autonomy","codependency","boundary","enmeshment","differentiation","fusion",
  "grief","mourning","acceptance","bargaining","anger","depression","recovery","relapse","healing","scarring",
  // FINAL 1800-1983
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
  "debt","credit","interest","principal","collateral","default","bankruptcy","solvency","dividend","investment"
];

const V = VOCAB.length; // 1984
const STEPS = 12;
const D = 384;          // embedding dim
const M = 768;          // SwiGLU hidden dim

const VOCAB_IDX = new Map<string, number>();
for (let i = 0; i < VOCAB.length; i++) {
  if (!VOCAB_IDX.has(VOCAB[i])) {
    VOCAB_IDX.set(VOCAB[i], i);
  }
}

const STOP = new Set(
  "i me my we our you your he she it they them the a an and or but in on at to for of is am are was were be been being have has had do does did will would shall should can could may might must not no nor so if then than that this these those what which who whom how when where why all each every some any few many much more most other another such".split(" ")
);


// ═══════════════════════════════════════════════════════════════
// MATH — pure TypeScript, no dependencies
// ═══════════════════════════════════════════════════════════════

function randn(): number {
  const u1 = Math.random() + 1e-12;
  const u2 = Math.random() + 1e-12;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.2831853 * u2);
}

function zeros(n: number): number[] {
  return new Array(n).fill(0.0);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vadd(a: number[], b: number[]): number[] {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function vsub(a: number[], b: number[]): number[] {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function vscale(a: number[], s: number): number[] {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
  return out;
}

function matmul_mv(W: number[], x: number[], rows: number, cols: number): number[] {
  const out = zeros(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0.0;
    for (let j = 0; j < cols; j++) {
      s += W[i * cols + j] * x[j];
    }
    out[i] = s;
  }
  return out;
}

function matmul_mtv(W: number[], x: number[], rows: number, cols: number): number[] {
  const out = zeros(cols);
  for (let j = 0; j < cols; j++) {
    let s = 0.0;
    for (let i = 0; i < rows; i++) {
      s += W[i * cols + j] * x[i];
    }
    out[j] = s;
  }
  return out;
}

function rmsnorm(x: number[], g: number[], n: number): number[] {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  ss = ss / n + 1e-5;
  const inv = 1.0 / Math.sqrt(ss);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = g[i] * x[i] * inv;
  return out;
}

function silu(x: number): number {
  return x > -20 ? x / (1.0 + Math.exp(-x)) : 0.0;
}

function softmax(x: number[]): number[] {
  let mx = x[0];
  for (let i = 1; i < x.length; i++) if (x[i] > mx) mx = x[i];
  const e = new Array(x.length);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    e[i] = Math.exp(x[i] - mx);
    s += e[i];
  }
  for (let i = 0; i < x.length; i++) e[i] /= s;
  return e;
}


// ═══════════════════════════════════════════════════════════════
// MODEL — 12 step-specific weight sets + shared embedding
// ═══════════════════════════════════════════════════════════════

class StepWeights {
  wr: number[];
  rms: number[];
  w_gate: number[];
  w_up: number[];
  w_down: number[];

  constructor() {
    const scale_d = Math.sqrt(2.0 / D);
    const scale_m = Math.sqrt(2.0 / M);
    this.wr = new Array(D * D);
    for (let i = 0; i < D * D; i++) this.wr[i] = randn() * scale_d;
    this.rms = new Array(D).fill(1.0);
    this.w_gate = new Array(D * M);
    for (let i = 0; i < D * M; i++) this.w_gate[i] = randn() * scale_d;
    this.w_up = new Array(D * M);
    for (let i = 0; i < D * M; i++) this.w_up[i] = randn() * scale_d;
    this.w_down = new Array(M * D);
    for (let i = 0; i < M * D; i++) this.w_down[i] = randn() * scale_m;
  }

  paramCount(): number {
    return D * D + D + D * M + D * M + M * D;
  }

  params(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.wr.length; i++) out.push(this.wr[i]);
    for (let i = 0; i < this.rms.length; i++) out.push(this.rms[i]);
    for (let i = 0; i < this.w_gate.length; i++) out.push(this.w_gate[i]);
    for (let i = 0; i < this.w_up.length; i++) out.push(this.w_up[i]);
    for (let i = 0; i < this.w_down.length; i++) out.push(this.w_down[i]);
    return out;
  }

  loadFrom(flat: number[], offset: number): number {
    let o = offset;
    this.wr = flat.slice(o, o + D * D); o += D * D;
    this.rms = flat.slice(o, o + D); o += D;
    this.w_gate = flat.slice(o, o + D * M); o += D * M;
    this.w_up = flat.slice(o, o + D * M); o += D * M;
    this.w_down = flat.slice(o, o + M * D); o += M * D;
    return o;
  }
}


class Penelope {
  embed: number[];
  steps: StepWeights[];

  constructor() {
    const scale = Math.sqrt(2.0 / V);
    this.embed = new Array(V * D);
    for (let i = 0; i < V * D; i++) this.embed[i] = randn() * scale;
    this.steps = [];
    for (let i = 0; i < STEPS; i++) this.steps.push(new StepWeights());
  }

  paramCount(): number {
    let total = V * D;
    for (const s of this.steps) total += s.paramCount();
    return total;
  }

  getEmbed(idx: number): number[] {
    return this.embed.slice(idx * D, (idx + 1) * D);
  }

  poolContext(wordIds: number[]): number[] {
    if (wordIds.length === 0) return zeros(D);
    const ctx = zeros(D);
    for (const wid of wordIds) {
      const e = this.getEmbed(wid);
      for (let j = 0; j < D; j++) ctx[j] += e[j];
    }
    const inv = 1.0 / wordIds.length;
    for (let j = 0; j < D; j++) ctx[j] *= inv;
    return ctx;
  }

  forwardStep(contextIds: number[], stepIdx: number): number[] {
    const sw = this.steps[stepIdx];
    const ctx = this.poolContext(contextIds);

    // RRPRAM resonance
    let query = matmul_mv(sw.wr, ctx, D, D);

    // RMSNorm
    query = rmsnorm(query, sw.rms, D);

    // SwiGLU
    const gate = matmul_mv(sw.w_gate, query, M, D);
    const up = matmul_mv(sw.w_up, query, M, D);
    const swiglu = new Array(M);
    for (let i = 0; i < M; i++) swiglu[i] = silu(gate[i]) * up[i];
    const hidden = matmul_mv(sw.w_down, swiglu, D, M);

    // Residual
    const out = vadd(query, hidden);

    // Logits = E @ out (tied weights)
    const logits = matmul_mv(this.embed, out, V, D);
    return logits;
  }

  save(path: string): void {
    const totalParams = this.paramCount();
    const headerBuf = Buffer.alloc(16);
    headerBuf.writeInt32LE(V, 0);
    headerBuf.writeInt32LE(D, 4);
    headerBuf.writeInt32LE(M, 8);
    headerBuf.writeInt32LE(STEPS, 12);

    const bodyBuf = Buffer.alloc(totalParams * 4);
    let off = 0;
    for (let i = 0; i < this.embed.length; i++) { bodyBuf.writeFloatLE(this.embed[i], off); off += 4; }
    for (const s of this.steps) {
      for (let i = 0; i < s.wr.length; i++) { bodyBuf.writeFloatLE(s.wr[i], off); off += 4; }
      for (let i = 0; i < s.rms.length; i++) { bodyBuf.writeFloatLE(s.rms[i], off); off += 4; }
      for (let i = 0; i < s.w_gate.length; i++) { bodyBuf.writeFloatLE(s.w_gate[i], off); off += 4; }
      for (let i = 0; i < s.w_up.length; i++) { bodyBuf.writeFloatLE(s.w_up[i], off); off += 4; }
      for (let i = 0; i < s.w_down.length; i++) { bodyBuf.writeFloatLE(s.w_down[i], off); off += 4; }
    }

    const fd = fs.openSync(path, "w");
    fs.writeSync(fd, headerBuf);
    fs.writeSync(fd, bodyBuf);
    fs.closeSync(fd);

    const size = fs.statSync(path).size;
    console.log(`  saved ${path}: ${totalParams} params (${(size / 1e6).toFixed(1)}MB)`);
  }

  load(path: string): void {
    const data = fs.readFileSync(path);
    const v = data.readInt32LE(0);
    const d = data.readInt32LE(4);
    const m = data.readInt32LE(8);
    const st = data.readInt32LE(12);
    if (v !== V || d !== D || m !== M || st !== STEPS) {
      throw new Error(`config mismatch: file has V=${v} D=${d} M=${m} S=${st}`);
    }
    const numFloats = (data.length - 16) / 4;
    const flat: number[] = new Array(numFloats);
    for (let i = 0; i < numFloats; i++) flat[i] = data.readFloatLE(16 + i * 4);

    let o = 0;
    this.embed = flat.slice(o, o + V * D); o += V * D;
    for (const s of this.steps) {
      o = s.loadFrom(flat, o);
    }
    console.log(`  loaded ${path}: ${flat.length} params`);
  }
}


// ═══════════════════════════════════════════════════════════════
// TOKENIZER — map arbitrary text to word IDs in our 1984 vocab
// ═══════════════════════════════════════════════════════════════

function tokenize_text(text: string): number[] {
  const words = text.toLowerCase().match(/[a-z]+/g) || [];
  const ids: number[] = [];
  for (const w of words) {
    if (STOP.has(w) || w.length < 2) continue;
    if (VOCAB_IDX.has(w)) {
      ids.push(VOCAB_IDX.get(w)!);
    } else {
      let best = -1;
      let bestLen = 0;
      for (const [vw, vi] of VOCAB_IDX) {
        const ml = Math.min(w.length, vw.length);
        let plen = 0;
        for (let k = 0; k < ml; k++) {
          if (w[k] === vw[k]) plen++;
          else break;
        }
        if (plen > bestLen) {
          bestLen = plen;
          best = vi;
        }
      }
      if (best >= 0 && bestLen >= 3) {
        ids.push(best);
      }
    }
  }
  return ids;
}


// ═══════════════════════════════════════════════════════════════
// TRAINING — next-word prediction, step s predicts word[s+1]
// ═══════════════════════════════════════════════════════════════

function train(model: Penelope, dataPath: string, steps: number = 5000, lr: number = 3e-4): void {
  const text = fs.readFileSync(dataPath, "utf-8");
  const ids = tokenize_text(text);
  if (ids.length < STEPS + 2) {
    console.log(`  corpus too small: ${ids.length} words (need ${STEPS + 2}+)`);
    return;
  }

  console.log(`  corpus: ${text.length} chars → ${ids.length} vocab words`);
  console.log(`  model: ${model.paramCount().toLocaleString()} params (${(model.paramCount() * 4 / 1e6).toFixed(1)}MB f32)`);
  console.log(`  training: ${steps} steps, lr=${lr.toExponential(1)}`);

  const window = STEPS + 1;
  let bestLoss = Infinity;

  for (let step = 1; step <= steps; step++) {
    const start = Math.floor(Math.random() * (ids.length - window));
    const win = ids.slice(start, start + window);

    let totalLoss = 0.0;

    for (let s = 0; s < STEPS; s++) {
      const context = win.slice(0, s + 1);
      const target = win[s + 1];

      const logits = model.forwardStep(context, s);
      const probs = softmax(logits);
      let p = probs[target];
      if (p < 1e-10) p = 1e-10;
      totalLoss -= Math.log(p);

      // gradient: d_logits = probs - one_hot(target)
      const d_logits = [...probs];
      d_logits[target] -= 1.0;

      const sw = model.steps[s];
      const ctx = model.poolContext(context);

      // reconstruct forward
      const query = matmul_mv(sw.wr, ctx, D, D);
      const query_n = rmsnorm(query, sw.rms, D);
      const gate = matmul_mv(sw.w_gate, query_n, M, D);
      const up = matmul_mv(sw.w_up, query_n, M, D);
      const swiglu = new Array(M);
      for (let i = 0; i < M; i++) swiglu[i] = silu(gate[i]) * up[i];
      const hidden = matmul_mv(sw.w_down, swiglu, D, M);
      const out = vadd(query_n, hidden);

      // d_out from tied weights
      const d_out = zeros(D);
      for (let v = 0; v < V; v++) {
        if (Math.abs(d_logits[v]) < 1e-8) continue;
        const ev = model.getEmbed(v);
        for (let j = 0; j < D; j++) {
          d_out[j] += d_logits[v] * ev[j];
        }
      }

      // update embedding (SGD on tied weights)
      for (let v = 0; v < V; v++) {
        if (Math.abs(d_logits[v]) < 1e-8) continue;
        const base = v * D;
        for (let j = 0; j < D; j++) {
          model.embed[base + j] -= lr * d_logits[v] * out[j];
        }
      }

      // d_hidden (residual)
      const d_hidden = [...d_out];

      // backprop through w_down
      const d_swiglu = matmul_mtv(sw.w_down, d_hidden, D, M);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < D; j++) {
          sw.w_down[i * D + j] -= lr * swiglu[i] * d_hidden[j];
        }
      }

      // backprop through SwiGLU
      for (let i = 0; i < M; i++) {
        const sg = silu(gate[i]);
        const sig = gate[i] > -20 ? 1.0 / (1.0 + Math.exp(-gate[i])) : 0;
        const silu_grad = gate[i] > -20 ? sig * (1.0 + gate[i] * (1.0 - sig)) : 0;
        const d_gate_i = d_swiglu[i] * up[i] * silu_grad;
        const d_up_i = d_swiglu[i] * sg;

        for (let j = 0; j < D; j++) {
          sw.w_gate[i * D + j] -= lr * d_gate_i * query_n[j];
          sw.w_up[i * D + j] -= lr * d_up_i * query_n[j];
        }
      }

      // d_query_n (from SwiGLU input + residual)
      const d_qn = [...d_out];
      const d_gate_vec = new Array(M);
      for (let i = 0; i < M; i++) {
        const sig = gate[i] > -20 ? 1.0 / (1.0 + Math.exp(-gate[i])) : 0;
        const silu_grad = gate[i] > -20 ? sig * (1.0 + gate[i] * (1.0 - sig)) : 0;
        d_gate_vec[i] = d_swiglu[i] * up[i] * silu_grad;
      }
      const d_qn_gate = matmul_mtv(sw.w_gate, d_gate_vec, M, D);
      const d_up_vec = new Array(M);
      for (let i = 0; i < M; i++) d_up_vec[i] = d_swiglu[i] * silu(gate[i]);
      const d_qn_up = matmul_mtv(sw.w_up, d_up_vec, M, D);
      for (let j = 0; j < D; j++) d_qn[j] += d_qn_gate[j] + d_qn_up[j];

      // approx RMSNorm backward
      let ss = 0;
      for (let i = 0; i < D; i++) ss += query[i] * query[i];
      ss = ss / D + 1e-5;
      const inv = 1.0 / Math.sqrt(ss);
      const d_query = new Array(D);
      for (let i = 0; i < D; i++) d_query[i] = d_qn[i] * sw.rms[i] * inv;

      // update Wr
      for (let i = 0; i < D; i++) {
        if (Math.abs(d_query[i]) < 1e-8) continue;
        for (let j = 0; j < D; j++) {
          sw.wr[i * D + j] -= lr * d_query[i] * ctx[j];
        }
      }
    }

    const avgLoss = totalLoss / STEPS;
    if (avgLoss < bestLoss) bestLoss = avgLoss;

    if (step % 50 === 0 || step === 1) {
      console.log(`  step ${String(step).padStart(5)}/${steps}  loss=${avgLoss.toFixed(4)}  best=${bestLoss.toFixed(4)}`);
    }
  }

  console.log(`  training complete. best loss: ${bestLoss.toFixed(4)}`);
}


// ═══════════════════════════════════════════════════════════════
// DARIO FIELD — live co-occurrence overlay
// ═══════════════════════════════════════════════════════════════

class DarioField {
  cooc: Map<string, number>;
  bigrams: Map<string, Map<string, number>>;
  destiny: number[];
  trauma: number;
  prophecyTarget: number | null;
  prophecyAge: number;
  chambers: Record<string, number>;
  decay: Record<string, number>;

  constructor() {
    this.cooc = new Map();
    this.bigrams = new Map();
    this.destiny = new Array(8).fill(0.0);
    this.trauma = 0.0;
    this.prophecyTarget = null;
    this.prophecyAge = 0;
    this.chambers = { fear: 0, love: 0, rage: 0, void: 0, flow: 0, complex: 0 };
    this.decay = { fear: 0.95, love: 0.95, rage: 0.93, void: 0.96, flow: 0.94, complex: 0.97 };
  }

  updateCooc(w1: number, w2: number): void {
    const k = `${Math.min(w1, w2)}|${Math.max(w1, w2)}`;
    this.cooc.set(k, (this.cooc.get(k) || 0) + 1.0);
  }

  getCooc(w1: number, w2: number): number {
    const k = `${Math.min(w1, w2)}|${Math.max(w1, w2)}`;
    return this.cooc.get(k) || 0.0;
  }

  updateChambers(stepIdx: number): void {
    const C = this.chambers;
    const depth = stepIdx / STEPS;
    const phase = depth < 0.33 ? 0 : depth < 0.66 ? 1 : 2;
    if (phase === 0) C["flow"] += 0.05;
    if (phase === 1) C["fear"] += 0.04;
    if (phase === 2) C["void"] += 0.05;
    if (depth > 0.75) C["complex"] += 0.03;
    if (this.trauma > 0.3) C["rage"] += 0.04;
    const K = 0.02;
    const old: Record<string, number> = { ...C };
    for (const i in C) {
      for (const j in C) {
        if (i !== j) {
          C[i] += K * Math.sin(old[j] - old[i]);
        }
      }
    }
    for (const k in C) {
      C[k] = Math.max(0, Math.min(1, C[k] * (this.decay[k] ?? 0.95)));
    }
  }

  overlay(logits: number[], contextIds: number[], stepIdx: number): number[] {
    const C = this.chambers;
    const alphaMod = 1 + 0.3 * C["love"] - 0.2 * C["rage"] + 0.1 * C["flow"];
    const gammaMod = 1 + 0.4 * C["void"] + 0.2 * C["complex"];

    for (let v = 0; v < V; v++) {
      let h = 0.0;
      const recent = contextIds.slice(-8);
      for (const ci of recent) {
        h += this.getCooc(ci, v);
      }
      if (h > 0) {
        logits[v] += alphaMod * 0.3 * Math.min(h, 1.0);
      }

      if (this.prophecyTarget !== null && v === this.prophecyTarget) {
        logits[v] += 0.5 * Math.log(1 + this.prophecyAge);
      }

      const cat = word_category(v);
      let dMax = 0.01;
      for (const d of this.destiny) if (Math.abs(d) > dMax) dMax = Math.abs(d);
      logits[v] += gammaMod * 0.25 * this.destiny[cat] / dMax;
    }

    return logits;
  }
}


function word_category(idx: number): number {
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
// GENERATION — 12 steps, each picks one word
// ═══════════════════════════════════════════════════════════════

function find_seed(key: string): number {
  if (VOCAB_IDX.has(key)) return VOCAB_IDX.get(key)!;
  let best = 0;
  let bestScore = -1;
  for (const [w, i] of VOCAB_IDX) {
    let score = 0;
    if (w.includes(key) || key.includes(w)) score = 3;
    for (let k = 0; k < Math.min(w.length, key.length); k++) {
      if (w[k] === key[k]) score += 0.5;
      else break;
    }
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return bestScore > 0 ? best : Math.floor(Math.random() * 200);
}


function extract_key(text: string): string {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 1 && !STOP.has(w));
  if (words.length === 0) {
    const parts = text.split(/\s+/);
    return parts.length > 0 ? parts[0].toLowerCase() : "silence";
  }
  words.sort((a, b) => b.length - a.length);
  return words[0];
}


function run_chain(model: Penelope, field: DarioField, text: string): number[] {
  const key = extract_key(text);
  const seed = find_seed(key);

  const deepCats = [2, 5, 7];
  const tcat = deepCats[Math.floor(Math.random() * deepCats.length)];
  const ranges: [number, number][] = [
    [0, 100], [100, 200], [200, 300], [300, 350],
    [350, 450], [450, 550], [550, 650], [650, V]
  ];
  const [s, e] = ranges[tcat];
  field.prophecyTarget = Math.floor(Math.random() * (Math.min(e, V) - s)) + s;
  field.prophecyAge = 0;

  console.log(`\n  destined: ${VOCAB[field.prophecyTarget]}`);
  console.log(`\n  ${VOCAB[seed]}`);

  const chain = [seed];
  const forbidden = new Set([seed]);

  for (let step = 0; step < STEPS; step++) {
    field.updateChambers(step);
    field.prophecyAge++;

    // learned logits from step-specific weights
    let logits = model.forwardStep(chain, step);

    // Dario field overlay
    logits = field.overlay(logits, chain, step);

    // mask forbidden
    for (const f of forbidden) {
      logits[f] = -1e9;
    }

    // top-k sampling
    const probs = softmax(logits);
    const indexed: [number, number][] = [];
    for (let i = 0; i < probs.length; i++) indexed.push([i, probs[i]]);
    indexed.sort((a, b) => b[1] - a[1]);
    const topk = indexed.slice(0, 12);
    let total = 0.001;
    for (const [, p] of topk) total += Math.max(0, p);
    let r = Math.random() * total;
    let pick = topk[0][0];
    for (const [idx, p] of topk) {
      r -= Math.max(0, p);
      if (r <= 0) {
        pick = idx;
        break;
      }
    }

    chain.push(pick);
    forbidden.add(pick);

    // update field
    if (chain.length >= 2) {
      field.updateCooc(chain[chain.length - 2], pick);
      const cat = word_category(pick);
      field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat];
    }

    if (step > 7) {
      field.trauma = Math.min(1, field.trauma + 0.1);
    }
    field.trauma *= 0.97;

    const marker = step === STEPS - 1 ? "  *" : "   ";
    console.log(`${marker}${VOCAB[pick]}`);
  }

  const fulfilled = chain.includes(field.prophecyTarget!);
  const cats = new Set(chain.map(w => word_category(w))).size;
  console.log(`\n  drift ${cats}/8 · prophecy ${fulfilled ? "fulfilled" : "unfulfilled"}`);
  return chain;
}


// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

function main(): void {
  const args = process.argv.slice(2);
  let trainPath: string | null = null;
  let loadPath: string | null = null;
  let savePath: string | null = null;
  let trainSteps = 5000;
  let lr = 3e-4;
  let text: string | null = null;

  let i = 0;
  while (i < args.length) {
    if (args[i] === "--train" && i + 1 < args.length) {
      trainPath = args[i + 1]; i += 2;
    } else if (args[i] === "--load" && i + 1 < args.length) {
      loadPath = args[i + 1]; i += 2;
    } else if (args[i] === "--save" && i + 1 < args.length) {
      savePath = args[i + 1]; i += 2;
    } else if (args[i] === "--steps" && i + 1 < args.length) {
      trainSteps = parseInt(args[i + 1]); i += 2;
    } else if (args[i] === "--lr" && i + 1 < args.length) {
      lr = parseFloat(args[i + 1]); i += 2;
    } else {
      text = args.slice(i).join(" ");
      break;
    }
  }

  const model = new Penelope();
  const field = new DarioField();

  console.log();
  console.log(`  penelope — 1984 words, ${STEPS} steps, Dario Equation`);
  console.log(`  ${model.paramCount().toLocaleString()} trainable params`);
  console.log();

  if (loadPath && fs.existsSync(loadPath)) {
    model.load(loadPath);
  }

  if (trainPath) {
    train(model, trainPath, trainSteps, lr);
    if (savePath) {
      model.save(savePath);
    }
  }

  if (text) {
    run_chain(model, field, text);
  } else if (!trainPath) {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    const prompt = (): void => {
      rl.question("  > ", (answer) => {
        const trimmed = answer.trim();
        if (!trimmed) {
          prompt();
          return;
        }
        run_chain(model, field, trimmed);
        prompt();
      });
    };
    prompt();
  }

  if (savePath && !trainPath) {
    model.save(savePath);
  }
}

main();
