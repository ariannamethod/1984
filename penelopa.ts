// penelope.ts — 1984 words. 12 steps of resonance. Dario Equation.
// TypeScript version. Single file. No dependencies.
// By Arianna Method. הרזוננס לא נשבר

// ================================================================
// 1984 WORDS (same as Python/JS/C versions)
// ================================================================

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
] as const;

const STEPS = 12;
const STOP_WORDS = new Set([
  "i","me","my","we","our","you","your","he","she","it","they","them","the","a","an",
  "and","or","but","in","on","at","to","for","of","is","am","are","was","were","be",
  "been","being","have","has","had","do","does","did","will","would","shall","should",
  "can","could","may","might","must","not","no","nor","so","if","then","than","that",
  "this","these","those","what","which","who","whom","how","when","where","why","all",
  "each","every","some","any","few","many","much","more","most","other","another","such"
]);

// ================================================================
// DARIO FIELD STATE
// ================================================================

// Co-occurrence matrix (undirected)
const cooc = new Map<string, number>(); // key: "min|max" -> count
const bigrams = new Map<string, Map<string, number>>(); // prev -> {next: count}
let destiny = new Array(8).fill(0);
let trauma = 0;
let prophecyTarget: string | null = null;
let prophecyAge = 0;

// Kuramoto chambers
const chambers = {
  fear: 0,
  love: 0,
  rage: 0,
  void_: 0,
  flow: 0,
  complex: 0
};
const chamberDecay = {
  fear: 0.95,
  love: 0.95,
  rage: 0.93,
  void_: 0.96,
  flow: 0.94,
  complex: 0.97
};

// ================================================================
// UTILITIES
// ================================================================

function wordCategory(idx: number): number {
  if (idx < 100) return 0;
  if (idx < 200) return 1;
  if (idx < 300) return 2;
  if (idx < 350) return 3;
  if (idx < 450) return 4;
  if (idx < 550) return 5;
  if (idx < 650) return 6;
  return 7;
}

function updateCooc(a: string, b: string) {
  const key = a < b ? `${a}|${b}` : `${b}|${a}`;
  cooc.set(key, (cooc.get(key) || 0) + 1);
}

function getCooc(a: string, b: string): number {
  const key = a < b ? `${a}|${b}` : `${b}|${a}`;
  return cooc.get(key) || 0;
}

function updateBigram(prev: string, next: string) {
  if (!bigrams.has(prev)) bigrams.set(prev, new Map());
  const m = bigrams.get(prev)!;
  m.set(next, (m.get(next) || 0) + 1);
}

function updateChambers(stepIdx: number) {
  const depth = stepIdx / STEPS;
  const phase = depth < 0.33 ? 0 : depth < 0.66 ? 1 : 2;
  if (phase === 0) chambers.flow += 0.05;
  if (phase === 1) chambers.fear += 0.04;
  if (phase === 2) chambers.void_ += 0.05;
  if (depth > 0.75) chambers.complex += 0.03;
  if (trauma > 0.3) chambers.rage += 0.04;

  const K = 0.02;
  const old = { ...chambers };
  const names = Object.keys(chambers) as (keyof typeof chambers)[];
  for (const i of names) {
    for (const j of names) {
      if (i !== j) {
        (chambers as any)[i] += K * Math.sin(old[j] - old[i]);
      }
    }
  }

  for (const k of names) {
    (chambers as any)[k] *= chamberDecay[k];
    (chambers as any)[k] = Math.max(0, Math.min(1, (chambers as any)[k]));
  }
}

function darioScore(
  candidate: string,
  context: string[],
  prevWord: string | null,
  stepIdx: number
): number {
  const cidx = VOCAB.indexOf(candidate);
  if (cidx < 0) return -Infinity;

  const alphaMod = 1 + 0.3 * chambers.love - 0.2 * chambers.rage + 0.1 * chambers.flow;
  const gammaMod = 1 + 0.4 * chambers.void_ + 0.2 * chambers.complex;

  // B: bigram
  let B = 0;
  if (prevWord && bigrams.has(prevWord)) {
    const m = bigrams.get(prevWord)!;
    const maxVal = Math.max(...m.values(), 0.001);
    B = (m.get(candidate) || 0) / maxVal;
  }

  // H: Hebbian co-occurrence with context
  let H = 0;
  for (let i = 0; i < context.length; i++) {
    H += getCooc(context[i], candidate) * (1 / (context.length - i));
  }
  H = Math.min(H, 1);

  // F: prophecy
  let F = 0;
  if (prophecyTarget && candidate === prophecyTarget) {
    F = 0.5 * Math.log(1 + prophecyAge);
  }

  // A: destiny attraction (category alignment)
  const cat = wordCategory(cidx);
  const dMax = Math.max(...destiny.map(Math.abs), 0.01);
  let A = destiny[cat] / dMax;

  // T: trauma (deeper words in later steps)
  let T = 0;
  if (trauma > 0.3 && stepIdx > 6) {
    if (cidx >= 200 && cidx < 300) T = trauma * 1.5;
    if (cidx >= 450 && cidx < 550) T = trauma * 1.2;
    if (cidx >= 930 && cidx < 1000) T = trauma * 1.0;
  }

  // wormhole: occasional category jump
  let wormhole = 0;
  if (Math.random() < 0.08 && stepIdx > 3 && stepIdx < 10) {
    const jumpCat = Math.floor(Math.random() * 8);
    if (cat === jumpCat) wormhole = 0.5;
  }

  // step-specific vibe
  const vibes = [1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 0.8, 0.7, 1.4];
  const vibe = vibes[stepIdx] || 1.0;
  const tau = 0.7 + stepIdx * 0.08;

  const score = (B * 8 + alphaMod * 0.3 * H + 0.15 * F + gammaMod * 0.25 * A + T + wormhole) * vibe / tau;
  return score;
}

function selectWord(
  context: string[],
  prevWord: string | null,
  stepIdx: number,
  forbidden: Set<string>
): string {
  const scores: { word: string; score: number }[] = [];
  for (const w of VOCAB) {
    if (forbidden.has(w)) continue;
    scores.push({ word: w, score: darioScore(w, context, prevWord, stepIdx) });
  }

  // add noise
  const noise = 0.3 * (1 - stepIdx / STEPS);
  for (const s of scores) s.score += Math.random() * noise;

  scores.sort((a, b) => b.score - a.score);

  const k = Math.min(12, scores.length);
  const topk = scores.slice(0, k);
  const total = topk.reduce((s, x) => s + Math.max(0, x.score), 0.001);
  let r = Math.random() * total;
  for (const t of topk) {
    r -= Math.max(0, t.score);
    if (r <= 0) return t.word;
  }
  return topk[0].word;
}

function extractKey(text: string): string {
  const words = text.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/)
    .filter(w => w.length > 1 && !STOP_WORDS.has(w));
  if (words.length === 0) return text.toLowerCase().split(/\s+/)[0] || "silence";
  words.sort((a, b) => b.length - a.length || words.indexOf(b) - words.indexOf(a));
  return words[0];
}

function findSeed(key: string): string {
  const exact = VOCAB.find(w => w === key);
  if (exact) return exact;

  let best = "";
  let bestScore = -1;
  for (const w of VOCAB) {
    let score = 0;
    if (w.includes(key) || key.includes(w)) score = 3;
    for (let i = 0; i < Math.min(w.length, key.length); i++) {
      if (w[i] === key[i]) score += 0.5;
      else break;
    }
    if (score > bestScore) {
      bestScore = score;
      best = w;
    }
  }
  if (best && bestScore > 0) return best;
  return VOCAB[Math.floor(Math.random() * 200)]; // fallback
}

function makeProphecy(seed: string): string {
  const deepCats = [2, 5, 7]; // emotion, abstract, other
  const targetCat = deepCats[Math.floor(Math.random() * deepCats.length)];
  const catStart = [0, 100, 200, 300, 350, 450, 550, 650][targetCat] || 450;
  const catEnd = [100, 200, 300, 350, 450, 550, 650, 1984][targetCat] || 550;
  const idx = catStart + Math.floor(Math.random() * (catEnd - catStart));
  return VOCAB[idx];
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ================================================================
// UI & MAIN CHAIN
// ================================================================

let running = false;

async function runChain(userText: string) {
  const column = document.getElementById('column');
  const metrics = document.getElementById('metrics');
  const prophecyEl = document.getElementById('prophecy');
  const waiting = document.getElementById('waiting');
  if (!column || !metrics || !prophecyEl || !waiting) return;

  column.innerHTML = '';
  metrics.classList.remove('visible');
  prophecyEl.classList.remove('visible');
  waiting.style.display = 'none';

  const keyWord = extractKey(userText);
  const seed = findSeed(keyWord);

  const prophesied = makeProphecy(seed);
  prophecyTarget = prophesied;
  prophecyAge = 0;
  prophecyEl.textContent = `destined: ${prophesied}`;
  setTimeout(() => prophecyEl.classList.add('visible'), 300);

  const inputWords = new Set(userText.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/));
  const forbidden = new Set([...inputWords, seed]);

  // show seed
  await showWord(column, seed, 'seed', 0);

  const chain: string[] = [seed];
  let prevWord = seed;
  let totalDebt = 0;

  for (let step = 0; step < STEPS; step++) {
    await sleep(1200 + step * 150);
    updateChambers(step);
    prophecyAge++;

    const word = selectWord(chain, prevWord, step, forbidden);
    chain.push(word);
    forbidden.add(word);

    updateCooc(prevWord, word);
    updateBigram(prevWord, word);
    const widx = VOCAB.indexOf(word);
    if (widx >= 0) {
      const cat = wordCategory(widx);
      destiny[cat] = 0.3 * 1 + 0.7 * destiny[cat];
    }

    if (word !== prophecyTarget) {
      totalDebt += 0.1 * Math.log(1 + prophecyAge);
    } else {
      totalDebt = Math.max(0, totalDebt - 1);
    }

    if (step > 7) trauma = Math.min(1, trauma + 0.1);
    trauma *= 0.97;

    // ingest into field (cooc with recent context)
    const recent = chain.slice(-4);
    for (const ctx of recent) updateCooc(ctx, word);

    const cls = step === STEPS - 1 ? 'final' : 'step';
    await showWord(column, word, cls, step + 1);
    prevWord = word;
  }

  const fulfilled = chain.includes(prophecyTarget!);

  await sleep(600);
  const cats = chain.map(w => wordCategory(VOCAB.indexOf(w))).filter(c => c >= 0);
  const uniqueCats = new Set(cats).size;
  metrics.innerHTML = `debt ${totalDebt.toFixed(2)} · resonance ${(cooc.size / 100).toFixed(2)} · drift ${uniqueCats}/8 · prophecy ${fulfilled ? 'fulfilled' : 'unfulfilled'}`;
  metrics.classList.add('visible');

  if (fulfilled) {
    prophecyEl.textContent = `destined: ${prophesied} ✓`;
  } else {
    prophecyEl.textContent = `destined: ${prophesied} — unfulfilled`;
  }
}

function showWord(container: HTMLElement, word: string, cls: string, stepNum: number): Promise<void> {
  return new Promise(resolve => {
    const div = document.createElement('div');
    div.className = `word-line ${cls}`;
    div.textContent = word;
    container.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    setTimeout(resolve, 100);
  });
}

// ================================================================
// BOOTSTRAP UI
// ================================================================

function initUI() {
  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Penelope · TypeScript</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,700;1,400&display=swap');
    :root {
      --bg: #fafaf8;
      --fg: #1a1a1a;
      --dim: #777;
      --ghost: #999;
      --accent: #1a1a1a;
      --serif: 'EB Garamond', Georgia, serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--fg);
      font-family: var(--serif);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #logo {
      position: fixed;
      top: 24px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 11px;
      letter-spacing: 8px;
      text-transform: uppercase;
      color: #888;
      user-select: none;
      font-weight: 500;
    }
    #eight {
      position: fixed;
      top: 18px;
      right: 28px;
      font-size: 22px;
      color: #999;
      font-weight: 700;
      user-select: none;
      opacity: 0.4;
      letter-spacing: 2px;
    }
    #arena {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      align-items: center;
      padding: 80px 40px 120px;
      overflow: hidden;
    }
    #prophecy {
      position: fixed;
      top: 60px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 12px;
      font-style: italic;
      color: var(--ghost);
      letter-spacing: 2px;
      opacity: 0;
      transition: opacity 1.5s ease;
      text-align: center;
    }
    #prophecy.visible { opacity: 1; }
    #column {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      width: 100%;
    }
    .word-line {
      font-size: 23px;
      font-weight: 400;
      letter-spacing: 1px;
      opacity: 0;
      transform: translateY(12px);
      animation: word-arrive 0.8s ease forwards;
      text-align: center;
      line-height: 1.3;
    }
    .word-line.seed {
      font-weight: 700;
      font-size: 27px;
      color: var(--accent);
    }
    .word-line.step { color: var(--fg); }
    .word-line.final {
      font-weight: 700;
      font-size: 25px;
      border-bottom: 1px solid var(--dim);
      padding-bottom: 4px;
    }
    @keyframes word-arrive {
      0% { opacity: 0; transform: translateY(12px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    #metrics {
      position: fixed;
      bottom: 68px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 10px;
      letter-spacing: 2px;
      color: var(--ghost);
      text-align: center;
      opacity: 0;
      transition: opacity 1s ease;
    }
    #metrics.visible { opacity: 1; }
    #input-bar {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-top: 1px solid #ccc;
      background: var(--bg);
    }
    #input-bar input {
      background: none;
      border: none;
      border-bottom: 1px solid #bbb;
      outline: none;
      font-family: var(--serif);
      font-size: 15px;
      color: var(--fg);
      text-align: center;
      width: 400px;
      letter-spacing: 0.5px;
      padding-bottom: 2px;
    }
    #input-bar input::placeholder {
      color: #999;
      font-style: italic;
    }
    #input-bar input:focus {
      border-bottom-color: var(--fg);
    }
    #send-btn {
      background: none;
      border: none;
      cursor: pointer;
      padding: 4px 8px;
      margin-left: 8px;
      color: #999;
      font-size: 20px;
      line-height: 1;
      transition: color 0.2s ease;
      user-select: none;
    }
    #send-btn:hover { color: var(--fg); }
    #send-btn:active { color: var(--fg); transform: scale(0.9); }
    #send-btn:disabled { opacity: 0.3; cursor: default; }
    #waiting {
      font-size: 14px;
      color: var(--ghost);
      font-style: italic;
      letter-spacing: 1px;
    }
  </style>
</head>
<body>
  <div id="logo">penelope</div>
  <div id="eight">12</div>
  <div id="prophecy"></div>
  <div id="arena">
    <div id="waiting">waiting</div>
    <div id="column"></div>
  </div>
  <div id="metrics"></div>
  <div id="input-bar">
    <input type="text" id="input" placeholder="speak" autocomplete="off" spellcheck="false">
    <button id="send-btn" aria-label="send">→</button>
  </div>
</body>
</html>
  `;

  document.documentElement.innerHTML = html;

  const input = document.getElementById('input') as HTMLInputElement;
  const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;

  async function submit() {
    if (running) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;
    running = true;

    await runChain(text);

    running = false;
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }

  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
  sendBtn.addEventListener('click', submit);
  input.focus();
}

// ================================================================
// START
// ================================================================

if (typeof window !== 'undefined') {
  window.onload = initUI;
}

export {}; // make it a module
