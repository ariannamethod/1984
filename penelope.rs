// penelope.rs — 1984 words. 12 steps of resonance. Dario Equation.
//
// Trainable resonance engine in Rust. Not a transformer. A mirror that learns.
//
// Input:  text → vocab word IDs (prefix matching)
// Attend: RRPRAM resonance + SwiGLU per step (how it thinks)
// Output: word-level from 1984 vocab (gibberish impossible)
//
// 12 learned step-weights (~1.03M each). Each step has its own lens.
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
//   score(w) = B + α·H + β·F + γ·A + T      (Dario Equation)
//
//   rustc -O penelope.rs -o penelope_rs
//   ./penelope_rs                                  # interactive
//   ./penelope_rs "darkness eats the city"         # single chain
//   ./penelope_rs --train corpus.txt               # train 5000 steps
//   ./penelope_rs --train corpus.txt --steps 1000  # train N steps
//   ./penelope_rs --load penelope.bin              # load weights
//   ./penelope_rs --save penelope.bin              # save after
//
// By Arianna Method. הרזוננס לא נשבר

use std::collections::HashMap;
use std::convert::TryInto;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

const NWORDS: usize = 1990;
const NSTEPS: usize = 12;
const DIM: usize = 384;
const HDIM: usize = 768;

// ═══════════════════════════════════════════════════════════════
// 1984 WORDS
// ═══════════════════════════════════════════════════════════════

const VOCAB: [&str; NWORDS] = [
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
"debt","credit","interest","principal","collateral","default","bankruptcy","solvency","dividend","investment",
];

// ═══════════════════════════════════════════════════════════════
// STOPWORDS
// ═══════════════════════════════════════════════════════════════

const STOPS: [&str; 79] = [
"i","me","my","we","our","you","your","he","she","it","they","them","the","a","an",
"and","or","but","in","on","at","to","for","of","is","am","are","was","were","be",
"been","being","have","has","had","do","does","did","will","would","shall","should",
"can","could","may","might","must","not","no","nor","so","if","then","than","that",
"this","these","those","what","which","who","whom","how","when","where","why","all",
"each","every","some","any","few","many","much","more","most","other","another","such",
];

// ═══════════════════════════════════════════════════════════════
// PRNG — xoshiro256** (no deps)
// ═══════════════════════════════════════════════════════════════

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = [seed ^ 0x123456789abcdef0, seed.wrapping_mul(6364136223846793005),
                     seed ^ 0xfedcba9876543210, seed.wrapping_mul(1442695040888963407)];
        for v in s.iter_mut() { *v = v.wrapping_add(0x9e3779b97f4a7c15); }
        Rng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn randf(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn randn(&mut self) -> f32 {
        let u1 = self.randf() + 1e-12;
        let u2 = self.randf() + 1e-12;
        (-2.0 * u1.ln()).sqrt() * (6.2831853 * u2).cos()
    }

    fn randint(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

// ═══════════════════════════════════════════════════════════════
// MATH UTILS
// ═══════════════════════════════════════════════════════════════

fn silu(x: f32) -> f32 {
    if x > -20.0 { x / (1.0 + (-x).exp()) } else { 0.0 }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&v| v / s).collect()
}

fn matmul_mv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for i in 0..rows {
        let mut s = 0.0f32;
        let base = i * cols;
        for j in 0..cols { s += w[base + j] * x[j]; }
        out[i] = s;
    }
    out
}

fn matmul_mtv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for j in 0..cols {
        let mut s = 0.0f32;
        for i in 0..rows { s += w[i * cols + j] * x[i]; }
        out[j] = s;
    }
    out
}

fn rmsnorm(x: &[f32], g: &[f32]) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-5;
    let inv = 1.0 / ss.sqrt();
    (0..n).map(|i| g[i] * x[i] * inv).collect()
}

// ═══════════════════════════════════════════════════════════════
// MODEL — 12 step-specific weight sets + shared embedding
// ═══════════════════════════════════════════════════════════════

struct StepWeights {
    wr: Vec<f32>,      // [DIM*DIM]
    rms: Vec<f32>,     // [DIM]
    w_gate: Vec<f32>,  // [DIM*HDIM]
    w_up: Vec<f32>,    // [DIM*HDIM]
    w_down: Vec<f32>,  // [HDIM*DIM]
}

impl StepWeights {
    fn new(rng: &mut Rng) -> Self {
        let sd = (2.0f32 / DIM as f32).sqrt();
        let sm = (2.0f32 / HDIM as f32).sqrt();
        StepWeights {
            wr:     (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            rms:    vec![1.0; DIM],
            w_gate: (0..DIM*HDIM).map(|_| rng.randn() * sd).collect(),
            w_up:   (0..DIM*HDIM).map(|_| rng.randn() * sd).collect(),
            w_down: (0..HDIM*DIM).map(|_| rng.randn() * sm).collect(),
        }
    }

    fn param_count() -> usize {
        DIM*DIM + DIM + DIM*HDIM + DIM*HDIM + HDIM*DIM
    }
}

struct Penelope {
    embed: Vec<f32>,  // [NWORDS*DIM]
    steps: Vec<StepWeights>,
}

impl Penelope {
    fn new(rng: &mut Rng) -> Self {
        let sv = (2.0f32 / NWORDS as f32).sqrt();
        Penelope {
            embed: (0..NWORDS*DIM).map(|_| rng.randn() * sv).collect(),
            steps: (0..NSTEPS).map(|_| StepWeights::new(rng)).collect(),
        }
    }

    fn param_count() -> usize {
        NWORDS * DIM + NSTEPS * StepWeights::param_count()
    }

    fn pool_context(&self, ids: &[usize]) -> Vec<f32> {
        let mut ctx = vec![0.0f32; DIM];
        if ids.is_empty() { return ctx; }
        for &id in ids {
            let base = id * DIM;
            for j in 0..DIM { ctx[j] += self.embed[base + j]; }
        }
        let inv = 1.0 / ids.len() as f32;
        for v in ctx.iter_mut() { *v *= inv; }
        ctx
    }

    fn forward_step(&self, ctx_ids: &[usize], step: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let sw = &self.steps[step];
        let ctx = self.pool_context(ctx_ids);

        let query = matmul_mv(&sw.wr, &ctx, DIM, DIM);
        let query_n = rmsnorm(&query, &sw.rms);
        let gate = matmul_mv(&sw.w_gate, &query_n, HDIM, DIM);
        let up = matmul_mv(&sw.w_up, &query_n, HDIM, DIM);
        let swiglu: Vec<f32> = (0..HDIM).map(|i| silu(gate[i]) * up[i]).collect();
        let hidden = matmul_mv(&sw.w_down, &swiglu, DIM, HDIM);
        let out: Vec<f32> = (0..DIM).map(|i| query_n[i] + hidden[i]).collect();
        let logits = matmul_mv(&self.embed, &out, NWORDS, DIM);

        (logits, query, query_n, gate, up, swiglu, out)
    }

    fn save(&self, path: &str) {
        let mut data: Vec<u8> = Vec::new();
        for &v in &[NWORDS as u32, DIM as u32, HDIM as u32, NSTEPS as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.embed { data.extend_from_slice(&v.to_le_bytes()); }
        for sw in &self.steps {
            for &v in &sw.wr { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &sw.rms { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &sw.w_gate { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &sw.w_up { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &sw.w_down { data.extend_from_slice(&v.to_le_bytes()); }
        }
        fs::write(path, &data).expect("save failed");
        let expected = 16 + Self::param_count() * 4;
        println!("  saved {}: {} params ({:.1}MB) [{}]", path, Self::param_count(),
                 data.len() as f64 / 1e6,
                 if data.len() == expected { "OK" } else { "SIZE MISMATCH!" });
    }

    fn load(&mut self, path: &str) -> bool {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("  cannot open {}: {}", path, e); return false; }
        };
        if data.len() < 16 { eprintln!("  file too small"); return false; }
        let v = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let h = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let s = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if v != NWORDS || d != DIM || h != HDIM || s != NSTEPS {
            eprintln!("  config mismatch: V={} D={} M={} S={}", v, d, h, s);
            return false;
        }
        let floats: Vec<f32> = data[16..].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut o = 0;
        self.embed = floats[o..o+NWORDS*DIM].to_vec(); o += NWORDS*DIM;
        for sw in &mut self.steps {
            sw.wr = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            sw.rms = floats[o..o+DIM].to_vec(); o += DIM;
            sw.w_gate = floats[o..o+DIM*HDIM].to_vec(); o += DIM*HDIM;
            sw.w_up = floats[o..o+DIM*HDIM].to_vec(); o += DIM*HDIM;
            sw.w_down = floats[o..o+HDIM*DIM].to_vec(); o += HDIM*DIM;
        }
        println!("  loaded {}: {} params", path, Self::param_count());
        true
    }
}

// ═══════════════════════════════════════════════════════════════
// TOKENIZER
// ═══════════════════════════════════════════════════════════════

fn build_vocab_idx() -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (i, &w) in VOCAB.iter().enumerate() {
        let clean = w.trim_end_matches(|c: char| c.is_ascii_digit());
        m.entry(clean.to_string()).or_insert(i);
    }
    m
}

fn is_stop(w: &str) -> bool {
    STOPS.contains(&w)
}

fn tokenize_text(text: &str, idx: &HashMap<String, usize>) -> Vec<usize> {
    let mut ids = Vec::new();
    for word in text.to_lowercase().split(|c: char| !c.is_alphabetic()) {
        if word.len() < 2 || is_stop(word) { continue; }
        if let Some(&i) = idx.get(word) {
            ids.push(i);
        } else {
            let mut best = None;
            let mut best_len = 0usize;
            for (&ref vw, &vi) in idx.iter() {
                let plen = word.chars().zip(vw.chars()).take_while(|(a,b)| a == b).count();
                if plen > best_len { best_len = plen; best = Some(vi); }
            }
            if best_len >= 3 { if let Some(b) = best { ids.push(b); } }
        }
    }
    ids
}

fn word_category(idx: usize) -> usize {
    if idx < 100 { 0 }
    else if idx < 200 { 1 }
    else if idx < 300 { 2 }
    else if idx < 350 { 3 }
    else if idx < 450 { 4 }
    else if idx < 550 { 5 }
    else if idx < 650 { 6 }
    else { 7 }
}

fn vocab_display(idx: usize) -> &'static str {
    VOCAB[idx]
}

// ═══════════════════════════════════════════════════════════════
// DARIO FIELD
// ═══════════════════════════════════════════════════════════════

struct DarioField {
    cooc: HashMap<(usize,usize), f32>,
    destiny: [f32; 8],
    trauma: f32,
    prophecy_target: Option<usize>,
    prophecy_age: usize,
    chambers: [f32; 6],  // fear, love, rage, void, flow, complex
}

const CH_DECAY: [f32; 6] = [0.95, 0.95, 0.93, 0.96, 0.94, 0.97];

impl DarioField {
    fn new() -> Self {
        DarioField {
            cooc: HashMap::new(),
            destiny: [0.0; 8],
            trauma: 0.0,
            prophecy_target: None,
            prophecy_age: 0,
            chambers: [0.0; 6],
        }
    }

    fn update_cooc(&mut self, a: usize, b: usize) {
        let key = if a < b { (a, b) } else { (b, a) };
        *self.cooc.entry(key).or_insert(0.0) += 1.0;
    }

    fn get_cooc(&self, a: usize, b: usize) -> f32 {
        let key = if a < b { (a, b) } else { (b, a) };
        *self.cooc.get(&key).unwrap_or(&0.0)
    }

    fn update_chambers(&mut self, step: usize) {
        let depth = step as f32 / NSTEPS as f32;
        let phase = if depth < 0.33 { 0 } else if depth < 0.66 { 1 } else { 2 };
        if phase == 0 { self.chambers[4] += 0.05; }
        if phase == 1 { self.chambers[0] += 0.04; }
        if phase == 2 { self.chambers[3] += 0.05; }
        if depth > 0.75 { self.chambers[5] += 0.03; }
        if self.trauma > 0.3 { self.chambers[2] += 0.04; }

        let k = 0.02f32;
        let old = self.chambers;
        for i in 0..6 {
            for j in 0..6 {
                if i != j { self.chambers[i] += k * (old[j] - old[i]).sin(); }
            }
        }
        for i in 0..6 {
            self.chambers[i] = (self.chambers[i] * CH_DECAY[i]).clamp(0.0, 1.0);
        }
    }

    fn overlay(&self, logits: &mut [f32], ctx: &[usize]) {
        let alpha_mod = 1.0 + 0.3*self.chambers[1] - 0.2*self.chambers[2] + 0.1*self.chambers[4];
        let gamma_mod = 1.0 + 0.4*self.chambers[3] + 0.2*self.chambers[5];

        let start = if ctx.len() > 8 { ctx.len() - 8 } else { 0 };
        let d_max = self.destiny.iter().map(|d| d.abs()).fold(0.01f32, f32::max);

        for v in 0..NWORDS {
            // H: Hebbian
            let mut h = 0.0f32;
            for &ci in &ctx[start..] { h += self.get_cooc(ci, v); }
            if h > 1.0 { h = 1.0; }
            logits[v] += alpha_mod * 0.3 * h;

            // F: prophecy
            if let Some(pt) = self.prophecy_target {
                if v == pt { logits[v] += 0.5 * (1.0 + self.prophecy_age as f32).ln(); }
            }

            // A: destiny
            let cat = word_category(v);
            logits[v] += gamma_mod * 0.25 * self.destiny[cat] / d_max;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TRAINING
// ═══════════════════════════════════════════════════════════════

fn train(model: &mut Penelope, data_path: &str, steps: usize, lr: f32) {
    let text = match fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("  cannot open {}: {}", data_path, e); return; }
    };
    let idx = build_vocab_idx();
    let ids = tokenize_text(&text, &idx);
    let window = NSTEPS + 1;
    if ids.len() < window + 1 {
        eprintln!("  corpus too small: {} words (need {}+)", ids.len(), window + 1);
        return;
    }

    println!("  corpus: {} bytes -> {} vocab words", text.len(), ids.len());
    println!("  model: {} params ({:.1}MB f32)", Penelope::param_count(),
             Penelope::param_count() as f64 * 4.0 / 1e6);
    println!("  training: {} steps, lr={:.1e}", steps, lr);

    let mut rng = Rng::new(42);
    let mut best_loss = f32::INFINITY;

    for step in 1..=steps {
        let start = rng.randint(ids.len() - window);
        let win = &ids[start..start+window];

        let mut total_loss = 0.0f32;

        for s in 0..NSTEPS {
            let ctx_ids = &win[..s+1];
            let target = win[s+1];

            let (logits, query, query_n, gate, up, swiglu, out) = model.forward_step(ctx_ids, s);
            let probs = softmax(&logits);
            let p = probs[target].max(1e-10);
            total_loss -= p.ln();

            // d_logits = probs - one_hot
            let mut d_logits = probs.clone();
            d_logits[target] -= 1.0;

            // d_out = E @ d_logits
            let mut d_out = vec![0.0f32; DIM];
            for v in 0..NWORDS {
                if d_logits[v].abs() < 1e-8 { continue; }
                for j in 0..DIM {
                    d_out[j] += d_logits[v] * model.embed[v * DIM + j];
                }
            }

            // update embedding
            for v in 0..NWORDS {
                if d_logits[v].abs() < 1e-8 { continue; }
                for j in 0..DIM {
                    model.embed[v * DIM + j] -= lr * d_logits[v] * out[j];
                }
            }

            // pre-compute context for Wr update (before mutable borrow)
            let ctx = model.pool_context(ctx_ids);

            // approx RMSNorm backward
            let ss_q: f32 = query.iter().map(|v| v * v).sum::<f32>() / DIM as f32 + 1e-5;
            let inv_q = 1.0 / ss_q.sqrt();
            let rms_copy: Vec<f32> = model.steps[s].rms.clone();
            let d_query: Vec<f32> = (0..DIM).map(|i| d_out[i] * rms_copy[i] * inv_q).collect();

            let d_swiglu = matmul_mtv(&model.steps[s].w_down, &d_out, DIM, HDIM);

            let sw = &mut model.steps[s];

            // backprop w_down
            for i in 0..HDIM {
                for j in 0..DIM {
                    sw.w_down[i * DIM + j] -= lr * swiglu[i] * d_out[j];
                }
            }

            // backprop SwiGLU
            for i in 0..HDIM {
                let sg = silu(gate[i]);
                let sig = if gate[i] > -20.0 { 1.0 / (1.0 + (-gate[i]).exp()) } else { 0.0 };
                let silu_grad = if gate[i] > -20.0 { sig * (1.0 + gate[i] * (1.0 - sig)) } else { 0.0 };
                let d_gate_i = d_swiglu[i] * up[i] * silu_grad;
                let d_up_i = d_swiglu[i] * sg;
                for j in 0..DIM {
                    sw.w_gate[i * DIM + j] -= lr * d_gate_i * query_n[j];
                    sw.w_up[i * DIM + j] -= lr * d_up_i * query_n[j];
                }
            }

            // update Wr
            for i in 0..DIM {
                if d_query[i].abs() < 1e-8 { continue; }
                for j in 0..DIM {
                    sw.wr[i * DIM + j] -= lr * d_query[i] * ctx[j];
                }
            }
        }

        let avg_loss = total_loss / NSTEPS as f32;
        if avg_loss < best_loss { best_loss = avg_loss; }

        if step % 50 == 0 || step == 1 {
            println!("  step {:5}/{} loss={:.4} best={:.4}", step, steps, avg_loss, best_loss);
        }
    }

    println!("  training complete. best loss: {:.4}", best_loss);
}

// ═══════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════

fn find_seed(key: &str, idx: &HashMap<String, usize>, rng: &mut Rng) -> usize {
    if let Some(&i) = idx.get(key) { return i; }
    let mut best = 0usize;
    let mut best_score = -1.0f32;
    for (vw, &vi) in idx.iter() {
        let mut score = 0.0f32;
        if vw.contains(key) || key.contains(vw.as_str()) { score = 3.0; }
        let plen = key.chars().zip(vw.chars()).take_while(|(a,b)| a == b).count();
        score += plen as f32 * 0.5;
        if score > best_score { best_score = score; best = vi; }
    }
    if best_score > 0.0 { best } else { rng.randint(200) }
}

fn extract_key(text: &str) -> String {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace()
        .filter(|w| w.len() > 1 && !is_stop(w)).collect();
    if words.is_empty() {
        lower.split_whitespace().next().unwrap_or("silence").to_string()
    } else {
        words.iter().max_by_key(|w| w.len()).unwrap().to_string()
    }
}

fn run_chain(model: &Penelope, field: &mut DarioField, text: &str, rng: &mut Rng) {
    let idx = build_vocab_idx();
    let key = extract_key(text);
    let seed = find_seed(&key, &idx, rng);

    let deep_cats = [2, 5, 7];
    let tcat = deep_cats[rng.randint(3)];
    let ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,NWORDS)];
    let (s, e) = ranges[tcat];
    field.prophecy_target = Some(s + rng.randint(e - s));
    field.prophecy_age = 0;

    println!("\n  destined: {}", vocab_display(field.prophecy_target.unwrap()));
    println!("\n  {}", vocab_display(seed));

    let mut chain = vec![seed];
    let mut forbidden = std::collections::HashSet::new();
    forbidden.insert(seed);

    let mut fulfilled = false;

    for step in 0..NSTEPS {
        field.update_chambers(step);
        field.prophecy_age += 1;

        let (mut logits, _, _, _, _, _, _) = model.forward_step(&chain, step);
        field.overlay(&mut logits, &chain);

        for &f in &forbidden { logits[f] = -1e9; }

        let probs = softmax(&logits);
        let mut top: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top.truncate(12);

        let total: f32 = top.iter().map(|(_, p)| p.max(0.0)).sum::<f32>() + 0.001;
        let mut r = rng.randf() * total;
        let mut pick = top[0].0;
        for &(idx, p) in &top {
            r -= p.max(0.0);
            if r <= 0.0 { pick = idx; break; }
        }

        chain.push(pick);
        forbidden.insert(pick);

        if chain.len() >= 2 {
            field.update_cooc(chain[chain.len()-2], pick);
        }
        let cat = word_category(pick);
        field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat];

        if Some(pick) == field.prophecy_target { fulfilled = true; }
        if step > 7 { field.trauma = (field.trauma + 0.1).min(1.0); }
        field.trauma *= 0.97;

        if step == NSTEPS - 1 {
            println!("  *{}", vocab_display(pick));
        } else {
            println!("   {}", vocab_display(pick));
        }
    }

    let cats: std::collections::HashSet<usize> = chain.iter().map(|&w| word_category(w)).collect();
    println!("\n  drift {}/8 \u{00b7} prophecy {}",
             cats.len(), if fulfilled { "fulfilled" } else { "unfulfilled" });
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut train_path: Option<String> = None;
    let mut load_path: Option<String> = None;
    let mut save_path: Option<String> = None;
    let mut train_steps = 5000usize;
    let mut lr = 3e-4f32;
    let mut text: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" if i+1 < args.len() => { train_path = Some(args[i+1].clone()); i += 2; }
            "--load" if i+1 < args.len() => { load_path = Some(args[i+1].clone()); i += 2; }
            "--save" if i+1 < args.len() => { save_path = Some(args[i+1].clone()); i += 2; }
            "--steps" if i+1 < args.len() => { train_steps = args[i+1].parse().unwrap_or(5000); i += 2; }
            "--lr" if i+1 < args.len() => { lr = args[i+1].parse().unwrap_or(3e-4); i += 2; }
            _ => { text = Some(args[i..].join(" ")); break; }
        }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut rng = Rng::new(seed);
    let mut model = Penelope::new(&mut rng);

    println!();
    println!("  penelope \u{2014} 1984 words, {} steps, Dario Equation", NSTEPS);
    println!("  {} trainable params ({:.1}MB f32)", Penelope::param_count(),
             Penelope::param_count() as f64 * 4.0 / 1e6);
    println!("  by Arianna Method");
    println!();

    if let Some(ref path) = load_path { model.load(path); }
    if let Some(ref path) = train_path {
        train(&mut model, path, train_steps, lr);
        if let Some(ref sp) = save_path { model.save(sp); }
    }

    let mut field = DarioField::new();

    if let Some(ref t) = text {
        run_chain(&model, &mut field, t, &mut rng);
    } else if train_path.is_none() {
        let stdin = io::stdin();
        loop {
            print!("  > ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
            let line = line.trim();
            if line.is_empty() { continue; }
            run_chain(&model, &mut field, line, &mut rng);
        }
    }

    if save_path.is_some() && train_path.is_none() {
        model.save(save_path.as_ref().unwrap());
    }
}
