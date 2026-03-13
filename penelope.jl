# penelope.jl — 1984 words. 12 steps of resonance. Dario Equation.
# Julia version. Single file. No dependencies.
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
"debt","credit","interest","principal","collateral","default","bankruptcy","solvency","dividend","investment",
]


const V = length(VOCAB)  # 1990 (1984 canonical + 6 overlapping entries)
const STEPS = 12
const D = 384            # embedding dim
const M = 768            # SwiGLU hidden dim

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
# MODEL — 12 step-specific weight sets + shared embedding
# ═══════════════════════════════════════════════════════════════

mutable struct StepWeights
    wr::Vector{Float64}       # D*D
    rms::Vector{Float64}      # D
    w_gate::Vector{Float64}   # D*M
    w_up::Vector{Float64}     # D*M
    w_down::Vector{Float64}   # M*D
end

function StepWeights()
    scale_d = sqrt(2.0 / D)
    scale_m = sqrt(2.0 / M)
    wr     = [_randn() * scale_d for _ in 1:(D * D)]
    rms    = ones(Float64, D)
    w_gate = [_randn() * scale_d for _ in 1:(D * M)]
    w_up   = [_randn() * scale_d for _ in 1:(D * M)]
    w_down = [_randn() * scale_m for _ in 1:(M * D)]
    return StepWeights(wr, rms, w_gate, w_up, w_down)
end

function step_param_count()
    return D*D + D + D*M + D*M + M*D
end

mutable struct Penelope
    embed::Vector{Float64}    # V*D
    steps::Vector{StepWeights}
end

function Penelope()
    scale = sqrt(2.0 / V)
    embed = [_randn() * scale for _ in 1:(V * D)]
    steps = [StepWeights() for _ in 1:STEPS]
    return Penelope(embed, steps)
end

function param_count(model::Penelope)
    return V * D + STEPS * step_param_count()
end

function get_embed(model::Penelope, idx::Int)
    # idx is 0-based
    base = idx * D + 1  # convert to 1-based Julia indexing
    return model.embed[base:base + D - 1]
end

function pool_context(model::Penelope, word_ids::Vector{Int})
    # word_ids are 0-based
    if isempty(word_ids)
        return _zeros(D)
    end
    ctx = _zeros(D)
    for wid in word_ids
        e = get_embed(model, wid)
        ctx = vadd(ctx, e)
    end
    return vscale(ctx, 1.0 / length(word_ids))
end

function forward_step(model::Penelope, context_ids::Vector{Int}, step_idx::Int)
    # step_idx is 0-based
    sw = model.steps[step_idx + 1]  # Julia 1-based
    ctx = pool_context(model, context_ids)

    # RRPRAM resonance: query = ctx @ Wr
    query = matmul_mv(sw.wr, ctx, D, D)

    # RMSNorm
    query = rmsnorm(query, sw.rms, D)

    # SwiGLU
    gate = matmul_mv(sw.w_gate, query, M, D)
    up = matmul_mv(sw.w_up, query, M, D)
    swiglu = [_silu(gate[i]) * up[i] for i in 1:M]
    hidden = matmul_mv(sw.w_down, swiglu, D, M)

    # Residual
    out = vadd(query, hidden)

    # Logits = E @ out (tied weights)
    logits = matmul_mv(model.embed, out, V, D)
    return logits
end

function save_model(model::Penelope, path::String)
    flat = copy(model.embed)
    for s in model.steps
        append!(flat, s.wr)
        append!(flat, s.rms)
        append!(flat, s.w_gate)
        append!(flat, s.w_up)
        append!(flat, s.w_down)
    end
    open(path, "w") do f
        write(f, Int32(V))
        write(f, Int32(D))
        write(f, Int32(M))
        write(f, Int32(STEPS))
        for v in flat
            write(f, Float32(v))
        end
    end
    sz = filesize(path) / 1e6
    println("  saved $path: $(length(flat)) params ($(round(sz, digits=1))MB)")
end

function load_model!(model::Penelope, path::String)
    open(path, "r") do f
        v = read(f, Int32)
        d = read(f, Int32)
        m = read(f, Int32)
        st = read(f, Int32)
        @assert v == V && d == D && m == M && st == STEPS "config mismatch: file has V=$v D=$d M=$m S=$st"
        total = V * D + STEPS * step_param_count()
        flat = Vector{Float64}(undef, total)
        for i in 1:total
            flat[i] = Float64(read(f, Float32))
        end
        o = 1  # 1-based offset
        model.embed = flat[o:o + V*D - 1]; o += V*D
        for s in model.steps
            s.wr     = flat[o:o + D*D - 1]; o += D*D
            s.rms    = flat[o:o + D - 1]; o += D
            s.w_gate = flat[o:o + D*M - 1]; o += D*M
            s.w_up   = flat[o:o + D*M - 1]; o += D*M
            s.w_down = flat[o:o + M*D - 1]; o += M*D
        end
    end
    println("  loaded $path")
end


# ═══════════════════════════════════════════════════════════════
# TOKENIZER — map arbitrary text to word IDs in our 1984 vocab
# ═══════════════════════════════════════════════════════════════

function tokenize_text(text::String)
    words = [m.match for m in eachmatch(r"[a-z]+", lowercase(text))]
    ids = Int[]
    for w in words
        if w in STOP || length(w) < 2
            continue
        end
        if haskey(VOCAB_IDX, w)
            push!(ids, VOCAB_IDX[w])
        else
            # prefix match
            best = -1
            best_len = 0
            for (vw, vi) in VOCAB_IDX
                ml = min(length(w), length(vw))
                plen = 0
                for k in 1:ml
                    if w[k] == vw[k]
                        plen += 1
                    else
                        break
                    end
                end
                if plen > best_len
                    best_len = plen
                    best = vi
                end
            end
            if best >= 0 && best_len >= 3
                push!(ids, best)
            end
        end
    end
    return ids  # 0-based
end


# ═══════════════════════════════════════════════════════════════
# TRAINING — next-word prediction, step s predicts word[s+1]
# ═══════════════════════════════════════════════════════════════

function train!(model::Penelope, data_path::String, steps::Int=5000, lr::Float64=3e-4)
    text = read(data_path, String)
    ids = tokenize_text(text)
    if length(ids) < STEPS + 2
        println("  corpus too small: $(length(ids)) words (need $(STEPS+2)+)")
        return
    end

    println("  corpus: $(length(text)) chars → $(length(ids)) vocab words")
    pc = param_count(model)
    println("  model: $(pc) params ($(round(pc*4/1e6, digits=1))MB f32)")
    println("  training: $steps steps, lr=$(@sprintf("%.1e", lr))")

    window = STEPS + 1  # 13 words
    best_loss = Inf

    for step in 1:steps
        start = rand(1:length(ids) - window + 1)
        win = ids[start:start + window - 1]

        total_loss = 0.0

        for s in 0:(STEPS - 1)
            context = win[1:s + 1]    # 0-based word IDs via 1-based Julia slice
            target = win[s + 2]       # 0-based target word ID

            logits = forward_step(model, context, s)
            probs = _softmax(logits)
            p = probs[target + 1]  # target is 0-based, probs is 1-based
            if p < 1e-10
                p = 1e-10
            end
            total_loss -= log(p)

            # gradient: d_logits = probs - one_hot(target)
            d_logits = copy(probs)
            d_logits[target + 1] -= 1.0

            # backprop
            sw = model.steps[s + 1]
            ctx = pool_context(model, context)

            # reconstruct forward
            query = matmul_mv(sw.wr, ctx, D, D)
            query_n = rmsnorm(query, sw.rms, D)
            gate = matmul_mv(sw.w_gate, query_n, M, D)
            up = matmul_mv(sw.w_up, query_n, M, D)
            swiglu = [_silu(gate[i]) * up[i] for i in 1:M]
            hidden = matmul_mv(sw.w_down, swiglu, D, M)
            out = vadd(query_n, hidden)

            # d_out from tied weights
            d_out = _zeros(D)
            for v in 1:V
                if abs(d_logits[v]) < 1e-8
                    continue
                end
                ev = get_embed(model, v - 1)  # v-1 for 0-based
                for j in 1:D
                    d_out[j] += d_logits[v] * ev[j]
                end
            end

            # update embedding (SGD on tied weights)
            for v in 1:V
                if abs(d_logits[v]) < 1e-8
                    continue
                end
                base = (v - 1) * D  # 0-based word index → flat offset (1-based array)
                for j in 1:D
                    model.embed[base + j] -= lr * d_logits[v] * out[j]
                end
            end

            # d_hidden
            d_hidden = copy(d_out)

            # backprop through w_down
            d_swiglu = matmul_mtv(sw.w_down, d_hidden, D, M)
            for i in 1:M
                for j in 1:D
                    sw.w_down[(i - 1) * D + j] -= lr * swiglu[i] * d_hidden[j]
                end
            end

            # backprop through SwiGLU
            for i in 1:M
                sg = _silu(gate[i])
                sig = gate[i] > -20.0 ? 1.0 / (1.0 + exp(-gate[i])) : 0.0
                silu_grad = gate[i] > -20.0 ? sig * (1.0 + gate[i] * (1.0 - sig)) : 0.0
                d_gate_i = d_swiglu[i] * up[i] * silu_grad
                d_up_i = d_swiglu[i] * sg

                for j in 1:D
                    sw.w_gate[(i - 1) * D + j] -= lr * d_gate_i * query_n[j]
                    sw.w_up[(i - 1) * D + j] -= lr * d_up_i * query_n[j]
                end
            end

            # d_query_n
            d_qn = copy(d_out)
            d_qn_gate_input = Vector{Float64}(undef, M)
            for i in 1:M
                g = gate[i]
                if g > -20.0
                    sig = 1.0 / (1.0 + exp(-g))
                    d_qn_gate_input[i] = d_swiglu[i] * up[i] * sig * (1.0 + g * (1.0 - sig))
                else
                    d_qn_gate_input[i] = 0.0
                end
            end
            d_qn_gate = matmul_mtv(sw.w_gate, d_qn_gate_input, M, D)
            d_qn_up = matmul_mtv(sw.w_up, [d_swiglu[i] * _silu(gate[i]) for i in 1:M], M, D)
            d_qn = vadd(d_qn, vadd(d_qn_gate, d_qn_up))

            # approx RMSNorm backward
            ss = 0.0
            for i in 1:D
                ss += query[i] * query[i]
            end
            ss = ss / D + 1e-5
            inv = 1.0 / sqrt(ss)
            d_query = [d_qn[i] * sw.rms[i] * inv for i in 1:D]

            # update Wr
            for i in 1:D
                if abs(d_query[i]) < 1e-8
                    continue
                end
                for j in 1:D
                    sw.wr[(i - 1) * D + j] -= lr * d_query[i] * ctx[j]
                end
            end
        end

        avg_loss = total_loss / STEPS
        if avg_loss < best_loss
            best_loss = avg_loss
        end

        if step % 50 == 0 || step == 1
            @printf("  step %5d/%d  loss=%.4f  best=%.4f\n", step, steps, avg_loss, best_loss)
        end
    end

    @printf("  training complete. best loss: %.4f\n", best_loss)
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
    depth = step_idx / STEPS
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

function overlay!(field::DarioField, logits::Vector{Float64}, context_ids::Vector{Int}, step_idx::Int)
    C = field.chambers
    alpha_mod = 1.0 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
    gamma_mod = 1.0 + 0.4*C["void"] + 0.2*C["complex"]

    # last 8 context ids
    ctx_tail = length(context_ids) > 8 ? context_ids[end-7:end] : context_ids

    for v in 0:(V - 1)
        h = 0.0
        for ci in ctx_tail
            h += get_cooc(field, ci, v)
        end
        if h > 0.0
            logits[v + 1] += alpha_mod * 0.3 * min(h, 1.0)
        end

        if field.prophecy_target !== nothing && v == field.prophecy_target
            logits[v + 1] += 0.5 * log(1.0 + field.prophecy_age)
        end

        cat = word_category(v)
        d_max = maximum(abs.(field.destiny)) + 0.01
        logits[v + 1] += gamma_mod * 0.25 * field.destiny[cat + 1] / d_max
    end

    return logits
end


function word_category(idx::Int)
    # idx is 0-based
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
# GENERATION — 12 steps, each picks one word
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

function run_chain(model::Penelope, field::DarioField, text::String)
    key = extract_key(text)
    seed = find_seed(key)

    deep_cats = [2, 5, 7]
    tcat = deep_cats[rand(1:length(deep_cats))]
    ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,V)]
    s_range, e_range = ranges[tcat + 1]  # tcat is category index, ranges is 1-based
    field.prophecy_target = rand(s_range:min(e_range - 1, V - 1))
    field.prophecy_age = 0

    println("\n  destined: $(VOCAB[field.prophecy_target + 1])")
    println("\n  $(VOCAB[seed + 1])")

    chain = [seed]
    forbidden = Set([seed])

    for step in 0:(STEPS - 1)
        update_chambers!(field, step)
        field.prophecy_age += 1

        # learned logits
        logits = forward_step(model, chain, step)

        # Dario field overlay
        logits = overlay!(field, logits, chain, step)

        # mask forbidden
        for f in forbidden
            logits[f + 1] = -1e9
        end

        # top-k sampling
        probs = _softmax(logits)
        indexed = sort(collect(enumerate(probs)), by=x -> -x[2])
        indexed = indexed[1:min(12, length(indexed))]
        total = sum(max(0.0, p) for (_, p) in indexed) + 0.001
        r = rand() * total
        pick = indexed[1][1] - 1  # convert back to 0-based
        for (idx_1based, p) in indexed
            r -= max(0.0, p)
            if r <= 0.0
                pick = idx_1based - 1  # 0-based
                break
            end
        end

        push!(chain, pick)
        push!(forbidden, pick)

        # update field
        if length(chain) >= 2
            update_cooc!(field, chain[end - 1], pick)
            cat = word_category(pick)
            field.destiny[cat + 1] = 0.3 + 0.7 * field.destiny[cat + 1]
        end

        if step > 7
            field.trauma = min(1.0, field.trauma + 0.1)
        end
        field.trauma *= 0.97

        marker = step == STEPS - 1 ? "  *" : "   "
        println("$(marker)$(VOCAB[pick + 1])")
    end

    fulfilled = field.prophecy_target in chain
    cats = length(Set(word_category(w) for w in chain))
    println("\n  drift $cats/8 · prophecy $(fulfilled ? "fulfilled" : "unfulfilled")")
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

    println()
    pc = param_count(model)
    # Format with commas
    pc_str = _format_int(pc)
    println("  penelope — 1984 words, $STEPS steps, Dario Equation")
    println("  $pc_str trainable params")
    println()

    if load_path !== nothing && isfile(load_path)
        load_model!(model, load_path)
    end

    if train_path !== nothing
        train!(model, train_path, train_steps, lr)
        if save_path !== nothing
            save_model(model, save_path)
        end
    end

    if text !== nothing
        run_chain(model, field, text)
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
            run_chain(model, field, String(line))
        end
    end

    if save_path !== nothing && train_path === nothing
        save_model(model, save_path)
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
