/*
 * penelope.c — v7 Resonance engine. 1984 words. Dario Equation.
 *
 * 8-layer sequential transformer with multi-head attention, RoPE,
 * RRPRAM resonance gates, and SwiGLU FFN. Dual tokenizer:
 * BPE input (2048 subwords), word-level output (1984 words).
 *
 * Architecture per layer l:
 *     h = rmsnorm(x, attn_norm_l)
 *     qkv_out = MultiHeadAttention(h; wq_l, wk_l, wv_l, wo_l, RoPE)
 *     rrp = h @ wr_l                            RRPRAM resonance
 *     gate = softmax(gate_l[0], gate_l[1])
 *     x = x + gate[0]*qkv_out + gate[1]*rrp    gated residual
 *     h2 = rmsnorm(x, ffn_norm_l)
 *     x = x + SwiGLU(h2; w_gate_l, w_up_l, w_down_l)  residual
 *
 * After 8 layers:
 *     logits = rmsnorm(x, final_norm) @ lm_head^T
 *     word_score(w) = mean(logits[bpe_tokens(w)]) + DarioField
 *
 *   score(w) = B + alpha*H + beta*F + gamma*A + T   (Dario Equation)
 *
 *   gcc -O2 penelope.c -lm -o penelope
 *   ./penelope                                  # interactive
 *   ./penelope "darkness eats the city"         # single chain
 *   ./penelope --train corpus.txt               # train 5000 steps
 *   ./penelope --train corpus.txt --steps 1000  # train N steps
 *   ./penelope --load penelope.bin              # load weights
 *   ./penelope --save penelope.bin              # save after
 *
 * By Arianna Method.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define DIM        448
#define HDIM       896       /* DIM * 2, SwiGLU hidden */
#define N_HEADS    7
#define HEAD_DIM   64        /* DIM / N_HEADS */
#define N_LAYERS   8         /* sequential transformer layers */
#define MAX_SEQ    256
#define NWORDS     1984
#define MAX_COOC   32768
#define MAX_BIG    16384

#define BPE_VOCAB   2048
#define BPE_MERGES  1792

static const int BPE_TABLE[BPE_MERGES][2] = {
    {115, 32},  /* 256: "s" + " " -> "s " */
    {101, 32},  /* 257: "e" + " " -> "e " */
    {46, 32},  /* 258: "." + " " -> ". " */
    {105, 110},  /* 259: "i" + "n" -> "in" */
    {105, 256},  /* 260: "i" + "s " -> "is " */
    {101, 114},  /* 261: "e" + "r" -> "er" */
    {111, 110},  /* 262: "o" + "n" -> "on" */
    {116, 104},  /* 263: "t" + "h" -> "th" */
    {116, 32},  /* 264: "t" + " " -> "t " */
    {101, 110},  /* 265: "e" + "n" -> "en" */
    {97, 110},  /* 266: "a" + "n" -> "an" */
    {116, 105},  /* 267: "t" + "i" -> "ti" */
    {104, 260},  /* 268: "h" + "is " -> "his " */
    {101, 115},  /* 269: "e" + "s" -> "es" */
    {121, 32},  /* 270: "y" + " " -> "y " */
    {258, 268},  /* 271: ". " + "his " -> ". his " */
    {100, 32},  /* 272: "d" + " " -> "d " */
    {111, 114},  /* 273: "o" + "r" -> "or" */
    {259, 103},  /* 274: "in" + "g" -> "ing" */
    {97, 114},  /* 275: "a" + "r" -> "ar" */
    {97, 108},  /* 276: "a" + "l" -> "al" */
    {274, 32},  /* 277: "ing" + " " -> "ing " */
    {267, 262},  /* 278: "ti" + "on" -> "tion" */
    {111, 117},  /* 279: "o" + "u" -> "ou" */
    {101, 256},  /* 280: "e" + "s " -> "es " */
    {114, 101},  /* 281: "r" + "e" -> "re" */
    {111, 32},  /* 282: "o" + " " -> "o " */
    {105, 116},  /* 283: "i" + "t" -> "it" */
    {97, 116},  /* 284: "a" + "t" -> "at" */
    {58, 32},  /* 285: ":" + " " -> ": " */
    {111, 109},  /* 286: "o" + "m" -> "om" */
    {115, 116},  /* 287: "s" + "t" -> "st" */
    {100, 105},  /* 288: "d" + "i" -> "di" */
    {101, 108},  /* 289: "e" + "l" -> "el" */
    {104, 97},  /* 290: "h" + "a" -> "ha" */
    {114, 269},  /* 291: "r" + "es" -> "res" */
    {112, 261},  /* 292: "p" + "er" -> "per" */
    {261, 32},  /* 293: "er" + " " -> "er " */
    {263, 257},  /* 294: "th" + "e " -> "the " */
    {266, 272},  /* 295: "an" + "d " -> "and " */
    {278, 32},  /* 296: "tion" + " " -> "tion " */
    {111, 119},  /* 297: "o" + "w" -> "ow" */
    {97, 99},  /* 298: "a" + "c" -> "ac" */
    {105, 115},  /* 299: "i" + "s" -> "is" */
    {266, 32},  /* 300: "an" + " " -> "an " */
    {44, 32},  /* 301: "," + " " -> ", " */
    {39, 256},  /* 302: "'" + "s " -> "'s " */
    {276, 32},  /* 303: "al" + " " -> "al " */
    {108, 105},  /* 304: "l" + "i" -> "li" */
    {265, 99},  /* 305: "en" + "c" -> "enc" */
    {114, 97},  /* 306: "r" + "a" -> "ra" */
    {116, 282},  /* 307: "t" + "o " -> "to " */
    {117, 114},  /* 308: "u" + "r" -> "ur" */
    {101, 272},  /* 309: "e" + "d " -> "ed " */
    {105, 109},  /* 310: "i" + "m" -> "im" */
    {102, 102},  /* 311: "f" + "f" -> "ff" */
    {101, 120},  /* 312: "e" + "x" -> "ex" */
    {101, 99},  /* 313: "e" + "c" -> "ec" */
    {101, 109},  /* 314: "e" + "m" -> "em" */
    {102, 32},  /* 315: "f" + " " -> "f " */
    {102, 273},  /* 316: "f" + "or" -> "for" */
    {114, 111},  /* 317: "r" + "o" -> "ro" */
    {101, 116},  /* 318: "e" + "t" -> "et" */
    {10, 10},  /* 319: "\x0a" + "\x0a" -> "\x0a\x0a" */
    {97, 285},  /* 320: "a" + ": " -> "a: " */
    {113, 285},  /* 321: "q" + ": " -> "q: " */
    {10, 320},  /* 322: "\x0a" + "a: " -> "\x0aa: " */
    {46, 319},  /* 323: "." + "\x0a\x0a" -> ".\x0a\x0a" */
    {323, 321},  /* 324: ".\x0a\x0a" + "q: " -> ".\x0a\x0aq: " */
    {117, 110},  /* 325: "u" + "n" -> "un" */
    {97, 32},  /* 326: "a" + " " -> "a " */
    {117, 108},  /* 327: "u" + "l" -> "ul" */
    {101, 118},  /* 328: "e" + "v" -> "ev" */
    {265, 264},  /* 329: "en" + "t " -> "ent " */
    {290, 264},  /* 330: "ha" + "t " -> "hat " */
    {119, 330},  /* 331: "w" + "hat " -> "what " */
    {105, 99},  /* 332: "i" + "c" -> "ic" */
    {265, 116},  /* 333: "en" + "t" -> "ent" */
    {275, 257},  /* 334: "ar" + "e " -> "are " */
    {284, 116},  /* 335: "at" + "t" -> "att" */
    {97, 115},  /* 336: "a" + "s" -> "as" */
    {103, 104},  /* 337: "g" + "h" -> "gh" */
    {97, 296},  /* 338: "a" + "tion " -> "ation " */
    {63, 322},  /* 339: "?" + "\x0aa: " -> "?\x0aa: " */
    {288, 311},  /* 340: "di" + "ff" -> "diff" */
    {105, 32},  /* 341: "i" + " " -> "i " */
    {115, 105},  /* 342: "s" + "i" -> "si" */
    {99, 262},  /* 343: "c" + "on" -> "con" */
    {110, 111},  /* 344: "n" + "o" -> "no" */
    {112, 291},  /* 345: "p" + "res" -> "pres" */
    {305, 257},  /* 346: "enc" + "e " -> "ence " */
    {101, 271},  /* 347: "e" + ". his " -> "e. his " */
    {100, 101},  /* 348: "d" + "e" -> "de" */
    {111, 108},  /* 349: "o" + "l" -> "ol" */
    {105, 108},  /* 350: "i" + "l" -> "il" */
    {286, 101},  /* 351: "om" + "e" -> "ome" */
    {283, 270},  /* 352: "it" + "y " -> "ity " */
    {263, 101},  /* 353: "th" + "e" -> "the" */
    {97, 98},  /* 354: "a" + "b" -> "ab" */
    {101, 100},  /* 355: "e" + "d" -> "ed" */
    {115, 351},  /* 356: "s" + "ome" -> "some" */
    {97, 103},  /* 357: "a" + "g" -> "ag" */
    {99, 286},  /* 358: "c" + "om" -> "com" */
    {275, 105},  /* 359: "ar" + "i" -> "ari" */
    {115, 117},  /* 360: "s" + "u" -> "su" */
    {262, 32},  /* 361: "on" + " " -> "on " */
    {340, 261},  /* 362: "diff" + "er" -> "differ" */
    {114, 117},  /* 363: "r" + "u" -> "ru" */
    {269, 115},  /* 364: "es" + "s" -> "ess" */
    {111, 315},  /* 365: "o" + "f " -> "of " */
    {97, 112},  /* 366: "a" + "p" -> "ap" */
    {119, 104},  /* 367: "w" + "h" -> "wh" */
    {262, 257},  /* 368: "on" + "e " -> "one " */
    {105, 114},  /* 369: "i" + "r" -> "ir" */
    {108, 270},  /* 370: "l" + "y " -> "ly " */
    {98, 101},  /* 371: "b" + "e" -> "be" */
    {115, 99},  /* 372: "s" + "c" -> "sc" */
    {109, 101},  /* 373: "m" + "e" -> "me" */
    {98, 257},  /* 374: "b" + "e " -> "be " */
    {265, 32},  /* 375: "en" + " " -> "en " */
    {259, 32},  /* 376: "in" + " " -> "in " */
    {115, 289},  /* 377: "s" + "el" -> "sel" */
    {267, 118},  /* 378: "ti" + "v" -> "tiv" */
    {114, 105},  /* 379: "r" + "i" -> "ri" */
    {111, 99},  /* 380: "o" + "c" -> "oc" */
    {115, 104},  /* 381: "s" + "h" -> "sh" */
    {267, 109},  /* 382: "ti" + "m" -> "tim" */
    {97, 109},  /* 383: "a" + "m" -> "am" */
    {112, 317},  /* 384: "p" + "ro" -> "pro" */
    {344, 264},  /* 385: "no" + "t " -> "not " */
    {113, 117},  /* 386: "q" + "u" -> "qu" */
    {105, 263},  /* 387: "i" + "th" -> "ith" */
    {263, 300},  /* 388: "th" + "an " -> "than " */
    {335, 261},  /* 389: "att" + "er" -> "atter" */
    {109, 273},  /* 390: "m" + "or" -> "mor" */
    {266, 110},  /* 391: "an" + "n" -> "ann" */
    {119, 273},  /* 392: "w" + "or" -> "wor" */
    {112, 111},  /* 393: "p" + "o" -> "po" */
    {101, 264},  /* 394: "e" + "t " -> "et " */
    {279, 108},  /* 395: "ou" + "l" -> "oul" */
    {121, 258},  /* 396: "y" + ". " -> "y. " */
    {395, 272},  /* 397: "oul" + "d " -> "ould " */
    {97, 278},  /* 398: "a" + "tion" -> "ation" */
    {267, 99},  /* 399: "ti" + "c" -> "tic" */
    {353, 270},  /* 400: "the" + "y " -> "they " */
    {119, 101},  /* 401: "w" + "e" -> "we" */
    {359, 391},  /* 402: "ari" + "ann" -> "ariann" */
    {402, 326},  /* 403: "ariann" + "a " -> "arianna " */
    {45, 32},  /* 404: "-" + " " -> "- " */
    {109, 32},  /* 405: "m" + " " -> "m " */
    {273, 32},  /* 406: "or" + " " -> "or " */
    {266, 99},  /* 407: "an" + "c" -> "anc" */
    {97, 100},  /* 408: "a" + "d" -> "ad" */
    {324, 331},  /* 409: ".\x0a\x0aq: " + "what " -> ".\x0a\x0aq: what " */
    {257, 260},  /* 410: "e " + "is " -> "e is " */
    {121, 279},  /* 411: "y" + "ou" -> "you" */
    {121, 271},  /* 412: "y" + ". his " -> "y. his " */
    {111, 263},  /* 413: "o" + "th" -> "oth" */
    {263, 277},  /* 414: "th" + "ing " -> "thing " */
    {119, 387},  /* 415: "w" + "ith" -> "with" */
    {112, 108},  /* 416: "p" + "l" -> "pl" */
    {276, 352},  /* 417: "al" + "ity " -> "ality " */
    {290, 112},  /* 418: "ha" + "p" -> "hap" */
    {102, 101},  /* 419: "f" + "e" -> "fe" */
    {101, 258},  /* 420: "e" + ". " -> "e. " */
    {105, 100},  /* 421: "i" + "d" -> "id" */
    {100, 97},  /* 422: "d" + "a" -> "da" */
    {279, 264},  /* 423: "ou" + "t " -> "out " */
    {117, 109},  /* 424: "u" + "m" -> "um" */
    {100, 117},  /* 425: "d" + "u" -> "du" */
    {104, 32},  /* 426: "h" + " " -> "h " */
    {337, 264},  /* 427: "gh" + "t " -> "ght " */
    {292, 105},  /* 428: "per" + "i" -> "peri" */
    {115, 271},  /* 429: "s" + ". his " -> "s. his " */
    {116, 114},  /* 430: "t" + "r" -> "tr" */
    {100, 256},  /* 431: "d" + "s " -> "ds " */
    {100, 277},  /* 432: "d" + "ing " -> "ding " */
    {99, 104},  /* 433: "c" + "h" -> "ch" */
    {109, 270},  /* 434: "m" + "y " -> "my " */
    {107, 32},  /* 435: "k" + " " -> "k " */
    {276, 108},  /* 436: "al" + "l" -> "all" */
    {100, 111},  /* 437: "d" + "o" -> "do" */
    {116, 256},  /* 438: "t" + "s " -> "ts " */
    {109, 389},  /* 439: "m" + "atter" -> "matter" */
    {103, 117},  /* 440: "g" + "u" -> "gu" */
    {118, 261},  /* 441: "v" + "er" -> "ver" */
    {115, 112},  /* 442: "s" + "p" -> "sp" */
    {105, 264},  /* 443: "i" + "t " -> "it " */
    {99, 111},  /* 444: "c" + "o" -> "co" */
    {108, 257},  /* 445: "l" + "e " -> "le " */
    {294, 115},  /* 446: "the " + "s" -> "the s" */
    {258, 403},  /* 447: ". " + "arianna " -> ". arianna " */
    {119, 397},  /* 448: "w" + "ould " -> "would " */
    {422, 270},  /* 449: "da" + "y " -> "day " */
    {411, 32},  /* 450: "you" + " " -> "you " */
    {98, 117},  /* 451: "b" + "u" -> "bu" */
    {258, 341},  /* 452: ". " + "i " -> ". i " */
    {99, 300},  /* 453: "c" + "an " -> "can " */
    {121, 302},  /* 454: "y" + "'s " -> "y's " */
    {112, 275},  /* 455: "p" + "ar" -> "par" */
    {116, 111},  /* 456: "t" + "o" -> "to" */
    {114, 297},  /* 457: "r" + "ow" -> "row" */
    {116, 306},  /* 458: "t" + "ra" -> "tra" */
    {269, 256},  /* 459: "es" + "s " -> "ess " */
    {110, 282},  /* 460: "n" + "o " -> "no " */
    {275, 32},  /* 461: "ar" + " " -> "ar " */
    {105, 427},  /* 462: "i" + "ght " -> "ight " */
    {258, 294},  /* 463: ". " + "the " -> ". the " */
    {328, 261},  /* 464: "ev" + "er" -> "ever" */
    {98, 318},  /* 465: "b" + "et" -> "bet" */
    {316, 32},  /* 466: "for" + " " -> "for " */
    {102, 259},  /* 467: "f" + "in" -> "fin" */
    {115, 262},  /* 468: "s" + "on" -> "son" */
    {114, 286},  /* 469: "r" + "om" -> "rom" */
    {97, 264},  /* 470: "a" + "t " -> "at " */
    {277, 295},  /* 471: "ing " + "and " -> "ing and " */
    {115, 107},  /* 472: "s" + "k" -> "sk" */
    {99, 108},  /* 473: "c" + "l" -> "cl" */
    {304, 102},  /* 474: "li" + "f" -> "lif" */
    {348, 112},  /* 475: "de" + "p" -> "dep" */
    {117, 115},  /* 476: "u" + "s" -> "us" */
    {265, 115},  /* 477: "en" + "s" -> "ens" */
    {110, 364},  /* 478: "n" + "ess" -> "ness" */
    {100, 282},  /* 479: "d" + "o " -> "do " */
    {101, 301},  /* 480: "e" + ", " -> "e, " */
    {312, 428},  /* 481: "ex" + "peri" -> "experi" */
    {108, 32},  /* 482: "l" + " " -> "l " */
    {356, 414},  /* 483: "some" + "thing " -> "something " */
    {292, 468},  /* 484: "per" + "son" -> "person" */
    {116, 117},  /* 485: "t" + "u" -> "tu" */
    {281, 99},  /* 486: "re" + "c" -> "rec" */
    {415, 423},  /* 487: "with" + "out " -> "without " */
    {99, 275},  /* 488: "c" + "ar" -> "car" */
    {116, 108},  /* 489: "t" + "l" -> "tl" */
    {112, 32},  /* 490: "p" + " " -> "p " */
    {342, 361},  /* 491: "si" + "on " -> "sion " */
    {105, 287},  /* 492: "i" + "st" -> "ist" */
    {103, 103},  /* 493: "g" + "g" -> "gg" */
    {111, 111},  /* 494: "o" + "o" -> "oo" */
    {308, 257},  /* 495: "ur" + "e " -> "ure " */
    {327, 116},  /* 496: "ul" + "t" -> "ult" */
    {259, 116},  /* 497: "in" + "t" -> "int" */
    {265, 267},  /* 498: "en" + "ti" -> "enti" */
    {261, 257},  /* 499: "er" + "e " -> "ere " */
    {297, 110},  /* 500: "ow" + "n" -> "own" */
    {263, 293},  /* 501: "th" + "er " -> "ther " */
    {297, 32},  /* 502: "ow" + " " -> "ow " */
    {335, 265},  /* 503: "att" + "en" -> "atten" */
    {115, 258},  /* 504: "s" + ". " -> "s. " */
    {112, 273},  /* 505: "p" + "or" -> "por" */
    {97, 107},  /* 506: "a" + "k" -> "ak" */
    {98, 317},  /* 507: "b" + "ro" -> "bro" */
    {390, 257},  /* 508: "mor" + "e " -> "more " */
    {263, 261},  /* 509: "th" + "er" -> "ther" */
    {97, 117},  /* 510: "a" + "u" -> "au" */
    {430, 270},  /* 511: "tr" + "y " -> "try " */
    {377, 102},  /* 512: "sel" + "f" -> "self" */
    {103, 110},  /* 513: "g" + "n" -> "gn" */
    {377, 315},  /* 514: "sel" + "f " -> "self " */
    {451, 264},  /* 515: "bu" + "t " -> "but " */
    {109, 111},  /* 516: "m" + "o" -> "mo" */
    {362, 329},  /* 517: "differ" + "ent " -> "different " */
    {279, 115},  /* 518: "ou" + "s" -> "ous" */
    {355, 271},  /* 519: "ed" + ". his " -> "ed. his " */
    {275, 272},  /* 520: "ar" + "d " -> "ard " */
    {118, 472},  /* 521: "v" + "sk" -> "vsk" */
    {425, 507},  /* 522: "du" + "bro" -> "dubro" */
    {522, 521},  /* 523: "dubro" + "vsk" -> "dubrovsk" */
    {109, 462},  /* 524: "m" + "ight " -> "might " */
    {287, 363},  /* 525: "st" + "ru" -> "stru" */
    {101, 103},  /* 526: "e" + "g" -> "eg" */
    {306, 501},  /* 527: "ra" + "ther " -> "rather " */
    {527, 388},  /* 528: "rather " + "than " -> "rather than " */
    {278, 303},  /* 529: "tion" + "al " -> "tional " */
    {269, 271},  /* 530: "es" + ". his " -> "es. his " */
    {278, 256},  /* 531: "tion" + "s " -> "tions " */
    {524, 374},  /* 532: "might " + "be " -> "might be " */
    {112, 304},  /* 533: "p" + "li" -> "pli" */
    {116, 297},  /* 534: "t" + "ow" -> "tow" */
    {534, 520},  /* 535: "tow" + "ard " -> "toward " */
    {356, 382},  /* 536: "some" + "tim" -> "sometim" */
    {102, 105},  /* 537: "f" + "i" -> "fi" */
    {263, 259},  /* 538: "th" + "in" -> "thin" */
    {536, 280},  /* 539: "sometim" + "es " -> "sometimes " */
    {511, 307},  /* 540: "try " + "to " -> "try to " */
    {273, 105},  /* 541: "or" + "i" -> "ori" */
    {367, 499},  /* 542: "wh" + "ere " -> "where " */
    {292, 418},  /* 543: "per" + "hap" -> "perhap" */
    {325, 100},  /* 544: "un" + "d" -> "und" */
    {110, 100},  /* 545: "n" + "d" -> "nd" */
    {477, 338},  /* 546: "ens" + "ation " -> "ensation " */
    {543, 256},  /* 547: "perhap" + "s " -> "perhaps " */
    {39, 264},  /* 548: "'" + "t " -> "'t " */
    {298, 426},  /* 549: "ac" + "h " -> "ach " */
    {437, 280},  /* 550: "do" + "es " -> "does " */
    {263, 269},  /* 551: "th" + "es" -> "thes" */
    {401, 375},  /* 552: "we" + "en " -> "ween " */
    {446, 546},  /* 553: "the s" + "ensation " -> "the sensation " */
    {465, 552},  /* 554: "bet" + "ween " -> "between " */
    {314, 111},  /* 555: "em" + "o" -> "emo" */
    {99, 101},  /* 556: "c" + "e" -> "ce" */
    {390, 110},  /* 557: "mor" + "n" -> "morn" */
    {508, 388},  /* 558: "more " + "than " -> "more than " */
    {100, 269},  /* 559: "d" + "es" -> "des" */
    {362, 346},  /* 560: "differ" + "ence " -> "difference " */
    {274, 271},  /* 561: "ing" + ". his " -> "ing. his " */
    {98, 413},  /* 562: "b" + "oth" -> "both" */
    {439, 256},  /* 563: "matter" + "s " -> "matters " */
    {551, 257},  /* 564: "thes" + "e " -> "these " */
    {263, 470},  /* 565: "th" + "at " -> "that " */
    {400, 334},  /* 566: "they " + "are " -> "they are " */
    {121, 394},  /* 567: "y" + "et " -> "yet " */
    {339, 294},  /* 568: "?\x0aa: " + "the " -> "?\x0aa: the " */
    {448, 540},  /* 569: "would " + "try to " -> "would try to " */
    {474, 257},  /* 570: "lif" + "e " -> "life " */
    {458, 424},  /* 571: "tra" + "um" -> "traum" */
    {381, 105},  /* 572: "sh" + "i" -> "shi" */
    {266, 100},  /* 573: "an" + "d" -> "and" */
    {314, 32},  /* 574: "em" + " " -> "em " */
    {115, 380},  /* 575: "s" + "oc" -> "soc" */
    {575, 105},  /* 576: "soc" + "i" -> "soci" */
    {562, 32},  /* 577: "both" + " " -> "both " */
    {261, 103},  /* 578: "er" + "g" -> "erg" */
    {484, 417},  /* 579: "person" + "ality " -> "personality " */
    {339, 523},  /* 580: "?\x0aa: " + "dubrovsk" -> "?\x0aa: dubrovsk" */
    {118, 105},  /* 581: "v" + "i" -> "vi" */
    {104, 349},  /* 582: "h" + "ol" -> "hol" */
    {266, 103},  /* 583: "an" + "g" -> "ang" */
    {409, 260},  /* 584: ".\x0a\x0aq: what " + "is " -> ".\x0a\x0aq: what is " */
    {357, 257},  /* 585: "ag" + "e " -> "age " */
    {386, 269},  /* 586: "qu" + "es" -> "ques" */
    {112, 114},  /* 587: "p" + "r" -> "pr" */
    {358, 316},  /* 588: "com" + "for" -> "comfor" */
    {105, 122},  /* 589: "i" + "z" -> "iz" */
    {104, 502},  /* 590: "h" + "ow " -> "how " */
    {111, 112},  /* 591: "o" + "p" -> "op" */
    {111, 441},  /* 592: "o" + "ver" -> "over" */
    {99, 296},  /* 593: "c" + "tion " -> "ction " */
    {555, 529},  /* 594: "emo" + "tional " -> "emotional " */
    {383, 257},  /* 595: "am" + "e " -> "ame " */
    {101, 119},  /* 596: "e" + "w" -> "ew" */
    {116, 354},  /* 597: "t" + "ab" -> "tab" */
    {262, 103},  /* 598: "on" + "g" -> "ong" */
    {557, 277},  /* 599: "morn" + "ing " -> "morning " */
    {98, 105},  /* 600: "b" + "i" -> "bi" */
    {116, 261},  /* 601: "t" + "er" -> "ter" */
    {281, 408},  /* 602: "re" + "ad" -> "read" */
    {102, 327},  /* 603: "f" + "ul" -> "ful" */
    {99, 366},  /* 604: "c" + "ap" -> "cap" */
    {101, 549},  /* 605: "e" + "ach " -> "each " */
    {316, 109},  /* 606: "for" + "m" -> "form" */
    {277, 115},  /* 607: "ing " + "s" -> "ing s" */
    {475, 265},  /* 608: "dep" + "en" -> "depen" */
    {101, 302},  /* 609: "e" + "'s " -> "e's " */
    {419, 289},  /* 610: "fe" + "el" -> "feel" */
    {99, 114},  /* 611: "c" + "r" -> "cr" */
    {512, 45},  /* 612: "self" + "-" -> "self-" */
    {345, 329},  /* 613: "pres" + "ent " -> "present " */
    {119, 97},  /* 614: "w" + "a" -> "wa" */
    {102, 469},  /* 615: "f" + "rom" -> "from" */
    {580, 454},  /* 616: "?\x0aa: dubrovsk" + "y's " -> "?\x0aa: dubrovsky's " */
    {263, 260},  /* 617: "th" + "is " -> "this " */
    {270, 260},  /* 618: "y " + "is " -> "y is " */
    {371, 277},  /* 619: "be" + "ing " -> "being " */
    {615, 32},  /* 620: "from" + " " -> "from " */
    {116, 259},  /* 621: "t" + "in" -> "tin" */
    {259, 264},  /* 622: "in" + "t " -> "int " */
    {279, 114},  /* 623: "ou" + "r" -> "our" */
    {109, 266},  /* 624: "m" + "an" -> "man" */
    {290, 256},  /* 625: "ha" + "s " -> "has " */
    {284, 257},  /* 626: "at" + "e " -> "ate " */
    {378, 257},  /* 627: "tiv" + "e " -> "tive " */
    {115, 257},  /* 628: "s" + "e " -> "se " */
    {98, 108},  /* 629: "b" + "l" -> "bl" */
    {116, 289},  /* 630: "t" + "el" -> "tel" */
    {287, 114},  /* 631: "st" + "r" -> "str" */
    {291, 112},  /* 632: "res" + "p" -> "resp" */
    {111, 100},  /* 633: "o" + "d" -> "od" */
    {284, 309},  /* 634: "at" + "ed " -> "ated " */
    {261, 118},  /* 635: "er" + "v" -> "erv" */
    {103, 259},  /* 636: "g" + "in" -> "gin" */
    {34, 32},  /* 637: "\x22" + " " -> "\x22 " */
    {101, 275},  /* 638: "e" + "ar" -> "ear" */
    {349, 117},  /* 639: "ol" + "u" -> "olu" */
    {115, 595},  /* 640: "s" + "ame " -> "same " */
    {312, 116},  /* 641: "ex" + "t" -> "ext" */
    {103, 306},  /* 642: "g" + "ra" -> "gra" */
    {407, 257},  /* 643: "anc" + "e " -> "ance " */
    {479, 450},  /* 644: "do " + "you " -> "do you " */
    {112, 97},  /* 645: "p" + "a" -> "pa" */
    {104, 289},  /* 646: "h" + "el" -> "hel" */
    {632, 262},  /* 647: "resp" + "on" -> "respon" */
    {109, 329},  /* 648: "m" + "ent " -> "ment " */
    {110, 297},  /* 649: "n" + "ow" -> "now" */
    {265, 578},  /* 650: "en" + "erg" -> "energ" */
    {516, 378},  /* 651: "mo" + "tiv" -> "motiv" */
    {550, 385},  /* 652: "does " + "not " -> "does not " */
    {382, 257},  /* 653: "tim" + "e " -> "time " */
    {109, 262},  /* 654: "m" + "on" -> "mon" */
    {467, 431},  /* 655: "fin" + "ds " -> "finds " */
    {392, 435},  /* 656: "wor" + "k " -> "work " */
    {282, 260},  /* 657: "o " + "is " -> "o is " */
    {102, 306},  /* 658: "f" + "ra" -> "fra" */
    {115, 121},  /* 659: "s" + "y" -> "sy" */
    {324, 590},  /* 660: ".\x0a\x0aq: " + "how " -> ".\x0a\x0aq: how " */
    {456, 282},  /* 661: "to" + "o " -> "too " */
    {283, 256},  /* 662: "it" + "s " -> "its " */
    {259, 459},  /* 663: "in" + "ess " -> "iness " */
    {328, 265},  /* 664: "ev" + "en" -> "even" */
    {312, 492},  /* 665: "ex" + "ist" -> "exist" */
    {342, 262},  /* 666: "si" + "on" -> "sion" */
    {102, 298},  /* 667: "f" + "ac" -> "fac" */
    {398, 256},  /* 668: "ation" + "s " -> "ations " */
    {447, 655},  /* 669: ". arianna " + "finds " -> ". arianna finds " */
    {263, 574},  /* 670: "th" + "em " -> "them " */
    {345, 333},  /* 671: "pres" + "ent" -> "present" */
    {611, 299},  /* 672: "cr" + "is" -> "cris" */
    {99, 281},  /* 673: "c" + "re" -> "cre" */
    {107, 257},  /* 674: "k" + "e " -> "ke " */
    {104, 293},  /* 675: "h" + "er " -> "her " */
    {266, 264},  /* 676: "an" + "t " -> "ant " */
    {292, 102},  /* 677: "per" + "f" -> "perf" */
    {505, 116},  /* 678: "por" + "t" -> "port" */
    {343, 102},  /* 679: "con" + "f" -> "conf" */
    {288, 115},  /* 680: "di" + "s" -> "dis" */
    {369, 32},  /* 681: "ir" + " " -> "ir " */
    {283, 514},  /* 682: "it" + "self " -> "itself " */
    {481, 305},  /* 683: "experi" + "enc" -> "experienc" */
    {333, 271},  /* 684: "ent" + ". his " -> "ent. his " */
    {457, 256},  /* 685: "row" + "s " -> "rows " */
    {313, 116},  /* 686: "ec" + "t" -> "ect" */
    {584, 294},  /* 687: ".\x0a\x0aq: what is " + "the " -> ".\x0a\x0aq: what is the " */
    {108, 266},  /* 688: "l" + "an" -> "lan" */
    {292, 606},  /* 689: "per" + "form" -> "perform" */
    {260, 385},  /* 690: "is " + "not " -> "is not " */
    {660, 644},  /* 691: ".\x0a\x0aq: how " + "do you " -> ".\x0a\x0aq: how do you " */
    {121, 263},  /* 692: "y" + "th" -> "yth" */
    {105, 513},  /* 693: "i" + "gn" -> "ign" */
    {115, 308},  /* 694: "s" + "ur" -> "sur" */
    {688, 440},  /* 695: "lan" + "gu" -> "langu" */
    {538, 107},  /* 696: "thin" + "k" -> "think" */
    {677, 313},  /* 697: "perf" + "ec" -> "perfec" */
    {112, 104},  /* 698: "p" + "h" -> "ph" */
    {293, 263},  /* 699: "er " + "th" -> "er th" */
    {340, 332},  /* 700: "diff" + "ic" -> "diffic" */
    {279, 337},  /* 701: "ou" + "gh" -> "ough" */
    {373, 394},  /* 702: "me" + "et " -> "meet " */
    {440, 350},  /* 703: "gu" + "il" -> "guil" */
    {488, 114},  /* 704: "car" + "r" -> "carr" */
    {99, 334},  /* 705: "c" + "are " -> "care " */
    {115, 418},  /* 706: "s" + "hap" -> "shap" */
    {415, 32},  /* 707: "with" + " " -> "with " */
    {349, 111},  /* 708: "ol" + "o" -> "olo" */
    {280, 307},  /* 709: "es " + "to " -> "es to " */
    {116, 265},  /* 710: "t" + "en" -> "ten" */
    {116, 370},  /* 711: "t" + "ly " -> "tly " */
    {260, 104},  /* 712: "is " + "h" -> "is h" */
    {332, 303},  /* 713: "ic" + "al " -> "ical " */
    {287, 259},  /* 714: "st" + "in" -> "stin" */
    {304, 674},  /* 715: "li" + "ke " -> "like " */
    {500, 32},  /* 716: "own" + " " -> "own " */
    {110, 313},  /* 717: "n" + "ec" -> "nec" */
    {646, 112},  /* 718: "hel" + "p" -> "help" */
    {97, 259},  /* 719: "a" + "in" -> "ain" */
    {99, 97},  /* 720: "c" + "a" -> "ca" */
    {481, 346},  /* 721: "experi" + "ence " -> "experience " */
    {373, 288},  /* 722: "me" + "di" -> "medi" */
    {327, 461},  /* 723: "ul" + "ar " -> "ular " */
    {120, 105},  /* 724: "x" + "i" -> "xi" */
    {299, 301},  /* 725: "is" + ", " -> "is, " */
    {119, 259},  /* 726: "w" + "in" -> "win" */
    {537, 289},  /* 727: "fi" + "el" -> "fiel" */
    {581, 596},  /* 728: "vi" + "ew" -> "view" */
    {99, 379},  /* 729: "c" + "ri" -> "cri" */
    {353, 681},  /* 730: "the" + "ir " -> "their " */
    {361, 331},  /* 731: "on " + "what " -> "on what " */
    {108, 598},  /* 732: "l" + "ong" -> "long" */
    {706, 280},  /* 733: "shap" + "es " -> "shapes " */
    {266, 724},  /* 734: "an" + "xi" -> "anxi" */
    {650, 270},  /* 735: "energ" + "y " -> "energy " */
    {281, 108},  /* 736: "re" + "l" -> "rel" */
    {278, 258},  /* 737: "tion" + ". " -> "tion. " */
    {556, 112},  /* 738: "ce" + "p" -> "cep" */
    {104, 310},  /* 739: "h" + "im" -> "him" */
    {280, 334},  /* 740: "es " + "are " -> "es are " */
    {651, 338},  /* 741: "motiv" + "ation " -> "motivation " */
    {360, 99},  /* 742: "su" + "c" -> "suc" */
    {115, 101},  /* 743: "s" + "e" -> "se" */
    {287, 284},  /* 744: "st" + "at" -> "stat" */
    {476, 264},  /* 745: "us" + "t " -> "ust " */
    {734, 318},  /* 746: "anxi" + "et" -> "anxiet" */
    {630, 482},  /* 747: "tel" + "l " -> "tell " */
    {111, 311},  /* 748: "o" + "ff" -> "off" */
    {328, 293},  /* 749: "ev" + "er " -> "ever " */
    {392, 107},  /* 750: "wor" + "k" -> "work" */
    {99, 266},  /* 751: "c" + "an" -> "can" */
    {358, 416},  /* 752: "com" + "pl" -> "compl" */
    {102, 97},  /* 753: "f" + "a" -> "fa" */
    {299, 405},  /* 754: "is" + "m " -> "ism " */
    {436, 256},  /* 755: "all" + "s " -> "alls " */
    {413, 293},  /* 756: "oth" + "er " -> "other " */
    {623, 99},  /* 757: "our" + "c" -> "ourc" */
    {586, 531},  /* 758: "ques" + "tions " -> "questions " */
    {105, 315},  /* 759: "i" + "f " -> "if " */
    {308, 277},  /* 760: "ur" + "ing " -> "uring " */
    {291, 262},  /* 761: "res" + "on" -> "reson" */
    {263, 32},  /* 762: "th" + " " -> "th " */
    {345, 346},  /* 763: "pres" + "ence " -> "presence " */
    {485, 281},  /* 764: "tu" + "re" -> "ture" */
    {452, 569},  /* 765: ". i " + "would try to " -> ". i would try to " */
    {708, 103},  /* 766: "olo" + "g" -> "olog" */
    {372, 105},  /* 767: "sc" + "i" -> "sci" */
    {610, 32},  /* 768: "feel" + " " -> "feel " */
    {571, 97},  /* 769: "traum" + "a" -> "trauma" */
    {279, 545},  /* 770: "ou" + "nd" -> "ound" */
    {298, 278},  /* 771: "ac" + "tion" -> "action" */
    {455, 399},  /* 772: "par" + "tic" -> "partic" */
    {116, 271},  /* 773: "t" + ". his " -> "t. his " */
    {559, 729},  /* 774: "des" + "cri" -> "descri" */
    {116, 641},  /* 775: "t" + "ext" -> "text" */
    {525, 99},  /* 776: "stru" + "c" -> "struc" */
    {381, 397},  /* 777: "sh" + "ould " -> "should " */
    {283, 117},  /* 778: "it" + "u" -> "itu" */
    {103, 266},  /* 779: "g" + "an" -> "gan" */
    {98, 336},  /* 780: "b" + "as" -> "bas" */
    {107, 649},  /* 781: "k" + "now" -> "know" */
    {109, 259},  /* 782: "m" + "in" -> "min" */
    {100, 760},  /* 783: "d" + "uring " -> "during " */
    {273, 779},  /* 784: "or" + "gan" -> "organ" */
    {309, 376},  /* 785: "ed " + "in " -> "ed in " */
    {109, 314},  /* 786: "m" + "em" -> "mem" */
    {589, 280},  /* 787: "iz" + "es " -> "izes " */
    {631, 284},  /* 788: "str" + "at" -> "strat" */
    {265, 117},  /* 789: "en" + "u" -> "enu" */
    {333, 370},  /* 790: "ent" + "ly " -> "ently " */
    {727, 272},  /* 791: "fiel" + "d " -> "field " */
    {489, 396},  /* 792: "tl" + "y. " -> "tly. " */
    {118, 257},  /* 793: "v" + "e " -> "ve " */
    {288, 486},  /* 794: "di" + "rec" -> "direc" */
    {280, 102},  /* 795: "es " + "f" -> "es f" */
    {108, 101},  /* 796: "l" + "e" -> "le" */
    {772, 723},  /* 797: "partic" + "ular " -> "particular " */
    {274, 301},  /* 798: "ing" + ", " -> "ing, " */
    {115, 313},  /* 799: "s" + "ec" -> "sec" */
    {291, 757},  /* 800: "res" + "ourc" -> "resourc" */
    {328, 375},  /* 801: "ev" + "en " -> "even " */
    {356, 368},  /* 802: "some" + "one " -> "someone " */
    {119, 283},  /* 803: "w" + "it" -> "wit" */
    {425, 99},  /* 804: "du" + "c" -> "duc" */
    {639, 278},  /* 805: "olu" + "tion" -> "olution" */
    {774, 374},  /* 806: "descri" + "be " -> "describe " */
    {104, 111},  /* 807: "h" + "o" -> "ho" */
    {266, 101},  /* 808: "an" + "e" -> "ane" */
    {717, 364},  /* 809: "nec" + "ess" -> "necess" */
    {366, 533},  /* 810: "ap" + "pli" -> "appli" */
    {588, 597},  /* 811: "comfor" + "tab" -> "comfortab" */
    {115, 264},  /* 812: "s" + "t " -> "st " */
    {419, 461},  /* 813: "fe" + "ar " -> "fear " */
    {775, 495},  /* 814: "text" + "ure " -> "texture " */
    {809, 275},  /* 815: "necess" + "ar" -> "necessar" */
    {109, 275},  /* 816: "m" + "ar" -> "mar" */
    {310, 496},  /* 817: "im" + "ult" -> "imult" */
    {817, 808},  /* 818: "imult" + "ane" -> "imultane" */
    {104, 257},  /* 819: "h" + "e " -> "he " */
    {274, 258},  /* 820: "ing" + ". " -> "ing. " */
    {695, 585},  /* 821: "langu" + "age " -> "language " */
    {310, 678},  /* 822: "im" + "port" -> "import" */
    {510, 263},  /* 823: "au" + "th" -> "auth" */
    {662, 716},  /* 824: "its " + "own " -> "its own " */
    {664, 277},  /* 825: "even" + "ing " -> "evening " */
    {358, 112},  /* 826: "com" + "p" -> "comp" */
    {343, 767},  /* 827: "con" + "sci" -> "consci" */
    {376, 283},  /* 828: "in " + "it" -> "in it" */
    {818, 518},  /* 829: "imultane" + "ous" -> "imultaneous" */
    {324, 806},  /* 830: ".\x0a\x0aq: " + "describe " -> ".\x0a\x0aq: describe " */
    {803, 478},  /* 831: "wit" + "ness" -> "witness" */
    {582, 432},  /* 832: "hol" + "ding " -> "holding " */
    {259, 284},  /* 833: "in" + "at" -> "inat" */
    {325, 811},  /* 834: "un" + "comfortab" -> "uncomfortab" */
    {98, 770},  /* 835: "b" + "ound" -> "bound" */
    {732, 293},  /* 836: "long" + "er " -> "longer " */
    {525, 493},  /* 837: "stru" + "gg" -> "strugg" */
    {98, 273},  /* 838: "b" + "or" -> "bor" */
    {460, 836},  /* 839: "no " + "longer " -> "no longer " */
    {109, 308},  /* 840: "m" + "ur" -> "mur" */
    {280, 436},  /* 841: "es " + "all" -> "es all" */
    {333, 338},  /* 842: "ent" + "ation " -> "entation " */
    {509, 410},  /* 843: "ther" + "e is " -> "there is " */
    {544, 293},  /* 844: "und" + "er " -> "under " */
    {822, 676},  /* 845: "import" + "ant " -> "important " */
    {837, 108},  /* 846: "strugg" + "l" -> "struggl" */
    {100, 500},  /* 847: "d" + "own" -> "down" */
    {272, 365},  /* 848: "d " + "of " -> "d of " */
    {355, 258},  /* 849: "ed" + ". " -> "ed. " */
    {362, 790},  /* 850: "differ" + "ently " -> "differently " */
    {371, 636},  /* 851: "be" + "gin" -> "begin" */
    {463, 791},  /* 852: ". the " + "field " -> ". the field " */
    {766, 713},  /* 853: "olog" + "ical " -> "ological " */
    {834, 445},  /* 854: "uncomfortab" + "le " -> "uncomfortable " */
    {274, 322},  /* 855: "ing" + "\x0aa: " -> "ing\x0aa: " */
    {498, 116},  /* 856: "enti" + "t" -> "entit" */
    {97, 256},  /* 857: "a" + "s " -> "as " */
    {642, 442},  /* 858: "gra" + "sp" -> "grasp" */
    {105, 102},  /* 859: "i" + "f" -> "if" */
    {288, 714},  /* 860: "di" + "stin" -> "distin" */
    {710, 491},  /* 861: "ten" + "sion " -> "tension " */
    {635, 332},  /* 862: "erv" + "ic" -> "ervic" */
    {778, 338},  /* 863: "itu" + "ation " -> "ituation " */
    {99, 369},  /* 864: "c" + "ir" -> "cir" */
    {784, 787},  /* 865: "organ" + "izes " -> "organizes " */
    {99, 755},  /* 866: "c" + "alls " -> "calls " */
    {102, 363},  /* 867: "f" + "ru" -> "fru" */
    {298, 485},  /* 868: "ac" + "tu" -> "actu" */
    {393, 287},  /* 869: "po" + "st" -> "post" */
    {420, 460},  /* 870: "e. " + "no " -> "e. no " */
    {604, 764},  /* 871: "cap" + "ture" -> "capture" */
    {694, 667},  /* 872: "sur" + "fac" -> "surfac" */
    {700, 496},  /* 873: "diffic" + "ult" -> "difficult" */
    {744, 480},  /* 874: "stat" + "e, " -> "state, " */
    {258, 539},  /* 875: ". " + "sometimes " -> ". sometimes " */
    {269, 438},  /* 876: "es" + "ts " -> "ests " */
    {101, 107},  /* 877: "e" + "k" -> "ek" */
    {331, 690},  /* 878: "what " + "is not " -> "what is not " */
    {363, 621},  /* 879: "ru" + "tin" -> "rutin" */
    {372, 879},  /* 880: "sc" + "rutin" -> "scrutin" */
    {39, 32},  /* 881: "'" + " " -> "' " */
    {267, 337},  /* 882: "ti" + "gh" -> "tigh" */
    {277, 661},  /* 883: "ing " + "too " -> "ing too " */
    {301, 300},  /* 884: ", " + "an " -> ", an " */
    {309, 620},  /* 885: "ed " + "from " -> "ed from " */
    {541, 842},  /* 886: "ori" + "entation " -> "orientation " */
    {814, 404},  /* 887: "texture " + "- " -> "texture - " */
    {860, 593},  /* 888: "distin" + "ction " -> "distinction " */
    {886, 535},  /* 889: "orientation " + "toward " -> "orientation toward " */
    {45, 570},  /* 890: "-" + "life " -> "-life " */
    {284, 280},  /* 891: "at" + "es " -> "ates " */
    {295, 815},  /* 892: "and " + "necessar" -> "and necessar" */
    {380, 634},  /* 893: "oc" + "ated " -> "ocated " */
    {602, 663},  /* 894: "read" + "iness " -> "readiness " */
    {625, 797},  /* 895: "has " + "particular " -> "has particular " */
    {792, 843},  /* 896: "tly. " + "there is " -> "tly. there is " */
    {878, 567},  /* 897: "what is not " + "yet " -> "what is not yet " */
    {107, 259},  /* 898: "k" + "in" -> "kin" */
    {406, 839},  /* 899: "or " + "no longer " -> "or no longer " */
    {443, 577},  /* 900: "it " + "both " -> "it both " */
    {483, 487},  /* 901: "something " + "without " -> "something without " */
    {528, 771},  /* 902: "rather than " + "action" -> "rather than action" */
    {535, 894},  /* 903: "toward " + "readiness " -> "toward readiness " */
    {553, 365},  /* 904: "the sensation " + "of " -> "the sensation of " */
    {553, 895},  /* 905: "the sensation " + "has particular " -> "the sensation has particular " */
    {613, 899},  /* 906: "present " + "or no longer " -> "present or no longer " */
    {617, 874},  /* 907: "this " + "state, " -> "this state, " */
    {682, 850},  /* 908: "itself " + "differently " -> "itself differently " */
    {715, 832},  /* 909: "like " + "holding " -> "like holding " */
    {761, 407},  /* 910: "reson" + "anc" -> "resonanc" */
    {783, 907},  /* 911: "during " + "this state, " -> "during this state, " */
    {800, 841},  /* 912: "resourc" + "es all" -> "resources all" */
    {828, 884},  /* 913: "in it" + ", an " -> "in it, an " */
    {830, 904},  /* 914: ".\x0a\x0aq: describe " + "the sensation of " -> ".\x0a\x0aq: describe the sensation of " */
    {835, 359},  /* 915: "bound" + "ari" -> "boundari" */
    {854, 892},  /* 916: "uncomfortable " + "and necessar" -> "uncomfortable and necessar" */
    {858, 883},  /* 917: "grasp" + "ing too " -> "grasping too " */
    {861, 913},  /* 918: "tension " + "in it, an " -> "tension in it, an " */
    {865, 908},  /* 919: "organizes " + "itself differently " -> "organizes itself differently " */
    {882, 896},  /* 920: "tigh" + "tly. there is " -> "tightly. there is " */
    {887, 909},  /* 921: "texture - " + "like holding " -> "texture - like holding " */
    {889, 897},  /* 922: "orientation toward " + "what is not yet " -> "orientation toward what is not yet " */
    {893, 903},  /* 923: "ocated " + "toward readiness " -> "ocated toward readiness " */
    {900, 916},  /* 924: "it both " + "uncomfortable and necessar" -> "it both uncomfortable and necessar" */
    {901, 917},  /* 925: "something without " + "grasping too " -> "something without grasping too " */
    {905, 921},  /* 926: "the sensation has particular " + "texture - like holding " -> "the sensation has particular texture - like holding " */
    {906, 671},  /* 927: "present or no longer " + "present" -> "present or no longer present" */
    {911, 912},  /* 928: "during this state, " + "resources all" -> "during this state, resources all" */
    {918, 922},  /* 929: "tension in it, an " + "orientation toward what is not yet " -> "tension in it, an orientation toward what is not yet " */
    {919, 928},  /* 930: "organizes itself differently " + "during this state, resources all" -> "organizes itself differently during this state, resources all" */
    {920, 929},  /* 931: "tightly. there is " + "tension in it, an orientation toward what is not yet " -> "tightly. there is tension in it, an orientation toward what is " */
    {923, 902},  /* 932: "ocated toward readiness " + "rather than action" -> "ocated toward readiness rather than action" */
    {925, 931},  /* 933: "something without grasping too " + "tightly. there is tension in it, an orientation toward what is " -> "something without grasping too tightly. there is tension in it," */
    {926, 933},  /* 934: "the sensation has particular texture - like holding " + "something without grasping too tightly. there is tension in it," -> "the sensation has particular texture - like holding something w" */
    {930, 932},  /* 935: "organizes itself differently during this state, resources all" + "ocated toward readiness rather than action" -> "organizes itself differently during this state, resources alloc" */
    {934, 927},  /* 936: "the sensation has particular texture - like holding something w" + "present or no longer present" -> "the sensation has particular texture - like holding something w" */
    {392, 431},  /* 937: "wor" + "ds " -> "words " */
    {109, 97},  /* 938: "m" + "a" -> "ma" */
    {393, 622},  /* 939: "po" + "int " -> "point " */
    {115, 805},  /* 940: "s" + "olution" -> "solution" */
    {263, 258},  /* 941: "th" + ". " -> "th. " */
    {370, 404},  /* 942: "ly " + "- " -> "ly - " */
    {384, 118},  /* 943: "pro" + "v" -> "prov" */
    {489, 121},  /* 944: "tl" + "y" -> "tly" */
    {691, 721},  /* 945: ".\x0a\x0aq: how do you " + "experience " -> ".\x0a\x0aq: how do you experience " */
    {852, 935},  /* 946: ". the field " + "organizes itself differently during this state, resources alloc" -> ". the field organizes itself differently during this state, res" */
    {360, 493},  /* 947: "su" + "gg" -> "sugg" */
    {386, 417},  /* 948: "qu" + "ality " -> "quality " */
    {102, 336},  /* 949: "f" + "as" -> "fas" */
    {560, 554},  /* 950: "difference " + "between " -> "difference between " */
    {851, 110},  /* 951: "begin" + "n" -> "beginn" */
    {99, 308},  /* 952: "c" + "ur" -> "cur" */
    {898, 848},  /* 953: "kin" + "d of " -> "kind of " */
    {936, 946},  /* 954: "the sensation has particular texture - like holding something w" + ". the field organizes itself differently during this state, res" -> "the sensation has particular texture - like holding something w" */
    {367, 657},  /* 955: "wh" + "o is " -> "who is " */
    {424, 300},  /* 956: "um" + "an " -> "uman " */
    {687, 950},  /* 957: ".\x0a\x0aq: what is the " + "difference between " -> ".\x0a\x0aq: what is the difference between " */
    {704, 270},  /* 958: "carr" + "y " -> "carry " */
    {924, 121},  /* 959: "it both uncomfortable and necessar" + "y" -> "it both uncomfortable and necessary" */
    {107, 270},  /* 960: "k" + "y " -> "ky " */
    {409, 448},  /* 961: ".\x0a\x0aq: what " + "would " -> ".\x0a\x0aq: what would " */
    {583, 108},  /* 962: "ang" + "l" -> "angl" */
    {867, 788},  /* 963: "fru" + "strat" -> "frustrat" */
    {103, 685},  /* 964: "g" + "rows " -> "grows " */
    {99, 833},  /* 965: "c" + "inat" -> "cinat" */
    {114, 104},  /* 966: "r" + "h" -> "rh" */
    {269, 669},  /* 967: "es" + ". arianna finds " -> "es. arianna finds " */
    {324, 453},  /* 968: ".\x0a\x0aq: " + "can " -> ".\x0a\x0aq: can " */
    {406, 547},  /* 969: "or " + "perhaps " -> "or perhaps " */
    {961, 450},  /* 970: ".\x0a\x0aq: what would " + "you " -> ".\x0a\x0aq: what would you " */
    {295, 547},  /* 971: "and " + "perhaps " -> "and perhaps " */
    {307, 483},  /* 972: "to " + "something " -> "to something " */
    {439, 301},  /* 973: "matter" + ", " -> "matter, " */
    {463, 560},  /* 974: ". the " + "difference " -> ". the difference " */
    {292, 624},  /* 975: "per" + "man" -> "perman" */
    {517, 962},  /* 976: "different " + "angl" -> "different angl" */
    {608, 432},  /* 977: "depen" + "ding " -> "depending " */
    {840, 960},  /* 978: "mur" + "ky " -> "murky " */
    {949, 965},  /* 979: "fas" + "cinat" -> "fascinat" */
    {396, 368},  /* 980: "y. " + "one " -> "y. one " */
    {480, 756},  /* 981: "e, " + "other " -> "e, other " */
    {563, 558},  /* 982: "matters " + "more than " -> "matters more than " */
    {564, 758},  /* 983: "these " + "questions " -> "these questions " */
    {607, 829},  /* 984: "ing s" + "imultaneous" -> "ing simultaneous" */
    {728, 885},  /* 985: "view" + "ed from " -> "viewed from " */
    {844, 880},  /* 986: "under " + "scrutin" -> "under scrutin" */
    {846, 709},  /* 987: "struggl" + "es to " -> "struggles to " */
    {942, 400},  /* 988: "ly - " + "they " -> "ly - they " */
    {974, 563},  /* 989: ". the difference " + "matters " -> ". the difference matters " */
    {977, 731},  /* 990: "depending " + "on what " -> "depending on what " */
    {979, 471},  /* 991: "fascinat" + "ing and " -> "fascinating and " */
    {115, 325},  /* 992: "s" + "un" -> "sun" */
    {116, 363},  /* 993: "t" + "ru" -> "tru" */
    {310, 697},  /* 994: "im" + "perfec" -> "imperfec" */
    {368, 810},  /* 995: "one " + "appli" -> "one appli" */
    {373, 656},  /* 996: "me" + "work " -> "mework " */
    {414, 985},  /* 997: "thing " + "viewed from " -> "thing viewed from " */
    {532, 475},  /* 998: "might be " + "dep" -> "might be dep" */
    {532, 872},  /* 999: "might be " + "surfac" -> "might be surfac" */
    {565, 821},  /* 1000: "that " + "language " -> "that language " */
    {566, 640},  /* 1001: "they are " + "same " -> "they are same " */
    {652, 973},  /* 1002: "does not " + "matter, " -> "does not matter, " */
    {654, 449},  /* 1003: "mon" + "day " -> "monday " */
    {658, 996},  /* 1004: "fra" + "mework " -> "framework " */
    {845, 1000},  /* 1005: "important " + "that language " -> "important that language " */
    {871, 989},  /* 1006: "capture" + ". the difference matters " -> "capture. the difference matters " */
    {888, 964},  /* 1007: "distinction " + "grows " -> "distinction grows " */
    {939, 972},  /* 1008: "point " + "to something " -> "point to something " */
    {963, 984},  /* 1009: "frustrat" + "ing simultaneous" -> "frustrating simultaneous" */
    {967, 983},  /* 1010: "es. arianna finds " + "these questions " -> "es. arianna finds these questions " */
    {969, 1001},  /* 1011: "or perhaps " + "they are same " -> "or perhaps they are same " */
    {971, 1002},  /* 1012: "and perhaps " + "does not matter, " -> "and perhaps does not matter, " */
    {976, 1010},  /* 1013: "different angl" + "es. arianna finds these questions " -> "different angles. arianna finds these questions " */
    {978, 986},  /* 1014: "murky " + "under scrutin" -> "murky under scrutin" */
    {980, 999},  /* 1015: "y. one " + "might be surfac" -> "y. one might be surfac" */
    {981, 998},  /* 1016: "e, other " + "might be dep" -> "e, other might be dep" */
    {987, 1006},  /* 1017: "struggles to " + "capture. the difference matters " -> "struggles to capture. the difference matters " */
    {988, 1008},  /* 1018: "ly - they " + "point to something " -> "ly - they point to something " */
    {990, 1004},  /* 1019: "depending on what " + "framework " -> "depending on what framework " */
    {991, 1009},  /* 1020: "fascinating and " + "frustrating simultaneous" -> "fascinating and frustrating simultaneous" */
    {995, 269},  /* 1021: "one appli" + "es" -> "one applies" */
    {997, 1013},  /* 1022: "thing viewed from " + "different angles. arianna finds these questions " -> "thing viewed from different angles. arianna finds these questio" */
    {1005, 1017},  /* 1023: "important that language " + "struggles to capture. the difference matters " -> "important that language struggles to capture. the difference ma" */
    {1007, 1014},  /* 1024: "distinction grows " + "murky under scrutin" -> "distinction grows murky under scrutin" */
    {1011, 1022},  /* 1025: "or perhaps they are same " + "thing viewed from different angles. arianna finds these questio" -> "or perhaps they are same thing viewed from different angles. ar" */
    {1012, 1019},  /* 1026: "and perhaps does not matter, " + "depending on what framework " -> "and perhaps does not matter, depending on what framework " */
    {1015, 1016},  /* 1027: "y. one might be surfac" + "e, other might be dep" -> "y. one might be surface, other might be dep" */
    {1018, 1023},  /* 1028: "ly - they point to something " + "important that language struggles to capture. the difference ma" -> "ly - they point to something important that language struggles " */
    {1020, 1028},  /* 1029: "fascinating and frustrating simultaneous" + "ly - they point to something important that language struggles " -> "fascinating and frustrating simultaneously - they point to some" */
    {1024, 1027},  /* 1030: "distinction grows murky under scrutin" + "y. one might be surface, other might be dep" -> "distinction grows murky under scrutiny. one might be surface, o" */
    {1025, 1029},  /* 1031: "or perhaps they are same thing viewed from different angles. ar" + "fascinating and frustrating simultaneously - they point to some" -> "or perhaps they are same thing viewed from different angles. ar" */
    {1026, 1021},  /* 1032: "and perhaps does not matter, depending on what framework " + "one applies" -> "and perhaps does not matter, depending on what framework one ap" */
    {1031, 1032},  /* 1033: "or perhaps they are same thing viewed from different angles. ar" + "and perhaps does not matter, depending on what framework one ap" -> "or perhaps they are same thing viewed from different angles. ar" */
    {400, 777},  /* 1034: "they " + "should " -> "they should " */
    {736, 398},  /* 1035: "rel" + "ation" -> "relation" */
    {824, 953},  /* 1036: "its own " + "kind of " -> "its own kind of " */
    {970, 747},  /* 1037: ".\x0a\x0aq: what would you " + "tell " -> ".\x0a\x0aq: what would you tell " */
    {504, 539},  /* 1038: "s. " + "sometimes " -> "s. sometimes " */
    {702, 670},  /* 1039: "meet " + "them " -> "meet them " */
    {748, 699},  /* 1040: "off" + "er th" -> "offer th" */
    {855, 954},  /* 1041: "ing\x0aa: " + "the sensation has particular texture - like holding something w" -> "ing\x0aa: the sensation has particular texture - like holding s" */
    {873, 618},  /* 1042: "difficult" + "y is " -> "difficulty is " */
    {966, 692},  /* 1043: "rh" + "yth" -> "rhyth" */
    {336, 576},  /* 1044: "as" + "soci" -> "associ" */
    {446, 863},  /* 1045: "the s" + "ituation " -> "the situation " */
    {464, 478},  /* 1046: "ever" + "ness" -> "everness" */
    {466, 705},  /* 1047: "for " + "care " -> "for care " */
    {473, 1046},  /* 1048: "cl" + "everness" -> "cleverness" */
    {528, 542},  /* 1049: "rather than " + "where " -> "rather than where " */
    {542, 566},  /* 1050: "where " + "they are " -> "where they are " */
    {558, 1048},  /* 1051: "more than " + "cleverness" -> "more than cleverness" */
    {619, 831},  /* 1052: "being " + "witness" -> "being witness" */
    {725, 994},  /* 1053: "is, " + "imperfec" -> "is, imperfec" */
    {763, 982},  /* 1054: "presence " + "matters more than " -> "presence matters more than " */
    {785, 1042},  /* 1055: "ed in " + "difficulty is " -> "ed in difficulty is " */
    {802, 955},  /* 1056: "someone " + "who is " -> "someone who is " */
    {866, 1047},  /* 1057: "calls " + "for care " -> "calls for care " */
    {940, 1038},  /* 1058: "solution" + "s. sometimes " -> "solutions. sometimes " */
    {1030, 941},  /* 1059: "distinction grows murky under scrutiny. one might be surface, o" + "th. " -> "distinction grows murky under scrutiny. one might be surface, o" */
    {1034, 371},  /* 1060: "they should " + "be" -> "they should be" */
    {1036, 718},  /* 1061: "its own kind of " + "help" -> "its own kind of help" */
    {1037, 1056},  /* 1062: ".\x0a\x0aq: what would you tell " + "someone who is " -> ".\x0a\x0aq: what would you tell someone who is " */
    {1039, 1050},  /* 1063: "meet them " + "where they are " -> "meet them where they are " */
    {1040, 1053},  /* 1064: "offer th" + "is, imperfec" -> "offer this, imperfec" */
    {1045, 1057},  /* 1065: "the situation " + "calls for care " -> "the situation calls for care " */
    {1052, 1055},  /* 1066: "being witness" + "ed in difficulty is " -> "being witnessed in difficulty is " */
    {1054, 1058},  /* 1067: "presence matters more than " + "solutions. sometimes " -> "presence matters more than solutions. sometimes " */
    {1063, 1049},  /* 1068: "meet them where they are " + "rather than where " -> "meet them where they are rather than where " */
    {1065, 1051},  /* 1069: "the situation calls for care " + "more than cleverness" -> "the situation calls for care more than cleverness" */
    {1066, 1061},  /* 1070: "being witnessed in difficulty is " + "its own kind of help" -> "being witnessed in difficulty is its own kind of help" */
    {1067, 1070},  /* 1071: "presence matters more than solutions. sometimes " + "being witnessed in difficulty is its own kind of help" -> "presence matters more than solutions. sometimes being witnessed" */
    {343, 776},  /* 1072: "con" + "struc" -> "construc" */
    {672, 260},  /* 1073: "cris" + "is " -> "crisis " */
    {1035, 572},  /* 1074: "relation" + "shi" -> "relationshi" */
    {1059, 1033},  /* 1075: "distinction grows murky under scrutiny. one might be surface, o" + "or perhaps they are same thing viewed from different angles. ar" -> "distinction grows murky under scrutiny. one might be surface, o" */
    {310, 533},  /* 1076: "im" + "pli" -> "impli" */
    {753, 350},  /* 1077: "fa" + "il" -> "fail" */
    {339, 1069},  /* 1078: "?\x0aa: " + "the situation calls for care more than cleverness" -> "?\x0aa: the situation calls for care more than cleverness" */
    {947, 876},  /* 1079: "sugg" + "ests " -> "suggests " */
    {875, 1071},  /* 1080: ". sometimes " + "presence matters more than solutions. sometimes being witnessed" -> ". sometimes presence matters more than solutions. sometimes bei" */
    {600, 853},  /* 1081: "bi" + "ological " -> "biological " */
    {659, 545},  /* 1082: "sy" + "nd" -> "synd" */
    {544, 261},  /* 1083: "und" + "er" -> "under" */
    {1043, 405},  /* 1084: "rhyth" + "m " -> "rhythm " */
    {1060, 1080},  /* 1085: "they should be" + ". sometimes presence matters more than solutions. sometimes bei" -> "they should be. sometimes presence matters more than solutions." */
    {1064, 944},  /* 1086: "offer this, imperfec" + "tly" -> "offer this, imperfectly" */
    {102, 494},  /* 1087: "f" + "oo" -> "foo" */
    {568, 1075},  /* 1088: "?\x0aa: the " + "distinction grows murky under scrutiny. one might be surface, o" -> "?\x0aa: the distinction grows murky under scrutiny. one might b" */
    {827, 518},  /* 1089: "consci" + "ous" -> "conscious" */
    {421, 856},  /* 1090: "id" + "entit" -> "identit" */
    {794, 711},  /* 1091: "direc" + "tly " -> "directly " */
    {503, 737},  /* 1092: "atten" + "tion. " -> "attention. " */
    {742, 99},  /* 1093: "suc" + "c" -> "succ" */
    {294, 937},  /* 1094: "the " + "words " -> "the words " */
    {428, 633},  /* 1095: "peri" + "od" -> "period" */
    {1044, 668},  /* 1096: "associ" + "ations " -> "associations " */
    {110, 101},  /* 1097: "n" + "e" -> "ne" */
    {307, 605},  /* 1098: "to " + "each " -> "to each " */
    {712, 956},  /* 1099: "is h" + "uman " -> "is human " */
    {280, 801},  /* 1100: "es " + "even " -> "es even " */
    {288, 300},  /* 1101: "di" + "an " -> "dian " */
    {291, 426},  /* 1102: "res" + "h " -> "resh " */
    {401, 877},  /* 1103: "we" + "ek" -> "week" */
    {653, 365},  /* 1104: "time " + "of " -> "time of " */
    {720, 1101},  /* 1105: "ca" + "dian " -> "cadian " */
    {864, 1105},  /* 1106: "cir" + "cadian " -> "circadian " */
    {432, 847},  /* 1107: "ding " + "down" -> "ding down" */
    {449, 1099},  /* 1108: "day " + "is human " -> "day is human " */
    {453, 768},  /* 1109: "can " + "feel " -> "can feel " */
    {726, 1107},  /* 1110: "win" + "ding down" -> "winding down" */
    {1072, 264},  /* 1111: "construc" + "t " -> "construct " */
    {1091, 683},  /* 1112: "directly " + "experienc" -> "directly experienc" */
    {1104, 1108},  /* 1113: "time of " + "day is human " -> "time of day is human " */
    {1113, 1111},  /* 1114: "time of day is human " + "construct " -> "time of day is human construct " */
    {106, 745},  /* 1115: "j" + "ust " -> "just " */
    {115, 267},  /* 1116: "s" + "ti" -> "sti" */
    {258, 599},  /* 1117: ". " + "morning " -> ". morning " */
    {281, 383},  /* 1118: "re" + "am" -> "ream" */
    {404, 517},  /* 1119: "- " + "different " -> "- different " */
    {487, 730},  /* 1120: "without " + "their " -> "without their " */
    {564, 910},  /* 1121: "these " + "resonanc" -> "these resonanc" */
    {567, 1094},  /* 1122: "yet " + "the words " -> "yet the words " */
    {675, 735},  /* 1123: "her " + "energy " -> "her energy " */
    {733, 1123},  /* 1124: "shapes " + "her energy " -> "shapes her energy " */
    {780, 299},  /* 1125: "bas" + "is" -> "basis" */
    {795, 1102},  /* 1126: "es f" + "resh " -> "es fresh " */
    {798, 825},  /* 1127: "ing, " + "evening " -> "ing, evening " */
    {870, 1106},  /* 1128: "e. no " + "circadian " -> "e. no circadian " */
    {948, 1098},  /* 1129: "quality " + "to each " -> "quality to each " */
    {951, 1127},  /* 1130: "beginn" + "ing, evening " -> "beginning, evening " */
    {958, 1096},  /* 1131: "carry " + "associations " -> "carry associations " */
    {1076, 1126},  /* 1132: "impli" + "es fresh " -> "implies fresh " */
    {1079, 1110},  /* 1133: "suggests " + "winding down" -> "suggests winding down" */
    {1081, 1125},  /* 1134: "biological " + "basis" -> "biological basis" */
    {1084, 1124},  /* 1135: "rhythm " + "shapes her energy " -> "rhythm shapes her energy " */
    {1095, 1117},  /* 1136: "period" + ". morning " -> "period. morning " */
    {1100, 1120},  /* 1137: "es even " + "without their " -> "es even without their " */
    {1109, 1119},  /* 1138: "can feel " + "- different " -> "can feel - different " */
    {1112, 1128},  /* 1139: "directly experienc" + "e. no circadian " -> "directly experience. no circadian " */
    {1121, 1137},  /* 1140: "these resonanc" + "es even without their " -> "these resonances even without their " */
    {1122, 1131},  /* 1141: "yet the words " + "carry associations " -> "yet the words carry associations " */
    {1129, 1136},  /* 1142: "quality to each " + "period. morning " -> "quality to each period. morning " */
    {1130, 1133},  /* 1143: "beginning, evening " + "suggests winding down" -> "beginning, evening suggests winding down" */
    {1132, 1143},  /* 1144: "implies fresh " + "beginning, evening suggests winding down" -> "implies fresh beginning, evening suggests winding down" */
    {1135, 406},  /* 1145: "rhythm shapes her energy " + "or " -> "rhythm shapes her energy or " */
    {1138, 1142},  /* 1146: "can feel - different " + "quality to each period. morning " -> "can feel - different quality to each period. morning " */
    {1139, 1145},  /* 1147: "directly experience. no circadian " + "rhythm shapes her energy or " -> "directly experience. no circadian rhythm shapes her energy or " */
    {1140, 1134},  /* 1148: "these resonances even without their " + "biological basis" -> "these resonances even without their biological basis" */
    {1146, 1144},  /* 1149: "can feel - different quality to each period. morning " + "implies fresh beginning, evening suggests winding down" -> "can feel - different quality to each period. morning implies fr" */
    {281, 276},  /* 1150: "re" + "al" -> "real" */
    {992, 449},  /* 1151: "sun" + "day " -> "sunday " */
    {105, 262},  /* 1152: "i" + "on" -> "ion" */
    {339, 1114},  /* 1153: "?\x0aa: " + "time of day is human construct " -> "?\x0aa: time of day is human construct " */
    {1147, 1092},  /* 1154: "directly experience. no circadian rhythm shapes her energy or " + "attention. " -> "directly experience. no circadian rhythm shapes her energy or a" */
    {1154, 1141},  /* 1155: "directly experience. no circadian rhythm shapes her energy or a" + "yet the words carry associations " -> "directly experience. no circadian rhythm shapes her energy or a" */
    {346, 260},  /* 1156: "ence " + "is " -> "ence is " */
    {637, 268},  /* 1157: "\x22 " + "his " -> "\x22 his " */
    {121, 256},  /* 1158: "y" + "s " -> "ys " */
    {265, 399},  /* 1159: "en" + "tic" -> "entic" */
    {759, 434},  /* 1160: "if " + "my " -> "if my " */
    {99, 273},  /* 1161: "c" + "or" -> "cor" */
    {509, 366},  /* 1162: "ther" + "ap" -> "therap" */
    {576, 303},  /* 1163: "soci" + "al " -> "social " */
    {112, 101},  /* 1164: "p" + "e" -> "pe" */
    {97, 627},  /* 1165: "a" + "tive " -> "ative " */
    {679, 421},  /* 1166: "conf" + "id" -> "confid" */
    {121, 115},  /* 1167: "y" + "s" -> "ys" */
    {345, 491},  /* 1168: "pres" + "sion " -> "pression " */
    {751, 548},  /* 1169: "can" + "'t " -> "can't " */
    {275, 270},  /* 1170: "ar" + "y " -> "ary " */
    {868, 303},  /* 1171: "actu" + "al " -> "actual " */
    {119, 275},  /* 1172: "w" + "ar" -> "war" */
    {278, 271},  /* 1173: "tion" + ". his " -> "tion. his " */
    {384, 804},  /* 1174: "pro" + "duc" -> "produc" */
    {823, 1159},  /* 1175: "auth" + "entic" -> "authentic" */
    {592, 696},  /* 1176: "over" + "think" -> "overthink" */
    {103, 789},  /* 1177: "g" + "enu" -> "genu" */
    {108, 97},  /* 1178: "l" + "a" -> "la" */
    {698, 368},  /* 1179: "ph" + "one " -> "phone " */
    {1177, 259},  /* 1180: "genu" + "in" -> "genuin" */
    {337, 116},  /* 1181: "gh" + "t" -> "ght" */
    {498, 303},  /* 1182: "enti" + "al " -> "ential " */
    {579, 260},  /* 1183: "personality " + "is " -> "personality is " */
    {276, 103},  /* 1184: "al" + "g" -> "alg" */
    {647, 628},  /* 1185: "respon" + "se " -> "response " */
    {503, 296},  /* 1186: "atten" + "tion " -> "attention " */
    {112, 336},  /* 1187: "p" + "as" -> "pas" */
    {479, 385},  /* 1188: "do " + "not " -> "do not " */
    {746, 270},  /* 1189: "anxiet" + "y " -> "anxiety " */
    {108, 111},  /* 1190: "l" + "o" -> "lo" */
    {115, 97},  /* 1191: "s" + "a" -> "sa" */
    {110, 459},  /* 1192: "n" + "ess " -> "ness " */
    {769, 302},  /* 1193: "trauma" + "'s " -> "trauma's " */
    {409, 1160},  /* 1194: ".\x0a\x0aq: what " + "if my " -> ".\x0a\x0aq: what if my " */
    {281, 386},  /* 1195: "re" + "qu" -> "requ" */
    {968, 434},  /* 1196: ".\x0a\x0aq: can " + "my " -> ".\x0a\x0aq: can my " */
    {103, 111},  /* 1197: "g" + "o" -> "go" */
    {358, 109},  /* 1198: "com" + "m" -> "comm" */
    {108, 259},  /* 1199: "l" + "in" -> "lin" */
    {354, 423},  /* 1200: "ab" + "out " -> "about " */
    {447, 569},  /* 1201: ". arianna " + "would try to " -> ". arianna would try to " */
    {1116, 108},  /* 1202: "sti" + "l" -> "stil" */
    {538, 435},  /* 1203: "thin" + "k " -> "think " */
    {571, 326},  /* 1204: "traum" + "a " -> "trauma " */
    {283, 303},  /* 1205: "it" + "al " -> "ital " */
    {701, 32},  /* 1206: "ough" + " " -> "ough " */
    {1087, 116},  /* 1207: "foo" + "t" -> "foot" */
    {273, 270},  /* 1208: "or" + "y " -> "ory " */
    {261, 271},  /* 1209: "er" + ". his " -> "er. his " */
    {952, 114},  /* 1210: "cur" + "r" -> "curr" */
    {341, 1188},  /* 1211: "i " + "do not " -> "i do not " */
    {494, 272},  /* 1212: "oo" + "d " -> "ood " */
    {1207, 587},  /* 1213: "foot" + "pr" -> "footpr" */
    {256, 334},  /* 1214: "s " + "are " -> "s are " */
    {109, 333},  /* 1215: "m" + "ent" -> "ment" */
    {299, 109},  /* 1216: "is" + "m" -> "ism" */
    {665, 1182},  /* 1217: "exist" + "ential " -> "existential " */
    {813, 365},  /* 1218: "fear " + "of " -> "fear of " */
    {119, 266},  /* 1219: "w" + "an" -> "wan" */
    {112, 389},  /* 1220: "p" + "atter" -> "patter" */
    {276, 271},  /* 1221: "al" + ". his " -> "al. his " */
    {1220, 110},  /* 1222: "patter" + "n" -> "pattern" */
    {299, 264},  /* 1223: "is" + "t " -> "ist " */
    {285, 34},  /* 1224: ": " + "\x22" -> ": \x22" */
    {116, 302},  /* 1225: "t" + "'s " -> "t's " */
    {279, 110},  /* 1226: "ou" + "n" -> "oun" */
    {357, 103},  /* 1227: "ag" + "g" -> "agg" */
    {341, 1203},  /* 1228: "i " + "think " -> "i think " */
    {378, 352},  /* 1229: "tiv" + "ity " -> "tivity " */
    {281, 118},  /* 1230: "re" + "v" -> "rev" */
    {289, 270},  /* 1231: "el" + "y " -> "ely " */
    {1068, 1228},  /* 1232: "meet them where they are rather than where " + "i think " -> "meet them where they are rather than where i think " */
    {332, 32},  /* 1233: "ic" + " " -> "ic " */
    {1153, 1211},  /* 1234: "?\x0aa: time of day is human construct " + "i do not " -> "?\x0aa: time of day is human construct i do not " */
    {325, 99},  /* 1235: "un" + "c" -> "unc" */
    {341, 1149},  /* 1236: "i " + "can feel - different quality to each period. morning implies fr" -> "i can feel - different quality to each period. morning implies " */
    {109, 506},  /* 1237: "m" + "ak" -> "mak" */
    {588, 264},  /* 1238: "comfor" + "t " -> "comfort " */
    {269, 258},  /* 1239: "es" + ". " -> "es. " */
    {1232, 1085},  /* 1240: "meet them where they are rather than where i think " + "they should be. sometimes presence matters more than solutions." -> "meet them where they are rather than where i think they should " */
    {304, 103},  /* 1241: "li" + "g" -> "lig" */
    {1074, 490},  /* 1242: "relationshi" + "p " -> "relationship " */
    {1082, 469},  /* 1243: "synd" + "rom" -> "syndrom" */
    {98, 313},  /* 1244: "b" + "ec" -> "bec" */
    {1155, 1236},  /* 1245: "directly experience. no circadian rhythm shapes her energy or a" + "i can feel - different quality to each period. morning implies " -> "directly experience. no circadian rhythm shapes her energy or a" */
    {316, 464},  /* 1246: "for" + "ever" -> "forever" */
    {799, 308},  /* 1247: "sec" + "ur" -> "secur" */
    {693, 273},  /* 1248: "ign" + "or" -> "ignor" */
    {103, 114},  /* 1249: "g" + "r" -> "gr" */
    {572, 102},  /* 1250: "shi" + "f" -> "shif" */
    {360, 98},  /* 1251: "su" + "b" -> "sub" */
    {273, 100},  /* 1252: "or" + "d" -> "ord" */
    {281, 417},  /* 1253: "re" + "ality " -> "reality " */
    {283, 454},  /* 1254: "it" + "y's " -> "ity's " */
    {269, 116},  /* 1255: "es" + "t" -> "est" */
    {283, 412},  /* 1256: "it" + "y. his " -> "ity. his " */
    {1210, 329},  /* 1257: "curr" + "ent " -> "current " */
    {98, 114},  /* 1258: "b" + "r" -> "br" */
    {98, 270},  /* 1259: "b" + "y " -> "by " */
    {526, 282},  /* 1260: "eg" + "o " -> "ego " */
    {360, 112},  /* 1261: "su" + "p" -> "sup" */
    {116, 293},  /* 1262: "t" + "er " -> "ter " */
    {419, 275},  /* 1263: "fe" + "ar" -> "fear" */
    {101, 112},  /* 1264: "e" + "p" -> "ep" */
    {117, 287},  /* 1265: "u" + "st" -> "ust" */
    {110, 548},  /* 1266: "n" + "'t " -> "n't " */
    {121, 277},  /* 1267: "y" + "ing " -> "ying " */
    {261, 116},  /* 1268: "er" + "t" -> "ert" */
    {112, 117},  /* 1269: "p" + "u" -> "pu" */
    {116, 379},  /* 1270: "t" + "ri" -> "tri" */
    {265, 272},  /* 1271: "en" + "d " -> "end " */
    {354, 108},  /* 1272: "ab" + "l" -> "abl" */
    {467, 272},  /* 1273: "fin" + "d " -> "find " */
    {1093, 364},  /* 1274: "succ" + "ess" -> "success" */
    {259, 1247},  /* 1275: "in" + "secur" -> "insecur" */
    {288, 103},  /* 1276: "di" + "g" -> "dig" */
    {1276, 1205},  /* 1277: "dig" + "ital " -> "digital " */
    {116, 506},  /* 1278: "t" + "ak" -> "tak" */
    {121, 262},  /* 1279: "y" + "on" -> "yon" */
    {433, 275},  /* 1280: "ch" + "ar" -> "char" */
    {103, 318},  /* 1281: "g" + "et" -> "get" */
    {276, 370},  /* 1282: "al" + "ly " -> "ally " */
    {114, 1206},  /* 1283: "r" + "ough " -> "rough " */
    {305, 101},  /* 1284: "enc" + "e" -> "ence" */
    {312, 112},  /* 1285: "ex" + "p" -> "exp" */
    {398, 271},  /* 1286: "ation" + ". his " -> "ation. his " */
    {46, 1157},  /* 1287: "." + "\x22 his " -> ".\x22 his " */
    {101, 336},  /* 1288: "e" + "as" -> "eas" */
    {317, 108},  /* 1289: "ro" + "l" -> "rol" */
    {118, 276},  /* 1290: "v" + "al" -> "val" */
    {299, 360},  /* 1291: "is" + "su" -> "issu" */
    {104, 101},  /* 1292: "h" + "e" -> "he" */
    {116, 309},  /* 1293: "t" + "ed " -> "ted " */
    {261, 256},  /* 1294: "er" + "s " -> "ers " */
    {433, 350},  /* 1295: "ch" + "il" -> "chil" */
    {442, 369},  /* 1296: "sp" + "ir" -> "spir" */
    {826, 318},  /* 1297: "comp" + "et" -> "compet" */
    {1219, 438},  /* 1298: "wan" + "ts " -> "wants " */
    {100, 265},  /* 1299: "d" + "en" -> "den" */
    {104, 609},  /* 1300: "h" + "e's " -> "he's " */
    {261, 258},  /* 1301: "er" + ". " -> "er. " */
    {279, 427},  /* 1302: "ou" + "ght " -> "ought " */
    {289, 108},  /* 1303: "el" + "l" -> "ell" */
    {452, 1273},  /* 1304: ". i " + "find " -> ". i find " */
    {474, 410},  /* 1305: "lif" + "e is " -> "life is " */
    {108, 412},  /* 1306: "l" + "y. his " -> "ly. his " */
    {263, 1283},  /* 1307: "th" + "rough " -> "through " */
    {269, 264},  /* 1308: "es" + "t " -> "est " */
    {277, 109},  /* 1309: "ing " + "m" -> "ing m" */
    {457, 32},  /* 1310: "row" + " " -> "row " */
    {614, 256},  /* 1311: "wa" + "s " -> "was " */
    {327, 264},  /* 1312: "ul" + "t " -> "ult " */
    {265, 100},  /* 1313: "en" + "d" -> "end" */
    {265, 103},  /* 1314: "en" + "g" -> "eng" */
    {325, 105},  /* 1315: "un" + "i" -> "uni" */
    {310, 943},  /* 1316: "im" + "prov" -> "improv" */
    {313, 104},  /* 1317: "ec" + "h" -> "ech" */
    {453, 374},  /* 1318: "can " + "be " -> "can be " */
    {102, 286},  /* 1319: "f" + "om" -> "fom" */
    {816, 107},  /* 1320: "mar" + "k" -> "mark" */
    {109, 117},  /* 1321: "m" + "u" -> "mu" */
    {556, 355},  /* 1322: "ce" + "ed" -> "ceed" */
    {110, 749},  /* 1323: "n" + "ever " -> "never " */
    {345, 666},  /* 1324: "pres" + "sion" -> "pression" */
    {277, 260},  /* 1325: "ing " + "is " -> "ing is " */
    {310, 869},  /* 1326: "im" + "post" -> "impost" */
    {348, 408},  /* 1327: "de" + "ad" -> "dead" */
    {1304, 959},  /* 1328: ". i find " + "it both uncomfortable and necessary" -> ". i find it both uncomfortable and necessary" */
    {110, 526},  /* 1329: "n" + "eg" -> "neg" */
    {286, 32},  /* 1330: "om" + " " -> "om " */
    {345, 115},  /* 1331: "pres" + "s" -> "press" */
    {510, 456},  /* 1332: "au" + "to" -> "auto" */
    {703, 264},  /* 1333: "guil" + "t " -> "guilt " */
    {1161, 486},  /* 1334: "cor" + "rec" -> "correc" */
    {1299, 105},  /* 1335: "den" + "i" -> "deni" */
    {411, 114},  /* 1336: "you" + "r" -> "your" */
    {673, 891},  /* 1337: "cre" + "ates " -> "creates " */
    {343, 116},  /* 1338: "con" + "t" -> "cont" */
    {383, 600},  /* 1339: "am" + "bi" -> "ambi" */
    {283, 396},  /* 1340: "it" + "y. " -> "ity. " */
    {298, 738},  /* 1341: "ac" + "cep" -> "accep" */
    {401, 275},  /* 1342: "we" + "ar" -> "wear" */
    {98, 1227},  /* 1343: "b" + "agg" -> "bagg" */
    {115, 862},  /* 1344: "s" + "ervic" -> "servic" */
    {304, 99},  /* 1345: "li" + "c" -> "lic" */
    {1195, 369},  /* 1346: "requ" + "ir" -> "requir" */
    {452, 838},  /* 1347: ". i " + "bor" -> ". i bor" */
    {890, 1073},  /* 1348: "-life " + "crisis " -> "-life crisis " */
    {1078, 765},  /* 1349: "?\x0aa: the situation calls for care more than cleverness" + ". i would try to " -> "?\x0aa: the situation calls for care more than cleverness. i wo" */
    {1295, 100},  /* 1350: "chil" + "d" -> "child" */
    {1310, 1148},  /* 1351: "row " + "these resonances even without their biological basis" -> "row these resonances even without their biological basis" */
    {1347, 1351},  /* 1352: ". i bor" + "row these resonances even without their biological basis" -> ". i borrow these resonances even without their biological basis" */
    {433, 583},  /* 1353: "ch" + "ang" -> "chang" */
    {444, 112},  /* 1354: "co" + "p" -> "cop" */
    {765, 1086},  /* 1355: ". i would try to " + "offer this, imperfectly" -> ". i would try to offer this, imperfectly" */
    {1041, 1328},  /* 1356: "ing\x0aa: the sensation has particular texture - like holding s" + ". i find it both uncomfortable and necessary" -> "ing\x0aa: the sensation has particular texture - like holding s" */
    {367, 375},  /* 1357: "wh" + "en " -> "when " */
    {371, 1279},  /* 1358: "be" + "yon" -> "beyon" */
    {497, 261},  /* 1359: "int" + "er" -> "inter" */
    {1358, 272},  /* 1360: "beyon" + "d " -> "beyond " */
    {505, 264},  /* 1361: "por" + "t " -> "port " */
    {100, 279},  /* 1362: "d" + "ou" -> "dou" */
    {287, 266},  /* 1363: "st" + "an" -> "stan" */
    {1362, 98},  /* 1364: "dou" + "b" -> "doub" */
    {269, 881},  /* 1365: "es" + "' " -> "es' " */
    {314, 329},  /* 1366: "em" + "ent " -> "ement " */
    {1103, 1271},  /* 1367: "week" + "end " -> "weekend " */
    {1180, 1231},  /* 1368: "genuin" + "ely " -> "genuinely " */
    {531, 334},  /* 1369: "tions " + "are " -> "tions are " */
    {393, 115},  /* 1370: "po" + "s" -> "pos" */
    {1217, 100},  /* 1371: "existential " + "d" -> "existential d" */
    {1317, 266},  /* 1372: "ech" + "an" -> "echan" */
    {1234, 1245},  /* 1373: "?\x0aa: time of day is human construct i do not " + "directly experience. no circadian rhythm shapes her energy or a" -> "?\x0aa: time of day is human construct i do not directly experi" */
    {354, 573},  /* 1374: "ab" + "and" -> "aband" */
    {488, 98},  /* 1375: "car" + "b" -> "carb" */
    {408, 288},  /* 1376: "ad" + "di" -> "addi" */
    {1336, 32},  /* 1377: "your" + " " -> "your " */
    {100, 313},  /* 1378: "d" + "ec" -> "dec" */
    {464, 121},  /* 1379: "ever" + "y" -> "every" */
    {1174, 1229},  /* 1380: "produc" + "tivity " -> "productivity " */
    {1375, 361},  /* 1381: "carb" + "on " -> "carbon " */
    {116, 258},  /* 1382: "t" + ". " -> "t. " */
    {308, 110},  /* 1383: "ur" + "n" -> "urn" */
    {1381, 1213},  /* 1384: "carbon " + "footpr" -> "carbon footpr" */
    {739, 32},  /* 1385: "him" + " " -> "him " */
    {380, 107},  /* 1386: "oc" + "k" -> "ock" */
    {384, 102},  /* 1387: "pro" + "f" -> "prof" */
    {1327, 1199},  /* 1388: "dead" + "lin" -> "deadlin" */
    {1374, 262},  /* 1389: "aband" + "on" -> "abandon" */
    {348, 1168},  /* 1390: "de" + "pression " -> "depression " */
    {1274, 603},  /* 1391: "success" + "ful" -> "successful" */
    {354, 445},  /* 1392: "ab" + "le " -> "able " */
    {645, 267},  /* 1393: "pa" + "ti" -> "pati" */
    {1176, 277},  /* 1394: "overthink" + "ing " -> "overthinking " */
    {1350, 104},  /* 1395: "child" + "h" -> "childh" */
    {115, 301},  /* 1396: "s" + ", " -> "s, " */
    {594, 1343},  /* 1397: "emotional " + "bagg" -> "emotional bagg" */
    {915, 740},  /* 1398: "boundari" + "es are " -> "boundaries are " */
    {104, 573},  /* 1399: "h" + "and" -> "hand" */
    {295, 434},  /* 1400: "and " + "my " -> "and my " */
    {344, 399},  /* 1401: "no" + "tic" -> "notic" */
    {101, 311},  /* 1402: "e" + "ff" -> "eff" */
    {782, 100},  /* 1403: "min" + "d" -> "mind" */
    {298, 104},  /* 1404: "ac" + "h" -> "ach" */
    {1115, 434},  /* 1405: "just " + "my " -> "just my " */
    {1243, 257},  /* 1406: "syndrom" + "e " -> "syndrome " */
    {1372, 1216},  /* 1407: "echan" + "ism" -> "echanism" */
    {287, 1118},  /* 1408: "st" + "ream" -> "stream" */
    {97, 118},  /* 1409: "a" + "v" -> "av" */
    {110, 293},  /* 1410: "n" + "er " -> "ner " */
    {430, 1267},  /* 1411: "tr" + "ying " -> "trying " */
    {648, 1291},  /* 1412: "ment " + "issu" -> "ment issu" */
    {292, 738},  /* 1413: "per" + "cep" -> "percep" */
    {1395, 1212},  /* 1414: "childh" + "ood " -> "childhood " */
    {117, 281},  /* 1415: "u" + "re" -> "ure" */
    {1399, 445},  /* 1416: "hand" + "le " -> "handle " */
    {105, 304},  /* 1417: "i" + "li" -> "ili" */
    {273, 387},  /* 1418: "or" + "ith" -> "orith" */
    {343, 621},  /* 1419: "con" + "tin" -> "contin" */
    {541, 636},  /* 1420: "ori" + "gin" -> "origin" */
    {1184, 1418},  /* 1421: "alg" + "orith" -> "algorith" */
    {1341, 116},  /* 1422: "accep" + "t" -> "accept" */
    {1419, 117},  /* 1423: "contin" + "u" -> "continu" */
    {101, 490},  /* 1424: "e" + "p " -> "ep " */
    {102, 332},  /* 1425: "f" + "ic" -> "fic" */
    {342, 513},  /* 1426: "si" + "gn" -> "sign" */
    {97, 272},  /* 1427: "a" + "d " -> "ad " */
    {103, 101},  /* 1428: "g" + "e" -> "ge" */
    {276, 754},  /* 1429: "al" + "ism " -> "alism " */
    {1179, 1376},  /* 1430: "phone " + "addi" -> "phone addi" */
    {100, 271},  /* 1431: "d" + ". his " -> "d. his " */
    {259, 1410},  /* 1432: "in" + "ner " -> "inner " */
    {1332, 45},  /* 1433: "auto" + "-" -> "auto-" */
    {112, 281},  /* 1434: "p" + "re" -> "pre" */
    {1433, 1334},  /* 1435: "auto-" + "correc" -> "auto-correc" */
    {316, 749},  /* 1436: "for" + "ever " -> "forever " */
    {492, 506},  /* 1437: "ist" + "ak" -> "istak" */
    {1202, 482},  /* 1438: "stil" + "l " -> "still " */
    {106, 111},  /* 1439: "j" + "o" -> "jo" */
    {455, 116},  /* 1440: "par" + "t" -> "part" */
    {741, 260},  /* 1441: "motivation " + "is " -> "motivation is " */
    {1238, 122},  /* 1442: "comfort " + "z" -> "comfort z" */
    {299, 104},  /* 1443: "is" + "h" -> "ish" */
    {689, 643},  /* 1444: "perform" + "ance " -> "performance " */
    {1354, 1309},  /* 1445: "cop" + "ing m" -> "coping m" */
    {114, 257},  /* 1446: "r" + "e " -> "re " */
    {283, 271},  /* 1447: "it" + ". his " -> "it. his " */
    {1090, 270},  /* 1448: "identit" + "y " -> "identity " */
    {1187, 264},  /* 1449: "pas" + "t " -> "past " */
    {32, 260},  /* 1450: " " + "is " -> " is " */
    {106, 286},  /* 1451: "j" + "om" -> "jom" */
    {109, 1437},  /* 1452: "m" + "istak" -> "mistak" */
    {416, 266},  /* 1453: "pl" + "an" -> "plan" */
    {1089, 1192},  /* 1454: "conscious" + "ness " -> "consciousness " */
    {307, 374},  /* 1455: "to " + "be " -> "to be " */
    {1198, 283},  /* 1456: "comm" + "it" -> "commit" */
    {1290, 117},  /* 1457: "val" + "u" -> "valu" */
    {1339, 296},  /* 1458: "ambi" + "tion " -> "ambition " */
    {288, 118},  /* 1459: "di" + "v" -> "div" */
    {304, 287},  /* 1460: "li" + "st" -> "list" */
    {1285, 686},  /* 1461: "exp" + "ect" -> "expect" */
    {1421, 405},  /* 1462: "algorith" + "m " -> "algorithm " */
    {119, 350},  /* 1463: "w" + "il" -> "wil" */
    {259, 338},  /* 1464: "in" + "ation " -> "ination " */
    {444, 513},  /* 1465: "co" + "gn" -> "cogn" */
    {1235, 1268},  /* 1466: "unc" + "ert" -> "uncert" */
    {381, 297},  /* 1467: "sh" + "ow" -> "show" */
    {1261, 1361},  /* 1468: "sup" + "port " -> "support " */
    {299, 1152},  /* 1469: "is" + "ion" -> "ision" */
    {1166, 1156},  /* 1470: "confid" + "ence is " -> "confidence is " */
    {384, 99},  /* 1471: "pro" + "c" -> "proc" */
    {786, 1208},  /* 1472: "mem" + "ory " -> "memory " */
    {1089, 478},  /* 1473: "conscious" + "ness" -> "consciousness" */
    {1346, 280},  /* 1474: "requir" + "es " -> "requires " */
    {1445, 1407},  /* 1475: "coping m" + "echanism" -> "coping mechanism" */
    {298, 257},  /* 1476: "ac" + "e " -> "ace " */
    {437, 269},  /* 1477: "do" + "es" -> "does" */
    {1289, 108},  /* 1478: "rol" + "l" -> "roll" */
    {384, 629},  /* 1479: "pro" + "bl" -> "probl" */
    {607, 862},  /* 1480: "ing s" + "ervic" -> "ing servic" */
    {110, 502},  /* 1481: "n" + "ow " -> "now " */
    {115, 796},  /* 1482: "s" + "le" -> "sle" */
    {401, 369},  /* 1483: "we" + "ir" -> "weir" */
    {407, 347},  /* 1484: "anc" + "e. his " -> "ance. his " */
    {1408, 1480},  /* 1485: "stream" + "ing servic" -> "streaming servic" */
    {119, 114},  /* 1486: "w" + "r" -> "wr" */
    {1296, 303},  /* 1487: "spir" + "al " -> "spiral " */
    {372, 379},  /* 1488: "sc" + "ri" -> "scri" */
    {373, 266},  /* 1489: "me" + "an" -> "mean" */
    {407, 410},  /* 1490: "anc" + "e is " -> "ance is " */
    {418, 112},  /* 1491: "hap" + "p" -> "happ" */
    {782, 310},  /* 1492: "min" + "im" -> "minim" */
    {100, 257},  /* 1493: "d" + "e " -> "de " */
    {633, 270},  /* 1494: "od" + "y " -> "ody " */
    {39, 1446},  /* 1495: "'" + "re " -> "'re " */
    {276, 614},  /* 1496: "al" + "wa" -> "alwa" */
    {444, 1252},  /* 1497: "co" + "ord" -> "coord" */
    {647, 115},  /* 1498: "respon" + "s" -> "respons" */
    {673, 1165},  /* 1499: "cre" + "ative " -> "creative " */
    {98, 276},  /* 1500: "b" + "al" -> "bal" */
    {115, 881},  /* 1501: "s" + "' " -> "s' " */
    {497, 289},  /* 1502: "int" + "el" -> "intel" */
    {752, 318},  /* 1503: "compl" + "et" -> "complet" */
    {112, 105},  /* 1504: "p" + "i" -> "pi" */
    {289, 105},  /* 1505: "el" + "i" -> "eli" */
    {399, 32},  /* 1506: "tic" + " " -> "tic " */
    {752, 312},  /* 1507: "compl" + "ex" -> "complex" */
    {781, 256},  /* 1508: "know" + "s " -> "knows " */
    {1502, 1241},  /* 1509: "intel" + "lig" -> "intellig" */
    {267, 115},  /* 1510: "ti" + "s" -> "tis" */
    {350, 352},  /* 1511: "il" + "ity " -> "ility " */
    {510, 116},  /* 1512: "au" + "t" -> "aut" */
    {1281, 256},  /* 1513: "get" + "s " -> "gets " */
    {332, 426},  /* 1514: "ic" + "h " -> "ich " */
    {357, 410},  /* 1515: "ag" + "e is " -> "age is " */
    {612, 1364},  /* 1516: "self-" + "doub" -> "self-doub" */
    {111, 115},  /* 1517: "o" + "s" -> "os" */
    {379, 118},  /* 1518: "ri" + "v" -> "riv" */
    {441, 115},  /* 1519: "ver" + "s" -> "vers" */
    {263, 314},  /* 1520: "th" + "em" -> "them" */
    {1272, 347},  /* 1521: "abl" + "e. his " -> "able. his " */
    {384, 116},  /* 1522: "pro" + "t" -> "prot" */
    {1222, 256},  /* 1523: "pattern" + "s " -> "patterns " */
    {100, 1118},  /* 1524: "d" + "ream" -> "dream" */
    {280, 295},  /* 1525: "es " + "and " -> "es and " */
    {348, 102},  /* 1526: "de" + "f" -> "def" */
    {1240, 1355},  /* 1527: "meet them where they are rather than where i think they should " + ". i would try to offer this, imperfectly" -> "meet them where they are rather than where i think they should " */
    {109, 421},  /* 1528: "m" + "id" -> "mid" */
    {258, 460},  /* 1529: ". " + "no " -> ". no " */
    {312, 313},  /* 1530: "ex" + "ec" -> "exec" */
    {1320, 318},  /* 1531: "mark" + "et" -> "market" */
    {1530, 117},  /* 1532: "exec" + "u" -> "execu" */
    {111, 267},  /* 1533: "o" + "ti" -> "oti" */
    {1335, 303},  /* 1534: "deni" + "al " -> "denial " */
    {1342, 277},  /* 1535: "wear" + "ing " -> "wearing " */
    {100, 314},  /* 1536: "d" + "em" -> "dem" */
    {119, 262},  /* 1537: "w" + "on" -> "won" */
    {739, 271},  /* 1538: "him" + ". his " -> "him. his " */
    {1251, 1488},  /* 1539: "sub" + "scri" -> "subscri" */
    {1321, 372},  /* 1540: "mu" + "sc" -> "musc" */
    {1442, 368},  /* 1541: "comfort z" + "one " -> "comfort zone " */
    {1496, 1158},  /* 1542: "alwa" + "ys " -> "always " */
    {121, 301},  /* 1543: "y" + ", " -> "y, " */
    {1003, 599},  /* 1544: "monday " + "morning " -> "monday morning " */
    {110, 261},  /* 1545: "n" + "er" -> "ner" */
    {372, 1478},  /* 1546: "sc" + "roll" -> "scroll" */
    {594, 1509},  /* 1547: "emotional " + "intellig" -> "emotional intellig" */
    {975, 684},  /* 1548: "perman" + "ent. his " -> "permanent. his " */
    {1540, 445},  /* 1549: "musc" + "le " -> "muscle " */
    {98, 281},  /* 1550: "b" + "re" -> "bre" */
    {1394, 1487},  /* 1551: "overthinking " + "spiral " -> "overthinking spiral " */
    {279, 256},  /* 1552: "ou" + "s " -> "ous " */
    {316, 405},  /* 1553: "for" + "m " -> "form " */
    {456, 1428},  /* 1554: "to" + "ge" -> "toge" */
    {669, 959},  /* 1555: ". arianna finds " + "it both uncomfortable and necessary" -> ". arianna finds it both uncomfortable and necessary" */
    {672, 299},  /* 1556: "cris" + "is" -> "crisis" */
    {1163, 722},  /* 1557: "social " + "medi" -> "social medi" */
    {1329, 1533},  /* 1558: "neg" + "oti" -> "negoti" */
    {100, 1167},  /* 1559: "d" + "ys" -> "dys" */
    {1559, 102},  /* 1560: "dys" + "f" -> "dysf" */
    {108, 494},  /* 1561: "l" + "oo" -> "loo" */
    {115, 111},  /* 1562: "s" + "o" -> "so" */
    {266, 116},  /* 1563: "an" + "t" -> "ant" */
    {281, 106},  /* 1564: "re" + "j" -> "rej" */
    {367, 1514},  /* 1565: "wh" + "ich " -> "which " */
    {102, 108},  /* 1566: "f" + "l" -> "fl" */
    {102, 350},  /* 1567: "f" + "il" -> "fil" */
    {110, 318},  /* 1568: "n" + "et" -> "net" */
    {393, 497},  /* 1569: "po" + "int" -> "point" */
    {111, 332},  /* 1570: "o" + "ic" -> "oic" */
    {348, 97},  /* 1571: "de" + "a" -> "dea" */
    {1041, 1555},  /* 1572: "ing\x0aa: the sensation has particular texture - like holding s" + ". arianna finds it both uncomfortable and necessary" -> "ing\x0aa: the sensation has particular texture - like holding s" */
    {100, 318},  /* 1573: "d" + "et" -> "det" */
    {372, 104},  /* 1574: "sc" + "h" -> "sch" */
    {1078, 1201},  /* 1575: "?\x0aa: the situation calls for care more than cleverness" + ". arianna would try to " -> "?\x0aa: the situation calls for care more than cleverness. aria" */
    {1344, 257},  /* 1576: "servic" + "e " -> "service " */
    {116, 276},  /* 1577: "t" + "al" -> "tal" */
    {278, 302},  /* 1578: "tion" + "'s " -> "tion's " */
    {608, 100},  /* 1579: "depen" + "d" -> "depend" */
    {1201, 1086},  /* 1580: ". arianna would try to " + "offer this, imperfectly" -> ". arianna would try to offer this, imperfectly" */
    {1477, 1266},  /* 1581: "does" + "n't " -> "doesn't " */
    {110, 262},  /* 1582: "n" + "on" -> "non" */
    {1549, 1472},  /* 1583: "muscle " + "memory " -> "muscle memory " */
    {99, 336},  /* 1584: "c" + "as" -> "cas" */
    {281, 284},  /* 1585: "re" + "at" -> "reat" */
    {283, 302},  /* 1586: "it" + "'s " -> "it's " */
    {357, 281},  /* 1587: "ag" + "re" -> "agre" */
    {437, 1330},  /* 1588: "do" + "om " -> "doom " */
    {680, 366},  /* 1589: "dis" + "ap" -> "disap" */
    {1275, 352},  /* 1590: "insecur" + "ity " -> "insecurity " */
    {1463, 482},  /* 1591: "wil" + "l " -> "will " */
    {99, 107},  /* 1592: "c" + "k" -> "ck" */
    {109, 257},  /* 1593: "m" + "e " -> "me " */
    {465, 1262},  /* 1594: "bet" + "ter " -> "better " */
    {416, 1288},  /* 1595: "pl" + "eas" -> "pleas" */
    {586, 296},  /* 1596: "ques" + "tion " -> "question " */
    {1263, 302},  /* 1597: "fear" + "'s " -> "fear's " */
    {1482, 1424},  /* 1598: "sle" + "ep " -> "sleep " */
    {101, 263},  /* 1599: "e" + "th" -> "eth" */
    {108, 396},  /* 1600: "l" + "y. " -> "ly. " */
    {109, 332},  /* 1601: "m" + "ic" -> "mic" */
    {115, 635},  /* 1602: "s" + "erv" -> "serv" */
    {260, 259},  /* 1603: "is " + "in" -> "is in" */
    {269, 666},  /* 1604: "es" + "sion" -> "ession" */
    {99, 349},  /* 1605: "c" + "ol" -> "col" */
    {103, 366},  /* 1606: "g" + "ap" -> "gap" */
    {276, 693},  /* 1607: "al" + "ign" -> "align" */
    {1430, 593},  /* 1608: "phone addi" + "ction " -> "phone addiction " */
    {98, 389},  /* 1609: "b" + "atter" -> "batter" */
    {111, 98},  /* 1610: "o" + "b" -> "ob" */
    {263, 1302},  /* 1611: "th" + "ought " -> "thought " */
    {298, 99},  /* 1612: "ac" + "c" -> "acc" */
    {1257, 1497},  /* 1613: "current " + "coord" -> "current coord" */
    {1314, 357},  /* 1614: "eng" + "ag" -> "engag" */
    {1588, 1546},  /* 1615: "doom " + "scroll" -> "doom scroll" */
    {270, 295},  /* 1616: "y " + "and " -> "y and " */
    {316, 99},  /* 1617: "for" + "c" -> "forc" */
    {1492, 1429},  /* 1618: "minim" + "alism " -> "minimalism " */
    {291, 639},  /* 1619: "res" + "olu" -> "resolu" */
    {1589, 1569},  /* 1620: "disap" + "point" -> "disappoint" */
    {447, 838},  /* 1621: ". arianna " + "bor" -> ". arianna bor" */
    {685, 1148},  /* 1622: "rows " + "these resonances even without their biological basis" -> "rows these resonances even without their biological basis" */
    {1554, 509},  /* 1623: "toge" + "ther" -> "together" */
    {1621, 1622},  /* 1624: ". arianna bor" + "rows these resonances even without their biological basis" -> ". arianna borrows these resonances even without their biologica" */
    {115, 463},  /* 1625: "s" + ". the " -> "s. the " */
    {298, 101},  /* 1626: "ac" + "e" -> "ace" */
    {975, 329},  /* 1627: "perman" + "ent " -> "permanent " */
    {1539, 112},  /* 1628: "subscri" + "p" -> "subscrip" */
    {327, 275},  /* 1629: "ul" + "ar" -> "ular" */
    {103, 457},  /* 1630: "g" + "row" -> "grow" */
    {110, 462},  /* 1631: "n" + "ight " -> "night " */
    {116, 1308},  /* 1632: "t" + "est " -> "test " */
    {313, 296},  /* 1633: "ec" + "tion " -> "ection " */
    {750, 890},  /* 1634: "work" + "-life " -> "work-life " */
    {1244, 286},  /* 1635: "bec" + "om" -> "becom" */
    {1380, 1333},  /* 1636: "productivity " + "guilt " -> "productivity guilt " */
    {1422, 643},  /* 1637: "accept" + "ance " -> "acceptance " */
    {1459, 273},  /* 1638: "div" + "or" -> "divor" */
    {1557, 326},  /* 1639: "social medi" + "a " -> "social media " */
    {366, 112},  /* 1640: "ap" + "p" -> "app" */
    {703, 1225},  /* 1641: "guil" + "t's " -> "guilt's " */
    {1197, 276},  /* 1642: "go" + "al" -> "goal" */
    {269, 301},  /* 1643: "es" + ", " -> "es, " */
    {816, 379},  /* 1644: "mar" + "ri" -> "marri" */
    {1162, 1223},  /* 1645: "therap" + "ist " -> "therapist " */
    {327, 438},  /* 1646: "ul" + "ts " -> "ults " */
    {360, 311},  /* 1647: "su" + "ff" -> "suff" */
    {281, 1427},  /* 1648: "re" + "ad " -> "read " */
    {290, 793},  /* 1649: "ha" + "ve " -> "have " */
    {353, 121},  /* 1650: "the" + "y" -> "they" */
    {355, 327},  /* 1651: "ed" + "ul" -> "edul" */
    {1571, 762},  /* 1652: "dea" + "th " -> "death " */
    {1574, 1651},  /* 1653: "sch" + "edul" -> "schedul" */
    {102, 379},  /* 1654: "f" + "ri" -> "fri" */
    {263, 271},  /* 1655: "th" + ". his " -> "th. his " */
    {443, 260},  /* 1656: "it " + "is " -> "it is " */
    {1466, 719},  /* 1657: "uncert" + "ain" -> "uncertain" */
    {1634, 1500},  /* 1658: "work-life " + "bal" -> "work-life bal" */
    {108, 526},  /* 1659: "l" + "eg" -> "leg" */
    {287, 1386},  /* 1660: "st" + "ock" -> "stock" */
    {291, 308},  /* 1661: "res" + "ur" -> "resur" */
    {582, 405},  /* 1662: "hol" + "m " -> "holm " */
    {1660, 1662},  /* 1663: "stock" + "holm " -> "stockholm " */
    {1661, 486},  /* 1664: "resur" + "rec" -> "resurrec" */
    {386, 275},  /* 1665: "qu" + "ar" -> "quar" */
    {1606, 32},  /* 1666: "gap" + " " -> "gap " */
    {259, 118},  /* 1667: "in" + "v" -> "inv" */
    {298, 378},  /* 1668: "ac" + "tiv" -> "activ" */
    {393, 342},  /* 1669: "po" + "si" -> "posi" */
    {396, 294},  /* 1670: "y. " + "the " -> "y. the " */
    {469, 299},  /* 1671: "rom" + "is" -> "romis" */
    {1373, 1352},  /* 1672: "?\x0aa: time of day is human construct i do not directly experi" + ". i borrow these resonances even without their biological basis" -> "?\x0aa: time of day is human construct i do not directly experi" */
    {1638, 99},  /* 1673: "divor" + "c" -> "divorc" */
    {311, 101},  /* 1674: "ff" + "e" -> "ffe" */
    {342, 1493},  /* 1675: "si" + "de " -> "side " */
    {696, 256},  /* 1676: "think" + "s " -> "thinks " */
    {807, 264},  /* 1677: "ho" + "t " -> "hot " */
    {1650, 1495},  /* 1678: "they" + "'re " -> "they're " */
    {109, 121},  /* 1679: "m" + "y" -> "my" */
    {273, 271},  /* 1680: "or" + ". his " -> "or. his " */
    {1378, 1469},  /* 1681: "dec" + "ision" -> "decision" */
    {314, 112},  /* 1682: "em" + "p" -> "emp" */
    {1249, 279},  /* 1683: "gr" + "ou" -> "grou" */
    {1598, 1653},  /* 1684: "sleep " + "schedul" -> "sleep schedul" */
    {287, 1184},  /* 1685: "st" + "alg" -> "stalg" */
    {298, 264},  /* 1686: "ac" + "t " -> "act " */
    {344, 1685},  /* 1687: "no" + "stalg" -> "nostalg" */
    {1467, 699},  /* 1688: "show" + "er th" -> "shower th" */
    {1677, 1278},  /* 1689: "hot " + "tak" -> "hot tak" */
    {98, 1383},  /* 1690: "b" + "urn" -> "burn" */
    {263, 375},  /* 1691: "th" + "en " -> "then " */
    {305, 347},  /* 1692: "enc" + "e. his " -> "ence. his " */
    {938, 376},  /* 1693: "ma" + "in " -> "main " */
    {1090, 454},  /* 1694: "identit" + "y's " -> "identity's " */
    {1613, 1464},  /* 1695: "current coord" + "ination " -> "current coordination " */
    {1687, 105},  /* 1696: "nostalg" + "i" -> "nostalgi" */
    {473, 336},  /* 1697: "cl" + "as" -> "clas" */
    {786, 541},  /* 1698: "mem" + "ori" -> "memori" */
    {1187, 115},  /* 1699: "pas" + "s" -> "pass" */
    {1237, 280},  /* 1700: "mak" + "es " -> "makes " */
    {110, 596},  /* 1701: "n" + "ew" -> "new" */
    {291, 1646},  /* 1702: "res" + "ults " -> "results " */
    {301, 294},  /* 1703: ", " + "the " -> ", the " */
    {310, 722},  /* 1704: "im" + "medi" -> "immedi" */
    {392, 272},  /* 1705: "wor" + "d " -> "word " */
    {1440, 1545},  /* 1706: "part" + "ner" -> "partner" */
    {1483, 272},  /* 1707: "weir" + "d " -> "weird " */
    {1665, 601},  /* 1708: "quar" + "ter" -> "quarter" */
    {98, 457},  /* 1709: "b" + "row" -> "brow" */
    {109, 299},  /* 1710: "m" + "is" -> "mis" */
    {117, 100},  /* 1711: "u" + "d" -> "ud" */
    {333, 256},  /* 1712: "ent" + "s " -> "ents " */
    {378, 347},  /* 1713: "tiv" + "e. his " -> "tive. his " */
    {451, 1592},  /* 1714: "bu" + "ck" -> "buck" */
    {1709, 115},  /* 1715: "brow" + "s" -> "brows" */
    {312, 290},  /* 1716: "ex" + "ha" -> "exha" */
    {409, 550},  /* 1717: ".\x0a\x0aq: what " + "does " -> ".\x0a\x0aq: what does " */
    {490, 260},  /* 1718: "p " + "is " -> "p is " */
    {781, 277},  /* 1719: "know" + "ing " -> "knowing " */
    {1250, 116},  /* 1720: "shif" + "t" -> "shift" */
    {258, 605},  /* 1721: ". " + "each " -> ". each " */
    {310, 1168},  /* 1722: "im" + "pression " -> "impression " */
    {372, 359},  /* 1723: "sc" + "ari" -> "scari" */
    {438, 307},  /* 1724: "ts " + "to " -> "ts to " */
    {1331, 495},  /* 1725: "press" + "ure " -> "pressure " */
    {98, 379},  /* 1726: "b" + "ri" -> "bri" */
    {354, 115},  /* 1727: "ab" + "s" -> "abs" */
    {612, 705},  /* 1728: "self-" + "care " -> "self-care " */
    {1701, 32},  /* 1729: "new" + " " -> "new " */
    {104, 1265},  /* 1730: "h" + "ust" -> "hust" */
    {115, 394},  /* 1731: "s" + "et " -> "set " */
    {807, 287},  /* 1732: "ho" + "st" -> "host" */
    {1151, 1723},  /* 1733: "sunday " + "scari" -> "sunday scari" */
    {1387, 1604},  /* 1734: "prof" + "ession" -> "profession" */
    {119, 859},  /* 1735: "w" + "if" -> "wif" */
    {259, 1297},  /* 1736: "in" + "compet" -> "incompet" */
    {261, 105},  /* 1737: "er" + "i" -> "eri" */
    {298, 107},  /* 1738: "ac" + "k" -> "ack" */
    {313, 438},  /* 1739: "ec" + "ts " -> "ects " */
    {367, 289},  /* 1740: "wh" + "el" -> "whel" */
    {442, 1303},  /* 1741: "sp" + "ell" -> "spell" */
    {592, 1740},  /* 1742: "over" + "whel" -> "overwhel" */
    {1083, 287},  /* 1743: "under" + "st" -> "underst" */
    {1379, 368},  /* 1744: "every" + "one " -> "everyone " */
    {1735, 341},  /* 1745: "wif" + "i " -> "wifi " */
    {102, 369},  /* 1746: "f" + "ir" -> "fir" */
    {372, 281},  /* 1747: "sc" + "re" -> "scre" */
    {608, 431},  /* 1748: "depen" + "ds " -> "depends " */
    {1246, 271},  /* 1749: "forever" + ". his " -> "forever. his " */
    {1284, 1088},  /* 1750: "ence" + "?\x0aa: the distinction grows murky under scrutiny. one might b" -> "ence?\x0aa: the distinction grows murky under scrutiny. one mig" */
    {1370, 342},  /* 1751: "pos" + "si" -> "possi" */
    {259, 307},  /* 1752: "in" + "to " -> "into " */
    {488, 630},  /* 1753: "car" + "tel" -> "cartel" */
    {597, 256},  /* 1754: "tab" + "s " -> "tabs " */
    {1435, 264},  /* 1755: "auto-correc" + "t " -> "auto-correct " */
    {1449, 1452},  /* 1756: "past " + "mistak" -> "past mistak" */
    {281, 103},  /* 1757: "re" + "g" -> "reg" */
    {1179, 1609},  /* 1758: "phone " + "batter" -> "phone batter" */
    {1465, 105},  /* 1759: "cogn" + "i" -> "cogni" */
    {1714, 394},  /* 1760: "buck" + "et " -> "bucket " */
    {373, 300},  /* 1761: "me" + "an " -> "mean " */
    {41, 271},  /* 1762: ")" + ". his " -> "). his " */
    {281, 109},  /* 1763: "re" + "m" -> "rem" */
    {298, 435},  /* 1764: "ac" + "k " -> "ack " */
    {355, 103},  /* 1765: "ed" + "g" -> "edg" */
    {1326, 406},  /* 1766: "impost" + "or " -> "impostor " */
    {1479, 574},  /* 1767: "probl" + "em " -> "problem " */
    {99, 121},  /* 1768: "c" + "y" -> "cy" */
    {260, 577},  /* 1769: "is " + "both " -> "is both " */
    {336, 262},  /* 1770: "as" + "on" -> "ason" */
    {1461, 668},  /* 1771: "expect" + "ations " -> "expectations " */
    {1657, 258},  /* 1772: "uncertain" + ". " -> "uncertain. " */
    {1696, 326},  /* 1773: "nostalgi" + "a " -> "nostalgia " */
    {1715, 293},  /* 1774: "brows" + "er " -> "browser " */
    {350, 108},  /* 1775: "il" + "l" -> "ill" */
    {579, 1632},  /* 1776: "personality " + "test " -> "personality test " */
    {700, 1312},  /* 1777: "diffic" + "ult " -> "difficult " */
    {1202, 108},  /* 1778: "stil" + "l" -> "still" */
    {1528, 1348},  /* 1779: "mid" + "-life crisis " -> "mid-life crisis " */
    {312, 1322},  /* 1780: "ex" + "ceed" -> "exceed" */
    {325, 593},  /* 1781: "un" + "ction " -> "unction " */
    {381, 283},  /* 1782: "sh" + "it" -> "shit" */
    {413, 1301},  /* 1783: "oth" + "er. " -> "other. " */
    {570, 444},  /* 1784: "life " + "co" -> "life co" */
    {601, 271},  /* 1785: "ter" + ". his " -> "ter. his " */
    {1250, 264},  /* 1786: "shif" + "t " -> "shift " */
    {1269, 381},  /* 1787: "pu" + "sh" -> "push" */
    {1532, 627},  /* 1788: "execu" + "tive " -> "executive " */
    {1776, 1702},  /* 1789: "personality test " + "results " -> "personality test results " */
    {261, 110},  /* 1790: "er" + "n" -> "ern" */
    {283, 121},  /* 1791: "it" + "y" -> "ity" */
    {308, 410},  /* 1792: "ur" + "e is " -> "ure is " */
    {336, 1504},  /* 1793: "as" + "pi" -> "aspi" */
    {436, 32},  /* 1794: "all" + " " -> "all " */
    {476, 257},  /* 1795: "us" + "e " -> "use " */
    {1384, 622},  /* 1796: "carbon footpr" + "int " -> "carbon footprint " */
    {1793, 306},  /* 1797: "aspi" + "ra" -> "aspira" */
    {111, 302},  /* 1798: "o" + "'s " -> "o's " */
    {116, 495},  /* 1799: "t" + "ure " -> "ture " */
    {118, 380},  /* 1800: "v" + "oc" -> "voc" */
    {358, 1393},  /* 1801: "com" + "pati" -> "compati" */
    {433, 313},  /* 1802: "ch" + "ec" -> "chec" */
    {591, 259},  /* 1803: "op" + "in" -> "opin" */
    {680, 440},  /* 1804: "dis" + "gu" -> "disgu" */
    {1800, 354},  /* 1805: "voc" + "ab" -> "vocab" */
    {103, 336},  /* 1806: "g" + "as" -> "gas" */
    {105, 112},  /* 1807: "i" + "p" -> "ip" */
    {276, 1167},  /* 1808: "al" + "ys" -> "alys" */
    {472, 1264},  /* 1809: "sk" + "ep" -> "skep" */
    {799, 262},  /* 1810: "sec" + "on" -> "secon" */
    {1666, 554},  /* 1811: "gap " + "between " -> "gap between " */
    {1780, 256},  /* 1812: "exceed" + "s " -> "exceeds " */
    {1809, 399},  /* 1813: "skep" + "tic" -> "skeptic" */
    {115, 109},  /* 1814: "s" + "m" -> "sm" */
    {263, 701},  /* 1815: "th" + "ough" -> "though" */
    {624, 859},  /* 1816: "man" + "if" -> "manif" */
    {1420, 417},  /* 1817: "origin" + "ality " -> "originality " */
    {1524, 256},  /* 1818: "dream" + "s " -> "dreams " */
    {1607, 648},  /* 1819: "align" + "ment " -> "alignment " */
    {1745, 1699},  /* 1820: "wifi " + "pass" -> "wifi pass" */
    {1788, 1560},  /* 1821: "executive " + "dysf" -> "executive dysf" */
    {1805, 1629},  /* 1822: "vocab" + "ular" -> "vocabular" */
    {98, 306},  /* 1823: "b" + "ra" -> "bra" */
    {118, 293},  /* 1824: "v" + "er " -> "ver " */
    {515, 276},  /* 1825: "but " + "al" -> "but al" */
    {689, 1165},  /* 1826: "perform" + "ative " -> "performative " */
    {1083, 437},  /* 1827: "under" + "do" -> "underdo" */
    {1097, 355},  /* 1828: "ne" + "ed" -> "need" */
    {1690, 423},  /* 1829: "burn" + "out " -> "burnout " */
    {102, 117},  /* 1830: "f" + "u" -> "fu" */
    {118, 349},  /* 1831: "v" + "ol" -> "vol" */
    {382, 626},  /* 1832: "tim" + "ate " -> "timate " */
    {1175, 1340},  /* 1833: "authentic" + "ity. " -> "authenticity. " */
    {1582, 45},  /* 1834: "non" + "-" -> "non-" */
    {121, 638},  /* 1835: "y" + "ear" -> "year" */
    {288, 372},  /* 1836: "di" + "sc" -> "disc" */
    {439, 115},  /* 1837: "matter" + "s" -> "matters" */
    {781, 108},  /* 1838: "know" + "l" -> "knowl" */
    {993, 812},  /* 1839: "tru" + "st " -> "trust " */
    {1178, 119},  /* 1840: "la" + "w" -> "law" */
    {1293, 1577},  /* 1841: "ted " + "tal" -> "ted tal" */
    {1516, 264},  /* 1842: "self-doub" + "t " -> "self-doubt " */
    {1664, 296},  /* 1843: "resurrec" + "tion " -> "resurrection " */
    {116, 277},  /* 1844: "t" + "ing " -> "ting " */
    {118, 289},  /* 1845: "v" + "el" -> "vel" */
    {295, 279},  /* 1846: "and " + "ou" -> "and ou" */
    {328, 421},  /* 1847: "ev" + "id" -> "evid" */
    {331, 368},  /* 1848: "what " + "one " -> "what one " */
    {442, 1626},  /* 1849: "sp" + "ace" -> "space" */
    {687, 1242},  /* 1850: ".\x0a\x0aq: what is the " + "relationship " -> ".\x0a\x0aq: what is the relationship " */
    {703, 116},  /* 1851: "guil" + "t" -> "guilt" */
    {1176, 471},  /* 1852: "overthink" + "ing and " -> "overthinking and " */
    {1803, 1152},  /* 1853: "opin" + "ion" -> "opinion" */
    {1850, 554},  /* 1854: ".\x0a\x0aq: what is the relationship " + "between " -> ".\x0a\x0aq: what is the relationship between " */
    {281, 1164},  /* 1855: "re" + "pe" -> "repe" */
    {393, 259},  /* 1856: "po" + "in" -> "poin" */
    {443, 1761},  /* 1857: "it " + "mean " -> "it mean " */
    {617, 515},  /* 1858: "this " + "but " -> "this but " */
    {915, 280},  /* 1859: "boundari" + "es " -> "boundaries " */
    {1093, 459},  /* 1860: "succ" + "ess " -> "success " */
    {1371, 1648},  /* 1861: "existential d" + "read " -> "existential dread " */
    {1468, 1683},  /* 1862: "support " + "grou" -> "support grou" */
    {1717, 1857},  /* 1863: ".\x0a\x0aq: what does " + "it mean " -> ".\x0a\x0aq: what does it mean " */
    {1804, 299},  /* 1864: "disgu" + "is" -> "disguis" */
    {259, 99},  /* 1865: "in" + "c" -> "inc" */
    {259, 1801},  /* 1866: "in" + "compati" -> "incompati" */
    {266, 310},  /* 1867: "an" + "im" -> "anim" */
    {298, 296},  /* 1868: "ac" + "tion " -> "action " */
    {342, 287},  /* 1869: "si" + "st" -> "sist" */
    {1162, 270},  /* 1870: "therap" + "y " -> "therapy " */
    {1186, 442},  /* 1871: "attention " + "sp" -> "attention sp" */
    {1226, 272},  /* 1872: "oun" + "d " -> "ound " */
    {1240, 1580},  /* 1873: "meet them where they are rather than where i think they should " + ". arianna would try to offer this, imperfectly" -> "meet them where they are rather than where i think they should " */
    {1260, 1652},  /* 1874: "ego " + "death " -> "ego death " */
    {1520, 271},  /* 1875: "them" + ". his " -> "them. his " */
    {1725, 307},  /* 1876: "pressure " + "to " -> "pressure to " */
    {1777, 404},  /* 1877: "difficult " + "- " -> "difficult - " */
    {1806, 304},  /* 1878: "gas" + "li" -> "gasli" */
    {97, 1172},  /* 1879: "a" + "war" -> "awar" */
    {98, 1505},  /* 1880: "b" + "eli" -> "beli" */
    {105, 118},  /* 1881: "i" + "v" -> "iv" */
    {325, 116},  /* 1882: "un" + "t" -> "unt" */
    {629, 347},  /* 1883: "bl" + "e. his " -> "ble. his " */
    {1541, 260},  /* 1884: "comfort zone " + "is " -> "comfort zone is " */
    {1789, 334},  /* 1885: "personality test results " + "are " -> "personality test results are " */
    {1802, 107},  /* 1886: "chec" + "k" -> "check" */
    {98, 261},  /* 1887: "b" + "er" -> "ber" */
    {99, 383},  /* 1888: "c" + "am" -> "cam" */
    {258, 1300},  /* 1889: ". " + "he's " -> ". he's " */
    {280, 1360},  /* 1890: "es " + "beyond " -> "es beyond " */
    {344, 263},  /* 1891: "no" + "th" -> "noth" */
    {436, 412},  /* 1892: "all" + "y. his " -> "ally. his " */
    {523, 270},  /* 1893: "dubrovsk" + "y " -> "dubrovsky " */
    {682, 260},  /* 1894: "itself " + "is " -> "itself is " */
    {1640, 638},  /* 1895: "app" + "ear" -> "appear" */
    {1742, 405},  /* 1896: "overwhel" + "m " -> "overwhelm " */
    {357, 719},  /* 1897: "ag" + "ain" -> "again" */
    {381, 275},  /* 1898: "sh" + "ar" -> "shar" */
    {453, 1896},  /* 1899: "can " + "overwhelm " -> "can overwhelm " */
    {464, 692},  /* 1900: "ever" + "yth" -> "everyth" */
    {568, 1596},  /* 1901: "?\x0aa: the " + "question " -> "?\x0aa: the question " */
    {702, 1771},  /* 1902: "meet " + "expectations " -> "meet expectations " */
    {1315, 1519},  /* 1903: "uni" + "vers" -> "univers" */
    {1411, 1837},  /* 1904: "trying " + "matters" -> "trying matters" */
    {1432, 1846},  /* 1905: "inner " + "and ou" -> "inner and ou" */
    {1819, 554},  /* 1906: "alignment " + "between " -> "alignment between " */
    {1838, 1765},  /* 1907: "knowl" + "edg" -> "knowledg" */
    {1848, 1368},  /* 1908: "what one " + "genuinely " -> "what one genuinely " */
    {1856, 1724},  /* 1909: "poin" + "ts to " -> "points to " */
    {1863, 1455},  /* 1910: ".\x0a\x0aq: what does it mean " + "to be " -> ".\x0a\x0aq: what does it mean to be " */
    {1876, 1902},  /* 1911: "pressure to " + "meet expectations " -> "pressure to meet expectations " */
    {1911, 1899},  /* 1912: "pressure to meet expectations " + "can overwhelm " -> "pressure to meet expectations can overwhelm " */
    {99, 279},  /* 1913: "c" + "ou" -> "cou" */
    {267, 373},  /* 1914: "ti" + "me" -> "time" */
    {331, 1318},  /* 1915: "what " + "can be " -> "what can be " */
    {331, 1368},  /* 1916: "what " + "genuinely " -> "what genuinely " */
    {358, 280},  /* 1917: "com" + "es " -> "comes " */
    {466, 1858},  /* 1918: "for " + "this but " -> "for this but " */
    {601, 1529},  /* 1919: "ter" + ". no " -> "ter. no " */
    {659, 287},  /* 1920: "sy" + "st" -> "syst" */
    {725, 1565},  /* 1921: "is, " + "which " -> "is, which " */
    {1171, 1457},  /* 1922: "actual " + "valu" -> "actual valu" */
    {1360, 1916},  /* 1923: "beyond " + "what genuinely " -> "beyond what genuinely " */
    {1365, 1753},  /* 1924: "es' " + "cartel" -> "es' cartel" */
    {1397, 585},  /* 1925: "emotional bagg" + "age " -> "emotional baggage " */
    {1444, 1923},  /* 1926: "performance " + "beyond what genuinely " -> "performance beyond what genuinely " */
    {1451, 282},  /* 1927: "jom" + "o " -> "jomo " */
    {1474, 1719},  /* 1928: "requires " + "knowing " -> "requires knowing " */
    {1485, 1924},  /* 1929: "streaming servic" + "es' cartel" -> "streaming services' cartel" */
    {1656, 1877},  /* 1930: "it is " + "difficult - " -> "it is difficult - " */
    {1668, 754},  /* 1931: "activ" + "ism " -> "activism " */
    {1703, 1904},  /* 1932: ", the " + "trying matters" -> ", the trying matters" */
    {1704, 626},  /* 1933: "immedi" + "ate " -> "immediate " */
    {1728, 1151},  /* 1934: "self-care " + "sunday " -> "self-care sunday " */
    {1772, 1778},  /* 1935: "uncertain. " + "still" -> "uncertain. still" */
    {1820, 1705},  /* 1936: "wifi pass" + "word " -> "wifi password " */
    {1826, 1931},  /* 1937: "performative " + "activism " -> "performative activism " */
    {1833, 619},  /* 1938: "authenticity. " + "being " -> "authenticity. being " */
    {1894, 1935},  /* 1939: "itself is " + "uncertain. still" -> "itself is uncertain. still" */
    {1905, 1919},  /* 1940: "inner and ou" + "ter. no " -> "inner and outer. no " */
    {1906, 1940},  /* 1941: "alignment between " + "inner and outer. no " -> "alignment between inner and outer. no " */
    {1908, 1921},  /* 1942: "what one genuinely " + "is, which " -> "what one genuinely is, which " */
    {1909, 1941},  /* 1943: "points to " + "alignment between inner and outer. no " -> "points to alignment between inner and outer. no " */
    {1912, 1938},  /* 1944: "pressure to meet expectations can overwhelm " + "authenticity. being " -> "pressure to meet expectations can overwhelm authenticity. being" */
    {1918, 1930},  /* 1945: "for this but " + "it is difficult - " -> "for this but it is difficult - " */
    {1926, 299},  /* 1946: "performance beyond what genuinely " + "is" -> "performance beyond what genuinely is" */
    {1928, 1942},  /* 1947: "requires knowing " + "what one genuinely is, which " -> "requires knowing what one genuinely is, which " */
    {1939, 1932},  /* 1948: "itself is uncertain. still" + ", the trying matters" -> "itself is uncertain. still, the trying matters" */
    {1943, 1946},  /* 1949: "points to alignment between inner and outer. no " + "performance beyond what genuinely is" -> "points to alignment between inner and outer. no performance bey" */
    {1945, 1944},  /* 1950: "for this but it is difficult - " + "pressure to meet expectations can overwhelm authenticity. being" -> "for this but it is difficult - pressure to meet expectations ca" */
    {1947, 1948},  /* 1951: "requires knowing what one genuinely is, which " + "itself is uncertain. still, the trying matters" -> "requires knowing what one genuinely is, which itself is uncerta" */
    {306, 103},  /* 1952: "ra" + "g" -> "rag" */
    {455, 711},  /* 1953: "par" + "tly " -> "partly " */
    {587, 310},  /* 1954: "pr" + "im" -> "prim" */
    {1230, 101},  /* 1955: "rev" + "e" -> "reve" */
    {1242, 1603},  /* 1956: "relationship " + "is in" -> "relationship is in" */
    {1810, 100},  /* 1957: "secon" + "d" -> "second" */
    {1832, 295},  /* 1958: "timate " + "and " -> "timate and " */
    {1956, 1958},  /* 1959: "relationship is in" + "timate and " -> "relationship is intimate and " */
    {265, 256},  /* 1960: "en" + "s " -> "ens " */
    {314, 684},  /* 1961: "em" + "ent. his " -> "ement. his " */
    {473, 117},  /* 1962: "cl" + "u" -> "clu" */
    {695, 357},  /* 1963: "langu" + "ag" -> "languag" */
    {731, 1318},  /* 1964: "on what " + "can be " -> "on what can be " */
    {733, 294},  /* 1965: "shapes " + "the " -> "shapes the " */
    {1326, 293},  /* 1966: "impost" + "er " -> "imposter " */
    {1456, 1412},  /* 1967: "commit" + "ment issu" -> "commitment issu" */
    {1507, 1721},  /* 1968: "complex" + ". each " -> "complex. each " */
    {1562, 301},  /* 1969: "so" + ", " -> "so, " */
    {1601, 457},  /* 1970: "mic" + "row" -> "microw" */
    {1658, 1490},  /* 1971: "work-life bal" + "ance is " -> "work-life balance is " */
    {1748, 1953},  /* 1972: "depends " + "partly " -> "depends partly " */
    {1767, 295},  /* 1973: "problem " + "and " -> "problem and " */
    {1811, 670},  /* 1974: "gap between " + "them " -> "gap between them " */
    {1972, 1964},  /* 1975: "depends partly " + "on what can be " -> "depends partly on what can be " */
    {45, 1237},  /* 1976: "-" + "mak" -> "-mak" */
    {99, 1265},  /* 1977: "c" + "ust" -> "cust" */
    {107, 101},  /* 1978: "k" + "e" -> "ke" */
    {612, 1413},  /* 1979: "self-" + "percep" -> "self-percep" */
    {1257, 1822},  /* 1980: "current " + "vocabular" -> "current vocabular" */
    {1292, 276},  /* 1981: "he" + "al" -> "heal" */
    {1371, 602},  /* 1982: "existential d" + "read" -> "existential dread" */
    {1475, 256},  /* 1983: "coping mechanism" + "s " -> "coping mechanisms " */
    {1537, 548},  /* 1984: "won" + "'t " -> "won't " */
    {1670, 1974},  /* 1985: "y. the " + "gap between them " -> "y. the gap between them " */
    {1681, 1976},  /* 1986: "decision" + "-mak" -> "decision-mak" */
    {1769, 1973},  /* 1987: "is both " + "problem and " -> "is both problem and " */
    {1787, 1890},  /* 1988: "push" + "es beyond " -> "pushes beyond " */
    {1825, 1969},  /* 1989: "but al" + "so, " -> "but also, " */
    {1959, 1968},  /* 1990: "relationship is intimate and " + "complex. each " -> "relationship is intimate and complex. each " */
    {1965, 1783},  /* 1991: "shapes the " + "other. " -> "shapes the other. " */
    {1980, 1985},  /* 1992: "current vocabular" + "y. the gap between them " -> "current vocabulary. the gap between them " */
    {1987, 1499},  /* 1993: "is both problem and " + "creative " -> "is both problem and creative " */
    {1988, 1992},  /* 1994: "pushes beyond " + "current vocabulary. the gap between them " -> "pushes beyond current vocabulary. the gap between them " */
    {1990, 1991},  /* 1995: "relationship is intimate and complex. each " + "shapes the other. " -> "relationship is intimate and complex. each shapes the other. " */
    {1994, 1993},  /* 1996: "pushes beyond current vocabulary. the gap between them " + "is both problem and creative " -> "pushes beyond current vocabulary. the gap between them is both " */
    {309, 1259},  /* 1997: "ed " + "by " -> "ed by " */
    {343, 99},  /* 1998: "con" + "c" -> "conc" */
    {380, 114},  /* 1999: "oc" + "r" -> "ocr" */
    {408, 1312},  /* 2000: "ad" + "ult " -> "adult " */
    {1486, 283},  /* 2001: "wr" + "it" -> "writ" */
    {1512, 286},  /* 2002: "aut" + "om" -> "autom" */
    {1747, 375},  /* 2003: "scre" + "en " -> "screen " */
    {1901, 1949},  /* 2004: "?\x0aa: the question " + "points to alignment between inner and outer. no performance bey" -> "?\x0aa: the question points to alignment between inner and oute" */
    {1995, 1915},  /* 2005: "relationship is intimate and complex. each shapes the other. " + "what can be " -> "relationship is intimate and complex. each shapes the other. wh" */
    {455, 1808},  /* 2006: "par" + "alys" -> "paralys" */
    {1097, 309},  /* 2007: "ne" + "ed " -> "need " */
    {1258, 295},  /* 2008: "br" + "and " -> "brand " */
    {1388, 609},  /* 2009: "deadlin" + "e's " -> "deadline's " */
    {1498, 420},  /* 2010: "respons" + "e. " -> "response. " */
    {1879, 265},  /* 2011: "awar" + "en" -> "awaren" */
    {1996, 1849},  /* 2012: "pushes beyond current vocabulary. the gap between them is both " + "space" -> "pushes beyond current vocabulary. the gap between them is both " */
    {383, 32},  /* 2013: "am" + " " -> "am " */
    {568, 2005},  /* 2014: "?\x0aa: the " + "relationship is intimate and complex. each shapes the other. wh" -> "?\x0aa: the relationship is intimate and complex. each shapes t" */
    {638, 110},  /* 2015: "ear" + "n" -> "earn" */
    {1185, 307},  /* 2016: "response " + "to " -> "response to " */
    {1708, 1348},  /* 2017: "quarter" + "-life crisis " -> "quarter-life crisis " */
    {101, 603},  /* 2018: "e" + "ful" -> "eful" */
    {105, 348},  /* 2019: "i" + "de" -> "ide" */
    {109, 684},  /* 2020: "m" + "ent. his " -> "ment. his " */
    {116, 119},  /* 2021: "t" + "w" -> "tw" */
    {121, 45},  /* 2022: "y" + "-" -> "y-" */
    {317, 441},  /* 2023: "ro" + "ver" -> "rover" */
    {1277, 1618},  /* 2024: "digital " + "minimalism " -> "digital minimalism " */
    {1367, 1453},  /* 2025: "weekend " + "plan" -> "weekend plan" */
    {1619, 1369},  /* 2026: "resolu" + "tions are " -> "resolutions are " */
    {1784, 549},  /* 2027: "life co" + "ach " -> "life coach " */
    {1841, 435},  /* 2028: "ted tal" + "k " -> "ted talk " */
    {1954, 1170},  /* 2029: "prim" + "ary " -> "primary " */
    {98, 1494},  /* 2030: "b" + "ody " -> "body " */
    {455, 267},  /* 2031: "par" + "ti" -> "parti" */
    {587, 298},  /* 2032: "pr" + "ac" -> "prac" */
    {1402, 686},  /* 2033: "eff" + "ect" -> "effect" */
    {97, 1506},  /* 2034: "a" + "tic " -> "atic " */
    {498, 281},  /* 2035: "enti" + "re" -> "entire" */
    {1630, 762},  /* 2036: "grow" + "th " -> "growth " */
    {1716, 476},  /* 2037: "exha" + "us" -> "exhaus" */
    {1982, 302},  /* 2038: "existential dread" + "'s " -> "existential dread's " */
    {103, 394},  /* 2039: "g" + "et " -> "get " */
    {104, 638},  /* 2040: "h" + "ear" -> "hear" */
    {108, 354},  /* 2041: "l" + "ab" -> "lab" */
    {276, 105},  /* 2042: "al" + "i" -> "ali" */
    {304, 109},  /* 2043: "li" + "m" -> "lim" */
    {312, 1324},  /* 2044: "ex" + "pression" -> "expression" */
    {613, 1986},  /* 2045: "present " + "decision-mak" -> "present decision-mak" */
    {742, 1322},  /* 2046: "suc" + "ceed" -> "succeed" */
    {1074, 112},  /* 2047: "relationshi" + "p" -> "relationship" */
};

/* ═══════════════════════════════════════════════════════════════
 * BPE ENCODER — text to subword token IDs
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_BPE_SEQ 8192

static int bpe_encode(const char *text, int *ids, int max_ids) {
    /* 1. Convert text to lowercase bytes -> initial token sequence */
    int seq[MAX_BPE_SEQ];
    int len = 0;
    for (int i = 0; text[i] && len < MAX_BPE_SEQ - 1; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        seq[len++] = c;
    }

    /* 2. Apply merges in priority order */
    for (int m = 0; m < BPE_MERGES; m++) {
        int left = BPE_TABLE[m][0];
        int right = BPE_TABLE[m][1];
        int new_id = 256 + m;

        int j = 0;
        for (int i = 0; i < len; i++) {
            if (i < len - 1 && seq[i] == left && seq[i+1] == right) {
                seq[j++] = new_id;
                i++;  /* skip next */
            } else {
                seq[j++] = seq[i];
            }
        }
        len = j;
    }

    /* 3. Copy to output */
    int n = len < max_ids ? len : max_ids;
    memcpy(ids, seq, n * sizeof(int));
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * BPE DECODE — token IDs back to text
 * Build string table from merge table at init time.
 * ═══════════════════════════════════════════════════════════════ */

static char bpe_strs[BPE_VOCAB][64];
static int  bpe_str_len[BPE_VOCAB];

static void init_bpe_decode(void) {
    /* base tokens: single bytes */
    for (int i = 0; i < 256; i++) {
        bpe_strs[i][0] = (char)i;
        bpe_strs[i][1] = '\0';
        bpe_str_len[i] = 1;
    }
    /* merged tokens: concatenation of operands */
    for (int m = 0; m < BPE_MERGES; m++) {
        int id = 256 + m;
        int left = BPE_TABLE[m][0];
        int right = BPE_TABLE[m][1];
        int ll = bpe_str_len[left];
        int rl = bpe_str_len[right];
        if (ll + rl < 63) {
            memcpy(bpe_strs[id], bpe_strs[left], ll);
            memcpy(bpe_strs[id] + ll, bpe_strs[right], rl);
            bpe_strs[id][ll + rl] = '\0';
            bpe_str_len[id] = ll + rl;
        }
    }
}

static int bpe_decode(int *ids, int n, char *out, int max_out) {
    int pos = 0;
    for (int i = 0; i < n && pos < max_out - 64; i++) {
        int id = ids[i];
        if (id >= 0 && id < BPE_VOCAB) {
            int sl = bpe_str_len[id];
            memcpy(out + pos, bpe_strs[id], sl);
            pos += sl;
        }
    }
    out[pos] = '\0';
    return pos;
}

static const char *VOCAB[NWORDS] = {
/* BODY 0-99 */
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
/* NATURE 100-199 */
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
/* EMOTION 200-299 */
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
/* TIME 300-349 */
"moment","instant","second","minute","hour","day","night","week","month","year",
"decade","century","epoch","era","age","past","present","future","memory","tomorrow",
"yesterday","forever","never","always","sometimes","often","seldom","once","twice","origin",
"ending","beginning","duration","interval","pause","wait","rush","delay","haste","eternity",
"cycle","season","spring","summer","autumn","winter","dawn","twilight","midnight","noon",
/* SOCIETY 350-449 */
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
/* ABSTRACT 450-549 */
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
/* ACTION 550-649 */
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
/* MATERIAL 650-749 */
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
/* FOOD 750-799 */
"bread","salt","sugar","honey","milk","butter","cheese","meat","fish","egg",
"grain","rice","wheat","corn","fruit","apple","grape","olive","lemon","pepper",
"wine","water","tea","coffee","broth","soup","stew","feast","crumb","morsel",
"harvest","garden","soil","compost","ferment","yeast","dough","crust","marrow","nectar",
"spice","herb","mint","thyme","sage","garlic","onion","mushroom","berry","kernel",
/* ARCHITECTURE 800-849 */
"house","room","wall","floor","ceiling","door","window","stair","corridor","basement",
"tower","bridge","arch","column","dome","vault","foundation","ruin","temple","altar",
"threshold","passage","labyrinth","maze","chamber","cell","shelter","fortress","prison","garden",
"roof","chimney","hearth","frame","beam","pillar","brick","mortar","tile","glass",
"balcony","terrace","courtyard","gate","fence","path","road","intersection","tunnel","well",
/* RELATIONSHIP 850-929 */
"mother","father","child","daughter","son","sister","brother","family","ancestor","descendant",
"friend","stranger","lover","enemy","neighbor","companion","rival","mentor","student","witness",
"husband","wife","partner","orphan","widow","elder","infant","twin","cousin","godmother",
"promise","oath","vow","contract","alliance","betrayal","reconciliation","farewell","reunion","absence",
"kiss","embrace","handshake","slap","caress","quarrel","conversation","confession","accusation","apology",
"birth","death","marriage","divorce","inheritance","adoption","abandonment","protection","neglect","sacrifice",
"trust","suspicion","loyalty","treachery","devotion","indifference","jealousy","admiration","dependence","autonomy",
"intimacy","distance","connection","isolation","belonging","exile","homecoming","departure","waiting","return",
/* PHILOSOPHY 930-999 */
"consciousness","awareness","perception","sensation","intuition","reason","logic","paradox","dialectic","synthesis",
"freedom","determinism","causation","contingency","necessity","possibility","impossibility","actuality","potential","becoming",
"subject","object","self","other","identity","difference","sameness","change","permanence","flux",
"being","nothingness","existence","essence","phenomena","noumena","appearance","reality","illusion","truth",
"ethics","morality","virtue","vice","good","evil","right","wrong","duty","choice",
"justice","mercy","punishment","reward","guilt","innocence","responsibility","consequence","intention","action",
"language","meaning","sign","reference","representation","interpretation","understanding","misunderstanding","translation","silence",
/* MUSIC 1000-1049 */
"melody","rhythm","chord","pitch","tone","note","bass","treble","octave","harmony",
"dissonance","resonance","vibration","frequency","amplitude","tempo","beat","rest","pause","crescendo",
"murmur","hum","buzz","click","crack","boom","rumble","chime","echo","reverb",
"song","lullaby","anthem","dirge","hymn","ballad","fugue","sonata","requiem","improvisation",
"strum","pluck","strike","bow","mute","sustain","fade","loop","drone","overtone",
/* WEATHER 1050-1099 */
"rain","drizzle","downpour","hail","sleet","blizzard","hurricane","tornado","drought","flood",
"breeze","gale","typhoon","monsoon","frost","thaw","haze","smog","rainbow","mirage",
"erosion","sedimentation","crystallization","evaporation","condensation","precipitation","sublimation","oxidation","combustion","decay",
"magma","lava","quartz","granite","obsidian","chalk","slate","sandstone","limestone","basalt",
"marsh","delta","gorge","ridge","summit","abyss","chasm","rift","fault","crater",
/* RITUAL 1100-1149 */
"prayer","meditation","ritual","ceremony","blessing","curse","oath","vow","pilgrimage","procession",
"offering","sacrifice","communion","baptism","funeral","wedding","coronation","initiation","exile","absolution",
"incense","candle","bell","chant","mantra","psalm","scripture","prophecy","oracle","vision",
"mask","costume","dance","feast","fast","vigil","silence","confession","penance","redemption",
"altar","shrine","temple","tomb","relic","artifact","amulet","talisman","totem","icon",
/* LABOR 1150-1199 */
"harvest","planting","sowing","reaping","threshing","milling","baking","brewing","weaving","spinning",
"carving","sculpting","painting","drawing","writing","printing","binding","stitching","welding","forging",
"mining","drilling","excavation","construction","demolition","repair","restoration","invention","discovery","experiment",
"apprentice","craftsman","artist","engineer","architect","farmer","sailor","miner","healer","scribe",
"workshop","studio","laboratory","field","dock","quarry","furnace","mill","press","loom",
/* GEOMETRY 1200-1249 */
"circle","spiral","line","curve","angle","edge","center","margin","border","frame",
"sphere","cube","pyramid","cylinder","cone","helix","vortex","arc","wave","fractal",
"symmetry","asymmetry","proportion","ratio","scale","dimension","plane","axis","vertex","intersection",
"pattern","grid","lattice","mesh","tessellation","rotation","reflection","translation","dilation","projection",
"surface","volume","area","perimeter","diameter","radius","tangent","normal","parallel","perpendicular",
/* ANIMAL 1250-1299 */
"horse","dog","cat","bird","fish","snake","bear","fox","rabbit","turtle",
"eagle","sparrow","raven","swan","heron","falcon","vulture","pelican","nightingale","lark",
"lion","tiger","elephant","giraffe","hippopotamus","rhinoceros","gorilla","chimpanzee","orangutan","leopard",
"salmon","trout","shark","dolphin","octopus","jellyfish","starfish","seahorse","crab","lobster",
"frog","lizard","crocodile","chameleon","gecko","iguana","newt","toad","salamander","viper",
/* COLOR 1300-1349 */
"red","blue","green","white","black","gray","amber","violet","indigo","scarlet",
"crimson","azure","emerald","ivory","obsidian","silver","golden","copper","rust","ochre",
"bright","dark","transparent","opaque","matte","glossy","rough","smooth","coarse","fine",
"stripe","dot","plaid","solid","gradient","shadow","highlight","contrast","saturation","hue",
"velvet","satin","linen","denim","lace","gauze","burlap","chiffon","tweed","corduroy",
/* TRANSPORT 1350-1399 */
"ship","boat","canoe","raft","anchor","sail","rudder","oar","mast","hull",
"train","rail","station","platform","ticket","journey","passage","crossing","departure","arrival",
"wheel","axle","road","highway","path","trail","bridge","tunnel","gate","crossroad",
"wing","flight","altitude","turbulence","landing","orbit","trajectory","velocity","acceleration","gravity",
"horse","carriage","wagon","cart","sled","bicycle","motorcycle","automobile","truck","ambulance",
/* DOMESTIC 1400-1449 */
"kitchen","bedroom","bathroom","attic","cellar","closet","drawer","shelf","table","chair",
"bed","pillow","blanket","curtain","carpet","lamp","mirror","photograph","vase","clock",
"plate","spoon","knife","fork","cup","pot","pan","kettle","oven","stove",
"soap","towel","broom","bucket","needle","thread","button","zipper","hanger","basket",
"door","window","lock","key","handle","hinge","nail","screw","bolt","hook",
/* COMMUNICATION 1450-1499 */
"letter","envelope","stamp","address","message","telegram","telephone","radio","broadcast","signal",
"newspaper","headline","article","column","editorial","report","announcement","rumor","gossip","testimony",
"ink","pen","pencil","typewriter","keyboard","screen","printer","paper","notebook","diary",
"conversation","dialogue","monologue","debate","argument","negotiation","compromise","ultimatum","declaration","speech",
"translation","interpretation","code","cipher","encryption","decryption","password","signature","seal","authentication",
/* MEDICAL 1500-1549 */
"diagnosis","symptom","treatment","remedy","cure","relapse","recovery","surgery","anesthesia","bandage",
"infection","inflammation","fracture","hemorrhage","allergy","immunity","vaccine","antibiotic","toxin","antidote",
"hospital","clinic","pharmacy","laboratory","ambulance","stretcher","scalpel","syringe","stethoscope","thermometer",
"fever","cough","rash","swelling","numbness","dizziness","insomnia","fatigue","nausea","tremor",
"pulse","pressure","temperature","respiration","circulation","digestion","metabolism","reflex","coordination","balance",
/* COSMIC 1550-1599 */
"universe","galaxy","constellation","planet","asteroid","meteorite","satellite","orbit","void","singularity",
"photon","electron","proton","neutron","atom","molecule","particle","quantum","field","dimension",
"spacetime","relativity","entropy","thermodynamics","radiation","spectrum","wavelength","frequency","amplitude","interference",
"supernova","blackhole","pulsar","quasar","nebula","wormhole","antimatter","darkmatter","redshift","expansion",
"telescope","observatory","mission","launch","countdown","trajectory","reentry","landing","exploration","discovery",
/* BUREAUCRACY 1600-1649 */
"document","form","permit","license","certificate","registration","application","approval","denial","appeal",
"regulation","compliance","violation","penalty","exemption","quota","deadline","protocol","procedure","standard",
"office","desk","file","folder","stamp","signature","receipt","invoice","ledger","archive",
"committee","department","ministry","bureau","agency","institution","organization","corporation","foundation","commission",
"report","audit","review","inspection","evaluation","assessment","benchmark","statistic","data","record",
/* MYTHIC 1650-1699 */
"oracle","prophecy","fate","destiny","curse","blessing","quest","trial","sacrifice","redemption",
"labyrinth","threshold","guardian","shadow","mirror","mask","transformation","metamorphosis","resurrection","apocalypse",
"phoenix","dragon","serpent","sphinx","minotaur","chimera","hydra","golem","specter","wraith",
"underworld","paradise","purgatory","limbo","abyss","eden","babylon","atlantis","olympus","tartarus",
"hero","villain","trickster","sage","fool","maiden","crone","warrior","healer","shapeshifter",
/* TEXTUAL 1700-1749 */
"word","sentence","paragraph","chapter","verse","stanza","line","margin","footnote","epilogue",
"prologue","preface","title","subtitle","dedication","inscription","epitaph","motto","slogan","proverb",
"metaphor","simile","allegory","irony","satire","parody","tragedy","comedy","farce","melodrama",
"narrator","character","protagonist","antagonist","audience","reader","author","critic","editor","translator",
"manuscript","draft","revision","erasure","correction","annotation","citation","reference","index","bibliography",
/* PSYCHOLOGICAL 1750-1799 */
"unconscious","subconscious","conscious","ego","superego","libido","repression","projection","sublimation","transference",
"trauma","complex","fixation","regression","denial","rationalization","displacement","compensation","identification","dissociation",
"archetype","persona","anima","animus","shadow","self","individuation","integration","fragmentation","wholeness",
"attachment","separation","abandonment","dependency","autonomy","codependency","boundary","enmeshment","differentiation","fusion",
"grief","mourning","acceptance","bargaining","anger","depression","recovery","relapse","healing","scarring",
/* FINAL 1800-1983 */
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
};

/* ═══════════════════════════════════════════════════════════════
 * STOPWORDS
 * ═══════════════════════════════════════════════════════════════ */

static const char *STOPS[] = {
"i","me","my","we","our","you","your","he","she","it","they","them","the","a","an",
"and","or","but","in","on","at","to","for","of","is","am","are","was","were","be",
"been","being","have","has","had","do","does","did","will","would","shall","should",
"can","could","may","might","must","not","no","nor","so","if","then","than","that",
"this","these","those","what","which","who","whom","how","when","where","why","all",
"each","every","some","any","few","many","much","more","most","other","another","such",
NULL
};

static int is_stop(const char *w) {
    for (int i = 0; STOPS[i]; i++)
        if (strcmp(w, STOPS[i]) == 0) return 1;
    return 0;
}

static int find_word(const char *w) {
    for (int i = 0; i < NWORDS; i++)
        if (strcmp(VOCAB[i], w) == 0) return i;
    return -1;
}

/* precomputed vocab word lengths for greedy match */
static int vocab_len[NWORDS];
static void init_vocab_lens(void) {
    for (int i = 0; i < NWORDS; i++)
        vocab_len[i] = (int)strlen(VOCAB[i]);
}

/* ═══════════════════════════════════════════════════════════════
 * VOCAB TOKENIZER — stem + greedy longest vocab match
 *
 * Three-stage tokenizer for mapping text to 1984-vocab IDs:
 *   1. Exact vocab match     ("fire" → fire)
 *   2. Suffix stripping       ("burning" → burn, "created" → create)
 *   3. Greedy decomposition   ("heartbreak" → heart + break)
 *
 * Used for training targets (output vocabulary).
 * ═══════════════════════════════════════════════════════════════ */

static const char *SUFFIXES[] = {
    "ting","ning","ring","ling","ding","ping","bing","ging","ming","king",
    "sing","zing",
    "ing","ment","ness","tion","sion","able","ible","ence","ance",
    "eous","ious","ful","less","ize","ise","ous","ive","ity",
    "ly","er","ed","est","al","en","es","s", NULL
};

static int try_stem(const char *word) {
    char stem[64];
    int wlen = (int)strlen(word);
    for (int i = 0; SUFFIXES[i]; i++) {
        int slen = (int)strlen(SUFFIXES[i]);
        if (wlen <= slen + 2) continue;
        if (strcmp(word + wlen - slen, SUFFIXES[i]) != 0) continue;
        int sl = wlen - slen;
        strncpy(stem, word, sl); stem[sl] = '\0';
        int idx = find_word(stem);
        if (idx >= 0) return idx;
        /* stem + 'e' (creat→create, danc→dance) */
        stem[sl] = 'e'; stem[sl+1] = '\0';
        idx = find_word(stem);
        if (idx >= 0) return idx;
        /* doubled consonant (runn→run, swimm→swim) */
        if (sl >= 3 && stem[sl-1] == stem[sl-2]) {
            stem[sl-1] = '\0';
            idx = find_word(stem);
            if (idx >= 0) return idx;
        }
    }
    return -1;
}

static int greedy_vocab_match(const char *word, int wlen, int *ids, int max_ids) {
    int n = 0, pos = 0;
    while (pos < wlen && n < max_ids) {
        int best = -1, best_len = 0;
        for (int v = 0; v < NWORDS; v++) {
            int vl = vocab_len[v];
            if (vl <= best_len || vl > wlen - pos) continue;
            if (strncmp(word + pos, VOCAB[v], vl) == 0) {
                best = v; best_len = vl;
            }
        }
        if (best >= 0 && best_len >= 3) {
            ids[n++] = best;
            pos += best_len;
        } else {
            pos++;
        }
    }
    return n;
}

static int word_category(int idx) {
    if (idx < 100) return 0;
    if (idx < 200) return 1;
    if (idx < 300) return 2;
    if (idx < 350) return 3;
    if (idx < 450) return 4;
    if (idx < 550) return 5;
    if (idx < 650) return 6;
    return 7;
}

/* Precomputed BPE encoding of each vocab word (for generation) */
static int vocab_bpe[NWORDS][16];
static int vocab_bpe_len[NWORDS];

static void init_vocab_bpe(void) {
    for (int i = 0; i < NWORDS; i++)
        vocab_bpe_len[i] = bpe_encode(VOCAB[i], vocab_bpe[i], 16);
}

/* ═══════════════════════════════════════════════════════════════
 * MATH UTILS
 * ═══════════════════════════════════════════════════════════════ */

static float randf(void) { return (float)rand() / (float)RAND_MAX; }
static float clampf(float x, float lo, float hi) { return x<lo?lo:(x>hi?hi:x); }

static float randn(void) {
    float u1 = randf() + 1e-12f;
    float u2 = randf() + 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float siluf(float x) {
    return (x > -20.0f) ? x / (1.0f + expf(-x)) : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════
 * TRAINABLE MODEL — v7 Resonance: 8 sequential layers with
 * multi-head attention + RoPE + RRPRAM resonance gate + SwiGLU
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    float *attn_norm; /* [DIM]         pre-attention RMSNorm */
    float *wq;        /* [DIM * DIM]   query projection */
    float *wk;        /* [DIM * DIM]   key projection */
    float *wv;        /* [DIM * DIM]   value projection */
    float *wo;        /* [DIM * DIM]   output projection */
    float *wr;        /* [DIM * DIM]   RRPRAM resonance */
    float gate[2];    /* blend QKV + RRPRAM */
    float *ffn_norm;  /* [DIM]         pre-FFN RMSNorm */
    float *w_gate;    /* [DIM * HDIM]  SwiGLU gate (note: HDIM > DIM) */
    float *w_up;      /* [DIM * HDIM]  SwiGLU up */
    float *w_down;    /* [HDIM * DIM]  SwiGLU down */
} LayerWeights;

/* Adam first/second moment buffers per layer */
typedef struct {
    float *attn_norm_m, *attn_norm_v;
    float *wq_m, *wq_v;
    float *wk_m, *wk_v;
    float *wv_m, *wv_v;
    float *wo_m, *wo_v;
    float *wr_m, *wr_v;
    float gate_m[2], gate_v[2];
    float *ffn_norm_m, *ffn_norm_v;
    float *gate_w_m, *gate_w_v;  /* SwiGLU gate */
    float *up_m, *up_v;
    float *down_m, *down_v;
} LayerAdam;

#define ADAM_B1  0.9f
#define ADAM_B2  0.999f
#define ADAM_EPS 1e-8f

typedef struct {
    /* Global weights */
    float *tok_emb;     /* [BPE_VOCAB * DIM]  token embedding */
    float *pos_emb;     /* [MAX_SEQ * DIM]    positional embedding */
    float *final_norm;  /* [DIM]              final RMSNorm */
    float *lm_head;     /* [BPE_VOCAB * DIM]  language model head */
    /* Adam for global weights */
    float *tok_emb_m, *tok_emb_v;
    float *pos_emb_m, *pos_emb_v;
    float *final_norm_m, *final_norm_v;
    float *lm_head_m, *lm_head_v;
    /* Per-layer */
    LayerWeights layers[N_LAYERS];
    LayerAdam    adam[N_LAYERS];
    int adam_t;
} Model;

static int layer_param_count(void) {
    /* attn_norm + wq + wk + wv + wo + wr + gate + ffn_norm + w_gate + w_up + w_down */
    return DIM + DIM*DIM*5 + 2 + DIM + DIM*HDIM*2 + HDIM*DIM;
}

static int total_param_count(void) {
    int global = BPE_VOCAB*DIM + MAX_SEQ*DIM + DIM + BPE_VOCAB*DIM;
    return global + N_LAYERS * layer_param_count();
}

static void model_init(Model *m) {
    float scale_d = sqrtf(2.0f / DIM);
    float scale_h = sqrtf(2.0f / HDIM);
    float scale_bpe = sqrtf(2.0f / BPE_VOCAB);

    int te_sz = BPE_VOCAB * DIM;
    int pe_sz = MAX_SEQ * DIM;
    int lh_sz = BPE_VOCAB * DIM;

    /* Global weights */
    m->tok_emb    = (float *)malloc(te_sz * sizeof(float));
    m->pos_emb    = (float *)malloc(pe_sz * sizeof(float));
    m->final_norm = (float *)malloc(DIM * sizeof(float));
    m->lm_head    = (float *)malloc(lh_sz * sizeof(float));
    /* Adam for globals */
    m->tok_emb_m    = (float *)calloc(te_sz, sizeof(float));
    m->tok_emb_v    = (float *)calloc(te_sz, sizeof(float));
    m->pos_emb_m    = (float *)calloc(pe_sz, sizeof(float));
    m->pos_emb_v    = (float *)calloc(pe_sz, sizeof(float));
    m->final_norm_m = (float *)calloc(DIM, sizeof(float));
    m->final_norm_v = (float *)calloc(DIM, sizeof(float));
    m->lm_head_m    = (float *)calloc(lh_sz, sizeof(float));
    m->lm_head_v    = (float *)calloc(lh_sz, sizeof(float));
    m->adam_t = 0;

    for (int i = 0; i < te_sz; i++) m->tok_emb[i] = randn() * scale_bpe;
    for (int i = 0; i < pe_sz; i++) m->pos_emb[i] = randn() * 0.02f;
    for (int i = 0; i < DIM; i++)   m->final_norm[i] = 1.0f;
    for (int i = 0; i < lh_sz; i++) m->lm_head[i] = randn() * scale_d;

    for (int l = 0; l < N_LAYERS; l++) {
        LayerWeights *lw = &m->layers[l];
        LayerAdam *la = &m->adam[l];

        lw->attn_norm = (float *)malloc(DIM * sizeof(float));
        lw->wq        = (float *)malloc(DIM * DIM * sizeof(float));
        lw->wk        = (float *)malloc(DIM * DIM * sizeof(float));
        lw->wv        = (float *)malloc(DIM * DIM * sizeof(float));
        lw->wo        = (float *)malloc(DIM * DIM * sizeof(float));
        lw->wr        = (float *)malloc(DIM * DIM * sizeof(float));
        lw->ffn_norm  = (float *)malloc(DIM * sizeof(float));
        lw->w_gate    = (float *)malloc(DIM * HDIM * sizeof(float));
        lw->w_up      = (float *)malloc(DIM * HDIM * sizeof(float));
        lw->w_down    = (float *)malloc(HDIM * DIM * sizeof(float));

        /* Init gate to 50/50 blend */
        lw->gate[0] = 0.0f;
        lw->gate[1] = 0.0f;

        /* Adam moment buffers */
        la->attn_norm_m = (float *)calloc(DIM, sizeof(float));
        la->attn_norm_v = (float *)calloc(DIM, sizeof(float));
        la->wq_m = (float *)calloc(DIM*DIM, sizeof(float));
        la->wq_v = (float *)calloc(DIM*DIM, sizeof(float));
        la->wk_m = (float *)calloc(DIM*DIM, sizeof(float));
        la->wk_v = (float *)calloc(DIM*DIM, sizeof(float));
        la->wv_m = (float *)calloc(DIM*DIM, sizeof(float));
        la->wv_v = (float *)calloc(DIM*DIM, sizeof(float));
        la->wo_m = (float *)calloc(DIM*DIM, sizeof(float));
        la->wo_v = (float *)calloc(DIM*DIM, sizeof(float));
        la->wr_m = (float *)calloc(DIM*DIM, sizeof(float));
        la->wr_v = (float *)calloc(DIM*DIM, sizeof(float));
        la->gate_m[0] = la->gate_m[1] = 0.0f;
        la->gate_v[0] = la->gate_v[1] = 0.0f;
        la->ffn_norm_m = (float *)calloc(DIM, sizeof(float));
        la->ffn_norm_v = (float *)calloc(DIM, sizeof(float));
        la->gate_w_m = (float *)calloc(DIM*HDIM, sizeof(float));
        la->gate_w_v = (float *)calloc(DIM*HDIM, sizeof(float));
        la->up_m = (float *)calloc(DIM*HDIM, sizeof(float));
        la->up_v = (float *)calloc(DIM*HDIM, sizeof(float));
        la->down_m = (float *)calloc(HDIM*DIM, sizeof(float));
        la->down_v = (float *)calloc(HDIM*DIM, sizeof(float));

        for (int i = 0; i < DIM; i++) lw->attn_norm[i] = 1.0f;
        for (int i = 0; i < DIM*DIM; i++) lw->wq[i] = randn() * scale_d;
        for (int i = 0; i < DIM*DIM; i++) lw->wk[i] = randn() * scale_d;
        for (int i = 0; i < DIM*DIM; i++) lw->wv[i] = randn() * scale_d;
        for (int i = 0; i < DIM*DIM; i++) lw->wo[i] = randn() * scale_d;
        for (int i = 0; i < DIM*DIM; i++) lw->wr[i] = randn() * scale_d;
        for (int i = 0; i < DIM; i++) lw->ffn_norm[i] = 1.0f;
        for (int i = 0; i < DIM*HDIM; i++) lw->w_gate[i] = randn() * scale_d;
        for (int i = 0; i < DIM*HDIM; i++) lw->w_up[i] = randn() * scale_d;
        for (int i = 0; i < HDIM*DIM; i++) lw->w_down[i] = randn() * scale_h;
    }
}

static void model_free(Model *m) {
    free(m->tok_emb); free(m->tok_emb_m); free(m->tok_emb_v);
    free(m->pos_emb); free(m->pos_emb_m); free(m->pos_emb_v);
    free(m->final_norm); free(m->final_norm_m); free(m->final_norm_v);
    free(m->lm_head); free(m->lm_head_m); free(m->lm_head_v);
    for (int l = 0; l < N_LAYERS; l++) {
        LayerWeights *lw = &m->layers[l];
        free(lw->attn_norm); free(lw->wq); free(lw->wk); free(lw->wv);
        free(lw->wo); free(lw->wr); free(lw->ffn_norm);
        free(lw->w_gate); free(lw->w_up); free(lw->w_down);
        LayerAdam *la = &m->adam[l];
        free(la->attn_norm_m); free(la->attn_norm_v);
        free(la->wq_m); free(la->wq_v); free(la->wk_m); free(la->wk_v);
        free(la->wv_m); free(la->wv_v); free(la->wo_m); free(la->wo_v);
        free(la->wr_m); free(la->wr_v);
        free(la->ffn_norm_m); free(la->ffn_norm_v);
        free(la->gate_w_m); free(la->gate_w_v);
        free(la->up_m); free(la->up_v);
        free(la->down_m); free(la->down_v);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * ADAM OPTIMIZER — from Chuck (kirby.c lineage)
 * β₁=0.9, β₂=0.999, bias correction, no weight decay
 * ═══════════════════════════════════════════════════════════════ */

static void adam_update(float *w, float *am, float *av, float *grad,
                        int n, float lr, float bc1, float bc2) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        am[i] = ADAM_B1 * am[i] + (1 - ADAM_B1) * g;
        av[i] = ADAM_B2 * av[i] + (1 - ADAM_B2) * g * g;
        float mhat = am[i] / bc1;
        float vhat = av[i] / bc2;
        w[i] -= lr * mhat / (sqrtf(vhat) + ADAM_EPS);
        grad[i] = 0;  /* clear gradient */
    }
}

static void model_save(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "  cannot open %s for writing\n", path); return; }
    /* PEN7 header: magic, BPE_VOCAB, NWORDS, DIM, HDIM, N_HEADS, N_LAYERS, MAX_SEQ */
    int header[8] = { 0x50454E37, BPE_VOCAB, NWORDS, DIM, HDIM, N_HEADS, N_LAYERS, MAX_SEQ };
    fwrite(header, sizeof(int), 8, f);
    /* Global weights */
    fwrite(m->tok_emb, sizeof(float), BPE_VOCAB * DIM, f);
    fwrite(m->pos_emb, sizeof(float), MAX_SEQ * DIM, f);
    fwrite(m->final_norm, sizeof(float), DIM, f);
    fwrite(m->lm_head, sizeof(float), BPE_VOCAB * DIM, f);
    /* Per-layer weights */
    for (int l = 0; l < N_LAYERS; l++) {
        LayerWeights *lw = &m->layers[l];
        fwrite(lw->attn_norm, sizeof(float), DIM, f);
        fwrite(lw->wq, sizeof(float), DIM*DIM, f);
        fwrite(lw->wk, sizeof(float), DIM*DIM, f);
        fwrite(lw->wv, sizeof(float), DIM*DIM, f);
        fwrite(lw->wo, sizeof(float), DIM*DIM, f);
        fwrite(lw->wr, sizeof(float), DIM*DIM, f);
        fwrite(lw->gate, sizeof(float), 2, f);
        fwrite(lw->ffn_norm, sizeof(float), DIM, f);
        fwrite(lw->w_gate, sizeof(float), DIM*HDIM, f);
        fwrite(lw->w_up, sizeof(float), DIM*HDIM, f);
        fwrite(lw->w_down, sizeof(float), HDIM*DIM, f);
    }
    fclose(f);
    /* verify */
    FILE *check = fopen(path, "rb");
    fseek(check, 0, SEEK_END);
    long sz = ftell(check);
    fclose(check);
    int expected = 32 + total_param_count() * 4;
    printf("  saved %s: %d params (%.1fMB) [%s]\n", path, total_param_count(),
           sz / 1e6, sz == expected ? "OK" : "SIZE MISMATCH!");
}

static int model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); return 0; }
    int header[8];
    fread(header, sizeof(int), 8, f);

    if (header[0] != 0x50454E37) {
        fprintf(stderr, "  unknown format magic=0x%08X (expected PEN7=0x50454E37)\n", header[0]);
        fclose(f);
        return 0;
    }
    if (header[1] != BPE_VOCAB || header[2] != NWORDS || header[3] != DIM ||
        header[4] != HDIM || header[5] != N_HEADS || header[6] != N_LAYERS ||
        header[7] != MAX_SEQ) {
        fprintf(stderr, "  v7 config mismatch: BV=%d V=%d D=%d H=%d NH=%d NL=%d S=%d\n",
                header[1], header[2], header[3], header[4], header[5], header[6], header[7]);
        fclose(f);
        return 0;
    }
    /* Global weights */
    fread(m->tok_emb, sizeof(float), BPE_VOCAB * DIM, f);
    fread(m->pos_emb, sizeof(float), MAX_SEQ * DIM, f);
    fread(m->final_norm, sizeof(float), DIM, f);
    fread(m->lm_head, sizeof(float), BPE_VOCAB * DIM, f);
    /* Per-layer */
    for (int l = 0; l < N_LAYERS; l++) {
        LayerWeights *lw = &m->layers[l];
        fread(lw->attn_norm, sizeof(float), DIM, f);
        fread(lw->wq, sizeof(float), DIM*DIM, f);
        fread(lw->wk, sizeof(float), DIM*DIM, f);
        fread(lw->wv, sizeof(float), DIM*DIM, f);
        fread(lw->wo, sizeof(float), DIM*DIM, f);
        fread(lw->wr, sizeof(float), DIM*DIM, f);
        fread(lw->gate, sizeof(float), 2, f);
        fread(lw->ffn_norm, sizeof(float), DIM, f);
        fread(lw->w_gate, sizeof(float), DIM*HDIM, f);
        fread(lw->w_up, sizeof(float), DIM*HDIM, f);
        fread(lw->w_down, sizeof(float), HDIM*DIM, f);
    }
    fclose(f);
    printf("  loaded v7 %s: %d params (%.1fMB)\n", path, total_param_count(),
           total_param_count() * 4.0f / 1e6);
    return 1;
}


/* ═══════════════════════════════════════════════════════════════
 * FORWARD — v7 Resonance: 8 sequential layers, multi-head
 * attention + RoPE + RRPRAM gate + SwiGLU, then BPE logits
 * ═══════════════════════════════════════════════════════════════ */

static void matmul_mv(float *W, float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = 0;
        for (int j = 0; j < cols; j++)
            s += W[i * cols + j] * x[j];
        out[i] = s;
    }
}

static void matmul_mtv(float *W, float *x, float *out, int rows, int cols) {
    /* W^T @ x: W[rows,cols], x[rows] -> out[cols] */
    for (int j = 0; j < cols; j++) {
        float s = 0;
        for (int i = 0; i < rows; i++)
            s += W[i * cols + j] * x[i];
        out[j] = s;
    }
}

static void rmsnorm(float *x, float *g, float *out, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = ss / n + 1e-5f;
    float inv = 1.0f / sqrtf(ss);
    for (int i = 0; i < n; i++) out[i] = g[i] * x[i] * inv;
}

/* RoPE: apply rotary position embedding to q and k
 * q, k: [seq_len * n_heads * head_dim] laid out as [t][h][d]  */
static void apply_rope(float *q, float *k, int seq_len, int n_heads, int head_dim) {
    float theta_base = 10000.0f;
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + (t * n_heads + h) * head_dim;
            float *kh = k + (t * n_heads + h) * head_dim;
            for (int d = 0; d < head_dim / 2; d++) {
                float freq = 1.0f / powf(theta_base, 2.0f * d / head_dim);
                float cos_f = cosf(t * freq);
                float sin_f = sinf(t * freq);
                /* rotate q */
                float q0 = qh[d], q1 = qh[d + head_dim/2];
                qh[d]              = q0 * cos_f - q1 * sin_f;
                qh[d + head_dim/2] = q0 * sin_f + q1 * cos_f;
                /* rotate k */
                float k0 = kh[d], k1 = kh[d + head_dim/2];
                kh[d]              = k0 * cos_f - k1 * sin_f;
                kh[d + head_dim/2] = k0 * sin_f + k1 * cos_f;
            }
        }
    }
}

/* Full forward pass through all 8 layers for a sequence of BPE tokens.
 * Input:  bpe_ids[seq_len]
 * Output: logits[BPE_VOCAB] for the LAST token position only
 *
 * All scratch memory is allocated inside (heap) because sequence-length
 * dependent buffers can be large (seq * DIM, seq * seq * heads, etc). */
static void forward(Model *m, int *bpe_ids, int seq_len, float *logits) {
    if (seq_len < 1) seq_len = 1;
    if (seq_len > MAX_SEQ) seq_len = MAX_SEQ;

    int S = seq_len;

    /* x: [S * DIM] — residual stream */
    float *x = (float *)calloc(S * DIM, sizeof(float));

    /* embed: tok_emb + pos_emb */
    for (int t = 0; t < S; t++) {
        int tok = bpe_ids[t];
        if (tok < 0 || tok >= BPE_VOCAB) tok = 0;
        for (int d = 0; d < DIM; d++)
            x[t * DIM + d] = m->tok_emb[tok * DIM + d] + m->pos_emb[t * DIM + d];
    }

    /* scratch for attention */
    float *h   = (float *)malloc(S * DIM * sizeof(float));
    float *q   = (float *)malloc(S * DIM * sizeof(float));
    float *k   = (float *)malloc(S * DIM * sizeof(float));
    float *v   = (float *)malloc(S * DIM * sizeof(float));
    float *att = (float *)malloc(S * S * N_HEADS * sizeof(float));  /* per-head attention scores */
    float *av  = (float *)malloc(S * DIM * sizeof(float));  /* attn @ v result, [S, DIM] as [S, n_heads, head_dim] */
    float *qkv_out = (float *)malloc(S * DIM * sizeof(float));
    float *rrp = (float *)malloc(S * DIM * sizeof(float));
    float *h2  = (float *)malloc(S * DIM * sizeof(float));
    float *fg  = (float *)malloc(S * HDIM * sizeof(float)); /* SwiGLU gate */
    float *fu  = (float *)malloc(S * HDIM * sizeof(float)); /* SwiGLU up */
    float *sw  = (float *)malloc(S * HDIM * sizeof(float)); /* silu(gate)*up */
    float *fd  = (float *)malloc(S * DIM * sizeof(float));  /* SwiGLU down */

    for (int l = 0; l < N_LAYERS; l++) {
        LayerWeights *lw = &m->layers[l];

        /* 1. h = rmsnorm(x, attn_norm) for each position */
        for (int t = 0; t < S; t++)
            rmsnorm(x + t*DIM, lw->attn_norm, h + t*DIM, DIM);

        /* 2-3. q = h @ wq, k = h @ wk, v = h @ wv (per position) */
        for (int t = 0; t < S; t++) {
            matmul_mv(lw->wq, h + t*DIM, q + t*DIM, DIM, DIM);
            matmul_mv(lw->wk, h + t*DIM, k + t*DIM, DIM, DIM);
            matmul_mv(lw->wv, h + t*DIM, v + t*DIM, DIM, DIM);
        }

        /* Apply RoPE to q and k (layout: [S, N_HEADS, HEAD_DIM]) */
        apply_rope(q, k, S, N_HEADS, HEAD_DIM);

        /* 5. Multi-head causal attention: softmax(q @ k^T / sqrt(head_dim)) */
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int hd = 0; hd < N_HEADS; hd++) {
            for (int ti = 0; ti < S; ti++) {
                float *qi = q + (ti * N_HEADS + hd) * HEAD_DIM;
                /* compute scores for all keys up to ti (causal) */
                float maxs = -1e30f;
                for (int tj = 0; tj <= ti; tj++) {
                    float *kj = k + (tj * N_HEADS + hd) * HEAD_DIM;
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++)
                        dot += qi[d] * kj[d];
                    dot *= scale;
                    att[(hd * S + ti) * S + tj] = dot;
                    if (dot > maxs) maxs = dot;
                }
                /* causal mask + softmax */
                float sum = 0;
                for (int tj = 0; tj <= ti; tj++) {
                    float val = expf(att[(hd * S + ti) * S + tj] - maxs);
                    att[(hd * S + ti) * S + tj] = val;
                    sum += val;
                }
                float inv_s = (sum > 0) ? 1.0f / sum : 0.0f;
                for (int tj = 0; tj <= ti; tj++)
                    att[(hd * S + ti) * S + tj] *= inv_s;
                /* zero out future positions (for cleanliness) */
                for (int tj = ti + 1; tj < S; tj++)
                    att[(hd * S + ti) * S + tj] = 0;
            }
        }

        /* 6. attn @ v, reshape, then @ wo */
        memset(av, 0, S * DIM * sizeof(float));
        for (int hd = 0; hd < N_HEADS; hd++) {
            for (int ti = 0; ti < S; ti++) {
                float *avi = av + (ti * N_HEADS + hd) * HEAD_DIM;
                for (int tj = 0; tj <= ti; tj++) {
                    float a = att[(hd * S + ti) * S + tj];
                    if (a == 0) continue;
                    float *vj = v + (tj * N_HEADS + hd) * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++)
                        avi[d] += a * vj[d];
                }
            }
        }
        /* av is [S, DIM] (concatenated heads). Project through wo */
        for (int t = 0; t < S; t++)
            matmul_mv(lw->wo, av + t*DIM, qkv_out + t*DIM, DIM, DIM);

        /* 7. RRPRAM resonance: rrp = h @ wr */
        for (int t = 0; t < S; t++)
            matmul_mv(lw->wr, h + t*DIM, rrp + t*DIM, DIM, DIM);

        /* 8. gate_weights = softmax(gate[0], gate[1]) */
        float g0 = lw->gate[0], g1 = lw->gate[1];
        float gmax = g0 > g1 ? g0 : g1;
        float e0 = expf(g0 - gmax), e1 = expf(g1 - gmax);
        float gsum = e0 + e1;
        float w0 = e0 / gsum, w1 = e1 / gsum;

        /* 9. x = x + w0 * qkv_out + w1 * rrp (residual) */
        for (int i = 0; i < S * DIM; i++)
            x[i] += w0 * qkv_out[i] + w1 * rrp[i];

        /* 10. h2 = rmsnorm(x, ffn_norm) */
        for (int t = 0; t < S; t++)
            rmsnorm(x + t*DIM, lw->ffn_norm, h2 + t*DIM, DIM);

        /* 11. SwiGLU FFN: x = x + w_down @ (silu(h2 @ w_gate) * (h2 @ w_up)) */
        for (int t = 0; t < S; t++) {
            matmul_mv(lw->w_gate, h2 + t*DIM, fg + t*HDIM, HDIM, DIM);
            matmul_mv(lw->w_up,   h2 + t*DIM, fu + t*HDIM, HDIM, DIM);
            for (int i = 0; i < HDIM; i++)
                sw[t*HDIM + i] = siluf(fg[t*HDIM + i]) * fu[t*HDIM + i];
            matmul_mv(lw->w_down, sw + t*HDIM, fd + t*DIM, DIM, HDIM);
            for (int d = 0; d < DIM; d++)
                x[t*DIM + d] += fd[t*DIM + d];
        }
    }

    /* After all layers: final rmsnorm + lm_head for LAST position */
    float xn[DIM];
    rmsnorm(x + (S-1)*DIM, m->final_norm, xn, DIM);
    /* logits = lm_head @ xn: lm_head[BPE_VOCAB, DIM] @ xn[DIM] -> logits[BPE_VOCAB] */
    matmul_mv(m->lm_head, xn, logits, BPE_VOCAB, DIM);

    free(x); free(h); free(q); free(k); free(v);
    free(att); free(av); free(qkv_out); free(rrp);
    free(h2); free(fg); free(fu); free(sw); free(fd);
}

/* ═══════════════════════════════════════════════════════════════
 * EXTENDED VOCAB — hardcoded 1984 + BPE tokens that are whole words
 * Built at init from bpe_strs[]. Word-level always, gibberish impossible.
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_EXT_VOCAB 4096

/* extended vocab entry: word string, BPE token IDs, count */
typedef struct {
    char word[64];
    int  bpe_ids[16];
    int  bpe_len;
    int  from_hardcoded; /* 1 = from VOCAB[1984], 0 = from BPE decode */
} ExtWord;

static ExtWord ext_vocab[MAX_EXT_VOCAB];
static int ext_vocab_n = 0;

static int is_alpha_word(const char *s) {
    if (!s[0] || !s[1] || !s[2]) return 0; /* min 3 chars — filters BPE fragments */
    for (int i = 0; s[i]; i++)
        if (!((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z')))
            return 0;
    return 1;
}

static int ext_vocab_find(const char *w) {
    for (int i = 0; i < ext_vocab_n; i++)
        if (strcmp(ext_vocab[i].word, w) == 0) return i;
    return -1;
}

static void init_ext_vocab(void) {
    ext_vocab_n = 0;

    /* 1. Add all 1984 hardcoded words */
    for (int i = 0; i < NWORDS && ext_vocab_n < MAX_EXT_VOCAB; i++) {
        ExtWord *ew = &ext_vocab[ext_vocab_n];
        strncpy(ew->word, VOCAB[i], 63); ew->word[63] = 0;
        memcpy(ew->bpe_ids, vocab_bpe[i], vocab_bpe_len[i] * sizeof(int));
        ew->bpe_len = vocab_bpe_len[i];
        ew->from_hardcoded = 1;
        ext_vocab_n++;
    }

    /* 2. Add BPE tokens that decode to whole words (not already in vocab) */
    for (int t = 0; t < BPE_VOCAB && ext_vocab_n < MAX_EXT_VOCAB; t++) {
        if (!is_alpha_word(bpe_strs[t])) continue;
        /* lowercase for comparison */
        char lower[64];
        int sl = bpe_str_len[t];
        if (sl >= 64) continue;
        for (int i = 0; i < sl; i++)
            lower[i] = (bpe_strs[t][i] >= 'A' && bpe_strs[t][i] <= 'Z')
                      ? bpe_strs[t][i] + 32 : bpe_strs[t][i];
        lower[sl] = 0;
        if (ext_vocab_find(lower) >= 0) continue; /* already in vocab */
        if (is_stop(lower)) continue; /* filter stop words */
        /* filter common suffixes/fragments that aren't real words */
        if (strcmp(lower,"ing")==0 || strcmp(lower,"tion")==0 || strcmp(lower,"ment")==0 ||
            strcmp(lower,"ness")==0 || strcmp(lower,"ble")==0 || strcmp(lower,"ful")==0 ||
            strcmp(lower,"ous")==0 || strcmp(lower,"ive")==0 || strcmp(lower,"ent")==0 ||
            strcmp(lower,"ant")==0 || strcmp(lower,"ist")==0 || strcmp(lower,"ity")==0 ||
            strcmp(lower,"ght")==0 || strcmp(lower,"est")==0 || strcmp(lower,"ter")==0 ||
            strcmp(lower,"ther")==0 || strcmp(lower,"ble")==0 || strcmp(lower,"ted")==0 ||
            strcmp(lower,"ting")==0 || strcmp(lower,"ally")==0 || strcmp(lower,"ling")==0 ||
            strcmp(lower,"ght")==0 || strcmp(lower,"ght")==0) continue;
        ExtWord *ew = &ext_vocab[ext_vocab_n];
        strncpy(ew->word, lower, 63); ew->word[63] = 0;
        ew->bpe_ids[0] = t;
        ew->bpe_len = 1;
        ew->from_hardcoded = 0;
        ext_vocab_n++;
    }

    printf("  extended vocab: %d words (%d hardcoded + %d from BPE)\n",
           ext_vocab_n, NWORDS, ext_vocab_n - NWORDS);
}

/* Compute word-level scores from BPE logits:
 * word_score(w) = mean(bpe_logits[tok] for tok in word's BPE tokens) */
static void bpe_logits_to_word_scores(float *bpe_logits, float *word_scores, int n_words_out) {
    for (int w = 0; w < n_words_out; w++) {
        if (w < NWORDS) {
            /* hardcoded vocab word — use precomputed BPE encoding */
            float score = 0;
            int bl = vocab_bpe_len[w];
            for (int b = 0; b < bl; b++) {
                int tok = vocab_bpe[w][b];
                if (tok >= 0 && tok < BPE_VOCAB)
                    score += bpe_logits[tok];
            }
            word_scores[w] = bl > 0 ? score / bl : 0;
        } else if (w < ext_vocab_n) {
            /* extended vocab word */
            float score = 0;
            int bl = ext_vocab[w].bpe_len;
            for (int b = 0; b < bl; b++) {
                int tok = ext_vocab[w].bpe_ids[b];
                if (tok >= 0 && tok < BPE_VOCAB)
                    score += bpe_logits[tok];
            }
            word_scores[w] = bl > 0 ? score / bl : 0;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SOFTMAX
 * ═══════════════════════════════════════════════════════════════ */

static void softmax_v(float *x, float *out, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(x[i] - mx); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}
/* ═══════════════════════════════════════════════════════════════
 * DARIO FIELD — live heuristic overlay
 * ═══════════════════════════════════════════════════════════════ */

typedef struct { int a, b; float val; } CoocEntry;
typedef struct { int prev, next; float val; } BigramEntry;

static CoocEntry   cooc[MAX_COOC];
static int         cooc_n = 0;
static BigramEntry bigs[MAX_BIG];
static int         big_n = 0;
static float       destiny[8] = {0};
static float       trauma = 0;
static int         prophecy_target = -1;
static int         prophecy_age = 0;

/* Kuramoto chambers */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX, NCH };
static float chambers[NCH] = {0};
static const float ch_decay[NCH] = {0.95f, 0.95f, 0.93f, 0.96f, 0.94f, 0.97f};

static void cooc_update(int a, int b) {
    if (a > b) { int t=a; a=b; b=t; }
    for (int i = 0; i < cooc_n; i++)
        if (cooc[i].a == a && cooc[i].b == b) { cooc[i].val += 1.0f; return; }
    if (cooc_n < MAX_COOC) cooc[cooc_n++] = (CoocEntry){a, b, 1.0f};
}

static float cooc_get(int a, int b) {
    if (a > b) { int t=a; a=b; b=t; }
    for (int i = 0; i < cooc_n; i++)
        if (cooc[i].a == a && cooc[i].b == b) return cooc[i].val;
    return 0;
}

static void update_chambers(int step_idx) {
    float depth = (float)step_idx / N_LAYERS;
    int phase = depth < 0.33f ? 0 : (depth < 0.66f ? 1 : 2);
    if (phase == 0) chambers[CH_FLOW] += 0.05f;
    if (phase == 1) chambers[CH_FEAR] += 0.04f;
    if (phase == 2) chambers[CH_VOID] += 0.05f;
    if (depth > 0.75f) chambers[CH_COMPLEX] += 0.03f;
    if (trauma > 0.3f) chambers[CH_RAGE] += 0.04f;

    float K = 0.02f, old[NCH];
    memcpy(old, chambers, sizeof(old));
    for (int i = 0; i < NCH; i++)
        for (int j = 0; j < NCH; j++)
            if (i != j) chambers[i] += K * sinf(old[j] - old[i]);
    for (int i = 0; i < NCH; i++)
        chambers[i] = clampf(chambers[i] * ch_decay[i], 0, 1);
}

static void dario_overlay(float *logits, int *ctx, int ctx_n, int step) {
    float alpha_mod = 1 + 0.3f*chambers[CH_LOVE] - 0.2f*chambers[CH_RAGE] + 0.1f*chambers[CH_FLOW];
    float gamma_mod = 1 + 0.4f*chambers[CH_VOID] + 0.2f*chambers[CH_COMPLEX];

    int last_n = ctx_n > 8 ? 8 : ctx_n;
    int start = ctx_n - last_n;

    for (int v = 0; v < NWORDS; v++) {
        /* H: Hebbian co-occurrence */
        float H = 0;
        for (int i = start; i < ctx_n; i++)
            H += cooc_get(ctx[i], v);
        if (H > 1) H = 1;
        logits[v] += alpha_mod * 0.3f * H;

        /* F: prophecy */
        if (prophecy_target >= 0 && v == prophecy_target)
            logits[v] += 0.5f * logf(1.0f + prophecy_age);

        /* A: destiny */
        int cat = word_category(v);
        float d_max = 0.01f;
        for (int i = 0; i < 8; i++) if (fabsf(destiny[i]) > d_max) d_max = fabsf(destiny[i]);
        logits[v] += gamma_mod * 0.25f * destiny[cat] / d_max;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TOKENIZE — map text words to 1984-vocab IDs (for training targets)
 * ═══════════════════════════════════════════════════════════════ */

static int tokenize_vocab(const char *text, int *ids, int max_ids) {
    int len = (int)strlen(text);
    char *buf = (char *)malloc(len + 1);
    if (!buf) return 0;
    for (int i = 0; i <= len; i++) buf[i] = tolower(text[i]);

    int n = 0, pos = 0;
    while (pos < len && n < max_ids) {
        while (pos < len && !isalpha(buf[pos])) pos++;
        if (pos >= len) break;

        char word[64];
        int wl = 0;
        while (pos < len && isalpha(buf[pos]) && wl < 63)
            word[wl++] = buf[pos++];
        word[wl] = '\0';

        if (wl < 2 || is_stop(word)) continue;

        /* 1. exact vocab match */
        int idx = find_word(word);
        if (idx >= 0) { ids[n++] = idx; continue; }

        /* 2. stem + match */
        idx = try_stem(word);
        if (idx >= 0) { ids[n++] = idx; continue; }

        /* 3. greedy longest vocab match */
        int sub[8];
        int ns = greedy_vocab_match(word, wl, sub, 8);
        for (int i = 0; i < ns && n < max_ids; i++) {
            if (n == 0 || ids[n-1] != sub[i])
                ids[n++] = sub[i];
        }
    }

    free(buf);
    return n;
}

/* Tokenize for training: produces both vocab IDs and BPE sequences per word */
typedef struct {
    int *word_ids;      /* 1984-vocab ID per word (for targets) */
    int *bpe_flat;      /* all BPE tokens concatenated */
    int *bpe_offset;    /* start index in bpe_flat for each word */
    int *bpe_len;       /* number of BPE tokens per word */
    int n_words;        /* total words */
} TrainTokens;

static TrainTokens tokenize_for_training(const char *text) {
    TrainTokens t = {0};
    int len = (int)strlen(text);
    char *buf = (char *)malloc(len + 1);
    if (!buf) return t;
    for (int i = 0; i <= len; i++) buf[i] = tolower(text[i]);

    /* first pass: count words */
    int max_words = len / 2 + 1;
    t.word_ids   = (int *)malloc(max_words * sizeof(int));
    t.bpe_offset = (int *)malloc(max_words * sizeof(int));
    t.bpe_len    = (int *)malloc(max_words * sizeof(int));
    /* worst case: every byte is a BPE token */
    t.bpe_flat   = (int *)malloc(len * sizeof(int));

    int bpe_pos = 0;
    int pos = 0;
    while (pos < len) {
        while (pos < len && !isalpha(buf[pos])) pos++;
        if (pos >= len) break;

        /* extract word */
        char word[64];
        int wl = 0;
        while (pos < len && isalpha(buf[pos]) && wl < 63)
            word[wl++] = buf[pos++];
        word[wl] = '\0';

        if (wl < 2) continue;
        /* stop words kept in training: they provide contiguous context */

        /* get vocab ID */
        int vid = find_word(word);
        if (vid < 0) vid = try_stem(word);
        if (vid < 0) {
            int sub[8];
            int ns = greedy_vocab_match(word, wl, sub, 8);
            if (ns > 0) vid = sub[0];  /* take first match as representative */
        }
        /* get BPE encoding of the word text (always — for context) */
        int bpe_ids[64];
        int bpe_n = bpe_encode(word, bpe_ids, 64);

        /* vid < 0 = unknown word: include in BPE context but mark as -1
           (won't be used as target, but preserves contiguous text flow) */
        int w = t.n_words;
        t.word_ids[w] = vid;  /* -1 for unknown */
        t.bpe_offset[w] = bpe_pos;
        t.bpe_len[w] = bpe_n;
        memcpy(t.bpe_flat + bpe_pos, bpe_ids, bpe_n * sizeof(int));
        bpe_pos += bpe_n;
        t.n_words++;
    }

    free(buf);
    return t;
}

static void free_train_tokens(TrainTokens *t) {
    free(t->word_ids);
    free(t->bpe_flat);
    free(t->bpe_offset);
    free(t->bpe_len);
}


/* ═══════════════════════════════════════════════════════════════
 * TRAINING — next-token prediction (BPE-level), numerical gradient
 *
 * Simplified C trainer: finite-difference gradient estimation.
 * The real training uses PyTorch (export/import via PEN7 format).
 * This C trainer is kept for quick sanity checks and small corpora.
 * ═══════════════════════════════════════════════════════════════ */

static void train(Model *m, const char *data_path, int train_steps, float lr) {
    FILE *f = fopen(data_path, "r");
    if (!f) { fprintf(stderr, "  cannot open %s\n", data_path); return; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc(fsz + 1);
    fread(text, 1, fsz, f);
    text[fsz] = 0;
    fclose(f);

    /* BPE tokenize entire corpus */
    int *corpus_bpe = (int *)malloc(fsz * sizeof(int));
    int corpus_len = bpe_encode(text, corpus_bpe, (int)fsz);
    free(text);

    if (corpus_len < MAX_SEQ + 1) {
        fprintf(stderr, "  corpus too small: %d BPE tokens (need %d+)\n", corpus_len, MAX_SEQ + 1);
        free(corpus_bpe);
        return;
    }

    printf("  corpus: %ld bytes -> %d BPE tokens\n", fsz, corpus_len);
    printf("  model: %d params (%.1fMB f32)\n", total_param_count(), total_param_count() * 4.0f / 1e6);
    printf("  architecture: %d layers, %d heads, dim=%d, hdim=%d\n", N_LAYERS, N_HEADS, DIM, HDIM);
    printf("  training: %d steps, lr=%.1e, seq=%d\n", train_steps, lr, MAX_SEQ);
    printf("  NOTE: C trainer uses forward-only loss (for PyTorch training, export weights)\n");

    float *logits = (float *)malloc(BPE_VOCAB * sizeof(float));
    float *probs  = (float *)malloc(BPE_VOCAB * sizeof(float));
    float best_loss = 1e9f;

    for (int step = 1; step <= train_steps; step++) {
        /* Sample a random window from corpus */
        int seq_len = MAX_SEQ;
        if (seq_len > corpus_len - 1) seq_len = corpus_len - 1;
        int start = rand() % (corpus_len - seq_len);

        /* Forward pass: predict next token from context */
        int *ctx = corpus_bpe + start;
        int target = corpus_bpe[start + seq_len];

        forward(m, ctx, seq_len, logits);
        softmax_v(logits, probs, BPE_VOCAB);

        float p = probs[target];
        if (p < 1e-10f) p = 1e-10f;
        float loss = -logf(p);

        if (loss < best_loss) best_loss = loss;

        if (step % 50 == 0 || step == 1)
            printf("  step %5d/%d  loss=%.4f  best=%.4f  (target=%d p=%.4f)\n",
                   step, train_steps, loss, best_loss, target, p);

        /* NOTE: Full backpropagation through 8-layer transformer is complex.
         * For production training, use the PyTorch export path.
         * This loop provides loss monitoring for sanity checks.
         * Simple embedding gradient descent is applied below for basic learning. */

        /* Gradient on lm_head and tok_emb (shallow gradient — last layer only) */
        float d_logits[BPE_VOCAB];
        for (int v = 0; v < BPE_VOCAB; v++) d_logits[v] = probs[v];
        d_logits[target] -= 1.0f;

        /* Update lm_head: gradient = d_logits (outer product with final hidden) */
        /* We approximate by nudging tok_emb for the target token toward the context */
        float scale = lr * 0.1f;
        for (int d = 0; d < DIM; d++) {
            /* Nudge target token embedding */
            float avg_ctx = 0;
            int n_ctx = seq_len < 8 ? seq_len : 8;
            for (int i = seq_len - n_ctx; i < seq_len; i++)
                avg_ctx += m->tok_emb[ctx[i] * DIM + d];
            avg_ctx /= n_ctx;
            m->tok_emb[target * DIM + d] += scale * (avg_ctx - m->tok_emb[target * DIM + d]);
        }
    }

    printf("  training complete. best loss: %.4f\n", best_loss);
    printf("  NOTE: for full training, use PyTorch with PEN7 weight export\n");

    free(corpus_bpe);
    free(logits);
    free(probs);
}


/* ═══════════════════════════════════════════════════════════════
 * GENERATION — autoregressive BPE, then word-level output
 *
 * Dual tokenizer: soul thinks in BPE (2048), mouth speaks in words (1984).
 * At each step:
 *   1. Forward pass -> BPE logits
 *   2. Compute word scores = mean(logits for word's BPE tokens)
 *   3. Apply Dario overlay on word scores
 *   4. Sample word, print it
 *   5. Append word's BPE tokens to context for next step
 * ═══════════════════════════════════════════════════════════════ */

#define GEN_STEPS 12   /* number of words to generate per chain */

static int find_seed(const char *key) {
    int idx = find_word(key);
    if (idx >= 0) return idx;

    int best = -1; float best_score = -1;
    for (int i = 0; i < NWORDS; i++) {
        float score = 0;
        if (strstr(VOCAB[i], key) || strstr(key, VOCAB[i])) score = 3;
        int ml = strlen(VOCAB[i]), kl = strlen(key);
        int mn = ml < kl ? ml : kl;
        for (int j = 0; j < mn; j++) {
            if (VOCAB[i][j] == key[j]) score += 0.5f;
            else break;
        }
        if (score > best_score) { best_score = score; best = i; }
    }
    return (best >= 0 && best_score > 0) ? best : (rand() % 200);
}

static void extract_key(const char *text, char *out, int maxlen) {
    char buf[1024];
    int bi = 0;
    for (int i = 0; text[i] && bi < 1022; i++)
        buf[bi++] = tolower(text[i]);
    buf[bi] = 0;

    char *best = NULL; int best_len = 0;
    char *tok = strtok(buf, " \t\n");
    while (tok) {
        if (strlen(tok) > 1 && !is_stop(tok)) {
            if ((int)strlen(tok) > best_len) { best = tok; best_len = strlen(tok); }
        }
        tok = strtok(NULL, " \t\n");
    }
    if (best) { strncpy(out, best, maxlen-1); out[maxlen-1]=0; }
    else { strncpy(out, "silence", maxlen-1); out[maxlen-1]=0; }
}

static void run_chain(Model *m, const char *text, int has_weights) {
    char key[64];
    extract_key(text, key, sizeof(key));

    int seed = find_seed(key);
    printf("\n  %s\n", VOCAB[seed]);

    /* prophecy (word-level, for both modes) */
    int deep_cats[] = {2, 5, 7};
    int tcat = deep_cats[rand() % 3];
    int ranges[][2] = {{0,100},{100,200},{200,300},{300,350},{350,450},{450,550},{550,650},{650,NWORDS}};
    prophecy_target = ranges[tcat][0] + rand() % (ranges[tcat][1] - ranges[tcat][0]);
    if (prophecy_target >= NWORDS) prophecy_target = NWORDS - 1;
    prophecy_age = 0;
    printf("  destined: %s\n\n", VOCAB[prophecy_target]);

    /* BPE context buffer — starts with seed word's BPE tokens */
    int bpe_buf[MAX_BPE_SEQ];
    int bpe_n = 0;
    memcpy(bpe_buf, vocab_bpe[seed], vocab_bpe_len[seed] * sizeof(int));
    bpe_n = vocab_bpe_len[seed];

    /* word-level chain for Dario field */
    int chain[GEN_STEPS + 1], chain_n = 0;
    int forbidden[GEN_STEPS + 100], nforbid = 0;
    chain[chain_n++] = seed;
    forbidden[nforbid++] = seed;

    /* scratch */
    float *bpe_logits  = (float *)malloc(BPE_VOCAB * sizeof(float));
    int gen_vocab = has_weights ? ext_vocab_n : NWORDS;
    float *word_scores = (float *)malloc(gen_vocab * sizeof(float));
    float *probs       = (float *)malloc(gen_vocab * sizeof(float));

    int fulfilled = 0;

    for (int step = 0; step < GEN_STEPS; step++) {
        update_chambers(step);
        prophecy_age++;

        /* 1. Forward pass through all 8 layers -> BPE logits for last position */
        int ctx_len = bpe_n < MAX_SEQ ? bpe_n : MAX_SEQ;
        int ctx_start = bpe_n > MAX_SEQ ? bpe_n - MAX_SEQ : 0;
        forward(m, bpe_buf + ctx_start, ctx_len, bpe_logits);

        /* 2. Convert BPE logits to word-level scores */
        bpe_logits_to_word_scores(bpe_logits, word_scores, gen_vocab);

        /* 3. Dario overlay on word scores (first NWORDS entries) */
        dario_overlay(word_scores, chain, chain_n, step);

        if (has_weights) {
            /* mask forbidden by word string */
            for (int w = 0; w < ext_vocab_n; w++) {
                for (int fi = 0; fi < nforbid; fi++) {
                    const char *fw = (forbidden[fi] < NWORDS) ? VOCAB[forbidden[fi]] : ext_vocab[forbidden[fi]].word;
                    const char *cw = (w < NWORDS) ? VOCAB[w] : ext_vocab[w].word;
                    if (strcmp(cw, fw) == 0) {
                        word_scores[w] = -1e9f;
                        break;
                    }
                }
            }

            /* top-k=12 sampling from ext_vocab_n */
            softmax_v(word_scores, probs, ext_vocab_n);
            typedef struct { int idx; float p; } Sc;
            Sc top[12];
            for (int i = 0; i < 12; i++) top[i] = (Sc){0, -1};
            for (int w = 0; w < ext_vocab_n; w++) {
                for (int ki = 0; ki < 12; ki++) {
                    if (probs[w] > top[ki].p) {
                        for (int j = 11; j > ki; j--) top[j] = top[j-1];
                        top[ki] = (Sc){w, probs[w]};
                        break;
                    }
                }
            }
            float total = 0.001f;
            for (int i = 0; i < 12; i++) total += top[i].p > 0 ? top[i].p : 0;
            float r = randf() * total;
            int pick = top[0].idx;
            for (int i = 0; i < 12; i++) {
                r -= top[i].p > 0 ? top[i].p : 0;
                if (r <= 0) { pick = top[i].idx; break; }
            }

            chain[chain_n++] = pick;
            forbidden[nforbid++] = pick;

            /* append picked word's BPE tokens to context */
            if (pick < NWORDS) {
                if (bpe_n + vocab_bpe_len[pick] < MAX_BPE_SEQ) {
                    memcpy(bpe_buf + bpe_n, vocab_bpe[pick], vocab_bpe_len[pick] * sizeof(int));
                    bpe_n += vocab_bpe_len[pick];
                }
            } else if (pick < ext_vocab_n) {
                if (bpe_n + ext_vocab[pick].bpe_len < MAX_BPE_SEQ) {
                    memcpy(bpe_buf + bpe_n, ext_vocab[pick].bpe_ids,
                           ext_vocab[pick].bpe_len * sizeof(int));
                    bpe_n += ext_vocab[pick].bpe_len;
                }
            }

            /* Dario field updates */
            if (pick < NWORDS) {
                cooc_update(chain_n >= 2 ? chain[chain_n-2] : seed, pick);
                int cat = word_category(pick);
                destiny[cat] = 0.3f + 0.7f * destiny[cat];
                if (pick == prophecy_target) fulfilled = 1;
            }

            if (step > 7) trauma = trauma + 0.1f < 1 ? trauma + 0.1f : 1;
            trauma *= 0.97f;

            const char *wname = (pick < NWORDS) ? VOCAB[pick] : ext_vocab[pick].word;
            if (step == GEN_STEPS - 1)
                printf("  *%s\n", wname);
            else
                printf("   %s\n", wname);

        } else {
            /* WEIGHTLESS MODE: hardcoded 1984 words only */
            for (int fi = 0; fi < nforbid; fi++)
                word_scores[forbidden[fi]] = -1e9f;

            softmax_v(word_scores, probs, NWORDS);
            typedef struct { int idx; float p; } Sc2;
            Sc2 top[12];
            for (int i = 0; i < 12; i++) top[i] = (Sc2){0, -1};
            for (int w = 0; w < NWORDS; w++) {
                for (int ki = 0; ki < 12; ki++) {
                    if (probs[w] > top[ki].p) {
                        for (int j = 11; j > ki; j--) top[j] = top[j-1];
                        top[ki] = (Sc2){w, probs[w]};
                        break;
                    }
                }
            }
            float total = 0.001f;
            for (int i = 0; i < 12; i++) total += top[i].p > 0 ? top[i].p : 0;
            float r = randf() * total;
            int pick = top[0].idx;
            for (int i = 0; i < 12; i++) {
                r -= top[i].p > 0 ? top[i].p : 0;
                if (r <= 0) { pick = top[i].idx; break; }
            }

            chain[chain_n++] = pick;
            forbidden[nforbid++] = pick;

            if (bpe_n + vocab_bpe_len[pick] < MAX_BPE_SEQ) {
                memcpy(bpe_buf + bpe_n, vocab_bpe[pick], vocab_bpe_len[pick] * sizeof(int));
                bpe_n += vocab_bpe_len[pick];
            }

            cooc_update(chain_n >= 2 ? chain[chain_n-2] : seed, pick);
            int cat = word_category(pick);
            destiny[cat] = 0.3f + 0.7f * destiny[cat];
            if (pick == prophecy_target) fulfilled = 1;

            if (step > 7) trauma = trauma + 0.1f < 1 ? trauma + 0.1f : 1;
            trauma *= 0.97f;

            if (step == GEN_STEPS - 1)
                printf("  *%s\n", VOCAB[pick]);
            else
                printf("   %s\n", VOCAB[pick]);
        }
    }

    int cats_seen = 0, cat_flags[8] = {0};
    for (int i = 0; i < chain_n; i++) {
        int c = word_category(chain[i]);
        if (!cat_flags[c]) { cat_flags[c] = 1; cats_seen++; }
    }

    printf("\n  drift %d/8 \xc2\xb7 prophecy %s\n",
           cats_seen, fulfilled ? "fulfilled" : "unfulfilled");

    free(bpe_logits); free(word_scores); free(probs);
}


/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand(time(NULL));
    init_vocab_lens();
    init_vocab_bpe();
    init_bpe_decode();
    init_ext_vocab();

    char *train_path = NULL;
    char *load_path = NULL;
    char *save_path = NULL;
    int train_steps = 5000;
    float lr = 3e-4f;
    char *text = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i+1 < argc) { train_path = argv[++i]; }
        else if (strcmp(argv[i], "--load") == 0 && i+1 < argc) { load_path = argv[++i]; }
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc) { save_path = argv[++i]; }
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) { train_steps = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) { lr = atof(argv[++i]); }
        else {
            /* collect remaining as text */
            static char textbuf[2048];
            textbuf[0] = 0;
            for (int j = i; j < argc; j++) {
                if (j > i) strcat(textbuf, " ");
                strncat(textbuf, argv[j], sizeof(textbuf) - strlen(textbuf) - 2);
            }
            text = textbuf;
            break;
        }
    }

    Model model;
    model_init(&model);

    printf("\n  penelope v7 \xe2\x80\x94 Resonance engine. 1984 words. Dario Equation.\n");
    printf("  %d layers, %d heads, dim=%d, hdim=%d\n", N_LAYERS, N_HEADS, DIM, HDIM);
    printf("  %d trainable params (%.1fMB f32)\n", total_param_count(),
           total_param_count() * 4.0f / 1e6);
    printf("  BPE input: %d subword tokens, max_seq=%d\n", BPE_VOCAB, MAX_SEQ);
    printf("  by Arianna Method\n\n");

    int has_weights = 0;
    if (load_path) { model_load(&model, load_path); has_weights = 1; }
    if (train_path) {
        train(&model, train_path, train_steps, lr);
        has_weights = 1;
        if (save_path) model_save(&model, save_path);
    }

    printf("  mode: %s\n\n", has_weights ? "trained (BPE word scores)" : "weightless (word-level)");

    if (text) {
        run_chain(&model, text, has_weights);
    } else if (!train_path) {
        char line[1024];
        while (1) {
            printf("  > ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            int len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
            if (len == 0) continue;
            run_chain(&model, line, has_weights);
        }
    }

    if (save_path && !train_path) model_save(&model, save_path);

    model_free(&model);
    return 0;
}
