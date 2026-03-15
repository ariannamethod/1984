// penelope.rs -- v7 Resonance engine. 1984 words. Dario Equation.
//
// 8-layer sequential transformer with multi-head attention, RoPE,
// RRPRAM resonance gates, and SwiGLU FFN. Dual tokenizer:
// BPE input (2048 subwords), word-level output (1984 words).
//
// Architecture per layer l:
//     h = rmsnorm(x, attn_norm_l)
//     qkv_out = MultiHeadAttention(h; wq_l, wk_l, wv_l, wo_l, RoPE)
//     rrp = h @ wr_l                            RRPRAM resonance
//     gate = softmax(gate_l[0], gate_l[1])
//     x = x + gate[0]*qkv_out + gate[1]*rrp    gated residual
//     h2 = rmsnorm(x, ffn_norm_l)
//     x = x + SwiGLU(h2; w_gate_l, w_up_l, w_down_l)  residual
//
// After 8 layers:
//     logits = rmsnorm(x, final_norm) @ lm_head^T
//     word_score(w) = mean(logits[bpe_tokens(w)]) + DarioField
//
//   score(w) = B + alpha*H + beta*F + gamma*A + T   (Dario Equation)
//
//   rustc -O penelope.rs -o penelope_rs
//   ./penelope_rs                                  # interactive
//   ./penelope_rs "darkness eats the city"         # single chain
//   ./penelope_rs --train corpus.txt               # train 5000 steps
//   ./penelope_rs --train corpus.txt --steps 1000  # train N steps
//   ./penelope_rs --load penelope.bin              # load weights
//   ./penelope_rs --save penelope.bin              # save after
//
// By Arianna Method.

use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryInto;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

const DIM: usize = 448;
const HDIM: usize = 896;       // DIM * 2, SwiGLU hidden
const N_HEADS: usize = 7;
const HEAD_DIM: usize = 64;    // DIM / N_HEADS
const N_LAYERS: usize = 8;     // sequential transformer layers
const MAX_SEQ: usize = 256;
const NWORDS: usize = 1984;
const BPE_VOCAB: usize = 2048;
const BPE_MERGES: usize = 1792;
const MAX_BPE_SEQ: usize = 16384;
const MAX_EXT_VOCAB: usize = 4096;
const GEN_STEPS: usize = 12;

// ═══════════════════════════════════════════════════════════════
// BPE MERGE TABLE — 1792 merges, learned from English text
// ═══════════════════════════════════════════════════════════════

#[rustfmt::skip]
const BPE_TABLE: [(u16, u16); BPE_MERGES] = [
(115, 32), (101, 32), (46, 32), (105, 110), (105, 256), (101, 114), (111, 110), (116, 104),
(116, 32), (101, 110), (97, 110), (116, 105), (104, 260), (101, 115), (121, 32), (258, 268),
(100, 32), (111, 114), (259, 103), (97, 114), (97, 108), (274, 32), (267, 262), (111, 117),
(101, 256), (114, 101), (111, 32), (105, 116), (97, 116), (58, 32), (111, 109), (115, 116),
(100, 105), (101, 108), (104, 97), (114, 269), (112, 261), (261, 32), (263, 257), (266, 272),
(278, 32), (111, 119), (97, 99), (105, 115), (266, 32), (44, 32), (39, 256), (276, 32),
(108, 105), (265, 99), (114, 97), (116, 282), (117, 114), (101, 272), (105, 109), (102, 102),
(101, 120), (101, 99), (101, 109), (102, 32), (102, 273), (114, 111), (101, 116), (10, 10),
(97, 285), (113, 285), (10, 320), (46, 319), (323, 321), (117, 110), (97, 32), (117, 108),
(101, 118), (265, 264), (290, 264), (119, 330), (105, 99), (265, 116), (275, 257), (284, 116),
(97, 115), (103, 104), (97, 296), (63, 322), (288, 311), (105, 32), (115, 105), (99, 262),
(110, 111), (112, 291), (305, 257), (101, 271), (100, 101), (111, 108), (105, 108), (286, 101),
(283, 270), (263, 101), (97, 98), (101, 100), (115, 351), (97, 103), (99, 286), (275, 105),
(115, 117), (262, 32), (340, 261), (114, 117), (269, 115), (111, 315), (97, 112), (119, 104),
(262, 257), (105, 114), (108, 270), (98, 101), (115, 99), (109, 101), (98, 257), (265, 32),
(259, 32), (115, 289), (267, 118), (114, 105), (111, 99), (115, 104), (267, 109), (97, 109),
(112, 317), (344, 264), (113, 117), (105, 263), (263, 300), (335, 261), (109, 273), (266, 110),
(119, 273), (112, 111), (101, 264), (279, 108), (121, 258), (395, 272), (97, 278), (267, 99),
(353, 270), (119, 101), (359, 391), (402, 326), (45, 32), (109, 32), (273, 32), (266, 99),
(97, 100), (324, 331), (257, 260), (121, 279), (121, 271), (111, 263), (263, 277), (119, 387),
(112, 108), (276, 352), (290, 112), (102, 101), (101, 258), (105, 100), (100, 97), (279, 264),
(117, 109), (100, 117), (104, 32), (337, 264), (292, 105), (115, 271), (116, 114), (100, 256),
(100, 277), (99, 104), (109, 270), (107, 32), (276, 108), (100, 111), (116, 256), (109, 389),
(103, 117), (118, 261), (115, 112), (105, 264), (99, 111), (108, 257), (294, 115), (258, 403),
(119, 397), (422, 270), (411, 32), (98, 117), (258, 341), (99, 300), (121, 302), (112, 275),
(116, 111), (114, 297), (116, 306), (269, 256), (110, 282), (275, 32), (105, 427), (258, 294),
(328, 261), (98, 318), (316, 32), (102, 259), (115, 262), (114, 286), (97, 264), (277, 295),
(115, 107), (99, 108), (304, 102), (348, 112), (117, 115), (265, 115), (110, 364), (100, 282),
(101, 301), (312, 428), (108, 32), (356, 414), (292, 468), (116, 117), (281, 99), (415, 423),
(99, 275), (116, 108), (112, 32), (342, 361), (105, 287), (103, 103), (111, 111), (308, 257),
(327, 116), (259, 116), (265, 267), (261, 257), (297, 110), (263, 293), (297, 32), (335, 265),
(115, 258), (112, 273), (97, 107), (98, 317), (390, 257), (263, 261), (97, 117), (430, 270),
(377, 102), (103, 110), (377, 315), (451, 264), (109, 111), (362, 329), (279, 115), (355, 271),
(275, 272), (118, 472), (425, 507), (522, 521), (109, 462), (287, 363), (101, 103), (306, 501),
(527, 388), (278, 303), (269, 271), (278, 256), (524, 374), (112, 304), (116, 297), (534, 520),
(356, 382), (102, 105), (263, 259), (536, 280), (511, 307), (273, 105), (367, 499), (292, 418),
(325, 100), (110, 100), (477, 338), (543, 256), (39, 264), (298, 426), (437, 280), (263, 269),
(401, 375), (446, 546), (465, 552), (314, 111), (99, 101), (390, 110), (508, 388), (100, 269),
(362, 346), (274, 271), (98, 413), (439, 256), (551, 257), (263, 470), (400, 334), (121, 394),
(339, 294), (448, 540), (474, 257), (458, 424), (381, 105), (266, 100), (314, 32), (115, 380),
(575, 105), (562, 32), (261, 103), (484, 417), (339, 523), (118, 105), (104, 349), (266, 103),
(409, 260), (357, 257), (386, 269), (112, 114), (358, 316), (105, 122), (104, 502), (111, 112),
(111, 441), (99, 296), (555, 529), (383, 257), (101, 119), (116, 354), (262, 103), (557, 277),
(98, 105), (116, 261), (281, 408), (102, 327), (99, 366), (101, 549), (316, 109), (277, 115),
(475, 265), (101, 302), (419, 289), (99, 114), (512, 45), (345, 329), (119, 97), (102, 469),
(580, 454), (263, 260), (270, 260), (371, 277), (615, 32), (116, 259), (259, 264), (279, 114),
(109, 266), (290, 256), (284, 257), (378, 257), (115, 257), (98, 108), (116, 289), (287, 114),
(291, 112), (111, 100), (284, 309), (261, 118), (103, 259), (34, 32), (101, 275), (349, 117),
(115, 595), (312, 116), (103, 306), (407, 257), (479, 450), (112, 97), (104, 289), (632, 262),
(109, 329), (110, 297), (265, 578), (516, 378), (550, 385), (382, 257), (109, 262), (467, 431),
(392, 435), (282, 260), (102, 306), (115, 121), (324, 590), (456, 282), (283, 256), (259, 459),
(328, 265), (312, 492), (342, 262), (102, 298), (398, 256), (447, 655), (263, 574), (345, 333),
(611, 299), (99, 281), (107, 257), (104, 293), (266, 264), (292, 102), (505, 116), (343, 102),
(288, 115), (369, 32), (283, 514), (481, 305), (333, 271), (457, 256), (313, 116), (584, 294),
(108, 266), (292, 606), (260, 385), (660, 644), (121, 263), (105, 513), (115, 308), (688, 440),
(538, 107), (677, 313), (112, 104), (293, 263), (340, 332), (279, 337), (373, 394), (440, 350),
(488, 114), (99, 334), (115, 418), (415, 32), (349, 111), (280, 307), (116, 265), (116, 370),
(260, 104), (332, 303), (287, 259), (304, 674), (500, 32), (110, 313), (646, 112), (97, 259),
(99, 97), (481, 346), (373, 288), (327, 461), (120, 105), (299, 301), (119, 259), (537, 289),
(581, 596), (99, 379), (353, 681), (361, 331), (108, 598), (706, 280), (266, 724), (650, 270),
(281, 108), (278, 258), (556, 112), (104, 310), (280, 334), (651, 338), (360, 99), (115, 101),
(287, 284), (476, 264), (734, 318), (630, 482), (111, 311), (328, 293), (392, 107), (99, 266),
(358, 416), (102, 97), (299, 405), (436, 256), (413, 293), (623, 99), (586, 531), (105, 315),
(308, 277), (291, 262), (263, 32), (345, 346), (485, 281), (452, 569), (708, 103), (372, 105),
(610, 32), (571, 97), (279, 545), (298, 278), (455, 399), (116, 271), (559, 729), (116, 641),
(525, 99), (381, 397), (283, 117), (103, 266), (98, 336), (107, 649), (109, 259), (100, 760),
(273, 779), (309, 376), (109, 314), (589, 280), (631, 284), (265, 117), (333, 370), (727, 272),
(489, 396), (118, 257), (288, 486), (280, 102), (108, 101), (772, 723), (274, 301), (115, 313),
(291, 757), (328, 375), (356, 368), (119, 283), (425, 99), (639, 278), (774, 374), (104, 111),
(266, 101), (717, 364), (366, 533), (588, 597), (115, 264), (419, 461), (775, 495), (809, 275),
(109, 275), (310, 496), (817, 808), (104, 257), (274, 258), (695, 585), (310, 678), (510, 263),
(662, 716), (664, 277), (358, 112), (343, 767), (376, 283), (818, 518), (324, 806), (803, 478),
(582, 432), (259, 284), (325, 811), (98, 770), (732, 293), (525, 493), (98, 273), (460, 836),
(109, 308), (280, 436), (333, 338), (509, 410), (544, 293), (822, 676), (837, 108), (100, 500),
(272, 365), (355, 258), (362, 790), (371, 636), (463, 791), (766, 713), (834, 445), (274, 322),
(498, 116), (97, 256), (642, 442), (105, 102), (288, 714), (710, 491), (635, 332), (778, 338),
(99, 369), (784, 787), (99, 755), (102, 363), (298, 485), (393, 287), (420, 460), (604, 764),
(694, 667), (700, 496), (744, 480), (258, 539), (269, 438), (101, 107), (331, 690), (363, 621),
(372, 879), (39, 32), (267, 337), (277, 661), (301, 300), (309, 620), (541, 842), (814, 404),
(860, 593), (886, 535), (45, 570), (284, 280), (295, 815), (380, 634), (602, 663), (625, 797),
(792, 843), (878, 567), (107, 259), (406, 839), (443, 577), (483, 487), (528, 771), (535, 894),
(553, 365), (553, 895), (613, 899), (617, 874), (682, 850), (715, 832), (761, 407), (783, 907),
(800, 841), (828, 884), (830, 904), (835, 359), (854, 892), (858, 883), (861, 913), (865, 908),
(882, 896), (887, 909), (889, 897), (893, 903), (900, 916), (901, 917), (905, 921), (906, 671),
(911, 912), (918, 922), (919, 928), (920, 929), (923, 902), (925, 931), (926, 933), (930, 932),
(934, 927), (392, 431), (109, 97), (393, 622), (115, 805), (263, 258), (370, 404), (384, 118),
(489, 121), (691, 721), (852, 935), (360, 493), (386, 417), (102, 336), (560, 554), (851, 110),
(99, 308), (898, 848), (936, 946), (367, 657), (424, 300), (687, 950), (704, 270), (924, 121),
(107, 270), (409, 448), (583, 108), (867, 788), (103, 685), (99, 833), (114, 104), (269, 669),
(324, 453), (406, 547), (961, 450), (295, 547), (307, 483), (439, 301), (463, 560), (292, 624),
(517, 962), (608, 432), (840, 960), (949, 965), (396, 368), (480, 756), (563, 558), (564, 758),
(607, 829), (728, 885), (844, 880), (846, 709), (942, 400), (974, 563), (977, 731), (979, 471),
(115, 325), (116, 363), (310, 697), (368, 810), (373, 656), (414, 985), (532, 475), (532, 872),
(565, 821), (566, 640), (652, 973), (654, 449), (658, 996), (845, 1000), (871, 989), (888, 964),
(939, 972), (963, 984), (967, 983), (969, 1001), (971, 1002), (976, 1010), (978, 986), (980, 999),
(981, 998), (987, 1006), (988, 1008), (990, 1004), (991, 1009), (995, 269), (997, 1013), (1005, 1017),
(1007, 1014), (1011, 1022), (1012, 1019), (1015, 1016), (1018, 1023), (1020, 1028), (1024, 1027), (1025, 1029),
(1026, 1021), (1031, 1032), (400, 777), (736, 398), (824, 953), (970, 747), (504, 539), (702, 670),
(748, 699), (855, 954), (873, 618), (966, 692), (336, 576), (446, 863), (464, 478), (466, 705),
(473, 1046), (528, 542), (542, 566), (558, 1048), (619, 831), (725, 994), (763, 982), (785, 1042),
(802, 955), (866, 1047), (940, 1038), (1030, 941), (1034, 371), (1036, 718), (1037, 1056), (1039, 1050),
(1040, 1053), (1045, 1057), (1052, 1055), (1054, 1058), (1063, 1049), (1065, 1051), (1066, 1061), (1067, 1070),
(343, 776), (672, 260), (1035, 572), (1059, 1033), (310, 533), (753, 350), (339, 1069), (947, 876),
(875, 1071), (600, 853), (659, 545), (544, 261), (1043, 405), (1060, 1080), (1064, 944), (102, 494),
(568, 1075), (827, 518), (421, 856), (794, 711), (503, 737), (742, 99), (294, 937), (428, 633),
(1044, 668), (110, 101), (307, 605), (712, 956), (280, 801), (288, 300), (291, 426), (401, 877),
(653, 365), (720, 1101), (864, 1105), (432, 847), (449, 1099), (453, 768), (726, 1107), (1072, 264),
(1091, 683), (1104, 1108), (1113, 1111), (106, 745), (115, 267), (258, 599), (281, 383), (404, 517),
(487, 730), (564, 910), (567, 1094), (675, 735), (733, 1123), (780, 299), (795, 1102), (798, 825),
(870, 1106), (948, 1098), (951, 1127), (958, 1096), (1076, 1126), (1079, 1110), (1081, 1125), (1084, 1124),
(1095, 1117), (1100, 1120), (1109, 1119), (1112, 1128), (1121, 1137), (1122, 1131), (1129, 1136), (1130, 1133),
(1132, 1143), (1135, 406), (1138, 1142), (1139, 1145), (1140, 1134), (1146, 1144), (281, 276), (992, 449),
(105, 262), (339, 1114), (1147, 1092), (1154, 1141), (346, 260), (637, 268), (121, 256), (265, 399),
(759, 434), (99, 273), (509, 366), (576, 303), (112, 101), (97, 627), (679, 421), (121, 115),
(345, 491), (751, 548), (275, 270), (868, 303), (119, 275), (278, 271), (384, 804), (823, 1159),
(592, 696), (103, 789), (108, 97), (698, 368), (1177, 259), (337, 116), (498, 303), (579, 260),
(276, 103), (647, 628), (503, 296), (112, 336), (479, 385), (746, 270), (108, 111), (115, 97),
(110, 459), (769, 302), (409, 1160), (281, 386), (968, 434), (103, 111), (358, 109), (108, 259),
(354, 423), (447, 569), (1116, 108), (538, 435), (571, 326), (283, 303), (701, 32), (1087, 116),
(273, 270), (261, 271), (952, 114), (341, 1188), (494, 272), (1207, 587), (256, 334), (109, 333),
(299, 109), (665, 1182), (813, 365), (119, 266), (112, 389), (276, 271), (1220, 110), (299, 264),
(285, 34), (116, 302), (279, 110), (357, 103), (341, 1203), (378, 352), (281, 118), (289, 270),
(1068, 1228), (332, 32), (1153, 1211), (325, 99), (341, 1149), (109, 506), (588, 264), (269, 258),
(1232, 1085), (304, 103), (1074, 490), (1082, 469), (98, 313), (1155, 1236), (316, 464), (799, 308),
(693, 273), (103, 114), (572, 102), (360, 98), (273, 100), (281, 417), (283, 454), (269, 116),
(283, 412), (1210, 329), (98, 114), (98, 270), (526, 282), (360, 112), (116, 293), (419, 275),
(101, 112), (117, 287), (110, 548), (121, 277), (261, 116), (112, 117), (116, 379), (265, 272),
(354, 108), (467, 272), (1093, 364), (259, 1247), (288, 103), (1276, 1205), (116, 506), (121, 262),
(433, 275), (103, 318), (276, 370), (114, 1206), (305, 101), (312, 112), (398, 271), (46, 1157),
(101, 336), (317, 108), (118, 276), (299, 360), (104, 101), (116, 309), (261, 256), (433, 350),
(442, 369), (826, 318), (1219, 438), (100, 265), (104, 609), (261, 258), (279, 427), (289, 108),
(452, 1273), (474, 410), (108, 412), (263, 1283), (269, 264), (277, 109), (457, 32), (614, 256),
(327, 264), (265, 100), (265, 103), (325, 105), (310, 943), (313, 104), (453, 374), (102, 286),
(816, 107), (109, 117), (556, 355), (110, 749), (345, 666), (277, 260), (310, 869), (348, 408),
(1304, 959), (110, 526), (286, 32), (345, 115), (510, 456), (703, 264), (1161, 486), (1299, 105),
(411, 114), (673, 891), (343, 116), (383, 600), (283, 396), (298, 738), (401, 275), (98, 1227),
(115, 862), (304, 99), (1195, 369), (452, 838), (890, 1073), (1078, 765), (1295, 100), (1310, 1148),
(1347, 1351), (433, 583), (444, 112), (765, 1086), (1041, 1328), (367, 375), (371, 1279), (497, 261),
(1358, 272), (505, 264), (100, 279), (287, 266), (1362, 98), (269, 881), (314, 329), (1103, 1271),
(1180, 1231), (531, 334), (393, 115), (1217, 100), (1317, 266), (1234, 1245), (354, 573), (488, 98),
(408, 288), (1336, 32), (100, 313), (464, 121), (1174, 1229), (1375, 361), (116, 258), (308, 110),
(1381, 1213), (739, 32), (380, 107), (384, 102), (1327, 1199), (1374, 262), (348, 1168), (1274, 603),
(354, 445), (645, 267), (1176, 277), (1350, 104), (115, 301), (594, 1343), (915, 740), (104, 573),
(295, 434), (344, 399), (101, 311), (782, 100), (298, 104), (1115, 434), (1243, 257), (1372, 1216),
(287, 1118), (97, 118), (110, 293), (430, 1267), (648, 1291), (292, 738), (1395, 1212), (117, 281),
(1399, 445), (105, 304), (273, 387), (343, 621), (541, 636), (1184, 1418), (1341, 116), (1419, 117),
(101, 490), (102, 332), (342, 513), (97, 272), (103, 101), (276, 754), (1179, 1376), (100, 271),
(259, 1410), (1332, 45), (112, 281), (1433, 1334), (316, 749), (492, 506), (1202, 482), (106, 111),
(455, 116), (741, 260), (1238, 122), (299, 104), (689, 643), (1354, 1309), (114, 257), (283, 271),
(1090, 270), (1187, 264), (32, 260), (106, 286), (109, 1437), (416, 266), (1089, 1192), (307, 374),
(1198, 283), (1290, 117), (1339, 296), (288, 118), (304, 287), (1285, 686), (1421, 405), (119, 350),
(259, 338), (444, 513), (1235, 1268), (381, 297), (1261, 1361), (299, 1152), (1166, 1156), (384, 99),
(786, 1208), (1089, 478), (1346, 280), (1445, 1407), (298, 257), (437, 269), (1289, 108), (384, 629),
(607, 862), (110, 502), (115, 796), (401, 369), (407, 347), (1408, 1480), (119, 114), (1296, 303),
(372, 379), (373, 266), (407, 410), (418, 112), (782, 310), (100, 257), (633, 270), (39, 1446),
(276, 614), (444, 1252), (647, 115), (673, 1165), (98, 276), (115, 881), (497, 289), (752, 318),
(112, 105), (289, 105), (399, 32), (752, 312), (781, 256), (1502, 1241), (267, 115), (350, 352),
(510, 116), (1281, 256), (332, 426), (357, 410), (612, 1364), (111, 115), (379, 118), (441, 115),
(263, 314), (1272, 347), (384, 116), (1222, 256), (100, 1118), (280, 295), (348, 102), (1240, 1355),
(109, 421), (258, 460), (312, 313), (1320, 318), (1530, 117), (111, 267), (1335, 303), (1342, 277),
(100, 314), (119, 262), (739, 271), (1251, 1488), (1321, 372), (1442, 368), (1496, 1158), (121, 301),
(1003, 599), (110, 261), (372, 1478), (594, 1509), (975, 684), (1540, 445), (98, 281), (1394, 1487),
(279, 256), (316, 405), (456, 1428), (669, 959), (672, 299), (1163, 722), (1329, 1533), (100, 1167),
(1559, 102), (108, 494), (115, 111), (266, 116), (281, 106), (367, 1514), (102, 108), (102, 350),
(110, 318), (393, 497), (111, 332), (348, 97), (1041, 1555), (100, 318), (372, 104), (1078, 1201),
(1344, 257), (116, 276), (278, 302), (608, 100), (1201, 1086), (1477, 1266), (110, 262), (1549, 1472),
(99, 336), (281, 284), (283, 302), (357, 281), (437, 1330), (680, 366), (1275, 352), (1463, 482),
(99, 107), (109, 257), (465, 1262), (416, 1288), (586, 296), (1263, 302), (1482, 1424), (101, 263),
(108, 396), (109, 332), (115, 635), (260, 259), (269, 666), (99, 349), (103, 366), (276, 693),
(1430, 593), (98, 389), (111, 98), (263, 1302), (298, 99), (1257, 1497), (1314, 357), (1588, 1546),
(270, 295), (316, 99), (1492, 1429), (291, 639), (1589, 1569), (447, 838), (685, 1148), (1554, 509),
(1621, 1622), (115, 463), (298, 101), (975, 329), (1539, 112), (327, 275), (103, 457), (110, 462),
(116, 1308), (313, 296), (750, 890), (1244, 286), (1380, 1333), (1422, 643), (1459, 273), (1557, 326),
(366, 112), (703, 1225), (1197, 276), (269, 301), (816, 379), (1162, 1223), (327, 438), (360, 311),
(281, 1427), (290, 793), (353, 121), (355, 327), (1571, 762), (1574, 1651), (102, 379), (263, 271),
(443, 260), (1466, 719), (1634, 1500), (108, 526), (287, 1386), (291, 308), (582, 405), (1660, 1662),
(1661, 486), (386, 275), (1606, 32), (259, 118), (298, 378), (393, 342), (396, 294), (469, 299),
(1373, 1352), (1638, 99), (311, 101), (342, 1493), (696, 256), (807, 264), (1650, 1495), (109, 121),
(273, 271), (1378, 1469), (314, 112), (1249, 279), (1598, 1653), (287, 1184), (298, 264), (344, 1685),
(1467, 699), (1677, 1278), (98, 1383), (263, 375), (305, 347), (938, 376), (1090, 454), (1613, 1464),
(1687, 105), (473, 336), (786, 541), (1187, 115), (1237, 280), (110, 596), (291, 1646), (301, 294),
(310, 722), (392, 272), (1440, 1545), (1483, 272), (1665, 601), (98, 457), (109, 299), (117, 100),
(333, 256), (378, 347), (451, 1592), (1709, 115), (312, 290), (409, 550), (490, 260), (781, 277),
(1250, 116), (258, 605), (310, 1168), (372, 359), (438, 307), (1331, 495), (98, 379), (354, 115),
(612, 705), (1701, 32), (104, 1265), (115, 394), (807, 287), (1151, 1723), (1387, 1604), (119, 859),
(259, 1297), (261, 105), (298, 107), (313, 438), (367, 289), (442, 1303), (592, 1740), (1083, 287),
(1379, 368), (1735, 341), (102, 369), (372, 281), (608, 431), (1246, 271), (1284, 1088), (1370, 342),
(259, 307), (488, 630), (597, 256), (1435, 264), (1449, 1452), (281, 103), (1179, 1609), (1465, 105),
(1714, 394), (373, 300), (41, 271), (281, 109), (298, 435), (355, 103), (1326, 406), (1479, 574),
(99, 121), (260, 577), (336, 262), (1461, 668), (1657, 258), (1696, 326), (1715, 293), (350, 108),
(579, 1632), (700, 1312), (1202, 108), (1528, 1348), (312, 1322), (325, 593), (381, 283), (413, 1301),
(570, 444), (601, 271), (1250, 264), (1269, 381), (1532, 627), (1776, 1702), (261, 110), (283, 121),
(308, 410), (336, 1504), (436, 32), (476, 257), (1384, 622), (1793, 306), (111, 302), (116, 495),
(118, 380), (358, 1393), (433, 313), (591, 259), (680, 440), (1800, 354), (103, 336), (105, 112),
(276, 1167), (472, 1264), (799, 262), (1666, 554), (1780, 256), (1809, 399), (115, 109), (263, 701),
(624, 859), (1420, 417), (1524, 256), (1607, 648), (1745, 1699), (1788, 1560), (1805, 1629), (98, 306),
(118, 293), (515, 276), (689, 1165), (1083, 437), (1097, 355), (1690, 423), (102, 117), (118, 349),
(382, 626), (1175, 1340), (1582, 45), (121, 638), (288, 372), (439, 115), (781, 108), (993, 812),
(1178, 119), (1293, 1577), (1516, 264), (1664, 296), (116, 277), (118, 289), (295, 279), (328, 421),
(331, 368), (442, 1626), (687, 1242), (703, 116), (1176, 471), (1803, 1152), (1850, 554), (281, 1164),
(393, 259), (443, 1761), (617, 515), (915, 280), (1093, 459), (1371, 1648), (1468, 1683), (1717, 1857),
(1804, 299), (259, 99), (259, 1801), (266, 310), (298, 296), (342, 287), (1162, 270), (1186, 442),
(1226, 272), (1240, 1580), (1260, 1652), (1520, 271), (1725, 307), (1777, 404), (1806, 304), (97, 1172),
(98, 1505), (105, 118), (325, 116), (629, 347), (1541, 260), (1789, 334), (1802, 107), (98, 261),
(99, 383), (258, 1300), (280, 1360), (344, 263), (436, 412), (523, 270), (682, 260), (1640, 638),
(1742, 405), (357, 719), (381, 275), (453, 1896), (464, 692), (568, 1596), (702, 1771), (1315, 1519),
(1411, 1837), (1432, 1846), (1819, 554), (1838, 1765), (1848, 1368), (1856, 1724), (1863, 1455), (1876, 1902),
(1911, 1899), (99, 279), (267, 373), (331, 1318), (331, 1368), (358, 280), (466, 1858), (601, 1529),
(659, 287), (725, 1565), (1171, 1457), (1360, 1916), (1365, 1753), (1397, 585), (1444, 1923), (1451, 282),
(1474, 1719), (1485, 1924), (1656, 1877), (1668, 754), (1703, 1904), (1704, 626), (1728, 1151), (1772, 1778),
(1820, 1705), (1826, 1931), (1833, 619), (1894, 1935), (1905, 1919), (1906, 1940), (1908, 1921), (1909, 1941),
(1912, 1938), (1918, 1930), (1926, 299), (1928, 1942), (1939, 1932), (1943, 1946), (1945, 1944), (1947, 1948),
(306, 103), (455, 711), (587, 310), (1230, 101), (1242, 1603), (1810, 100), (1832, 295), (1956, 1958),
(265, 256), (314, 684), (473, 117), (695, 357), (731, 1318), (733, 294), (1326, 293), (1456, 1412),
(1507, 1721), (1562, 301), (1601, 457), (1658, 1490), (1748, 1953), (1767, 295), (1811, 670), (1972, 1964),
(45, 1237), (99, 1265), (107, 101), (612, 1413), (1257, 1822), (1292, 276), (1371, 602), (1475, 256),
(1537, 548), (1670, 1974), (1681, 1976), (1769, 1973), (1787, 1890), (1825, 1969), (1959, 1968), (1965, 1783),
(1980, 1985), (1987, 1499), (1988, 1992), (1990, 1991), (1994, 1993), (309, 1259), (343, 99), (380, 114),
(408, 1312), (1486, 283), (1512, 286), (1747, 375), (1901, 1949), (1995, 1915), (455, 1808), (1097, 309),
(1258, 295), (1388, 609), (1498, 420), (1879, 265), (1996, 1849), (383, 32), (568, 2005), (638, 110),
(1185, 307), (1708, 1348), (101, 603), (105, 348), (109, 684), (116, 119), (121, 45), (317, 441),
(1277, 1618), (1367, 1453), (1619, 1369), (1784, 549), (1841, 435), (1954, 1170), (98, 1494), (455, 267),
(587, 298), (1402, 686), (97, 1506), (498, 281), (1630, 762), (1716, 476), (1982, 302), (103, 394),
(104, 638), (108, 354), (276, 105), (304, 109), (312, 1324), (613, 1986), (742, 1322), (1074, 112),
];

// ═══════════════════════════════════════════════════════════════
// BPE ENCODER
// ═══════════════════════════════════════════════════════════════

fn bpe_encode(text: &str) -> Vec<u16> {
    let mut seq: Vec<u16> = text.bytes()
        .map(|b| if b >= b'A' && b <= b'Z' { (b - b'A' + b'a') as u16 } else { b as u16 })
        .collect();
    for (m, &(left, right)) in BPE_TABLE.iter().enumerate() {
        let new_id = 256 + m as u16;
        let mut j = 0;
        let mut i = 0;
        let len = seq.len();
        while i < len {
            if i < len - 1 && seq[i] == left && seq[i + 1] == right {
                seq[j] = new_id;
                j += 1;
                i += 2;
            } else {
                seq[j] = seq[i];
                j += 1;
                i += 1;
            }
        }
        seq.truncate(j);
    }
    seq
}

/// Decode a single BPE token ID back to a string.
fn bpe_decode_token(id: u16) -> String {
    if (id as usize) < 256 {
        return String::from(id as u8 as char);
    }
    let m = id as usize - 256;
    if m < BPE_MERGES {
        let (left, right) = BPE_TABLE[m];
        let mut s = bpe_decode_token(left);
        s.push_str(&bpe_decode_token(right));
        return s;
    }
    String::new()
}

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
"debt","credit","interest","principal",
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

fn rmsnorm_slice(x: &[f32], g: &[f32]) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-5;
    let inv = 1.0 / ss.sqrt();
    (0..n).map(|i| g[i] * x[i] * inv).collect()
}

// ═══════════════════════════════════════════════════════════════
// MODEL -- v7 Resonance: 8 sequential layers with
// multi-head attention + RoPE + RRPRAM resonance gate + SwiGLU
// ═══════════════════════════════════════════════════════════════

struct LayerWeights {
    attn_norm: Vec<f32>, // [DIM]         pre-attention RMSNorm
    wq: Vec<f32>,        // [DIM * DIM]   query projection
    wk: Vec<f32>,        // [DIM * DIM]   key projection
    wv: Vec<f32>,        // [DIM * DIM]   value projection
    wo: Vec<f32>,        // [DIM * DIM]   output projection
    wr: Vec<f32>,        // [DIM * DIM]   RRPRAM resonance
    gate: [f32; 2],      // blend QKV + RRPRAM
    ffn_norm: Vec<f32>,  // [DIM]         pre-FFN RMSNorm
    w_gate: Vec<f32>,    // [DIM * HDIM]  SwiGLU gate (HDIM > DIM)
    w_up: Vec<f32>,      // [DIM * HDIM]  SwiGLU up
    w_down: Vec<f32>,    // [HDIM * DIM]  SwiGLU down
}

impl LayerWeights {
    fn new(rng: &mut Rng) -> Self {
        let sd = (2.0f32 / DIM as f32).sqrt();
        let sh = (2.0f32 / HDIM as f32).sqrt();
        LayerWeights {
            attn_norm: vec![1.0; DIM],
            wq:        (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            wk:        (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            wv:        (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            wo:        (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            wr:        (0..DIM*DIM).map(|_| rng.randn() * sd).collect(),
            gate:      [0.0, 0.0],
            ffn_norm:  vec![1.0; DIM],
            w_gate:    (0..DIM*HDIM).map(|_| rng.randn() * sd).collect(),
            w_up:      (0..DIM*HDIM).map(|_| rng.randn() * sd).collect(),
            w_down:    (0..HDIM*DIM).map(|_| rng.randn() * sh).collect(),
        }
    }

    fn param_count() -> usize {
        // attn_norm + wq + wk + wv + wo + wr + gate + ffn_norm + w_gate + w_up + w_down
        DIM + DIM*DIM*5 + 2 + DIM + DIM*HDIM*2 + HDIM*DIM
    }
}

struct Penelope {
    // Global weights
    tok_emb: Vec<f32>,     // [BPE_VOCAB * DIM]  token embedding
    pos_emb: Vec<f32>,     // [MAX_SEQ * DIM]    positional embedding
    final_norm: Vec<f32>,  // [DIM]              final RMSNorm
    lm_head: Vec<f32>,     // [BPE_VOCAB * DIM]  language model head
    // Per-layer
    layers: Vec<LayerWeights>,
}

impl Penelope {
    fn new(rng: &mut Rng) -> Self {
        let scale_d = (2.0f32 / DIM as f32).sqrt();
        let scale_bpe = (2.0f32 / BPE_VOCAB as f32).sqrt();
        Penelope {
            tok_emb:    (0..BPE_VOCAB*DIM).map(|_| rng.randn() * scale_bpe).collect(),
            pos_emb:    (0..MAX_SEQ*DIM).map(|_| rng.randn() * 0.02).collect(),
            final_norm: vec![1.0; DIM],
            lm_head:    (0..BPE_VOCAB*DIM).map(|_| rng.randn() * scale_d).collect(),
            layers:     (0..N_LAYERS).map(|_| LayerWeights::new(rng)).collect(),
        }
    }

    fn param_count() -> usize {
        let global = BPE_VOCAB * DIM + MAX_SEQ * DIM + DIM + BPE_VOCAB * DIM;
        global + N_LAYERS * LayerWeights::param_count()
    }

    /// Full forward pass through all 8 layers for a sequence of BPE tokens.
    /// Returns BPE logits [BPE_VOCAB] for the LAST token position.
    fn forward(&self, bpe_ids: &[u16]) -> Vec<f32> {
        let s = bpe_ids.len().max(1).min(MAX_SEQ);
        let seq_len = s;

        // x: [S * DIM] -- residual stream
        let mut x = vec![0.0f32; seq_len * DIM];

        // embed: tok_emb + pos_emb
        for t in 0..seq_len {
            let tok = bpe_ids[t] as usize;
            let tok = if tok >= BPE_VOCAB { 0 } else { tok };
            for d in 0..DIM {
                x[t * DIM + d] = self.tok_emb[tok * DIM + d] + self.pos_emb[t * DIM + d];
            }
        }

        // scratch buffers
        let mut h   = vec![0.0f32; seq_len * DIM];
        let mut q   = vec![0.0f32; seq_len * DIM];
        let mut k   = vec![0.0f32; seq_len * DIM];
        let mut v   = vec![0.0f32; seq_len * DIM];
        let mut att = vec![0.0f32; seq_len * seq_len * N_HEADS];
        let mut av  = vec![0.0f32; seq_len * DIM];
        let mut qkv_out = vec![0.0f32; seq_len * DIM];
        let mut rrp = vec![0.0f32; seq_len * DIM];
        let mut h2  = vec![0.0f32; seq_len * DIM];
        let mut fg  = vec![0.0f32; seq_len * HDIM];
        let mut fu  = vec![0.0f32; seq_len * HDIM];
        let mut sw  = vec![0.0f32; seq_len * HDIM];
        let mut fd  = vec![0.0f32; seq_len * DIM];

        for l in 0..N_LAYERS {
            let lw = &self.layers[l];

            // 1. h = rmsnorm(x, attn_norm) for each position
            for t in 0..seq_len {
                let hslice = rmsnorm_slice(&x[t*DIM..(t+1)*DIM], &lw.attn_norm);
                h[t*DIM..(t+1)*DIM].copy_from_slice(&hslice);
            }

            // 2-3. q = h @ wq, k = h @ wk, v = h @ wv (per position)
            for t in 0..seq_len {
                let ht = &h[t*DIM..(t+1)*DIM];
                let qt = matmul_mv(&lw.wq, ht, DIM, DIM);
                let kt = matmul_mv(&lw.wk, ht, DIM, DIM);
                let vt = matmul_mv(&lw.wv, ht, DIM, DIM);
                q[t*DIM..(t+1)*DIM].copy_from_slice(&qt);
                k[t*DIM..(t+1)*DIM].copy_from_slice(&kt);
                v[t*DIM..(t+1)*DIM].copy_from_slice(&vt);
            }

            // Apply RoPE to q and k (layout: [S, N_HEADS, HEAD_DIM])
            apply_rope(&mut q, &mut k, seq_len);

            // 5. Multi-head causal attention: softmax(q @ k^T / sqrt(head_dim))
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            for hd in 0..N_HEADS {
                for ti in 0..seq_len {
                    let qi_off = (ti * N_HEADS + hd) * HEAD_DIM;
                    let mut maxs = -1e30f32;
                    for tj in 0..=ti {
                        let kj_off = (tj * N_HEADS + hd) * HEAD_DIM;
                        let mut dot = 0.0f32;
                        for d in 0..HEAD_DIM {
                            dot += q[qi_off + d] * k[kj_off + d];
                        }
                        dot *= scale;
                        att[(hd * seq_len + ti) * seq_len + tj] = dot;
                        if dot > maxs { maxs = dot; }
                    }
                    // softmax
                    let mut sum = 0.0f32;
                    for tj in 0..=ti {
                        let val = (att[(hd * seq_len + ti) * seq_len + tj] - maxs).exp();
                        att[(hd * seq_len + ti) * seq_len + tj] = val;
                        sum += val;
                    }
                    let inv_s = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for tj in 0..=ti {
                        att[(hd * seq_len + ti) * seq_len + tj] *= inv_s;
                    }
                    for tj in (ti+1)..seq_len {
                        att[(hd * seq_len + ti) * seq_len + tj] = 0.0;
                    }
                }
            }

            // 6. attn @ v, reshape, then @ wo
            for val in av.iter_mut() { *val = 0.0; }
            for hd in 0..N_HEADS {
                for ti in 0..seq_len {
                    let avi_off = (ti * N_HEADS + hd) * HEAD_DIM;
                    for tj in 0..=ti {
                        let a = att[(hd * seq_len + ti) * seq_len + tj];
                        if a == 0.0 { continue; }
                        let vj_off = (tj * N_HEADS + hd) * HEAD_DIM;
                        for d in 0..HEAD_DIM {
                            av[avi_off + d] += a * v[vj_off + d];
                        }
                    }
                }
            }
            // av is [S, DIM] (concatenated heads). Project through wo
            for t in 0..seq_len {
                let out = matmul_mv(&lw.wo, &av[t*DIM..(t+1)*DIM], DIM, DIM);
                qkv_out[t*DIM..(t+1)*DIM].copy_from_slice(&out);
            }

            // 7. RRPRAM resonance: rrp = h @ wr
            for t in 0..seq_len {
                let out = matmul_mv(&lw.wr, &h[t*DIM..(t+1)*DIM], DIM, DIM);
                rrp[t*DIM..(t+1)*DIM].copy_from_slice(&out);
            }

            // 8. gate_weights = softmax(gate[0], gate[1])
            let g0 = lw.gate[0];
            let g1 = lw.gate[1];
            let gmax = g0.max(g1);
            let e0 = (g0 - gmax).exp();
            let e1 = (g1 - gmax).exp();
            let gsum = e0 + e1;
            let w0 = e0 / gsum;
            let w1 = e1 / gsum;

            // 9. x = x + w0 * qkv_out + w1 * rrp (residual)
            for i in 0..(seq_len * DIM) {
                x[i] += w0 * qkv_out[i] + w1 * rrp[i];
            }

            // 10. h2 = rmsnorm(x, ffn_norm)
            for t in 0..seq_len {
                let h2slice = rmsnorm_slice(&x[t*DIM..(t+1)*DIM], &lw.ffn_norm);
                h2[t*DIM..(t+1)*DIM].copy_from_slice(&h2slice);
            }

            // 11. SwiGLU FFN: x = x + w_down @ (silu(h2 @ w_gate) * (h2 @ w_up))
            for t in 0..seq_len {
                let h2t = &h2[t*DIM..(t+1)*DIM];
                let fgt = matmul_mv(&lw.w_gate, h2t, HDIM, DIM);
                let fut = matmul_mv(&lw.w_up, h2t, HDIM, DIM);
                for i in 0..HDIM {
                    fg[t*HDIM + i] = fgt[i];
                    fu[t*HDIM + i] = fut[i];
                    sw[t*HDIM + i] = silu(fgt[i]) * fut[i];
                }
                let fdt = matmul_mv(&lw.w_down, &sw[t*HDIM..(t+1)*HDIM], DIM, HDIM);
                for d in 0..DIM {
                    fd[t*DIM + d] = fdt[d];
                    x[t*DIM + d] += fdt[d];
                }
            }
        }

        // After all layers: final rmsnorm + lm_head for LAST position
        let xn = rmsnorm_slice(&x[(seq_len-1)*DIM..seq_len*DIM], &self.final_norm);
        matmul_mv(&self.lm_head, &xn, BPE_VOCAB, DIM)
    }

    fn save(&self, path: &str) {
        let mut data: Vec<u8> = Vec::new();
        // PEN7 header: magic, BPE_VOCAB, NWORDS, DIM, HDIM, N_HEADS, N_LAYERS, MAX_SEQ
        let header: [u32; 8] = [
            0x50454E37, BPE_VOCAB as u32, NWORDS as u32, DIM as u32,
            HDIM as u32, N_HEADS as u32, N_LAYERS as u32, MAX_SEQ as u32,
        ];
        for &v in &header { data.extend_from_slice(&v.to_le_bytes()); }
        // Global weights
        for &v in &self.tok_emb { data.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.pos_emb { data.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.final_norm { data.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.lm_head { data.extend_from_slice(&v.to_le_bytes()); }
        // Per-layer weights
        for lw in &self.layers {
            for &v in &lw.attn_norm { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.wq { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.wk { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.wv { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.wo { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.wr { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.gate { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.ffn_norm { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.w_gate { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.w_up { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lw.w_down { data.extend_from_slice(&v.to_le_bytes()); }
        }
        fs::write(path, &data).expect("save failed");
        let expected = 32 + Self::param_count() * 4;
        println!("  saved {}: {} params ({:.1}MB) [{}]", path, Self::param_count(),
                 data.len() as f64 / 1e6,
                 if data.len() == expected { "OK" } else { "SIZE MISMATCH!" });
    }

    fn load(&mut self, path: &str) -> bool {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("  cannot open {}: {}", path, e); return false; }
        };
        if data.len() < 32 { eprintln!("  file too small"); return false; }
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());

        if magic != 0x50454E37 {
            eprintln!("  unknown format magic=0x{:08X} (expected PEN7=0x50454E37)", magic);
            return false;
        }
        let bv = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let nw = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let d  = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let hd = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let nh = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let nl = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let ms = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
        if bv != BPE_VOCAB || nw != NWORDS || d != DIM || hd != HDIM ||
           nh != N_HEADS || nl != N_LAYERS || ms != MAX_SEQ {
            eprintln!("  v7 config mismatch: BV={} V={} D={} H={} NH={} NL={} S={}",
                      bv, nw, d, hd, nh, nl, ms);
            return false;
        }
        let floats: Vec<f32> = data[32..].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut o = 0;
        // Global weights
        self.tok_emb = floats[o..o+BPE_VOCAB*DIM].to_vec(); o += BPE_VOCAB*DIM;
        self.pos_emb = floats[o..o+MAX_SEQ*DIM].to_vec(); o += MAX_SEQ*DIM;
        self.final_norm = floats[o..o+DIM].to_vec(); o += DIM;
        self.lm_head = floats[o..o+BPE_VOCAB*DIM].to_vec(); o += BPE_VOCAB*DIM;
        // Per-layer
        for lw in &mut self.layers {
            lw.attn_norm = floats[o..o+DIM].to_vec(); o += DIM;
            lw.wq = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            lw.wk = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            lw.wv = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            lw.wo = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            lw.wr = floats[o..o+DIM*DIM].to_vec(); o += DIM*DIM;
            lw.gate = [floats[o], floats[o+1]]; o += 2;
            lw.ffn_norm = floats[o..o+DIM].to_vec(); o += DIM;
            lw.w_gate = floats[o..o+DIM*HDIM].to_vec(); o += DIM*HDIM;
            lw.w_up = floats[o..o+DIM*HDIM].to_vec(); o += DIM*HDIM;
            lw.w_down = floats[o..o+HDIM*DIM].to_vec(); o += HDIM*DIM;
        }
        println!("  loaded v7 {}: {} params ({:.1}MB)", path, Self::param_count(),
                 Self::param_count() as f64 * 4.0 / 1e6);
        true
    }
}

// ═══════════════════════════════════════════════════════════════
// ROPE -- rotary position embedding
// ═══════════════════════════════════════════════════════════════

fn apply_rope(q: &mut [f32], k: &mut [f32], seq_len: usize) {
    let theta_base: f32 = 10000.0;
    for t in 0..seq_len {
        for h in 0..N_HEADS {
            let base = (t * N_HEADS + h) * HEAD_DIM;
            for d in 0..(HEAD_DIM / 2) {
                let freq = 1.0 / theta_base.powf(2.0 * d as f32 / HEAD_DIM as f32);
                let cos_f = (t as f32 * freq).cos();
                let sin_f = (t as f32 * freq).sin();
                // rotate q
                let q0 = q[base + d];
                let q1 = q[base + d + HEAD_DIM/2];
                q[base + d]              = q0 * cos_f - q1 * sin_f;
                q[base + d + HEAD_DIM/2] = q0 * sin_f + q1 * cos_f;
                // rotate k
                let k0 = k[base + d];
                let k1 = k[base + d + HEAD_DIM/2];
                k[base + d]              = k0 * cos_f - k1 * sin_f;
                k[base + d + HEAD_DIM/2] = k0 * sin_f + k1 * cos_f;
            }
        }
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

const SUFFIXES: [&str; 38] = [
    "ting","ning","ring","ling","ding","ping","bing","ging","ming","king",
    "sing","zing",
    "ing","ment","ness","tion","sion","able","ible","ence","ance",
    "eous","ious","ful","less","ize","ise","ous","ive","ity",
    "ly","er","ed","est","al","en","es","s",
];

/// Strip suffix, try exact match on stem, stem+"e", or doubled-consonant removal.
fn try_stem(word: &str, idx: &HashMap<String, usize>) -> Option<usize> {
    let wlen = word.len();
    for &suffix in &SUFFIXES {
        let slen = suffix.len();
        if wlen <= slen + 2 { continue; }
        if !word.ends_with(suffix) { continue; }
        let sl = wlen - slen;
        let stem = &word[..sl];

        // exact stem
        if let Some(&i) = idx.get(stem) { return Some(i); }

        // stem + 'e' (creat→create, danc→dance)
        let mut stem_e = String::with_capacity(sl + 1);
        stem_e.push_str(stem);
        stem_e.push('e');
        if let Some(&i) = idx.get(stem_e.as_str()) { return Some(i); }

        // doubled consonant (runn→run, swimm→swim)
        let stem_bytes = stem.as_bytes();
        if sl >= 3 && stem_bytes[sl - 1] == stem_bytes[sl - 2] {
            let shortened = &stem[..sl - 1];
            if let Some(&i) = idx.get(shortened) { return Some(i); }
        }
    }
    None
}

/// Greedy longest vocab match within a word (BPE decomposition).
fn greedy_vocab_match(word: &str, idx: &HashMap<String, usize>) -> Vec<usize> {
    let mut ids = Vec::new();
    let wlen = word.len();
    let mut pos = 0;
    while pos < wlen && ids.len() < 8 {
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (vw, &vi) in idx.iter() {
            let vl = vw.len();
            if vl <= best_len || vl > wlen - pos { continue; }
            if word[pos..].starts_with(vw.as_str()) {
                best = Some(vi);
                best_len = vl;
            }
        }
        if let Some(b) = best {
            if best_len >= 3 {
                ids.push(b);
                pos += best_len;
            } else {
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
    ids
}

fn tokenize_vocab(text: &str, idx: &HashMap<String, usize>) -> Vec<usize> {
    let mut ids = Vec::new();
    for word in text.to_lowercase().split(|c: char| !c.is_alphabetic()) {
        if word.len() < 2 || is_stop(word) { continue; }

        // 1. exact vocab match
        if let Some(&i) = idx.get(word) {
            ids.push(i);
            continue;
        }

        // 2. stem + match
        if let Some(i) = try_stem(word, idx) {
            ids.push(i);
            continue;
        }

        // 3. greedy longest vocab match (BPE decomposition)
        let subs = greedy_vocab_match(word, idx);
        for &s in &subs {
            if ids.is_empty() || *ids.last().unwrap() != s {
                ids.push(s);
            }
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
        let depth = step as f32 / N_LAYERS as f32;
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
// BPE LOGITS TO WORD SCORES
// word_score(w) = mean(bpe_logits[tok] for tok in word's BPE tokens)
// ═══════════════════════════════════════════════════════════════

fn bpe_logits_to_word_scores(bpe_logits: &[f32], ext_vocab: &[ExtWord]) -> Vec<f32> {
    let mut scores = vec![0.0f32; ext_vocab.len()];
    for (w, ew) in ext_vocab.iter().enumerate() {
        let bl = ew.bpe_ids.len();
        let mut score = 0.0f32;
        for &tok in &ew.bpe_ids {
            let t = tok as usize;
            if t < BPE_VOCAB { score += bpe_logits[t]; }
        }
        scores[w] = if bl > 0 { score / bl as f32 } else { 0.0 };
    }
    scores
}

// ═══════════════════════════════════════════════════════════════
// TRAINING -- BPE-level next-token prediction
// ═══════════════════════════════════════════════════════════════

fn train(model: &mut Penelope, data_path: &str, steps: usize, lr: f32) {
    let text = match fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("  cannot open {}: {}", data_path, e); return; }
    };

    // BPE tokenize entire corpus
    let corpus_bpe = bpe_encode(&text);
    if corpus_bpe.len() < MAX_SEQ + 1 {
        eprintln!("  corpus too small: {} BPE tokens (need {}+)", corpus_bpe.len(), MAX_SEQ + 1);
        return;
    }

    println!("  corpus: {} bytes -> {} BPE tokens", text.len(), corpus_bpe.len());
    println!("  model: {} params ({:.1}MB f32)", Penelope::param_count(),
             Penelope::param_count() as f64 * 4.0 / 1e6);
    println!("  architecture: {} layers, {} heads, dim={}, hdim={}", N_LAYERS, N_HEADS, DIM, HDIM);
    println!("  training: {} steps, lr={:.1e}, seq={}", steps, lr, MAX_SEQ);
    println!("  NOTE: Rust trainer uses forward-only loss (for full training, export weights)");

    let mut rng = Rng::new(42);
    let mut best_loss = f32::INFINITY;

    for step in 1..=steps {
        // Sample a random window from corpus
        let seq_len = MAX_SEQ.min(corpus_bpe.len() - 1);
        let start = rng.randint(corpus_bpe.len() - seq_len);
        let ctx = &corpus_bpe[start..start + seq_len];
        let target = corpus_bpe[start + seq_len] as usize;

        // Forward pass: predict next token from context
        let logits = model.forward(ctx);
        let probs = softmax(&logits);

        let p = probs[if target < BPE_VOCAB { target } else { 0 }].max(1e-10);
        let loss = -p.ln();

        if loss < best_loss { best_loss = loss; }

        if step % 50 == 0 || step == 1 {
            println!("  step {:5}/{} loss={:.4} best={:.4} (target={} p={:.4})",
                     step, steps, loss, best_loss, target, p);
        }

        // Shallow gradient: nudge tok_emb for target toward context average
        let scale = lr * 0.1;
        let n_ctx = seq_len.min(8);
        for d in 0..DIM {
            let mut avg_ctx = 0.0f32;
            for i in (seq_len - n_ctx)..seq_len {
                avg_ctx += model.tok_emb[ctx[i] as usize * DIM + d];
            }
            avg_ctx /= n_ctx as f32;
            let tidx = if target < BPE_VOCAB { target } else { 0 };
            model.tok_emb[tidx * DIM + d] += scale * (avg_ctx - model.tok_emb[tidx * DIM + d]);
        }
    }

    println!("  training complete. best loss: {:.4}", best_loss);
    println!("  NOTE: for full training, use PyTorch with PEN7 weight export");
}

// ═══════════════════════════════════════════════════════════════
// GENERATION -- autoregressive BPE, then word-level output
//
// Dual tokenizer: soul thinks in BPE (2048), mouth speaks in words (1984).
// At each step:
//   1. Forward pass -> BPE logits
//   2. Compute word scores = mean(logits for word's BPE tokens)
//   3. Apply Dario overlay on word scores
//   4. Sample word, print it
//   5. Append word's BPE tokens to context for next step
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

/// Precompute BPE encoding for each vocab word (for generation).
fn init_vocab_bpe() -> Vec<Vec<u16>> {
    (0..NWORDS).map(|i| bpe_encode(VOCAB[i])).collect()
}

/// Extended vocab entry: word string + BPE token IDs + origin flag.
#[derive(Clone)]
struct ExtWord {
    word: String,
    bpe_ids: Vec<u16>,
    from_hardcoded: bool,
}

const SUFFIX_FRAGMENTS: [&str; 20] = [
    "ing","tion","ment","ness","ble","ful","ous","ive","ent","ant",
    "ist","ity","ght","est","ter","ther","ted","ting","ally","ling",
];

/// Build extended vocab: hardcoded 1984 words + BPE tokens that are whole words.
fn init_ext_vocab(vocab_bpe: &[Vec<u16>]) -> Vec<ExtWord> {
    let mut ext: Vec<ExtWord> = Vec::new();
    let mut existing: HashSet<String> = HashSet::new();

    // 1. Add all 1984 hardcoded words
    for i in 0..NWORDS {
        if ext.len() >= MAX_EXT_VOCAB { break; }
        let w = VOCAB[i].to_string();
        existing.insert(w.clone());
        ext.push(ExtWord {
            word: w,
            bpe_ids: vocab_bpe[i].clone(),
            from_hardcoded: true,
        });
    }

    // 2. Add BPE tokens that decode to whole words (not already in vocab)
    let suffix_set: HashSet<&str> = SUFFIX_FRAGMENTS.iter().copied().collect();
    for t in 0..BPE_VOCAB {
        if ext.len() >= MAX_EXT_VOCAB { break; }
        let s = bpe_decode_token(t as u16);
        if s.len() < 3 { continue; }
        if !s.chars().all(|c| c.is_ascii_alphabetic()) { continue; }
        let lower = s.to_lowercase();
        if existing.contains(&lower) { continue; }
        if is_stop(&lower) { continue; }
        if suffix_set.contains(lower.as_str()) { continue; }
        existing.insert(lower.clone());
        ext.push(ExtWord {
            word: lower,
            bpe_ids: vec![t as u16],
            from_hardcoded: false,
        });
    }

    println!("  extended vocab: {} words ({} hardcoded + {} from BPE)",
             ext.len(), NWORDS, ext.len() - NWORDS);
    ext
}

fn run_chain(model: &Penelope, field: &mut DarioField, text: &str, rng: &mut Rng,
             ext_vocab: &[ExtWord], _has_weights: bool) {
    let idx = build_vocab_idx();
    let key = extract_key(text);
    let seed = find_seed(&key, &idx, rng);

    // prophecy
    let deep_cats = [2, 5, 7];
    let tcat = deep_cats[rng.randint(3)];
    let ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,NWORDS)];
    let (s, e) = ranges[tcat];
    field.prophecy_target = Some(s + rng.randint(e - s));
    field.prophecy_age = 0;

    println!("\n  destined: {}", vocab_display(field.prophecy_target.unwrap()));
    println!("\n  {}", vocab_display(seed));

    let mut chain: Vec<usize> = vec![seed];
    let mut forbidden = HashSet::new();
    forbidden.insert(seed);

    // BPE context buffer -- starts with seed word's BPE tokens
    let mut bpe_buf: Vec<u16> = ext_vocab[seed].bpe_ids.clone();

    let mut fulfilled = false;

    for step in 0..GEN_STEPS {
        field.update_chambers(step);
        field.prophecy_age += 1;

        // 1. Forward pass through all 8 layers -> BPE logits for last position
        let ctx_len = bpe_buf.len().min(MAX_SEQ);
        let ctx_start = if bpe_buf.len() > MAX_SEQ { bpe_buf.len() - MAX_SEQ } else { 0 };
        let bpe_logits = model.forward(&bpe_buf[ctx_start..ctx_start+ctx_len]);

        // 2. Convert BPE logits to word-level scores
        let mut word_scores = bpe_logits_to_word_scores(&bpe_logits, ext_vocab);

        // 3. Dario overlay on word scores
        field.overlay(&mut word_scores, &chain);

        // mask forbidden
        for &f in &forbidden {
            if f < word_scores.len() { word_scores[f] = -1e9; }
        }

        // top-k=12 sampling
        let probs = softmax(&word_scores);
        let mut top: Vec<(usize, f32)> = probs.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
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

        // append picked word's BPE tokens to context
        if pick < ext_vocab.len() && bpe_buf.len() + ext_vocab[pick].bpe_ids.len() < MAX_BPE_SEQ {
            bpe_buf.extend_from_slice(&ext_vocab[pick].bpe_ids);
        }

        // Dario field updates
        if chain.len() >= 2 {
            field.update_cooc(chain[chain.len()-2], pick);
        }
        let cat = word_category(pick);
        field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat];
        if Some(pick) == field.prophecy_target { fulfilled = true; }
        if step > 7 { field.trauma = (field.trauma + 0.1).min(1.0); }
        field.trauma *= 0.97;

        let wname = if pick < NWORDS { VOCAB[pick] } else { &ext_vocab[pick].word };
        if step == GEN_STEPS - 1 {
            println!("  *{}", wname);
        } else {
            println!("   {}", wname);
        }
    }

    let cats: HashSet<usize> = chain.iter().map(|&w| word_category(w)).collect();
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
    println!("  penelope v7 \u{2014} Resonance engine. 1984 words. Dario Equation.");
    println!("  {} layers, {} heads, dim={}, hdim={}", N_LAYERS, N_HEADS, DIM, HDIM);
    println!("  {} trainable params ({:.1}MB f32)", Penelope::param_count(),
             Penelope::param_count() as f64 * 4.0 / 1e6);
    println!("  BPE input: {} subword tokens, max_seq={}", BPE_VOCAB, MAX_SEQ);
    println!("  by Arianna Method");
    println!();

    let has_weights = if let Some(ref path) = load_path { model.load(path) } else { false };
    if let Some(ref path) = train_path {
        train(&mut model, path, train_steps, lr);
        if let Some(ref sp) = save_path { model.save(sp); }
    }

    let vocab_bpe = init_vocab_bpe();
    let ext_vocab = init_ext_vocab(&vocab_bpe);
    let mut field = DarioField::new();

    println!("  mode: {}\n", if has_weights { "trained (BPE word scores)" } else { "weightless (word-level)" });

    if let Some(ref t) = text {
        run_chain(&model, &mut field, t, &mut rng, &ext_vocab, has_weights);
    } else if train_path.is_none() {
        let stdin = io::stdin();
        loop {
            print!("  > ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
            let line = line.trim();
            if line.is_empty() { continue; }
            run_chain(&model, &mut field, line, &mut rng, &ext_vocab, has_weights);
        }
    }

    if save_path.is_some() && train_path.is_none() {
        model.save(save_path.as_ref().unwrap());
    }
}
