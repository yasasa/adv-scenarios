import numpy as np

# Indices where the lidar scan will hit the car.
CAR_SHAPE = np.array([ 338,  339,  340,  341,  342,  343,  344,  345,  346,  347,  348,
        349,  350,  351,  370,  371,  372,  373,  374,  375,  376,  377,
        378,  379,  380,  381,  382,  383,  401,  402,  403,  404,  405,
        406,  407,  408,  409,  410,  411,  412,  413,  414,  415,  433,
        434,  435,  436,  437,  438,  439,  440,  441,  442,  443,  444,
        445,  446,  447,  464,  465,  466,  467,  468,  469,  470,  471,
        472,  473,  474,  475,  476,  477,  478,  479,  496,  497,  498,
        499,  500,  501,  502,  503,  504,  505,  506,  507,  508,  509,
        510,  511,  527,  528,  529,  530,  531,  532,  533,  534,  535,
        536,  537,  538,  539,  540,  541,  542,  543,  558,  559,  560,
        561,  562,  563,  564,  565,  566,  567,  568,  569,  570,  571,
        572,  573,  574,  575,  589,  590,  591,  592,  593,  594,  595,
        596,  597,  598,  599,  600,  601,  602,  603,  604,  605,  606,
        607,  621,  622,  623,  624,  625,  626,  627,  628,  629,  630,
        631,  632,  633,  634,  635,  636,  637,  638,  639,  652,  653,
        654,  655,  656,  657,  658,  659,  660,  661,  662,  663,  664,
        665,  666,  667,  668,  669,  670,  671,  683,  684,  685,  686,
        687,  688,  689,  690,  691,  692,  693,  694,  695,  696,  697,
        698,  699,  700,  701,  702,  703,  715,  716,  717,  718,  719,
        720,  721,  722,  723,  724,  725,  726,  727,  728,  729,  730,
        731,  732,  733,  734,  735,  747,  748,  749,  750,  751,  752,
        753,  754,  755,  756,  757,  758,  759,  760,  761,  762,  763,
        764,  765,  766,  767,  779,  780,  781,  782,  783,  784,  785,
        786,  787,  788,  789,  790,  791,  792,  793,  794,  795,  796,
        797,  798,  799,  811,  812,  813,  814,  815,  816,  817,  818,
        819,  820,  821,  822,  823,  824,  825,  826,  827,  828,  829,
        830,  831,  843,  844,  845,  846,  847,  848,  849,  850,  851,
        852,  853,  854,  855,  856,  857,  858,  859,  860,  861,  862,
        863,  875,  876,  877,  878,  879,  880,  881,  882,  883,  884,
        885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  895,
        907,  908,  909,  910,  911,  912,  913,  914,  915,  916,  917,
        918,  919,  920,  921,  922,  923,  924,  925,  926,  927,  939,
        940,  941,  942,  943,  944,  945,  946,  947,  948,  949,  950,
        951,  952,  953,  954,  955,  956,  957,  958,  959,  972,  973,
        974,  975,  976,  977,  978,  979,  980,  981,  982,  983,  984,
        985,  986,  987,  988,  989,  990,  991, 1005, 1006, 1007, 1008,
       1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
       1020, 1021, 1022, 1023, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
       1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054,
       1055, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079,
       1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1103, 1104, 1105,
       1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116,
       1117, 1118, 1119, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142,
       1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1168, 1169,
       1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180,
       1181, 1182, 1183, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208,
       1209, 1210, 1211, 1212, 1213, 1214, 1215, 1233, 1234, 1235, 1236,
       1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247,
       1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276,
       1277, 1278, 1279, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305,
       1306, 1307, 1308, 1309, 1310, 1311])

# Values of the lidar scan for above indices
CAR_VALUES = np.array([1.18718243, 1.1410687 , 1.14932084, 1.15681207, 1.16637492,
       1.17494154, 1.18576384, 1.19724655, 1.20937729, 1.2221247 ,
       1.23548758, 1.24942839, 1.26634693, 1.28150439, 1.18513274,
       1.08372426, 1.05809557, 1.06555569, 1.07353985, 1.08205581,
       1.09108508, 1.10060835, 1.11261594, 1.12320018, 1.13643432,
       1.15030468, 1.16478932, 1.17987096, 1.30564582, 1.18319798,
       1.08212495, 0.99762845, 0.98996544, 0.99785894, 1.00632489,
       1.01533592, 1.02488351, 1.03494859, 1.04553258, 1.05660248,
       1.06814969, 1.08255053, 1.09759581, 1.30340242, 1.18138254,
       1.08063352, 0.99637318, 0.92940342, 0.93576306, 0.94419992,
       0.95142502, 0.96090329, 0.96894062, 0.97940379, 0.9903999 ,
       1.00190639, 1.01630032, 1.02886617, 1.45159101, 1.3013047 ,
       1.17968202, 1.07923341, 0.99521315, 0.92359293, 0.88474631,
       0.89150482, 0.89865774, 0.90809137, 0.91611773, 0.92659307,
       0.93543077, 0.94690061, 0.95888513, 0.9713726 , 1.44914675,
       1.29936337, 1.17811394, 1.07793629, 0.99413085, 0.92268205,
       0.8613072 , 0.84809774, 0.8552345 , 0.86278391, 0.87272054,
       0.88114995, 0.88994545, 0.90140623, 0.91340768, 0.92342794,
       1.63516593, 1.44598377, 1.29758656, 1.17667413, 1.07675219,
       0.99313706, 0.92184556, 0.86058944, 0.81265461, 0.81983006,
       0.82743531, 0.83544195, 0.84386301, 0.85266215, 0.86185515,
       0.87379479, 0.88628668, 1.87766385, 1.63253844, 1.44395542,
       1.29598606, 1.17536473, 1.07568598, 0.99223584, 0.92108864,
       0.85994709, 0.80662328, 0.78785378, 0.79536444, 0.80329484,
       0.81165725, 0.82041383, 0.83187389, 0.84150922, 0.85150552,
       2.20604038, 1.87454915, 1.63021684, 1.44215214, 1.29455221,
       1.17421114, 1.07472241, 0.99143839, 0.92041713, 0.85936326,
       0.80612522, 0.76390445, 0.76960325, 0.77752769, 0.78588384,
       0.79466408, 0.80384606, 0.81342471, 0.82337773, 2.20223546,
       1.87182999, 1.62817287, 1.4405781 , 1.29226625, 1.17319298,
       1.07388473, 0.99074227, 0.9198181 , 0.85886085, 0.8056969 ,
       0.7590993 , 0.74956465, 0.75556064, 0.76387936, 0.77263278,
       0.78180844, 0.79138535, 0.80134696, 2.67268443, 2.19898891,
       1.86879647, 1.62642765, 1.43923175, 1.29118764, 1.17232656,
       1.07316971, 0.99013954, 0.9193182 , 0.85843325, 0.80532128,
       0.75877827, 0.73103297, 0.73894352, 0.74730575, 0.75610471,
       0.76302755, 0.77257091, 0.78250086, 3.40467262, 2.66869593,
       2.19630551, 1.86688185, 1.62499011, 1.43810868, 1.29030585,
       1.17160559, 1.0725801 , 0.98964691, 0.9189018 , 0.85807467,
       0.80500966, 0.75850606, 0.71743602, 0.72534394, 0.73372078,
       0.74033237, 0.74947608, 0.75904536, 0.76901585, 3.39954448,
       2.66555357, 2.19419241, 1.8653667 , 1.62385309, 1.43724012,
       1.2896117 , 1.1710465 , 1.07211196, 0.98925954, 0.91856462,
       0.85778755, 0.80477273, 0.7582981 , 0.71707779, 0.71450412,
       0.72286481, 0.72946119, 0.73860115, 0.74817282, 0.75814295,
       3.39582634, 2.66328502, 2.19266939, 1.86427379, 1.62303782,
       1.43659949, 1.28909826, 1.17063272, 1.07177794, 0.98897946,
       0.91832709, 0.85759187, 0.80459297, 0.75814676, 0.71694058,
       0.70824641, 0.71455616, 0.72332096, 0.73022842, 0.73976296,
       0.74970978, 3.39352155, 2.66188192, 2.19172478, 1.86359537,
       1.62253058, 1.43620396, 1.28879261, 1.17037344, 1.07156754,
       0.98879743, 0.91817969, 0.85746002, 0.80448943, 0.75805241,
       0.71686059, 0.70242345, 0.7107752 , 0.71737587, 0.72651905,
       0.73610073, 0.7435959 , 3.39264441, 2.66134286, 2.19135547,
       1.86333406, 1.62232852, 1.43605328, 1.2886678 , 1.17028213,
       1.071486  , 0.98872948, 0.9181205 , 0.85741019, 0.80444252,
       0.7580151 , 0.71683097, 0.70097291, 0.70934266, 0.71594846,
       0.72511864, 0.73230886, 0.74222052, 3.39318037, 2.66166711,
       2.19157934, 1.86349559, 1.62245119, 1.43614018, 1.28874135,
       1.17034018, 1.07153714, 0.98877054, 0.91815561, 0.85743952,
       0.80447274, 0.75803727, 0.71685219, 0.70185775, 0.71021754,
       0.71681678, 0.72597921, 0.73556149, 0.74306309, 3.39513397,
       2.66286254, 2.19238281, 1.86406291, 1.62288141, 1.43648422,
       1.28900623, 1.1705544 , 1.07171071, 0.98892421, 0.91828299,
       0.85754907, 0.80456066, 0.75811976, 0.7169202 , 0.7051028 ,
       0.71342725, 0.72220129, 0.72911698, 0.73866922, 0.74863642,
       3.39851213, 2.66493082, 2.19377542, 1.86506426, 1.62363231,
       1.43705845, 1.28947103, 1.170928  , 1.07202196, 0.98918104,
       0.91850442, 0.85773355, 0.80472147, 0.75825369, 0.71703547,
       0.71276659, 0.72113657, 0.72775459, 0.73691583, 0.74649781,
       0.75649804, 3.40330148, 2.66786075, 2.19574594, 1.86647832,
       1.62468934, 1.43787384, 1.29011428, 1.17145598, 1.07245564,
       0.98954868, 0.91881144, 0.85799569, 0.80494875, 0.75845277,
       0.71720815, 0.72296834, 0.72926617, 0.7379961 , 0.74717087,
       0.75677091, 0.76677269, 2.67164183, 2.19828415, 1.86829984,
       1.62605858, 1.43893433, 1.29095721, 1.17213452, 1.0730145 ,
       0.99001437, 0.91920608, 0.85833812, 0.80523819, 0.75870711,
       0.72794205, 0.73588264, 0.74216998, 0.75090069, 0.76005757,
       0.76963311, 0.77960426, 2.20140028, 1.87052131, 1.62772298,
       1.44022608, 1.29198658, 1.17297232, 1.0737015 , 0.99058747,
       0.91969222, 0.8587476 , 0.80559534, 0.75901729, 0.74382406,
       0.75173324, 0.7600987 , 0.76889354, 0.77810752, 0.78772485,
       0.79773188, 2.20507312, 1.87385225, 1.62968624, 1.44175279,
       1.29423344, 1.17395306, 1.07451212, 0.99126256, 0.92026263,
       0.85923767, 0.80601674, 0.75937903, 0.76489049, 0.77286375,
       0.78127259, 0.78789616, 0.79932243, 0.8089596 , 0.81896037,
       1.87687671, 1.63195086, 1.44350207, 1.29562354, 1.17507493,
       1.07544672, 0.99203569, 0.92092055, 0.85979861, 0.80649555,
       0.78213048, 0.78967929, 0.7976709 , 0.80607957, 0.81490475,
       0.82411575, 0.83371007, 0.8436814 , 1.63450503, 1.44547153,
       1.29718494, 1.17633951, 1.07648468, 0.99290824, 0.92166102,
       0.86042887, 0.80703586, 0.81117582, 0.81873113, 0.82670826,
       0.8350817 , 0.84385282, 0.85530597, 0.86492515, 0.87739819,
       1.63733339, 1.44766355, 1.29891992, 1.17775571, 1.07764518,
       0.99387479, 0.92248094, 0.8611272 , 0.83815587, 0.84703529,
       0.85465139, 0.86265767, 0.87106872, 0.88204706, 0.89128542,
       0.90327632, 0.91329783, 1.45097971, 1.30082119, 1.17928398,
       1.07891142, 0.99494332, 0.92336559, 0.87320906, 0.8799333 ,
       0.88885307, 0.8964712 , 0.90447432, 0.91495079, 0.92377442,
       0.93523818, 0.94723547, 0.95973408, 1.30287957, 1.18095887,
       1.08028781, 0.99608982, 0.92434299, 0.92223251, 0.93065208,
       0.93785685, 0.94733143, 0.9553656 , 0.96584076, 0.9768492 ,
       0.98837435, 1.0003916 , 1.01289558, 1.30509079, 1.18275416,
       1.08176136, 0.99731737, 0.973957  , 0.98184383, 0.99030441,
       0.997531  , 1.00699019, 1.01697111, 1.02747607, 1.04068935,
       1.05227196, 1.06433284, 1.07933211, 1.18465364, 1.08333325,
       1.03890562, 1.04488313, 1.05278146, 1.06288862, 1.0719378 ,
       1.08150494, 1.09155345, 1.10419357, 1.11529076, 1.1291393 ,
       1.14360738, 1.15869081, 1.18667543, 1.11766231, 1.12456024,
       1.13343632, 1.14145243, 1.14996469, 1.16073644, 1.17218542,
       1.18428373, 1.19493127, 1.21038973, 1.2243489 , 1.23889673,
       1.25648928])
