#ifndef TORUS_MESH_H_
#define TORUS_MESH_H_

//*************************** NOT REALLY FAMOUS TORUS ********************************************//

#define Real btScalar
const int NUM_TRIANGLES = 600;
const int NUM_VERTICES = 300;
const int NUM_INDICES = NUM_TRIANGLES * 3;

static Real gVertices[NUM_VERTICES * 3] = {
	Real(2.5), Real(0), Real(0),
	Real(2.405), Real(0.294), Real(0),
	Real(2.155), Real(0.476), Real(0),
	Real(1.845), Real(0.476), Real(0),
	Real(1.595), Real(0.294), Real(0),
	Real(1.5), Real(0), Real(0),
	Real(1.595), Real(-0.294), Real(0),
	Real(1.845), Real(-0.476), Real(0),
	Real(2.155), Real(-0.476), Real(0),
	Real(2.405), Real(-0.294), Real(0),
	Real(2.445), Real(0), Real(0.52),
	Real(2.352), Real(0.294), Real(0.5),
	Real(2.107), Real(0.476), Real(0.448),
	Real(1.805), Real(0.476), Real(0.384),
	Real(1.561), Real(0.294), Real(0.332),
	Real(1.467), Real(0), Real(0.312),
	Real(1.561), Real(-0.294), Real(0.332),
	Real(1.805), Real(-0.476), Real(0.384),
	Real(2.107), Real(-0.476), Real(0.448),
	Real(2.352), Real(-0.294), Real(0.5),
	Real(2.284), Real(0), Real(1.017),
	Real(2.197), Real(0.294), Real(0.978),
	Real(1.968), Real(0.476), Real(0.876),
	Real(1.686), Real(0.476), Real(0.751),
	Real(1.458), Real(0.294), Real(0.649),
	Real(1.37), Real(0), Real(0.61),
	Real(1.458), Real(-0.294), Real(0.649),
	Real(1.686), Real(-0.476), Real(0.751),
	Real(1.968), Real(-0.476), Real(0.876),
	Real(2.197), Real(-0.294), Real(0.978),
	Real(2.023), Real(0), Real(1.469),
	Real(1.945), Real(0.294), Real(1.413),
	Real(1.743), Real(0.476), Real(1.266),
	Real(1.493), Real(0.476), Real(1.085),
	Real(1.291), Real(0.294), Real(0.938),
	Real(1.214), Real(0), Real(0.882),
	Real(1.291), Real(-0.294), Real(0.938),
	Real(1.493), Real(-0.476), Real(1.085),
	Real(1.743), Real(-0.476), Real(1.266),
	Real(1.945), Real(-0.294), Real(1.413),
	Real(1.673), Real(0), Real(1.858),
	Real(1.609), Real(0.294), Real(1.787),
	Real(1.442), Real(0.476), Real(1.601),
	Real(1.235), Real(0.476), Real(1.371),
	Real(1.068), Real(0.294), Real(1.186),
	Real(1.004), Real(0), Real(1.115),
	Real(1.068), Real(-0.294), Real(1.186),
	Real(1.235), Real(-0.476), Real(1.371),
	Real(1.442), Real(-0.476), Real(1.601),
	Real(1.609), Real(-0.294), Real(1.787),
	Real(1.25), Real(0), Real(2.165),
	Real(1.202), Real(0.294), Real(2.082),
	Real(1.077), Real(0.476), Real(1.866),
	Real(0.923), Real(0.476), Real(1.598),
	Real(0.798), Real(0.294), Real(1.382),
	Real(0.75), Real(0), Real(1.299),
	Real(0.798), Real(-0.294), Real(1.382),
	Real(0.923), Real(-0.476), Real(1.598),
	Real(1.077), Real(-0.476), Real(1.866),
	Real(1.202), Real(-0.294), Real(2.082),
	Real(0.773), Real(0), Real(2.378),
	Real(0.743), Real(0.294), Real(2.287),
	Real(0.666), Real(0.476), Real(2.049),
	Real(0.57), Real(0.476), Real(1.755),
	Real(0.493), Real(0.294), Real(1.517),
	Real(0.464), Real(0), Real(1.427),
	Real(0.493), Real(-0.294), Real(1.517),
	Real(0.57), Real(-0.476), Real(1.755),
	Real(0.666), Real(-0.476), Real(2.049),
	Real(0.743), Real(-0.294), Real(2.287),
	Real(0.261), Real(0), Real(2.486),
	Real(0.251), Real(0.294), Real(2.391),
	Real(0.225), Real(0.476), Real(2.143),
	Real(0.193), Real(0.476), Real(1.835),
	Real(0.167), Real(0.294), Real(1.587),
	Real(0.157), Real(0), Real(1.492),
	Real(0.167), Real(-0.294), Real(1.587),
	Real(0.193), Real(-0.476), Real(1.835),
	Real(0.225), Real(-0.476), Real(2.143),
	Real(0.251), Real(-0.294), Real(2.391),
	Real(-0.261), Real(0), Real(2.486),
	Real(-0.251), Real(0.294), Real(2.391),
	Real(-0.225), Real(0.476), Real(2.143),
	Real(-0.193), Real(0.476), Real(1.835),
	Real(-0.167), Real(0.294), Real(1.587),
	Real(-0.157), Real(0), Real(1.492),
	Real(-0.167), Real(-0.294), Real(1.587),
	Real(-0.193), Real(-0.476), Real(1.835),
	Real(-0.225), Real(-0.476), Real(2.143),
	Real(-0.251), Real(-0.294), Real(2.391),
	Real(-0.773), Real(0), Real(2.378),
	Real(-0.743), Real(0.294), Real(2.287),
	Real(-0.666), Real(0.476), Real(2.049),
	Real(-0.57), Real(0.476), Real(1.755),
	Real(-0.493), Real(0.294), Real(1.517),
	Real(-0.464), Real(0), Real(1.427),
	Real(-0.493), Real(-0.294), Real(1.517),
	Real(-0.57), Real(-0.476), Real(1.755),
	Real(-0.666), Real(-0.476), Real(2.049),
	Real(-0.743), Real(-0.294), Real(2.287),
	Real(-1.25), Real(0), Real(2.165),
	Real(-1.202), Real(0.294), Real(2.082),
	Real(-1.077), Real(0.476), Real(1.866),
	Real(-0.923), Real(0.476), Real(1.598),
	Real(-0.798), Real(0.294), Real(1.382),
	Real(-0.75), Real(0), Real(1.299),
	Real(-0.798), Real(-0.294), Real(1.382),
	Real(-0.923), Real(-0.476), Real(1.598),
	Real(-1.077), Real(-0.476), Real(1.866),
	Real(-1.202), Real(-0.294), Real(2.082),
	Real(-1.673), Real(0), Real(1.858),
	Real(-1.609), Real(0.294), Real(1.787),
	Real(-1.442), Real(0.476), Real(1.601),
	Real(-1.235), Real(0.476), Real(1.371),
	Real(-1.068), Real(0.294), Real(1.186),
	Real(-1.004), Real(0), Real(1.115),
	Real(-1.068), Real(-0.294), Real(1.186),
	Real(-1.235), Real(-0.476), Real(1.371),
	Real(-1.442), Real(-0.476), Real(1.601),
	Real(-1.609), Real(-0.294), Real(1.787),
	Real(-2.023), Real(0), Real(1.469),
	Real(-1.945), Real(0.294), Real(1.413),
	Real(-1.743), Real(0.476), Real(1.266),
	Real(-1.493), Real(0.476), Real(1.085),
	Real(-1.291), Real(0.294), Real(0.938),
	Real(-1.214), Real(0), Real(0.882),
	Real(-1.291), Real(-0.294), Real(0.938),
	Real(-1.493), Real(-0.476), Real(1.085),
	Real(-1.743), Real(-0.476), Real(1.266),
	Real(-1.945), Real(-0.294), Real(1.413),
	Real(-2.284), Real(0), Real(1.017),
	Real(-2.197), Real(0.294), Real(0.978),
	Real(-1.968), Real(0.476), Real(0.876),
	Real(-1.686), Real(0.476), Real(0.751),
	Real(-1.458), Real(0.294), Real(0.649),
	Real(-1.37), Real(0), Real(0.61),
	Real(-1.458), Real(-0.294), Real(0.649),
	Real(-1.686), Real(-0.476), Real(0.751),
	Real(-1.968), Real(-0.476), Real(0.876),
	Real(-2.197), Real(-0.294), Real(0.978),
	Real(-2.445), Real(0), Real(0.52),
	Real(-2.352), Real(0.294), Real(0.5),
	Real(-2.107), Real(0.476), Real(0.448),
	Real(-1.805), Real(0.476), Real(0.384),
	Real(-1.561), Real(0.294), Real(0.332),
	Real(-1.467), Real(0), Real(0.312),
	Real(-1.561), Real(-0.294), Real(0.332),
	Real(-1.805), Real(-0.476), Real(0.384),
	Real(-2.107), Real(-0.476), Real(0.448),
	Real(-2.352), Real(-0.294), Real(0.5),
	Real(-2.5), Real(0), Real(0),
	Real(-2.405), Real(0.294), Real(0),
	Real(-2.155), Real(0.476), Real(0),
	Real(-1.845), Real(0.476), Real(0),
	Real(-1.595), Real(0.294), Real(0),
	Real(-1.5), Real(0), Real(0),
	Real(-1.595), Real(-0.294), Real(0),
	Real(-1.845), Real(-0.476), Real(0),
	Real(-2.155), Real(-0.476), Real(0),
	Real(-2.405), Real(-0.294), Real(0),
	Real(-2.445), Real(0), Real(-0.52),
	Real(-2.352), Real(0.294), Real(-0.5),
	Real(-2.107), Real(0.476), Real(-0.448),
	Real(-1.805), Real(0.476), Real(-0.384),
	Real(-1.561), Real(0.294), Real(-0.332),
	Real(-1.467), Real(0), Real(-0.312),
	Real(-1.561), Real(-0.294), Real(-0.332),
	Real(-1.805), Real(-0.476), Real(-0.384),
	Real(-2.107), Real(-0.476), Real(-0.448),
	Real(-2.352), Real(-0.294), Real(-0.5),
	Real(-2.284), Real(0), Real(-1.017),
	Real(-2.197), Real(0.294), Real(-0.978),
	Real(-1.968), Real(0.476), Real(-0.876),
	Real(-1.686), Real(0.476), Real(-0.751),
	Real(-1.458), Real(0.294), Real(-0.649),
	Real(-1.37), Real(0), Real(-0.61),
	Real(-1.458), Real(-0.294), Real(-0.649),
	Real(-1.686), Real(-0.476), Real(-0.751),
	Real(-1.968), Real(-0.476), Real(-0.876),
	Real(-2.197), Real(-0.294), Real(-0.978),
	Real(-2.023), Real(0), Real(-1.469),
	Real(-1.945), Real(0.294), Real(-1.413),
	Real(-1.743), Real(0.476), Real(-1.266),
	Real(-1.493), Real(0.476), Real(-1.085),
	Real(-1.291), Real(0.294), Real(-0.938),
	Real(-1.214), Real(0), Real(-0.882),
	Real(-1.291), Real(-0.294), Real(-0.938),
	Real(-1.493), Real(-0.476), Real(-1.085),
	Real(-1.743), Real(-0.476), Real(-1.266),
	Real(-1.945), Real(-0.294), Real(-1.413),
	Real(-1.673), Real(0), Real(-1.858),
	Real(-1.609), Real(0.294), Real(-1.787),
	Real(-1.442), Real(0.476), Real(-1.601),
	Real(-1.235), Real(0.476), Real(-1.371),
	Real(-1.068), Real(0.294), Real(-1.186),
	Real(-1.004), Real(0), Real(-1.115),
	Real(-1.068), Real(-0.294), Real(-1.186),
	Real(-1.235), Real(-0.476), Real(-1.371),
	Real(-1.442), Real(-0.476), Real(-1.601),
	Real(-1.609), Real(-0.294), Real(-1.787),
	Real(-1.25), Real(0), Real(-2.165),
	Real(-1.202), Real(0.294), Real(-2.082),
	Real(-1.077), Real(0.476), Real(-1.866),
	Real(-0.923), Real(0.476), Real(-1.598),
	Real(-0.798), Real(0.294), Real(-1.382),
	Real(-0.75), Real(0), Real(-1.299),
	Real(-0.798), Real(-0.294), Real(-1.382),
	Real(-0.923), Real(-0.476), Real(-1.598),
	Real(-1.077), Real(-0.476), Real(-1.866),
	Real(-1.202), Real(-0.294), Real(-2.082),
	Real(-0.773), Real(0), Real(-2.378),
	Real(-0.743), Real(0.294), Real(-2.287),
	Real(-0.666), Real(0.476), Real(-2.049),
	Real(-0.57), Real(0.476), Real(-1.755),
	Real(-0.493), Real(0.294), Real(-1.517),
	Real(-0.464), Real(0), Real(-1.427),
	Real(-0.493), Real(-0.294), Real(-1.517),
	Real(-0.57), Real(-0.476), Real(-1.755),
	Real(-0.666), Real(-0.476), Real(-2.049),
	Real(-0.743), Real(-0.294), Real(-2.287),
	Real(-0.261), Real(0), Real(-2.486),
	Real(-0.251), Real(0.294), Real(-2.391),
	Real(-0.225), Real(0.476), Real(-2.143),
	Real(-0.193), Real(0.476), Real(-1.835),
	Real(-0.167), Real(0.294), Real(-1.587),
	Real(-0.157), Real(0), Real(-1.492),
	Real(-0.167), Real(-0.294), Real(-1.587),
	Real(-0.193), Real(-0.476), Real(-1.835),
	Real(-0.225), Real(-0.476), Real(-2.143),
	Real(-0.251), Real(-0.294), Real(-2.391),
	Real(0.261), Real(0), Real(-2.486),
	Real(0.251), Real(0.294), Real(-2.391),
	Real(0.225), Real(0.476), Real(-2.143),
	Real(0.193), Real(0.476), Real(-1.835),
	Real(0.167), Real(0.294), Real(-1.587),
	Real(0.157), Real(0), Real(-1.492),
	Real(0.167), Real(-0.294), Real(-1.587),
	Real(0.193), Real(-0.476), Real(-1.835),
	Real(0.225), Real(-0.476), Real(-2.143),
	Real(0.251), Real(-0.294), Real(-2.391),
	Real(0.773), Real(0), Real(-2.378),
	Real(0.743), Real(0.294), Real(-2.287),
	Real(0.666), Real(0.476), Real(-2.049),
	Real(0.57), Real(0.476), Real(-1.755),
	Real(0.493), Real(0.294), Real(-1.517),
	Real(0.464), Real(0), Real(-1.427),
	Real(0.493), Real(-0.294), Real(-1.517),
	Real(0.57), Real(-0.476), Real(-1.755),
	Real(0.666), Real(-0.476), Real(-2.049),
	Real(0.743), Real(-0.294), Real(-2.287),
	Real(1.25), Real(0), Real(-2.165),
	Real(1.202), Real(0.294), Real(-2.082),
	Real(1.077), Real(0.476), Real(-1.866),
	Real(0.923), Real(0.476), Real(-1.598),
	Real(0.798), Real(0.294), Real(-1.382),
	Real(0.75), Real(0), Real(-1.299),
	Real(0.798), Real(-0.294), Real(-1.382),
	Real(0.923), Real(-0.476), Real(-1.598),
	Real(1.077), Real(-0.476), Real(-1.866),
	Real(1.202), Real(-0.294), Real(-2.082),
	Real(1.673), Real(0), Real(-1.858),
	Real(1.609), Real(0.294), Real(-1.787),
	Real(1.442), Real(0.476), Real(-1.601),
	Real(1.235), Real(0.476), Real(-1.371),
	Real(1.068), Real(0.294), Real(-1.186),
	Real(1.004), Real(0), Real(-1.115),
	Real(1.068), Real(-0.294), Real(-1.186),
	Real(1.235), Real(-0.476), Real(-1.371),
	Real(1.442), Real(-0.476), Real(-1.601),
	Real(1.609), Real(-0.294), Real(-1.787),
	Real(2.023), Real(0), Real(-1.469),
	Real(1.945), Real(0.294), Real(-1.413),
	Real(1.743), Real(0.476), Real(-1.266),
	Real(1.493), Real(0.476), Real(-1.085),
	Real(1.291), Real(0.294), Real(-0.938),
	Real(1.214), Real(0), Real(-0.882),
	Real(1.291), Real(-0.294), Real(-0.938),
	Real(1.493), Real(-0.476), Real(-1.085),
	Real(1.743), Real(-0.476), Real(-1.266),
	Real(1.945), Real(-0.294), Real(-1.413),
	Real(2.284), Real(0), Real(-1.017),
	Real(2.197), Real(0.294), Real(-0.978),
	Real(1.968), Real(0.476), Real(-0.876),
	Real(1.686), Real(0.476), Real(-0.751),
	Real(1.458), Real(0.294), Real(-0.649),
	Real(1.37), Real(0), Real(-0.61),
	Real(1.458), Real(-0.294), Real(-0.649),
	Real(1.686), Real(-0.476), Real(-0.751),
	Real(1.968), Real(-0.476), Real(-0.876),
	Real(2.197), Real(-0.294), Real(-0.978),
	Real(2.445), Real(0), Real(-0.52),
	Real(2.352), Real(0.294), Real(-0.5),
	Real(2.107), Real(0.476), Real(-0.448),
	Real(1.805), Real(0.476), Real(-0.384),
	Real(1.561), Real(0.294), Real(-0.332),
	Real(1.467), Real(0), Real(-0.312),
	Real(1.561), Real(-0.294), Real(-0.332),
	Real(1.805), Real(-0.476), Real(-0.384),
	Real(2.107), Real(-0.476), Real(-0.448),
	Real(2.352), Real(-0.294), Real(-0.5)};

static int gIndices[NUM_TRIANGLES][3] = {
	{0, 1, 11},
	{1, 2, 12},
	{2, 3, 13},
	{3, 4, 14},
	{4, 5, 15},
	{5, 6, 16},
	{6, 7, 17},
	{7, 8, 18},
	{8, 9, 19},
	{9, 0, 10},
	{10, 11, 21},
	{11, 12, 22},
	{12, 13, 23},
	{13, 14, 24},
	{14, 15, 25},
	{15, 16, 26},
	{16, 17, 27},
	{17, 18, 28},
	{18, 19, 29},
	{19, 10, 20},
	{20, 21, 31},
	{21, 22, 32},
	{22, 23, 33},
	{23, 24, 34},
	{24, 25, 35},
	{25, 26, 36},
	{26, 27, 37},
	{27, 28, 38},
	{28, 29, 39},
	{29, 20, 30},
	{30, 31, 41},
	{31, 32, 42},
	{32, 33, 43},
	{33, 34, 44},
	{34, 35, 45},
	{35, 36, 46},
	{36, 37, 47},
	{37, 38, 48},
	{38, 39, 49},
	{39, 30, 40},
	{40, 41, 51},
	{41, 42, 52},
	{42, 43, 53},
	{43, 44, 54},
	{44, 45, 55},
	{45, 46, 56},
	{46, 47, 57},
	{47, 48, 58},
	{48, 49, 59},
	{49, 40, 50},
	{50, 51, 61},
	{51, 52, 62},
	{52, 53, 63},
	{53, 54, 64},
	{54, 55, 65},
	{55, 56, 66},
	{56, 57, 67},
	{57, 58, 68},
	{58, 59, 69},
	{59, 50, 60},
	{60, 61, 71},
	{61, 62, 72},
	{62, 63, 73},
	{63, 64, 74},
	{64, 65, 75},
	{65, 66, 76},
	{66, 67, 77},
	{67, 68, 78},
	{68, 69, 79},
	{69, 60, 70},
	{70, 71, 81},
	{71, 72, 82},
	{72, 73, 83},
	{73, 74, 84},
	{74, 75, 85},
	{75, 76, 86},
	{76, 77, 87},
	{77, 78, 88},
	{78, 79, 89},
	{79, 70, 80},
	{80, 81, 91},
	{81, 82, 92},
	{82, 83, 93},
	{83, 84, 94},
	{84, 85, 95},
	{85, 86, 96},
	{86, 87, 97},
	{87, 88, 98},
	{88, 89, 99},
	{89, 80, 90},
	{90, 91, 101},
	{91, 92, 102},
	{92, 93, 103},
	{93, 94, 104},
	{94, 95, 105},
	{95, 96, 106},
	{96, 97, 107},
	{97, 98, 108},
	{98, 99, 109},
	{99, 90, 100},
	{100, 101, 111},
	{101, 102, 112},
	{102, 103, 113},
	{103, 104, 114},
	{104, 105, 115},
	{105, 106, 116},
	{106, 107, 117},
	{107, 108, 118},
	{108, 109, 119},
	{109, 100, 110},
	{110, 111, 121},
	{111, 112, 122},
	{112, 113, 123},
	{113, 114, 124},
	{114, 115, 125},
	{115, 116, 126},
	{116, 117, 127},
	{117, 118, 128},
	{118, 119, 129},
	{119, 110, 120},
	{120, 121, 131},
	{121, 122, 132},
	{122, 123, 133},
	{123, 124, 134},
	{124, 125, 135},
	{125, 126, 136},
	{126, 127, 137},
	{127, 128, 138},
	{128, 129, 139},
	{129, 120, 130},
	{130, 131, 141},
	{131, 132, 142},
	{132, 133, 143},
	{133, 134, 144},
	{134, 135, 145},
	{135, 136, 146},
	{136, 137, 147},
	{137, 138, 148},
	{138, 139, 149},
	{139, 130, 140},
	{140, 141, 151},
	{141, 142, 152},
	{142, 143, 153},
	{143, 144, 154},
	{144, 145, 155},
	{145, 146, 156},
	{146, 147, 157},
	{147, 148, 158},
	{148, 149, 159},
	{149, 140, 150},
	{150, 151, 161},
	{151, 152, 162},
	{152, 153, 163},
	{153, 154, 164},
	{154, 155, 165},
	{155, 156, 166},
	{156, 157, 167},
	{157, 158, 168},
	{158, 159, 169},
	{159, 150, 160},
	{160, 161, 171},
	{161, 162, 172},
	{162, 163, 173},
	{163, 164, 174},
	{164, 165, 175},
	{165, 166, 176},
	{166, 167, 177},
	{167, 168, 178},
	{168, 169, 179},
	{169, 160, 170},
	{170, 171, 181},
	{171, 172, 182},
	{172, 173, 183},
	{173, 174, 184},
	{174, 175, 185},
	{175, 176, 186},
	{176, 177, 187},
	{177, 178, 188},
	{178, 179, 189},
	{179, 170, 180},
	{180, 181, 191},
	{181, 182, 192},
	{182, 183, 193},
	{183, 184, 194},
	{184, 185, 195},
	{185, 186, 196},
	{186, 187, 197},
	{187, 188, 198},
	{188, 189, 199},
	{189, 180, 190},
	{190, 191, 201},
	{191, 192, 202},
	{192, 193, 203},
	{193, 194, 204},
	{194, 195, 205},
	{195, 196, 206},
	{196, 197, 207},
	{197, 198, 208},
	{198, 199, 209},
	{199, 190, 200},
	{200, 201, 211},
	{201, 202, 212},
	{202, 203, 213},
	{203, 204, 214},
	{204, 205, 215},
	{205, 206, 216},
	{206, 207, 217},
	{207, 208, 218},
	{208, 209, 219},
	{209, 200, 210},
	{210, 211, 221},
	{211, 212, 222},
	{212, 213, 223},
	{213, 214, 224},
	{214, 215, 225},
	{215, 216, 226},
	{216, 217, 227},
	{217, 218, 228},
	{218, 219, 229},
	{219, 210, 220},
	{220, 221, 231},
	{221, 222, 232},
	{222, 223, 233},
	{223, 224, 234},
	{224, 225, 235},
	{225, 226, 236},
	{226, 227, 237},
	{227, 228, 238},
	{228, 229, 239},
	{229, 220, 230},
	{230, 231, 241},
	{231, 232, 242},
	{232, 233, 243},
	{233, 234, 244},
	{234, 235, 245},
	{235, 236, 246},
	{236, 237, 247},
	{237, 238, 248},
	{238, 239, 249},
	{239, 230, 240},
	{240, 241, 251},
	{241, 242, 252},
	{242, 243, 253},
	{243, 244, 254},
	{244, 245, 255},
	{245, 246, 256},
	{246, 247, 257},
	{247, 248, 258},
	{248, 249, 259},
	{249, 240, 250},
	{250, 251, 261},
	{251, 252, 262},
	{252, 253, 263},
	{253, 254, 264},
	{254, 255, 265},
	{255, 256, 266},
	{256, 257, 267},
	{257, 258, 268},
	{258, 259, 269},
	{259, 250, 260},
	{260, 261, 271},
	{261, 262, 272},
	{262, 263, 273},
	{263, 264, 274},
	{264, 265, 275},
	{265, 266, 276},
	{266, 267, 277},
	{267, 268, 278},
	{268, 269, 279},
	{269, 260, 270},
	{270, 271, 281},
	{271, 272, 282},
	{272, 273, 283},
	{273, 274, 284},
	{274, 275, 285},
	{275, 276, 286},
	{276, 277, 287},
	{277, 278, 288},
	{278, 279, 289},
	{279, 270, 280},
	{280, 281, 291},
	{281, 282, 292},
	{282, 283, 293},
	{283, 284, 294},
	{284, 285, 295},
	{285, 286, 296},
	{286, 287, 297},
	{287, 288, 298},
	{288, 289, 299},
	{289, 280, 290},
	{290, 291, 1},
	{291, 292, 2},
	{292, 293, 3},
	{293, 294, 4},
	{294, 295, 5},
	{295, 296, 6},
	{296, 297, 7},
	{297, 298, 8},
	{298, 299, 9},
	{299, 290, 0},
	{0, 11, 10},
	{1, 12, 11},
	{2, 13, 12},
	{3, 14, 13},
	{4, 15, 14},
	{5, 16, 15},
	{6, 17, 16},
	{7, 18, 17},
	{8, 19, 18},
	{9, 10, 19},
	{10, 21, 20},
	{11, 22, 21},
	{12, 23, 22},
	{13, 24, 23},
	{14, 25, 24},
	{15, 26, 25},
	{16, 27, 26},
	{17, 28, 27},
	{18, 29, 28},
	{19, 20, 29},
	{20, 31, 30},
	{21, 32, 31},
	{22, 33, 32},
	{23, 34, 33},
	{24, 35, 34},
	{25, 36, 35},
	{26, 37, 36},
	{27, 38, 37},
	{28, 39, 38},
	{29, 30, 39},
	{30, 41, 40},
	{31, 42, 41},
	{32, 43, 42},
	{33, 44, 43},
	{34, 45, 44},
	{35, 46, 45},
	{36, 47, 46},
	{37, 48, 47},
	{38, 49, 48},
	{39, 40, 49},
	{40, 51, 50},
	{41, 52, 51},
	{42, 53, 52},
	{43, 54, 53},
	{44, 55, 54},
	{45, 56, 55},
	{46, 57, 56},
	{47, 58, 57},
	{48, 59, 58},
	{49, 50, 59},
	{50, 61, 60},
	{51, 62, 61},
	{52, 63, 62},
	{53, 64, 63},
	{54, 65, 64},
	{55, 66, 65},
	{56, 67, 66},
	{57, 68, 67},
	{58, 69, 68},
	{59, 60, 69},
	{60, 71, 70},
	{61, 72, 71},
	{62, 73, 72},
	{63, 74, 73},
	{64, 75, 74},
	{65, 76, 75},
	{66, 77, 76},
	{67, 78, 77},
	{68, 79, 78},
	{69, 70, 79},
	{70, 81, 80},
	{71, 82, 81},
	{72, 83, 82},
	{73, 84, 83},
	{74, 85, 84},
	{75, 86, 85},
	{76, 87, 86},
	{77, 88, 87},
	{78, 89, 88},
	{79, 80, 89},
	{80, 91, 90},
	{81, 92, 91},
	{82, 93, 92},
	{83, 94, 93},
	{84, 95, 94},
	{85, 96, 95},
	{86, 97, 96},
	{87, 98, 97},
	{88, 99, 98},
	{89, 90, 99},
	{90, 101, 100},
	{91, 102, 101},
	{92, 103, 102},
	{93, 104, 103},
	{94, 105, 104},
	{95, 106, 105},
	{96, 107, 106},
	{97, 108, 107},
	{98, 109, 108},
	{99, 100, 109},
	{100, 111, 110},
	{101, 112, 111},
	{102, 113, 112},
	{103, 114, 113},
	{104, 115, 114},
	{105, 116, 115},
	{106, 117, 116},
	{107, 118, 117},
	{108, 119, 118},
	{109, 110, 119},
	{110, 121, 120},
	{111, 122, 121},
	{112, 123, 122},
	{113, 124, 123},
	{114, 125, 124},
	{115, 126, 125},
	{116, 127, 126},
	{117, 128, 127},
	{118, 129, 128},
	{119, 120, 129},
	{120, 131, 130},
	{121, 132, 131},
	{122, 133, 132},
	{123, 134, 133},
	{124, 135, 134},
	{125, 136, 135},
	{126, 137, 136},
	{127, 138, 137},
	{128, 139, 138},
	{129, 130, 139},
	{130, 141, 140},
	{131, 142, 141},
	{132, 143, 142},
	{133, 144, 143},
	{134, 145, 144},
	{135, 146, 145},
	{136, 147, 146},
	{137, 148, 147},
	{138, 149, 148},
	{139, 140, 149},
	{140, 151, 150},
	{141, 152, 151},
	{142, 153, 152},
	{143, 154, 153},
	{144, 155, 154},
	{145, 156, 155},
	{146, 157, 156},
	{147, 158, 157},
	{148, 159, 158},
	{149, 150, 159},
	{150, 161, 160},
	{151, 162, 161},
	{152, 163, 162},
	{153, 164, 163},
	{154, 165, 164},
	{155, 166, 165},
	{156, 167, 166},
	{157, 168, 167},
	{158, 169, 168},
	{159, 160, 169},
	{160, 171, 170},
	{161, 172, 171},
	{162, 173, 172},
	{163, 174, 173},
	{164, 175, 174},
	{165, 176, 175},
	{166, 177, 176},
	{167, 178, 177},
	{168, 179, 178},
	{169, 170, 179},
	{170, 181, 180},
	{171, 182, 181},
	{172, 183, 182},
	{173, 184, 183},
	{174, 185, 184},
	{175, 186, 185},
	{176, 187, 186},
	{177, 188, 187},
	{178, 189, 188},
	{179, 180, 189},
	{180, 191, 190},
	{181, 192, 191},
	{182, 193, 192},
	{183, 194, 193},
	{184, 195, 194},
	{185, 196, 195},
	{186, 197, 196},
	{187, 198, 197},
	{188, 199, 198},
	{189, 190, 199},
	{190, 201, 200},
	{191, 202, 201},
	{192, 203, 202},
	{193, 204, 203},
	{194, 205, 204},
	{195, 206, 205},
	{196, 207, 206},
	{197, 208, 207},
	{198, 209, 208},
	{199, 200, 209},
	{200, 211, 210},
	{201, 212, 211},
	{202, 213, 212},
	{203, 214, 213},
	{204, 215, 214},
	{205, 216, 215},
	{206, 217, 216},
	{207, 218, 217},
	{208, 219, 218},
	{209, 210, 219},
	{210, 221, 220},
	{211, 222, 221},
	{212, 223, 222},
	{213, 224, 223},
	{214, 225, 224},
	{215, 226, 225},
	{216, 227, 226},
	{217, 228, 227},
	{218, 229, 228},
	{219, 220, 229},
	{220, 231, 230},
	{221, 232, 231},
	{222, 233, 232},
	{223, 234, 233},
	{224, 235, 234},
	{225, 236, 235},
	{226, 237, 236},
	{227, 238, 237},
	{228, 239, 238},
	{229, 230, 239},
	{230, 241, 240},
	{231, 242, 241},
	{232, 243, 242},
	{233, 244, 243},
	{234, 245, 244},
	{235, 246, 245},
	{236, 247, 246},
	{237, 248, 247},
	{238, 249, 248},
	{239, 240, 249},
	{240, 251, 250},
	{241, 252, 251},
	{242, 253, 252},
	{243, 254, 253},
	{244, 255, 254},
	{245, 256, 255},
	{246, 257, 256},
	{247, 258, 257},
	{248, 259, 258},
	{249, 250, 259},
	{250, 261, 260},
	{251, 262, 261},
	{252, 263, 262},
	{253, 264, 263},
	{254, 265, 264},
	{255, 266, 265},
	{256, 267, 266},
	{257, 268, 267},
	{258, 269, 268},
	{259, 260, 269},
	{260, 271, 270},
	{261, 272, 271},
	{262, 273, 272},
	{263, 274, 273},
	{264, 275, 274},
	{265, 276, 275},
	{266, 277, 276},
	{267, 278, 277},
	{268, 279, 278},
	{269, 270, 279},
	{270, 281, 280},
	{271, 282, 281},
	{272, 283, 282},
	{273, 284, 283},
	{274, 285, 284},
	{275, 286, 285},
	{276, 287, 286},
	{277, 288, 287},
	{278, 289, 288},
	{279, 280, 289},
	{280, 291, 290},
	{281, 292, 291},
	{282, 293, 292},
	{283, 294, 293},
	{284, 295, 294},
	{285, 296, 295},
	{286, 297, 296},
	{287, 298, 297},
	{288, 299, 298},
	{289, 290, 299},
	{290, 1, 0},
	{291, 2, 1},
	{292, 3, 2},
	{293, 4, 3},
	{294, 5, 4},
	{295, 6, 5},
	{296, 7, 6},
	{297, 8, 7},
	{298, 9, 8},
	{299, 0, 9},
};

#endif
