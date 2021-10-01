Print.X3dPrecision = 1E-15;
Geometry.Tolerance = 1E-12;
Mesh.PreserveNumberingMsh2 = 1;

DefineConstant[
radius = {9, Name "radius of cell"}
length = {100, Name "length of cell (body)"}
pad = {1, Name "bounding box padding in all directions"}
Hein = {5.0, Name "Superellipse exponent"}
];

// The length units here micro meters
SetFactory("OpenCASCADE");

// Create the main body of the cell. Well quarete and then rotate for rest
r = Hein;
npts = 20;  // Control points for the ellipse
size = 10;

pointFirst = newp;
Point(pointFirst) = {0, radius, -length/2, size};
P[] = {pointFirst};

P0 = newp;
dtheta = Pi/2/(npts-1);
For i In {1:npts-2}
    theta = i*dtheta;
    xi = radius*Exp((2./r)*Log(Sin(theta)));
    yi = radius*Exp((2./r)*Log(Cos(theta)));

    Point(P0 + i - 1) = {xi, yi, -length/2, size};
    P[] += {P0 + i - 1};
EndFor
pointLast = newp;
Point(pointLast) = {radius, 0, -length/2, size};
P[] += {pointLast};

// The arch
L = newl;
Bezier(L) = {P[]};

// The quarter by extrusion
origin = newp;
Point(origin) = {0, 0, -length/2, size};
//+

toFirst = newl;
Line(toFirst) = {pointFirst, origin};
toLast = newl;
Line(toLast) = {pointLast, origin};

loop = newl;
Line Loop(loop) = {L, toFirst, -toLast};

base = news;
Plane Surface(base) = {loop};

cellsphere = Extrude {{0, 1, 0}, {0, 0, -length/2}, Pi/2} {
	Surface{base};Layers{32};Recombine;
};

ext_length = -length/2 + radius;

cellext = Extrude {0, ext_length, 0} { Surface{7}; };

ei = BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; };
ei_mirror = Rotate {{0, 1, 0}, {0, 0, -length/2}, Pi/2} {
  Duplicata { Volume{1}; }
};

quarter = BooleanUnion{ Volume{1}; Delete; }{Volume {2}; Delete; };
quarter_mirror = Rotate {{0, 1, 0}, {0, 0, -length/2}, Pi} {
  Duplicata { Volume{1}; }
};

half = BooleanUnion{ Volume{1}; Delete; }{Volume {2}; Delete; };
half_mirror = Rotate {{0, 0, 1}, {0, ext_length, -length/2}, Pi} {
  Duplicata { Volume{1}; }
};

cell() = BooleanUnion{ Volume{1}; Delete; }{Volume {2}; Delete; };

// Enclose in a bounding box
// Bounding box
box = newv;

min_x = -radius - pad;
max_x = radius + pad;
min_y = -radius - pad;
max_y = radius + pad;
min_z = -length/2-pad;
max_z = length/2+pad;

Box(box) = {min_x, min_y, min_z, max_x-min_x, max_y-min_y, max_z-min_z}; 

//+
Rotate {{1, 0, 0}, {0, 0, 0}, Pi/2} {
  Volume{cell};
}

//+
Translate {0, -length/2, -ext_length} {
  Volume{1};
}

v() = BooleanFragments {Volume{box}; Delete; }{Volume{cell}; Delete; };
// Rotate to get ports in x, y
v[] = Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} {Volume{v[]}; };

// Tags
cell = v[0];
box = v[1];

Physical Volume(1) = {cell};
Physical Volume(0) = {box};

//+
Physical Surface("surface_xmin") = {65};
//+
Physical Surface("surface_xmax") = {63};
//+
Physical Surface("surface_zmin") = {66};
//+
Physical Surface("surface_zmax") = {61};
//+
Physical Surface("surface_ymin") = {62};
//+
Physical Surface("surface_ymax") = {64};

// X (this is gmsh 4.4.1 numbering, might need adjusting for other versions)

surfMaster = 65;
surfSlave = 63;
boundMaster[] = {144, 137, 136, 142};
boundSlave[] = {141, 139, 134, 140};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

// Y
surfMaster = 66;
surfSlave = 61;
boundMaster[] = {138, 141, 143, 144};
boundSlave[] = {133, 134, 135, 136};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

// Z
surfMaster = 62;
surfSlave = 64;
boundMaster[] = {137, 138, 139, 133};
boundSlave[] = {142, 143, 140, 135};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

cell_surf[] = Unique(Abs(Boundary{ Volume{cell}; }));  
Physical Surface(1) = {cell_surf[]};

//+
midpt = newp;
Point(midpt) = {0, 0, 0, 1.0};
Point{midpt} In Volume {1};

dist = 2;

midpt_x1 = newp;
Point(midpt_x1) = {dist, 0, 0, 1.0};
midpt_x2 = newp;
Point(midpt_x2) = {-dist, 0, 0, 1.0};

midpt_y1 = newp;
Point(midpt_y1) = {0, dist, 0, 1.0};
midpt_y2 = newp;
Point(midpt_y2) = {0, -dist, 0, 1.0};

midpt_z1 = newp;
Point(midpt_z1) = {0, 0, dist, 1.0};
midpt_z2 = newp;
Point(midpt_z2) = {0, 0, -dist, 1.0};

Point{midpt_x1, midpt_x2, midpt_y1, midpt_y2, midpt_z1, midpt_z2} In Volume{1};
