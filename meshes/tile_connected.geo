Print.X3dPrecision = 1E-15;
Geometry.Tolerance = 1E-12;
Mesh.PreserveNumberingMsh2 = 1;

DefineConstant[
radius = {9, Name "radius of cell"}
length = {100, Name "length of cell (body)"}
pad_x = {1, Name "bounding box padding in x directions"}
radius_x = {7, Name "radius of connection in x direction"}
pad_y = {1, Name "bounding box padding in y directions"}
pad_z = {3, Name "bounding box padding in z directions"}
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

Rotate {{1, 0, 0}, {0, 0, 0}, Pi/2} {
  Volume{half};
}

Translate {0, -length/2, -ext_length} {
  Volume{half};
}

half_mirror = Rotate {{1, 0, 0}, {0, 0, 0}, Pi} {
  Duplicata { Volume{1}; }
};


// Let's add ports in z direction. Just a cylinder
port_x = newv;
Cylinder(port_x) = {0, 0, -length/2 - pad_x, 0, 0, (length+2*pad_x), radius_x, 2*Pi};


cell() = BooleanUnion{ Volume{half, half_mirror}; Delete; }{Volume {port_x}; Delete; };

Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} {Volume{cell}; }


// Enclose in a bounding box
// Bounding box
box = newv;

min_x = -length/2 - pad_x;
max_x = length/2 + pad_x;
min_y = -radius - pad_y;
max_y = radius + pad_y;
min_z = -radius-pad_z;
max_z = radius+pad_z;

Box(box) = {min_x, min_y, min_z, max_x-min_x, max_y-min_y, max_z-min_z};

v() = BooleanFragments{ Volume{box}; Delete; }{Volume {cell}; Delete; };

// Tags
cell = v[0];
box = v[1];

Physical Volume(1) = {cell};
Physical Volume(0) = {box};

interfaces[] = Unique(Abs(Boundary{ Volume{cell}; }));  
boundary[] = Unique(Abs(Boundary{ Volume{box}; }));
boundary[] -= {interfaces[]};

Physical Surface(1) = {interfaces[]};


// Periodicity

// y direction
surfMaster = 15;
surfSlave = 18;
boundMaster[] = {34, 30, 36, 35};
boundSlave[] = {39, 33, 37, 41};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

// z direction
surfMaster = 16;
surfSlave = 17;
boundMaster[] = {31, 36, 38, 37};
boundSlave[] = {32, 34, 40, 39};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

// x direction
surfMaster = 19;
surfSlave = 14;
boundMaster[] = {35, 40, 41, 38, 19};
boundSlave[] = {30, 32, 33, 31, 29};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

// x connection
surfMaster = 7;
surfSlave = 13;
boundMaster[] = {19};
boundSlave[] = {29};
Periodic Surface surfSlave { boundSlave[] } = surfMaster { boundMaster[] };

