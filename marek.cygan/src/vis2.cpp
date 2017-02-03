int SHIFT = 0;
int VISUALIZE = 0;
int PIECES = -1;
int ROTATE = 0;

#include "CImg.h"
#include <bits/stdtr1c++.h>
#include <unistd.h>
#include <sys/time.h>

using namespace std;
using namespace cimg_library;

const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 }, yellow[] = {255, 255, 0};

typedef vector <int> VI;
typedef vector <VI> VVI;
typedef long long LL;
typedef vector <LL> VLL;
typedef vector <double> VD;
typedef vector <VD> VVD;
typedef vector <string> VS;
typedef vector <VS> VVS;
typedef pair<int,int> PII;
typedef vector <PII> VPII;
typedef istringstream ISS;
typedef pair<double,double> PDD;

#define ALL(x) x.begin(),x.end()
#define REP(i,n) for (int i=0; i<(n); ++i)
#define FOR(var,pocz,koniec) for (int var=(pocz); var<=(koniec); ++var)
#define FORD(var,pocz,koniec) for (int var=(pocz); var>=(koniec); --var)
#define FOREACH(it, X) for(__typeof((X).begin()) it = (X).begin(); it != (X).end(); ++it)
#define PB push_back
#define PF push_front
#define MP(a,b) make_pair(a,b)
#define ST first
#define ND second
#define SIZE(x) (int)x.size()

template<class T> string i2s(T x) {ostringstream o; o << x; return o.str();}
template<class T1,class T2> ostream& operator<<(ostream &os, pair<T1,T2> &p) {os << "(" << p.first << "," << p.second << ")"; return os;}
template<class T> ostream& operator<<(ostream &os, vector<T> &v) {os << "{"; REP(i, (int)v.size()) {if (i) os << ", "; os << v[i];} os << "}"; return os;}
#define DB(a) {cerr << #a << ": " << (a) << endl; fflush(stderr); }

namespace Mytime{
  double start_time;
  static double last_call = 0;
   
  double get_time() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec+tv.tv_usec*1e-6;
  }

  void print_time(string s) {
    double x = get_time();
    fprintf(stderr,"%s cur=%.6lf lap=%.6lf\n",s.c_str(),x,x-last_call);
    last_call = x;
  }

  void init_time() {
    start_time = get_time();
    last_call = start_time;
  }
}

#define STOPER(name) Stoper name(#name);

struct Stoper;
vector<Stoper*> stoper_pointers;

struct Stoper {
  double used_time;
  double last_call;
  string name;
  void start() {
    last_call = Mytime::get_time();
  }
  void stop() {
    used_time += Mytime::get_time() - last_call;
  }

  Stoper(string s="") {
    used_time = 0.0;
    name=s;
    stoper_pointers.PB(this);
  }
}; 

STOPER(st_whole);
STOPER(st_inter);
STOPER(st_find);

/************************************************************************/
/************************ Code starts here ******************************/
/************************************************************************/

string TRAIN_DATA_DIR = "competition1/spacenet_TrainData/";
string TEST_DATA_DIR = "competition1/spacenet_TestData/";

string train_csv_path = TRAIN_DATA_DIR+"vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv";
string train_images_path = TRAIN_DATA_DIR+"3band/3band_AOI_1_RIO_img";
string test_images_path = TEST_DATA_DIR+"3band/3band_AOI_2_RIO_img";
string train_labels_path = TRAIN_DATA_DIR+"labels/";
string train_heat_path = TRAIN_DATA_DIR+"heatmaps/";

const int TRAIN_IMAGES = 6940;
const int TEST_IMAGES = 2795;
const int IMAGE_WIDTH = 438;
const int IMAGE_HEIGHT = 406;
typedef vector<pair<double,double>> Polygon;
vector<Polygon> polygons[TRAIN_IMAGES * 2];
int seen[TRAIN_IMAGES * 2];
int strange_img[TRAIN_IMAGES];

Polygon translate(Polygon poly, double dx, double dy) {
  for (auto &p: poly) {
    p.ST += dx;
    p.ND += dy;
  }
  return poly;
}

double stripe(double x1, double x2, double y, double px1, double py1, double px2, double py2) {
  const double EPS = 1E-4;
  assert(px1 <= px2);
  assert(x1 <= x2);

  double xl = max(x1, px1);
  double xr = min(x2, px2);

  if (xl+EPS >= xr) return 0.0;
  double yl = (xl - px1) / (px2 - px1) * (py2 - py1) + py1;
  double yr = (xr - px1) / (px2 - px1) * (py2 - py1) + py1;
  if (y <= min(yl, yr)+EPS) return (xr - xl) * y;
  else if (y >= max(yl, yr)-EPS) return (xr - xl) * min(yl, yr) + (xr - xl) * fabs(yl - yr) / 2.0;
  else {
    double xx = (y - yl) / (yr - yl) * (xr - xl) + xl;
    assert(xx >= xl && xx <= xr);
    if (yl <= yr) return (xr - xl) * y - (xx - xl) * (y - yl) / 2.0;
    else return (xr - xl) * y - (xr - xx) * (y - yr) / 2.0;
  }
}


double poly_stripe(const Polygon &poly, double x1, double x2, double y) {
  double res = 0.0;
  REP(i, SIZE(poly)-1) {
    if (poly[i].ST < poly[i+1].ST) res += stripe(x1, x2, y, poly[i].ST, poly[i].ND, poly[i+1].ST, poly[i+1].ND);
    else res -= stripe(x1, x2, y, poly[i+1].ST, poly[i+1].ND, poly[i].ST, poly[i].ND);
  }
  return fabs(res);
}

double polygon_area(const Polygon &poly) {
  double res = 0.0;
  REP(i, SIZE(poly)-1) res += (poly[i+1].ST - poly[i].ST) * (poly[i].ND + poly[i+1].ND) / 2.0;
  return fabs(res);
}

struct RectDesc{
  double cx, cy;
  double w, h;
  double angle;

  RectDesc(double _cx, double _cy, double _w, double _h, double _angle) : cx(_cx), cy(_cy), w(_w), h(_h), angle(_angle) {}

  inline PDD rotate(double a, double x, double y) {
    return MP(x * cos(a) - y * sin(a), y * cos(a) + x * sin(a));
  }

  Polygon gen() {
    pair<double,double> t[5] = {MP(0.0, 0.0), MP(w, 0.0), MP(w, h), MP(0., h), MP(0.0, 0.0)};
    REP(i,5) {
      double x = t[i].ST - w/2.0;
      double y = t[i].ND - h/2.0;
      t[i] = rotate(angle, x, y);
      t[i].ST += cx;
      t[i].ND += cy;
    }
    return Polygon(t, t+5);
  }

  double intersect(const Polygon &poly) {
    Polygon tp = translate(poly, -cx, -cy);
    if (fabs(angle) > 1e-6) {
      for (auto &p : tp) p = rotate(-angle, p.ST, p.ND);
    }
    return poly_stripe(tp, -w/2, w/2, h/2) - poly_stripe(tp, -w/2, w/2, -h/2);
  }

  //new
  double IoU(const Polygon &poly) {
    double inter = intersect(poly);
    double area1 = w * h;
    double area2 = polygon_area(poly);
    return inter / (area1 + area2 - inter);
  }

  double IoU(RectDesc r) {
    return IoU(r.gen());
  }

  string gen_string(double width, double height) {
    Polygon poly = gen();
    string res = "\"POLYGON ((";
    assert(SIZE(poly) == 5);
    char txt[1000];
    REP(i, SIZE(poly)) {
      sprintf(txt, "%.2lf %.2lf 0.0", poly[i].ST * width, poly[i].ND * height);
      if (i) res += ",";
      res += txt;
    }
    res += "))\"";
    return res;
  }
};
vector<RectDesc> poly_descs;
vector<Polygon> gen_polys;

bool operator<(const RectDesc &r1, const RectDesc &r2) {
  return r1.cx < r2.cx;
}

VI positions(const char *txt, char c) {
  VI res;
  int i = 0;
  while (txt[i] != 10 && txt[i] != 13 && txt[i]) {
    if (txt[i] == c) res.PB(i);
    ++i;
  }
  return res;
}

Polygon parse_polygon(const char *txt, int &strange) {
  strange = 0;
  Polygon res;
  string s(txt);
  assert(s.substr(0, 10) == "POLYGON ((");
  VI commas = positions(txt, ',');
  commas.insert(commas.begin(), 9);
  int close = 0;
  while (txt[close] != ')') close++;
  static int calls = 0;
  calls++;
  for (auto i : commas) if (i < close) {
    double x, y;
    //if (calls < 10) DB(txt+i);
    x = atof(txt+i+1);
    int j = i+1;
    while (txt[j] != ' ') j++;
    y = atof(txt+j+1);
    j++;
    while (txt[j] != ' ') j++;
    j++;
    assert(txt[j] == '0');
    assert(txt[j+1] == ',' || txt[j+1] == ')');
    res.PB(MP(x,y));
  }
  assert(SIZE(res) > 3);
  if (calls < 2) {
    DB(txt);
    DB(res);
  }
  assert(s[SIZE(s)-1] == ')');
  assert(s[SIZE(s)-2] == ')');
  int j = 0;
  int a = 0, b = 0;
  while (txt[j]) {
    if (txt[j] == '(') a++;
    if (txt[j] == ')') b++;
    j++;
  }
  strange = a!=2;
  assert(a == b);
  /*
     TODO
  if (a != 2 || b != 2) {
    static int count = 0;
    count++;
    DB(a); DB(b);
    DB(count);
  }
  */
  //DB(txt);
  //DB(res);
  assert(fabs(res[0].ST - res.back().ST) < 1E-6);
  assert(fabs(res[0].ND - res.back().ND) < 1E-6);
  return res;
}

void load_csv() {
  const int T = 1000000;
  char txt[T];
  FILE *f = fopen(train_csv_path.c_str(), "r");
  fgets(txt, T-1, f);
  int lines = 0;
  REP(i, TRAIN_IMAGES) seen[i] = 0;
  while (fgets(txt, T-1, f) != NULL) {
    ++lines;
    VI comma_pos = positions(txt, ',');
    VI quote_pos = positions(txt, '\"');
    //DB(txt);
    txt[comma_pos[0]] = 0;
    txt[comma_pos[1]] = 0;
    VI underline_pos = positions(txt, '_');
    assert(SIZE(underline_pos) == 3);
    int num = atoi(txt+underline_pos[2]+1+3);
    seen[num] = 1;
    if (quote_pos.empty()) {
      assert(SIZE(comma_pos) == 3);
      assert(string(txt+comma_pos[1]+1) == "POLYGON EMPTY,POLYGON EMPTY\n");
      continue;
    }
    assert(SIZE(quote_pos) == 4);
    txt[quote_pos[1]] = 0;
    int strange;
    polygons[num].PB(parse_polygon(txt+quote_pos[0]+1, strange));
    strange_img[num] |= strange;
  }
  FOR(i,1,TRAIN_IMAGES) assert(seen[i] == 1);
  DB(lines);
  fclose(f);

  //SCALE
  FOR(i,1,TRAIN_IMAGES) REP(j, SIZE(polygons[i])) REP(k, SIZE(polygons[i][j])) {
    polygons[i][j][k].ST /= IMAGE_WIDTH; //TODO load this value from image
    polygons[i][j][k].ND /= IMAGE_HEIGHT;
    assert(polygons[i][j][k].ST <= 1.0+1E-2);
    assert(polygons[i][j][k].ND <= 1.0+1E-2);
    assert(polygons[i][j][k].ST >= -1E-2);
    assert(polygons[i][j][k].ND >= -1E-2);
  }
}

void mydraw_poly(CImg<unsigned char> &img, const Polygon &poly, const unsigned char *color, double opacity) {
  int w = img.width();
  int h = img.height();
  CImg<int> points(SIZE(poly),2);
  REP(k, SIZE(poly)){
    points(k,0) = (int)(poly[k].ST * w);
    points(k,1) = (int)(poly[k].ND * h);
  }
  img.draw_polygon(points, color, opacity);
  if (opacity < 0.01) {
    REP(k, SIZE(poly)-1) {
      pair<double,double> p1 = poly[k];
      pair<double,double> p2 = poly[k+1];
      img.draw_line((int)(p1.ST * w), (int)(p1.ND * h), (int)(p2.ST * w), (int)(p2.ND * h), color);
    }
  }
}

void draw_roi(CImg<unsigned char> &img, RectDesc roi, const unsigned char *color, double opacity) {
  mydraw_poly(img, roi.gen(), blue, 0.0);
}

void calc_bounding_box(const Polygon &p1, double *t) {
  t[0] = 1; t[1] = 0; t[2] = 1; t[3] = 0;
  for (auto p : p1) {
    t[0] = min(t[0], p.ST);
    t[1] = max(t[1], p.ST);
    t[2] = min(t[2], p.ND);
    t[3] = max(t[3], p.ND);
  }
}

#define POINTT double
#define POINTR double 
struct POINT {
  POINTT x,y;
  POINT(POINTT wx, POINTT wy) : x(wx), y(wy) {}
  POINT(PDD p) {x = p.ST; y = p.ND;}
};
#define Det(p1,p2,w) (POINTR(p2.x-p1.x)*POINTR(w.y-p1.y)-POINTR(p2.y-p1.y)*POINTR(w.x-p1.x))
int sgn(double x){ return x > 0 ? 1 : (x < 0 ? -1 : 0); }
inline bool SegmentCross(const POINT& p1, const POINT& p2, const POINT& l1, const POINT& l2) {
  return sgn(Det(p1,p2,l1))*sgn(Det(p1,p2,l2)) == -1 && sgn(Det(l1,l2,p1))*sgn(Det(l1,l2,p2)) == -1;
}
bool PointInPol(const Polygon &poly, double x, double y) {
  double x2 = 2.0;
  int acc = 0;
  REP(i, SIZE(poly)-1) {
    acc += SegmentCross(POINT(x,y), POINT(x2,y), poly[i], poly[i+1]);
  }
  return acc&1;
}


int inter_calls;

double intersection_area(const Polygon &p1, const Polygon &p2) {
  double t[2][4];
  calc_bounding_box(p1, t[0]);
  calc_bounding_box(p2, t[1]);
  if (t[0][0] > t[1][1] || t[1][0] > t[0][1]) return 0.0;
  if (t[0][2] > t[1][3] || t[1][2] > t[0][3]) return 0.0;
  inter_calls++;
  double minx = min(t[0][0],t[1][0]);
  double maxx = max(t[0][1],t[1][1]);
  double miny = min(t[0][2],t[1][2]);
  double maxy = max(t[0][3],t[1][3]);
  int c = 0;
  const int ITERS = 10000;
  REP(foo, ITERS) {
    double x = minx + (maxx - minx) * drand48();
    double y = miny + (maxy - miny) * drand48();
    int a = PointInPol(p1, x, y);
    int b = PointInPol(p2, x, y);
    c += a && b;
  }
  return (double)c / ITERS * (maxx - minx) * (maxy - miny);
}

void generate_polygons(int pieces, int centers, int angles, int scheme) {
  REP(cx,centers) REP(cy,centers) {
    if (scheme == 0) {
      for (auto w = 0.03; w <= 0.15; w *= 1.5) {
        for (auto h = 0.03; h <= 0.15; h *= 1.5) {
          REP(a, angles) {
            RectDesc rd((0.5 + cx) / pieces / centers, (0.5 + cy) / pieces / centers, w, h, acos(0.0) / angles * a);
            poly_descs.PB(rd);
            gen_polys.PB(rd.gen());
          }
        }
      }
    } else if (scheme == 1) {
      for (auto w = 0.02; w <= 0.15; w *= 1.22) {
        for (auto h = 0.02; h <= 0.15; h *= 1.22) {
          REP(a, angles) {
            RectDesc rd((0.5 + cx) / pieces / centers, (0.5 + cy) / pieces / centers, w, h, acos(0.0) / angles * a);
            poly_descs.PB(rd);
            gen_polys.PB(rd.gen());
          }
        }
      }
    } else if (scheme == 2) {
      for (auto w = 0.03; w <= 0.15; w *= 1.23) {
        for (auto h = 0.03; h <= 0.15; h *= 1.23) {
          REP(a, angles) {
            RectDesc rd((0.5 + cx) / pieces / centers, (0.5 + cy) / pieces / centers, w, h, acos(0.0) / angles * a);
            poly_descs.PB(rd);
            gen_polys.PB(rd.gen());
          }
        }
      }
    } else assert(0);
  }

  if (VISUALIZE) {
    CImgDisplay disp;
    const int SS = 300;
    const double S = 300;
    REP(i, SIZE(gen_polys)) {
      Polygon tp = translate(gen_polys[i], (pieces/2) * 1.0 / pieces, (pieces/2) * 1.0 / pieces);
      CImg<unsigned char> img(SS, SS, 1, 3);
      img.fill(0);
      img.draw_rectangle((int)(S / pieces * (pieces/2)), (int)(S / pieces * (pieces/2)), 
          (int)(S / pieces * (pieces/2+1)), (int)(S / pieces * (pieces/2+1)), green, 0.1);
      mydraw_poly(img, tp, red, 0.1);
      disp = img;
      usleep(200);
    //  while (!disp.is_closed()) {
    //    disp.wait();
    //    if (disp.is_keySPACE()) break;
    //  }
    }
  }
  DB(SIZE(gen_polys));
}

const int MAX_PIECES = 105;
const int MAX_RECTANGLES = 1300;
int mark_rect[MAX_PIECES][MAX_PIECES][MAX_RECTANGLES];
int mark_rect_which[MAX_PIECES][MAX_PIECES][MAX_RECTANGLES]; //image id (>= 1)
int mark_total[MAX_RECTANGLES];
int mark_best[MAX_RECTANGLES];

VI vi_from_s(char *txt) {
  VI res;
  int i = 0;
  int acc = 0;
  int digits = 0;
  while (txt[i]) {
    if (txt[i] == ',') {
      assert(digits);
      res.PB(acc);
      digits = 0;
      acc = 0;
    } else {
      assert(txt[i] >= '0' && txt[i] <= '9');
      acc = 10 * acc + txt[i] - '0';
      digits++;
    }
    i++;
  }
  assert(digits);
  res.PB(acc);
  return res;
}

void mark_train_images(int pieces) {
  const int T = 1000000;
  char txt[T];
  CImgDisplay main_disp;
  FOR(i,1,TRAIN_IMAGES) {
    DB(i);
    CImg<unsigned char> img((train_images_path+i2s(i)+".tif").c_str());
    main_disp = img;
    main_disp.set_title("%d", i);
    FILE *fl = fopen((train_labels_path+i2s(i)).c_str(), "r");
    int drawn = 0;
    REP(py,pieces) REP(px,pieces) {
      double dx = (double)px / pieces;
      double dy = (double)py / pieces;
      fscanf(fl, "%s", txt);
      VI v = vi_from_s(txt);
      int k = v[0];
      assert(SIZE(v) == 2*k+1);
      REP(i, k) {
        int a = v[2*i+1];
        mydraw_poly(img, translate(gen_polys[a], dx, dy), blue, 0.0);
        drawn++;
      }
    }
    DB(drawn);
    main_disp = img;
    while (drawn > 0 && !main_disp.is_closed()) {
      main_disp.wait();
      if (main_disp.is_keySPACE()) break;
    }
    fclose(fl);
  }
}

void pause(CImgDisplay &disp) {
  while (!disp.is_closed()) {
    disp.wait();
    if (disp.button() && disp.mouse_y()>=0) {
      //TODO
      const int y = disp.mouse_y();
      const int x = disp.mouse_x();
      DB(x);
      DB(y);
    }
    if (disp.is_keySPACE()) break;
  }
}

void mark_heatmaps(string filename) {
  FILE *f = fopen(filename.c_str(), "r");
  const int T = 1000000;
  char txt[T];
  CImgDisplay disp_img, disp_heat, disp_heat_truth, disp_max, disp_max2, disp_max3;
  while (fgets(txt, T-1, f) != NULL) {
    string s(txt);
    int test = s.find("test") != string::npos;
    s = s.substr(0, SIZE(s)-1); //remove endl
    DB(s);
    int j = SIZE(s)-1;
    while (!isdigit(s[j])) j--;
    int p = 1;
    int i = 0;
    while (isdigit(s[j])) {
      i += p * (s[j] - '0');
      p *= 10;
      j--;
    }
    DB(i);
    if (test) {
      CImg<unsigned char> img((test_images_path+i2s(i)+".tif").c_str());
      disp_img = img;
    } else {
      CImg<unsigned char> img((train_images_path+i2s(i)+".tif").c_str());
      REP(j, SIZE(polygons[i])) mydraw_poly(img, polygons[i][j], red, 0.5);
      disp_img = img;
    }
    disp_img.set_title("%d", i);

    if (test) {
      CImg<unsigned char> heat_truth(400,400,1,3,0);
      disp_heat_truth = heat_truth;
    } else {
      CImg<unsigned char> heat_truth((train_heat_path+i2s(i)+".jpg").c_str());
      disp_heat_truth = heat_truth;
    }

    CImg<unsigned char> heat(s.c_str());
    disp_heat = heat;
    CImg<unsigned char> heat_max(heat.width(), heat.height(), 1, 3, 0);
    CImg<unsigned char> heat_max2(heat.width(), heat.height(), 1, 3, 0);
    CImg<unsigned char> heat_max3(heat.width(), heat.height(), 1, 3, 0);
    REP(i,heat.width()) REP(j,heat.height()) {
      int a = (unsigned int)heat(i, j, 0, 0); 
      int b = (unsigned int)heat(i, j, 0, 1); 
      int c = (unsigned int)heat(i, j, 0, 2); 
      if (a >= b && a >= c) heat_max(i, j, 0, 0) = 255;
      else if (b >= a && b >= c) heat_max(i, j, 0, 1) = 255;
      else {
        assert(c >= a && c >= b);
        heat_max(i, j, 0, 2) = 255;
      }
    }

    REP(i,heat.width()) REP(j,heat.height()) {
      int a = (unsigned int)heat(i, j, 0, 0) * 2; 
      int b = (unsigned int)heat(i, j, 0, 1) * 2; 
      int c = (unsigned int)heat(i, j, 0, 2); 
      if (a >= b && a >= c) heat_max2(i, j, 0, 0) = 255;
      else if (b >= a && b >= c) heat_max2(i, j, 0, 1) = 255;
      else {
        assert(c >= a && c >= b);
        heat_max2(i, j, 0, 2) = 255;
      }
    }

    REP(i,heat.width()) REP(j,heat.height()) {
      int a = (unsigned int)heat(i, j, 0, 0); 
      int b = (unsigned int)heat(i, j, 0, 1) * 2; 
      int c = (unsigned int)heat(i, j, 0, 2); 
      if (a >= b && a >= c) heat_max3(i, j, 0, 0) = 255;
      else if (b >= a && b >= c) heat_max3(i, j, 0, 1) = 255;
      else {
        assert(c >= a && c >= b);
        heat_max3(i, j, 0, 2) = 255;
      }
    }
    disp_max = heat_max;
    disp_max2 = heat_max2;
    disp_max3 = heat_max3;
    pause(disp_img);
  }
}

void pause_disp(CImgDisplay &main_disp) {
  while (!main_disp.is_closed()) {
    main_disp.wait();
    if (main_disp.is_keySPACE()) break;
  }
}

int extract_id(string s) {
  int j = SIZE(s)-1;
  while (!isdigit(s[j])) j--;
  int p = 1;
  int i = 0;
  while (isdigit(s[j])) {
    i += p * (s[j] - '0');
    p *= 10;
    j--;
  }
  return i;
}

void extract_offsets(string s, double &scale, double &ox, double &oy) {
  int n = SIZE(s);
  scale = 1.0;
  ox = oy = 0.0;
  if (s[n-3] != '_') return;
  scale = 400.0 / (400.0 - (s[n-6] - 'a') * 20);
  oy = (s[n-4]-'a') / 400.0;
  ox = (s[n-2]-'a') / 400.0;
  fprintf(stderr, "offsets for %s are %.6lf %.6lf %.6lf\n", s.c_str(), scale, ox, oy);
}

int touched_nms[TRAIN_IMAGES][2];

void calc_nms(string nms_file, double threshold, double conf_threshold, int pieces) {
  DB(conf_threshold);
  FILE *f = fopen(nms_file.c_str(), "r");
  const int T = 1000000;
  char txt[T];
  CImgDisplay main_disp;
  CImg<unsigned char> img;
  vector<pair<double,int>> vpairs;
  int total_poly = 0; //false negatives
  FILE *ftest = NULL;
  vector<pair<double,RectDesc>> rois[TRAIN_IMAGES+1][2]; //id, train/test

  while (fgets(txt, T-1, f) != NULL) {
    string s(txt);
    double offset_x, offset_y;
    double scale;
    extract_offsets(s, scale, offset_x, offset_y);
    s = s.substr(0, SIZE(s)-1); //remove endl
    DB(s);
    if (VISUALIZE) main_disp.set_title(s.c_str());

    int test = s.find("test") != string::npos;
    int i = -1;
    i = extract_id(s);
    DB(i);
    touched_nms[i][test] = 1;

    FILE *fp = fopen(s.c_str(), "r");
    FILE *fp2 = fopen((s+".meta").c_str(), "r");
    if (fp == NULL) continue;
//    vector<pair<double,RectDesc>> rois;
    while (fgets(txt, T-1, fp) != NULL) {
      double conf;
      int px, py, rect;
      int r = sscanf(txt, "%lf %d %d %d", &conf, &py, &px, &rect);
      assert(r == 4);
      fgets(txt, T-1, fp2);
      double desc[5];
      sscanf(txt, "%lf %lf %lf %lf %lf %d %d %d", desc, desc+1, desc+2, desc+3, desc+4, &py, &px, &rect);
      if (conf >= conf_threshold) {
        RectDesc roi = poly_descs[rect];
        roi.cx += 1.0 * px / pieces + offset_x; 
        roi.cy += 1.0 * py / pieces + offset_y; 
        //TODO shift, scale and rotate
        if (SHIFT) {
          double w = roi.w;
          double h = roi.h;
          roi.cx += desc[0] * w;
          roi.cy += desc[1] * h;
          roi.w *= exp(desc[2]);
          roi.h *= exp(desc[3]);
          roi.angle += desc[4];
        }
        roi.cx *= scale;
        roi.cy *= scale;
        roi.w *= scale;
        roi.h *= scale;
        rois[i][test].PB(MP(conf,roi));
      }
    }
    fclose(fp);
    fclose(fp2);
  }
  FOR(i,1,TRAIN_IMAGES) REP(test,2) if (touched_nms[i][test]) {
    DB(i);
    DB(test);
    int testw = 0, testh = 0;
    if (!test) total_poly += SIZE(polygons[i]);
    if (1 || VISUALIZE || test) {
      img = CImg<unsigned char>(((test ? test_images_path : train_images_path)+i2s(i)+".tif").c_str());
      testw = img.width();
      testh = img.height();
    }
    if (VISUALIZE) main_disp = img;
    if (test) {
      DB(testw);
      DB(testh);
    }

    sort(ALL(rois[i][test]));
    reverse(ALL(rois[i][test]));
    vector<RectDesc> picked;
    vector<double> picked_conf;
    for (auto p : rois[i][test]) {
      RectDesc roi = p.ND;
      //        DB(roi.conf); DB(roi.px); DB(roi.py); DB(roi.rect);
      int ok = 1;
      if (threshold < 1) {
        for (auto pr : picked) {
          if (pr.IoU(roi) >= threshold) {
            ok = 0;
            break;
          }
        }
      }
      if (ok) {
        picked.PB(roi);
        picked_conf.PB(p.ST);
      }
    }

    if (!test) {
      VI marked(SIZE(polygons[i]), 0);
      VD marked_iou(SIZE(polygons[i]), 0.);
      VD vconf(SIZE(polygons[i]), 0.);

      REP(foo, SIZE(picked)) {
        RectDesc roi = picked[foo];
        int num_marked = 0;
        REP(j, SIZE(polygons[i])) {
          double x = roi.IoU(polygons[i][j]);
          marked_iou[j] = max(marked_iou[j], x);
          if (!marked[j] && x >= 0.5) {
            marked[j] = 1;
            vconf[j] = picked_conf[foo];
            num_marked++;
          }
        }
        assert(num_marked <= 1);
        vpairs.PB(MP(picked_conf[foo], num_marked));
      }
      REP(j, SIZE(polygons[i])) {
        if (marked[j]) {
          DB("jest!!! " + i2s(marked_iou[j]) + i2s(vconf[j]));
        } else {
          DB("no :(" + i2s(marked_iou[j]));
        }
        if (VISUALIZE) mydraw_poly(img, polygons[i][j], marked[j] ? green : red, 0.5);
      }
    }
    if (ftest == NULL){
      ftest = fopen("submission.csv", "w");
      fprintf(ftest, "ImageId,BuildingId,PolygonWKT_Pix,Confidence\n");
    }
    int id = 0;
    REP(foo, SIZE(picked)) {
      RectDesc roi = picked[foo];
      ++id;
      fprintf(ftest, "AOI_%d_RIO_img%d,%d,%s,%.6lf\n", (test ? 2 : 1), i, id, roi.gen_string(testw, testh).c_str(), exp(picked_conf[foo]));
    }

      //DB(vpairs);
    if (VISUALIZE) {
      for (auto roi: picked) draw_roi(img, roi, blue, 0.0);
      main_disp = img;
      pause_disp(main_disp);
    }
  }
  if (ftest) fclose(ftest);
  sort(ALL(vpairs)); reverse(ALL(vpairs));
  int tp = 0, fp = 0;
  double bestf = 0.0, bestconf = 0.0;
  double fffinal = 0.0;
  for (auto p : vpairs) {
    if (p.ND) tp++;
    else fp++;
    int fn = total_poly - tp;
    double precision = (double)tp / (tp + fp);
    double recall = (double)tp / (tp + fn);
    double f1 = 2.0 * precision * recall / (precision + recall);
    fffinal = f1;
    if (f1 > bestf) {
      bestf = f1;
      bestconf = p.ST;
      DB(tp); DB(fp);
      DB(f1);
    }
    if ((tp + fp) % 1000 == 0) {
      fprintf(stderr, "STATUS tp=%d fp=%d fn=%d total_poly=%d precision=%.6lf\n", tp, fp, fn, total_poly, precision);
    }
  }
  DB(fffinal);
  DB(total_poly);
  DB(tp);
  DB(fp);
  DB(bestf);
  DB(bestconf);
  int ile = 0;
  for (auto p : vpairs) if (p.ST >= bestconf) ile++;
  DB(ile);
  fclose(f);
} 

const int HEAT_SIZE = 1000;
int heat_marked[HEAT_SIZE][HEAT_SIZE];
double heat_marked_dist[HEAT_SIZE][HEAT_SIZE];

inline double dist_from_segment(double x1, double y1, double x2, double y2, double x, double y) {
  x2 -= x1;
  y2 -= y1;
  x -= x1;
  y -= y1;
  double dot = x * x2 + y * y2;
  double d = hypot(x2, y2);
  double a = dot / d / d;
  if (a < 0 || a > 1) return 2.0;
  x -= a * x2;
  y -= a * y2;
  return hypot(x,y);
}

double dist_from_poly(const Polygon &poly, double x, double y) {
  double res = 2.0;
  for (auto p : poly) res = min(res, hypot(p.ST - x, p.ND - y));
  REP(i, SIZE(poly)-1) res = min(res, dist_from_segment(poly[i].ST, poly[i].ND, poly[i+1].ST, poly[i+1].ND, x, y));
  return res;
}

void gen_heatmaps(int size, double margin) {
  assert(size < HEAT_SIZE);
  CImg<unsigned char> image, heat;
  CImgDisplay disp_img, disp_heat;
  FOR(i,1,TRAIN_IMAGES) {
    DB(i);
    heat = CImg<unsigned char> (size, size, 1, 3, 0);
    heat.draw_rectangle(0,0,size,size,blue,1.0);
    if (VISUALIZE) {
      image = CImg<unsigned char> ((train_images_path+i2s(i)+".tif").c_str());
    }
    
    REP(j, SIZE(polygons[i])) {
      double t[4];
      calc_bounding_box(polygons[i][j], t);
      if (VISUALIZE) mydraw_poly(image, polygons[i][j], red, 0.5);
      int xl = max(0, (int)(t[0] * size));
      int xr = min(size-1, (int)(t[1] * size)+1);
      int yl = max(0, (int)(t[2] * size));
      int yr = min(size-1, (int)(t[3] * size)+1);
      FOR(x,xl,xr) FOR(y,yl,yr) {
        double xx = (double)x / size;
        double yy = (double)y / size;
        if (PointInPol(polygons[i][j], xx, yy)) {
          double d = dist_from_poly(polygons[i][j], xx, yy);
          if (heat_marked[x][y] == i) {
            DB("warning");
            DB(heat_marked_dist[x][y]);
            DB(d);
          }
//assert(heat_marked[x][y] != i || heat_marked[x][y] < 0.004 || d < 0.004);
          heat_marked[x][y] = i;
          heat_marked_dist[x][y] = d;
          heat(x,y,0,0) = 0;
          heat(x,y,0,1) = 0;
          heat(x,y,0,2) = 0;
          if (d >= margin) heat(x,y,0,0) = 255; 
          else if (d >= 2.0 / 3.0 * margin) {
            double r = (d - 2.0 / 3.0 * margin) / (margin / 3.0);
            heat(x,y,0,0) = (int)(r * 255); 
            heat(x,y,0,1) = (int)((1.0-r) * 255); 
          } else if (d >= margin / 3.0) {
            heat(x,y,0,1) = 255; 
          } else {
            double r = d / (margin / 3.0);
            heat(x,y,0,1) = (int)(r * 255); 
            heat(x,y,0,2) = (int)((1.0-r) * 255); 
          }
        }
      }
    }
    //DB(polygons[i]);

    if (VISUALIZE) {
      disp_img.set_title("%d", i);
      disp_img = image;
      disp_heat = heat;
      if (SIZE(polygons[i])) pause(disp_img);
    }
    heat.save((train_heat_path+i2s(i)+".jpg").c_str());
  }
}

inline double rate_desc(double *desc, const Polygon &poly) {
  RectDesc r(desc[0], desc[1], desc[2], desc[3], desc[4]);
  return r.IoU(poly);
}

RectDesc find_best_rect(const Polygon &poly, double &best) {
  //TODO here
  double t[4];
  calc_bounding_box(poly, t);
  double desc[5] = {(t[0] + t[1]) / 2.0, (t[2] + t[3]) / 2.0, t[1] - t[0], t[3] - t[2], 0.0};
  double sumx = 0, sumy = 0;
  REP(i, SIZE(poly)-1) sumx += poly[i].ST, sumy += poly[i].ND;
  sumx /= SIZE(poly)-1;
  sumy /= SIZE(poly)-1;
  desc[0] = sumx;
  desc[1] = sumy;

  vector<pair<int,double>> v;
  //TODO change to 5?
  REP(i,4+ROTATE) for (double dx = 0.1; dx >= 0.001; dx /= 3) {
    v.PB(MP(i,dx));
    v.PB(MP(i,-dx));
  }

  int last_change = 0;
  int i = 0;
  double pi2 = acos(0.0);

  best = rate_desc(desc, poly);
  //DB(best);
  int improving_moves = 0;
  int total_moves = 0;
  while (last_change < SIZE(v) * 3) {
    last_change++;
    if (i == SIZE(v)) {
      i = 0;
      random_shuffle(ALL(v));
    }
    total_moves++;
    desc[v[i].ST] += v[i].ND;
    int ok = desc[2] > 1e-3 && desc[3] > 1e-3 && fabs(desc[4]) <= pi2 / 4.0;
    double cand = ok ? rate_desc(desc, poly) : 0;
    if (cand > best) {
      best = cand;
      //DB(best);
      if (cand > best+1e-5) last_change = 0;
      improving_moves++;
    } else desc[v[i].ST] -= v[i].ND;
    i++;
  }
  DB(improving_moves);
  DB(total_moves);
  DB(best);
  assert(fabs(best - rate_desc(desc, poly)) < 1e-5);
  return RectDesc(desc[0], desc[1], desc[2], desc[3], desc[4]);
}

int main(int argc, char **argv) {
  int pieces = 9;
  int centers = 4;
  int angles = 15;
  int centers_slack = 0;
  int mark_train = 0;
  int scheme = 0;
  string nms_file, heat_file;
  double nms_threshold = 0.5, conf_threshold = -1e6;
  int heatmap = 0;
  double heat_margin;
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--scheme") {
      ++i;
      scheme = atoi(argv[i]);
    } else if (string(argv[i]) == "--rotate") {
      ROTATE = 1;
    } else if (string(argv[i]) == "--shift") {
      SHIFT = 1;
    } else if (string(argv[i]) == "--mark-heatmap") {
      ++i;
      heat_file = argv[i];
    } else if (string(argv[i]) == "--heatmap") {
      ++i;
      heatmap = atoi(argv[i]); //size
      ++i;
      heat_margin = atof(argv[i]);
    } else if (string(argv[i]) == "--nms") {
      ++i;
      nms_file = argv[i];
      ++i;
      nms_threshold = atof(argv[i]);
      ++i;
      conf_threshold = atof(argv[i]);
    } else if (string(argv[i]) == "--mark-train") {
      mark_train = 1;
    } else if (string(argv[i]) == "--vis") {
      VISUALIZE = 1;
    } else if (string(argv[i]) == "--pieces") {
      ++i;
      pieces = atoi(argv[i]);
    } else if (string(argv[i]) == "--centers-slack") {
      ++i;
      centers_slack = atoi(argv[i]);
    } else if (string(argv[i]) == "--centers") {
      ++i;
      centers = atoi(argv[i]);
    } else if (string(argv[i]) == "--angles") {
      ++i;
      angles = atoi(argv[i]);
    } else assert(0);
  }
  PIECES = pieces;
  assert(pieces < MAX_PIECES);
  DB(pieces);
  DB(centers);
  DB(centers_slack);
  DB(angles);
  DB(mark_train);
  DB(nms_file);
  DB(nms_threshold);
  DB(heatmap);
  DB(heat_margin);
  DB(heat_file);
  DB(scheme);

  generate_polygons(pieces, centers, angles, scheme);
  assert(SIZE(poly_descs) < MAX_RECTANGLES);

  if (mark_train) {
    mark_train_images(pieces);
    return 0;
  }

  load_csv();

  if (heat_file != "") {
    mark_heatmaps(heat_file);
  }

  if (heatmap) {
    gen_heatmaps(heatmap, heat_margin);
    return 0;
  }

  if (nms_file != "") {
    calc_nms(nms_file, nms_threshold, conf_threshold, pieces);
    return 0;
  }


  int i = 1;
  CImg<unsigned char> visu(500,400,1,3,0);
  CImgDisplay main_disp;
  int tp = 0, fp = 0;

  double start_time = Mytime::get_time();
  int tot_cnt = 0;
  double sum_inter = 0.0, sum_hypo = 0.0;
  CImg<unsigned char> image;
  vector<RectDesc> best_rect;
  while (i <= TRAIN_IMAGES) {
    DB(i);
 //DB(polygons[i]);
    if (VISUALIZE) {
      image = CImg<unsigned char> ((train_images_path+i2s(i)+".tif").c_str());
      assert(image.depth() == 1);
      assert(image.spectrum() == 3);
      DB(image.width());
      DB(image.height());
      DB(image.depth());
      DB(image.spectrum());
    }
    //DB(SIZE(polygons[i]));
    st_find.start();
    best_rect.clear();
    REP(j, SIZE(polygons[i])){
      double hypo;
      best_rect.PB(find_best_rect(polygons[i][j], hypo));
      if (VISUALIZE) {
        REP(k, SIZE(polygons[i][j])-1){
          pair<double,double> p1 = polygons[i][j][k];
          pair<double,double> p2 = polygons[i][j][k+1];
          image.draw_line((int)(p1.ST * IMAGE_WIDTH), (int)(p1.ND * IMAGE_HEIGHT), (int)(p2.ST * IMAGE_WIDTH), (int)(p2.ND * IMAGE_HEIGHT), red);
        }
      }

      double inter = 0.0;
      int best = 0;
      double area1 = polygon_area(polygons[i][j]);
      double sumx = 0.0, sumy = 0.0;
      REP(k, SIZE(polygons[i][j])-1){
        sumx += polygons[i][j][k].ST;
        sumy += polygons[i][j][k].ND;
      }
      sumx /= SIZE(polygons[i][j])-1;
      sumy /= SIZE(polygons[i][j])-1;

      if (VISUALIZE) mydraw_poly(image, best_rect[j].gen(), green, 0.5);

      double bdx, bdy;
      FOR(sx,-centers_slack,centers_slack) FOR(sy,-centers_slack,centers_slack) {
        int piece_x = (int)(sumx * pieces)+sx;
        int piece_y = (int)(sumy * pieces)+sy;
        if (piece_x < 0 || piece_x >= pieces || piece_y < 0 || piece_y >= pieces) continue;
        double dx = (double)piece_x / pieces;
        double dy = (double)piece_y / pieces;


        //DB(dx);
        //DB(dy);

        inter_calls = 0;
        Polygon poly = translate(polygons[i][j], -dx, -dy);
        REP(a, SIZE(gen_polys)) {
          double area2 = poly_descs[a].w * poly_descs[a].h;
          //double cand = 2.0 * intersection_area(poly, gen_polys[a]);
          st_inter.start();
          double cand = poly_descs[a].intersect(poly);
          st_inter.stop();
          cand /= (area1 + area2 - cand);
          if (cand > inter) {
            inter = cand;
            best = a;
            bdx = dx;
            bdy = dy;
          }
          if (cand >= 0.5) {
            /*
               if (mark_rect[piece_x][piece_y][a] == i) {
               DB(a);
               DB(gen_polys[a]);
               DB(polygons[i][j]);
               DB(polygons[i][mark_rect_which[piece_x][piece_y][a]-1]);
               mydraw_poly(image, polygons[i][j], IMAGE_WIDTH, IMAGE_HEIGHT, green, 0.5);
               mydraw_poly(image, polygons[i][mark_rect_which[piece_x][piece_y][a]-1], IMAGE_WIDTH, IMAGE_HEIGHT, yellow, 0.5);
               mydraw_poly(image, translate(gen_polys[best], dx, dy), IMAGE_WIDTH, IMAGE_HEIGHT, blue, 0.5);
               main_disp = image;
               while (!main_disp.is_closed()) main_disp.wait();
               }
             */
            assert(mark_rect[piece_x][piece_y][a] != i);
            mark_rect[piece_x][piece_y][a] = i;
            mark_rect_which[piece_x][piece_y][a] = j+1;
            mark_total[a]++;
            tot_cnt++;
          }
        }
      }
      if (inter >= 0.5){
        mark_best[best]++;
        sum_inter += inter;
        sum_hypo += hypo;
      }
      if (inter >= 0.5) tp++;
      else fp++;
      //if (inter >= 0.4 && hypo >= 0.5) tp++;
      //else fp++;
    //  DB(inter_calls);
      DB(inter);
      DB(hypo+1e-4 >= inter);
      //assert(hypo+1e-4 >= inter);
      if (VISUALIZE) {
        mydraw_poly(image, polygons[i][j], red, 0.5);
        mydraw_poly(image, translate(gen_polys[best], bdx, bdy), blue, 0.5);
      }
    }
    DB(sum_inter / tp);
    DB(sum_hypo / tp);
    /*
    if (i % 1000 == 0) {
      REP(a, SIZE(gen_polys)) fprintf(stderr, "markings[%d]=(%d,%d)\n", a, mark_total[a], mark_best[a]);
    }
    */
    st_find.stop();
    DB(st_find.used_time);
    DB(st_inter.used_time);
    DB(Mytime::get_time() - start_time);
    DB(tot_cnt);
    DB((double)tot_cnt / i);

    double precision = (double)tp / (tp + fp);
    double recall = (double)tp / (tp + fp);
    double f1 = 2.0 * precision * recall / (precision + recall);
    DB(tp); DB(fp);
    DB(f1);

    //assert(image.width() == 438 || image.width() == 439);
    //assert(image.height() == 406);
    if (SIZE(polygons[i]) > 0 && VISUALIZE) {
      main_disp = image;
      main_disp.set_title("%d", i);
      usleep(10);
      pause(main_disp);
    }
    FILE *f = fopen((TRAIN_DATA_DIR+"labels/"+i2s(i)).c_str(), "w");
    FILE *f2 = fopen((TRAIN_DATA_DIR+"labels/"+i2s(i)+".meta").c_str(), "w");
    REP(py, pieces) REP(px, pieces) {
      int cnt = 0;
      //DB(px); DB(py); DB(cnt);
      REP(a, SIZE(gen_polys)) if (mark_rect[px][py][a] == i) cnt++;
      fprintf(f, "%d", cnt);
      fprintf(f2, "%d", cnt);
      REP(a, SIZE(gen_polys)) if (mark_rect[px][py][a] == i) {
        fprintf(f, ",%d,%d", a, mark_rect_which[px][py][a]);
        int ww = mark_rect_which[px][py][a]-1;
        assert(ww >= 0 && ww < SIZE(polygons[i]));
        double desc[5];
        desc[0] = (best_rect[ww].cx - (1.0 * px / pieces + poly_descs[a].cx)) / poly_descs[a].w;
        desc[1] = (best_rect[ww].cy - (1.0 * py / pieces + poly_descs[a].cy)) / poly_descs[a].h;
        //desc[2] = (best_rect[ww].w - poly_descs[a].w) / poly_descs[a].w;
        //desc[3] = (best_rect[ww].h - poly_descs[a].h) / poly_descs[a].h;
        desc[2] = log(best_rect[ww].w / poly_descs[a].w);
        desc[3] = log(best_rect[ww].h / poly_descs[a].h);
        desc[4] = best_rect[ww].angle;
        int warn = 0;
        REP(j,4) if (fabs(desc[j]) > 1.0) warn = 1;
        if (warn) {
          REP(j,4) fprintf(stderr, "desc[%d]=%.6lf\n", j, desc[j]);
        }
        //REP(j,4) assert(desc[j] >= -2.0 && desc[j] <= 2.0);
        //TODO fill desc
        REP(j,5) fprintf(f2, ",%.6lf", desc[j]);
      }
      fprintf(f, "\n");
      fprintf(f2, "\n");
    }
    fclose(f);
    fclose(f2);
    ++i;
  }
  return 0;
}

