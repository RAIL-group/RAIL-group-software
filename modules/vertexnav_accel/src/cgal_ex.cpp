#include <cgal_ex.hpp>
#include <math.h>
#include <algorithm>
#include <limits>
#include <vector>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <sstream>

typedef Arrangement_2::Edge_const_iterator                              Edge_const_iterator;

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef Kernel::Point_2 Point_2;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_In;
 typedef Kernel_In::Point_2 Point_2_In;

VVD filter_by_poly_points_alt(VVD poly_points, VVD unfiltered_points) {
  int polyCorners = poly_points.size();
  int i, j = polyCorners - 1;

  std::vector<double> constant;
  std::vector<double> multiple;

  for (i=0; i < polyCorners; ++i) {
    if (poly_points[j][1] == poly_points[i][1]) {
      constant.push_back(poly_points[i][0]);
      multiple.push_back(0);
    } else {
      constant.push_back(
          poly_points[i][0]
          - (poly_points[i][1]*poly_points[j][0])/(poly_points[j][1]-poly_points[i][1])
          + (poly_points[i][1]*poly_points[i][0])/(poly_points[j][1]-poly_points[i][1]));
      multiple.push_back(
          (poly_points[j][0]-poly_points[i][0])/(poly_points[j][1]-poly_points[i][1]));
    }
    j = i;
  }

  VVD out_points;
  for (auto point : unfiltered_points) {
    j = polyCorners - 1;
    bool oddNodes = false;

    for (i=0; i < polyCorners; ++i) {
      if ((poly_points[i][1] < point[1] && poly_points[j][1] >= point[1]
           || poly_points[j][1]< point[1] && poly_points[i][1] >= point[1])) {
        oddNodes^=(point[1]*multiple[i]+constant[i] < point[0]); }
      j = i;
    }

    if (oddNodes) {
      out_points.push_back(point);
    }
  }

  return out_points;
}

VVD filter_by_poly_points_inf(VVD poly_points, VVD unfiltered_points,
                              double x, double y) {
  VVD poly_points_inf;
  for (auto point : poly_points) {
    poly_points_inf.push_back({
        point[0] + (point[0] - x)/2048,
        point[1] + (point[1] - y)/2048,
      });
  }
  return filter_by_poly_points_alt(poly_points_inf, unfiltered_points);
}

// void precalc_values() {

//   int i, j = polyCorners-1 ;

//   for(i=0; i<polyCorners; i++) {
//     if(polyY[j]==polyY[i]) {
//       constant[i]=polyX[i];
//       multiple[i]=0; }
//     else {
//       constant[i]=polyX[i]-(polyY[i]*polyX[j])/(polyY[j]-polyY[i])+(polyY[i]*polyX[i])/(polyY[j]-polyY[i]);
//       multiple[i]=(polyX[j]-polyX[i])/(polyY[j]-polyY[i]); }
//     j=i; }}

// bool pointInPolygon() {

//   int   i, j=polyCorners-1 ;
//   bool  oddNodes=NO      ;

//   for (i=0; i<polyCorners; i++) {
//     if ((polyY[i]< y && polyY[j]>=y
//          ||   polyY[j]< y && polyY[i]>=y)) {
//       oddNodes^=(y*multiple[i]+constant[i]<x); }
//     j=i; }

//   return oddNodes; }

FastWorld::FastWorld(const VVD &f_seg, const VEV &f_vert) : vertices(f_vert) {
  std::vector<Segment_2> segments;
  for (int i=0; i < f_seg.size(); ++i) {
    segments.push_back(Segment_2(
        Point_2(f_seg[i][0], f_seg[i][1]),
        Point_2(f_seg[i][2], f_seg[i][3])));
  }

  // insert geometry into the arrangement
  CGAL::insert(env, segments.begin(), segments.end());
  tev = std::make_shared<TEV>(env);
}

bool check_inside(const Point_2_In &pt,
                  const Point_2_In *pgn_begin,
                  const Point_2_In *pgn_end,
                  const Kernel_In &traits) {
  switch (CGAL::bounded_side_2(pgn_begin, pgn_end, pt, traits)) {
    case CGAL::ON_BOUNDED_SIDE:
      return true;
      break;
    case CGAL::ON_BOUNDARY:
      return true;
      break;
    case CGAL::ON_UNBOUNDED_SIDE:
      return false;
      break;
  }
}

bool compare(const Point2 &a, const Point2 &b) {
  return (fabs(a.x - b.x) < 0.00001) && (fabs(a.y - b.y) < 0.00001);
}

VEV filter_by_poly_points(const std::vector<Point_2_In> &filter_points,
                          const VEV &unfiltered_vertices) {
  VEV filtered_vertices;

  for (auto vert : unfiltered_vertices) {
    if (check_inside(Point_2_In(vert[0], vert[1]),
                     filter_points.data(),
                     filter_points.data()+filter_points.size(),
                     Kernel_In())) {
      filtered_vertices.push_back(vert);
    }
  }

  return filtered_vertices;
}

VEV FastWorld::getVisPoly(double qx, double qy) const {
  // find the face of the query point
  // (usually you may know that by other means)
  Point_2 q(qx, qy);
  Arrangement_2::Face_const_handle * face;
  CGAL::Arr_naive_point_location<Arrangement_2> pl(env);
  CGAL::Arr_point_location_result<Arrangement_2>::Type obj = pl.locate(q);
  // The query point locates in the interior of a face
  face = boost::get<Arrangement_2::Face_const_handle> (&obj);

  Arrangement_2 regular_output;
  // RSPV regular_visibility(env);
  tev->compute_visibility(q, *face, regular_output);

  std::vector<Seg> vis_seg;

  for (Edge_const_iterator eit = regular_output.edges_begin(); eit != regular_output.edges_end(); ++eit) {
    auto ps = eit->source()->point();
    auto pt = eit->target()->point();
    vis_seg.push_back(Seg(
        CGAL::to_double(ps.x()),
        CGAL::to_double(ps.y()),
        CGAL::to_double(pt.x()),
        CGAL::to_double(pt.y())));
  }

  std::sort(
      vis_seg.begin(),
      vis_seg.end(),
      [&](const Seg &sa, const Seg &sb) {
        double tha = atan2(sa.cy - qy, sa.cx - qx);
        double thb = atan2(sb.cy - qy, sb.cx - qx);
        return tha > thb;
      });

  auto near = [](const double a, const double b) {
                return fabs(a - b) < 0.001;
              };

  // Orient the segments
  for (int i=0; i < vis_seg.size()-1; ++i) {
    Seg *sa = &(vis_seg[i+0]);
    Seg *sb = &(vis_seg[i+1]);

    if (near(sa->x2, sb->x1) && near(sa->y2, sb->y1)) {
        // pass;
    } else if (near(sa->x2, sb->x2) && near(sa->y2, sb->y2)) {
        sb->swap();
    } else if (near(sa->x1, sb->x1) && near(sa->y1, sb->y1)) {
        sa->swap();
    } else if (near(sa->x1, sb->x2) && near(sa->y1, sb->y2)) {
        sa->swap();
        sb->swap();
    } else {
        std::cout << "Something's amiss\n";
    }
  }

  std::vector<Point_2_In> vis_points;
  for (auto seg : vis_seg) {
    vis_points.push_back(Point_2_In(seg.x1, seg.y1));
  }

  return filter_by_poly_points(vis_points, vertices);

  // VVD vis_points_alt;
  // for (auto seg : vis_seg) {
  //   vis_points_alt.push_back({seg.x1, seg.y1});
  // }

  // return filter_by_poly_points_inf(vis_points_alt, vertices, qx, qy);
}

VEV FastWorld::getVisPolyBounded(double qx, double qy,
                                 const VEV &poly_bound) const {
  VEV out_vertices = getVisPoly(qx, qy);

  // Now filter by the polygon
  std::vector<Point_2_In> filter_points;
  for (auto vert : poly_bound) {
    filter_points.push_back(Point_2_In(vert[0], vert[1]));
  }

  return filter_by_poly_points(filter_points, out_vertices);
}
