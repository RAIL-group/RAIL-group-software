#pragma once

#include <CGAL/Arrangement_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangular_expansion_visibility_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arr_naive_point_location.h>
#include <iostream>
#include <vector>
#include <types.hpp>
#include <memory>

// Define the used kernel and arrangement
typedef CGAL::Exact_predicates_exact_constructions_kernel       Kernel;
typedef Kernel::Point_2                                         Point_2;
typedef Kernel::Segment_2                                       Segment_2;
typedef CGAL::Arr_segment_traits_2<Kernel>                      Traits_2;
typedef CGAL::Arrangement_2<Traits_2>                           Arrangement_2;
typedef Arrangement_2::Halfedge_const_handle                    Halfedge_const_handle;
typedef Arrangement_2::Face_handle                              Face_handle;

// Define the used visibility class
typedef CGAL::Triangular_expansion_visibility_2<Arrangement_2, CGAL::Tag_true>  TEV;

struct Point2 {
  double x, y;
  Point2(double xl, double yl) {
    x = xl;
    y = yl;
  }

  bool operator == (Point2 const &b) {
    return (x == b.x) && (y == b.y);
  }
};


struct Seg {
  double x1, y1, x2, y2, cx, cy;
  Seg(const double &xl1, const double &yl1,
      const double &xl2, const double &yl2) :
      x1(xl1), y1(yl1), x2(xl2), y2(yl2),
      cx((x1+x2)/2), cy((y1+y2)/2) {
  }

  void swap() {
    double tmp_x = x1;
    double tmp_y = y1;
    x1 = x2;
    y1 = y2;
    x2 = tmp_x;
    y2 = tmp_y;
  }

};

struct FastWorld {
  Arrangement_2 env;
  std::shared_ptr<TEV> tev;
  VEV vertices;

  FastWorld() : vertices() {};
  FastWorld(const VVD &f_seg, const VEV &f_vert);
  FastWorld(const std::vector<Seg> &f_seg,
            const std::vector<Eigen::Vector2d> &f_vert);
  VEV getVisPoly(double x, double y) const;
  VEV getVisPolyBounded(double x, double y, const VEV &poly_bound) const;
  // ~FastWorld() {
  //   delete tev;
  // }
};
