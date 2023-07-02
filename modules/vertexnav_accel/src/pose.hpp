#include <math.h>

#pragma once

// Defines the Pose class
static int pose_count;

struct Pose {
 protected:
  static int class_id;
 public:
  int index;
  int robot_id;
  double x, y, yaw;

  Pose(double xi, double yi, double yawi, int n_robot_id): x(xi), y(yi),  yaw(yawi), robot_id(n_robot_id) {
    index = ++class_id;
  }

  Pose mul(const Pose &oth) const {
    return oth.rmul(*this);
  }

  Pose rmul(const Pose &oth) const;

  static Pose get_odom(const Pose &p_new, const Pose &p_old);

  Pose operator* (const Pose &oth) const {
    return mul(oth);
  }

};
