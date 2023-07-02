#include <pose.hpp>
#include <assert.h>

int Pose::class_id = 0;

Pose Pose::rmul(const Pose &oth) const {
  double sin_yaw = sin(yaw);
  double cos_yaw = cos(yaw);
  double xn = x + cos_yaw * oth.x - sin_yaw * oth.y;
  double yn = y + sin_yaw * oth.x + cos_yaw * oth.y;
  double yawn = fmod(yaw + oth.yaw, 2 * M_PI);
  return Pose(xn, yn, yawn, robot_id);
}

Pose Pose::get_odom(const Pose &p_new, const Pose &p_old) {
  assert(p_new.robot_id == p_old.robot_id);
  double dyaw = p_new.yaw - p_old.yaw;
  double dx = p_new.x - p_old.x;
  double dy = p_new.y - p_old.y;
  double sin_yaw = sin(p_old.yaw);
  double cos_yaw = cos(p_old.yaw);
  double x_odom = cos_yaw * dx + sin_yaw * dy;
  double y_odom = -sin_yaw * dx + cos_yaw * dy;
  return Pose(x_odom, y_odom, dyaw, p_old.robot_id);
}
