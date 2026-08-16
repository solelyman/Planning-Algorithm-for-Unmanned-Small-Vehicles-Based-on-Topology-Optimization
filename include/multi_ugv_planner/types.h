#ifndef MULTI_UGV_TYPES_H
#define MULTI_UGV_TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace MultiUGV
{

//2D Pose with heading
struct Pose2D
{
    double x = 0.0, y = 0.0, theta = 0.0;
    Pose2D() = default;
    Pose2D(double x, double y, double theta) : x(x), y(y), theta(theta) {}

    Eigen::Vector2d pos() const { return Eigen::Vector2d(x, y); }
    void setPos(const Eigen::Vector2d &p) { x = p.x(); y = p.y(); }
    static double normalizeAngle(double a)
    {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }
};

// Trajectory point
struct TrajectoryPoint
{
    double x = 0.0, y = 0.0, psi = 0.0, v = 0.0, omega = 0.0;
    double s = 0.0;  // progress along reference path
};

struct ControlPoint
{
    double a = 0.0, omega = 0.0;
};

// Trajectory
struct Trajectory
{
    std::vector<TrajectoryPoint> points;
    std::vector<ControlPoint> controls;
    double dt = 0.2;
    int n = 0;

    void clear() { points.clear(); controls.clear(); n = 0; }
    int size() const { return (int)points.size(); }
    void add(double x, double y, double psi, double v, double omega)
    {
        points.push_back({x, y, psi, v, omega, 0.0});
        n++;
    }
};

// Static obstacle
struct Obstacle
{
    double x = 0.0, y = 0.0, radius = 1.0;
};

// UGV state
struct UGVState
{
    double x = 0.0, y = 0.0, psi = 0.0;
    double v = 0.0, omega = 0.0;
    bool received = false;

    Eigen::Vector2d pos() const { return Eigen::Vector2d(x, y); }
};

} // namespace MultiUGV

#endif
