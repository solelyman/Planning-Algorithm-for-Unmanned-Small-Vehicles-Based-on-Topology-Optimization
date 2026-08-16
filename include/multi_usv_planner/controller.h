#ifndef MULTI_USV_CONTROLLER_H
#define MULTI_USV_CONTROLLER_H

#include <multi_usv_planner/types.h>
#include <utility>

namespace MultiUSV
{

class CommandFilter
{
public:
    CommandFilter() : omega_n_(2.2), zeta_n_(0.85), limit_(1.0), x1_(0.0), x2_(0.0) {}
    CommandFilter(double wn, double zeta, double limit)
        : omega_n_(wn), zeta_n_(zeta), limit_(limit), x1_(0.0), x2_(0.0) {}

    void reset(double val)
    {
        x1_ = std::clamp(val, -limit_, limit_);
        x2_ = 0.0;
    }

    void update(double input, double dt)
    {
        double bounded = std::clamp(input, -limit_, limit_);
        double err = x1_ - bounded;
        double x2_dot = -2.0 * zeta_n_ * omega_n_ * x2_ - omega_n_ * omega_n_ * err;
        x2_ += x2_dot * dt;
        x1_ += x2_ * dt;
    }

    double value() const { return x1_; }
    double derivative() const { return x2_; }

private:
    double omega_n_, zeta_n_, limit_, x1_, x2_;
};

class USVController
{
public:
    struct Params
    {
        // 模型
        double m11 = 28.0, m22 = 28.0, Izz = 10.0;
        double d_u = -16.45, d_r = -6.0;

        // 增益
        double lookahead_h = 8.0;
        double k_x = 2.0, k_u = 5.0, k_psi = 1.0, k_y = 0.5, k_r = 5.0;

        // 限幅
        double max_u = 2.0, max_r = 1.0;

        // 滤波器
        double filter_omega_n = 2.2, filter_zeta = 0.85;

        // 停止
        double stop_radius = 2.0;
    };

    USVController() = default;

    void setParams(const Params &p)
    {
        p_ = p;
        yaw_filter_ = CommandFilter(p.filter_omega_n, p.filter_zeta, p.max_r);
    }

    const Params &params() const { return p_; }

    // 重置滤波器（首次调用时）
    void reset(double initial_r)
    {
        yaw_filter_.reset(initial_r);
    }

    std::pair<double, double> compute(const Trajectory &traj,
                                      const USVState &state, double dt);

private:
    Params p_;
    CommandFilter yaw_filter_;
    double u_ff_ema_ = 0.0;
    double r_ff_ema_ = 0.0;
    double ff_alpha_ = 0.45;
};

} // namespace MultiUSV

#endif
