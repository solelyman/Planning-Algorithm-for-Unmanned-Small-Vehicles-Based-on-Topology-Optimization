#include <multi_usv_planner/controller.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace MultiUSV
{

std::pair<double, double> USVController::compute(const Trajectory &traj,
                                                  const USVState &state, double dt)
{
    if (traj.size() == 0) return {0.0, 0.0};

    dt = std::clamp(dt, 0.001, 0.1);
    int n = traj.size();

    int min_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < n; i++)
    {
        double dx = state.x - traj.points[i].x;
        double dy = state.y - traj.points[i].y;
        double d = dx * dx + dy * dy;
        if (d < min_dist) { min_dist = d; min_idx = i; }
    }

    // 弧长累计 lookahead
    double traj_len = 0.0;
    for (int i = 0; i < n - 1; i++)
    {
        double dx = traj.points[i + 1].x - traj.points[i].x;
        double dy = traj.points[i + 1].y - traj.points[i].y;
        traj_len += std::sqrt(dx * dx + dy * dy);
    }
    double effective_lookahead = std::min(p_.lookahead_h, 0.7 * std::max(traj_len, 1.0));

    double accum = 0.0;
    int ref_idx = min_idx;
    for (int i = min_idx; i < n - 1; i++)
    {
        double dx = traj.points[i + 1].x - traj.points[i].x;
        double dy = traj.points[i + 1].y - traj.points[i].y;
        accum += std::sqrt(dx * dx + dy * dy);
        ref_idx = i + 1;
        if (accum >= effective_lookahead) break;
    }
    const auto &ref = traj.points[ref_idx];

    double u_ff = ref.v;
    double r_ff = ref.omega;
    double psi_ref = ref.psi;

    u_ff_ema_ = ff_alpha_ * u_ff + (1.0 - ff_alpha_) * u_ff_ema_;
    r_ff_ema_ = ff_alpha_ * r_ff + (1.0 - ff_alpha_) * r_ff_ema_;

    double psi_traj = traj.points[min_idx].psi;
    double dx_ct = state.x - traj.points[min_idx].x;
    double dy_ct = state.y - traj.points[min_idx].y;
    double cy = std::cos(psi_traj), sy = std::sin(psi_traj);
    double ct_err = -sy * dx_ct + cy * dy_ct;

    double psi_err = Pose2D::normalizeAngle(state.psi - psi_ref);

    double u_des = u_ff_ema_ - p_.k_x * ct_err;
    u_des = std::clamp(u_des, 0.0, p_.max_u);

    double alpha_r = r_ff_ema_ - p_.k_psi * psi_err - p_.k_y * ct_err;
    yaw_filter_.update(alpha_r, dt);
    double r_des = yaw_filter_.value();
    double r_dot_des = yaw_filter_.derivative();

    double r_err = state.omega - r_des;
    double tau_r = -p_.d_r * r_des + p_.Izz * r_dot_des - p_.k_r * r_err;
    double r_cmd = std::clamp(tau_r, -p_.max_r, p_.max_r);

    return {u_des, r_cmd};
}

} // namespace MultiUSV
