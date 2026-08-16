#include <multi_usv_planner/mpc_solver.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace MultiUSV
{

namespace
{
double headingAlignmentScore(double err_abs)
{
    return std::clamp(0.5 * (std::cos(err_abs) + 1.0), 0.0, 1.0);
}

double computePreviewDistance(const MPCSolver::Params &params, int stage_index)
{
    double base = std::max(1.2, 0.35 * params.lookahead_distance);
    double grow = 0.18 * stage_index * params.desired_speed * params.dt;
    return std::clamp(base + grow, 1.2, std::max(3.5, params.lookahead_distance));
}

double stageOmegaLimit(const MPCSolver::Params &params, int stage_index)
{
    if (stage_index <= 0) return 0.55 * params.max_omega;
    if (stage_index == 1) return 0.70 * params.max_omega;
    if (stage_index == 2) return 0.85 * params.max_omega;
    return params.max_omega;
}

double seedOmegaTrustWidth(const MPCSolver::Params &params, int stage_index)
{
    if (stage_index <= 0) return 0.25 * params.max_omega;
    if (stage_index == 1) return 0.35 * params.max_omega;
    if (stage_index == 2) return 0.50 * params.max_omega;
    return params.max_omega;
}

struct StageReference
{
    double s = 0.0;
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    Eigen::Vector2d normal = Eigen::Vector2d::UnitY();
    double heading = 0.0;
    double speed_ref = 0.0;
};

StageReference buildStageReference(const ReferencePath &path,
                                   const MPCSolver::Params &params,
                                   const std::vector<double> &guidance_progress,
                                   double state_progress,
                                   double prev_progress,
                                   int stage_index)
{
    StageReference ref;
    if (path.empty())
        return ref;

    double path_len = path.length();
    double scheduled = (stage_index < static_cast<int>(guidance_progress.size()))
        ? guidance_progress[stage_index]
        : (guidance_progress.empty() ? 0.0 : guidance_progress.back());
    ref.s = std::clamp(std::max({scheduled, prev_progress, state_progress}), 0.0, path_len);
    ref.pos = path.sample(ref.s);

    double preview = computePreviewDistance(params, stage_index);
    double s_next = std::min(ref.s + preview, path_len);
    ref.heading = path.headingBetween(ref.s, s_next);
    ref.tangent = Eigen::Vector2d(std::cos(ref.heading), std::sin(ref.heading));
    ref.normal = Eigen::Vector2d(-ref.tangent.y(), ref.tangent.x());

    double remain = std::max(path_len - ref.s, 0.0);
    double remain_scale = std::clamp(remain / std::max(params.lookahead_distance, 1.0), 0.2, 1.0);
    ref.speed_ref = params.desired_speed * remain_scale;
    return ref;
}
} // namespace

double ReferencePath::length() const
{
    if (points.size() < 2) return 0.0;
    double total = 0.0;
    for (size_t i = 1; i < points.size(); i++)
        total += (points[i] - points[i - 1]).norm();
    return total;
}

int ReferencePath::findClosest(const Eigen::Vector2d &pos, double &s_out) const
{
    if (points.empty()) { s_out = 0.0; return -1; }
    if (points.size() == 1) { s_out = 0.0; return 0; }

    int best = 0;
    double best_dist = std::numeric_limits<double>::max();
    double accum = 0.0;
    s_out = 0.0;

    for (size_t i = 0; i + 1 < points.size(); i++)
    {
        Eigen::Vector2d a = points[i];
        Eigen::Vector2d b = points[i + 1];
        Eigen::Vector2d ab = b - a;
        double len2 = ab.squaredNorm();
        double t = 0.0;
        if (len2 > 1e-9)
            t = std::clamp((pos - a).dot(ab) / len2, 0.0, 1.0);
        Eigen::Vector2d proj = a + t * ab;
        double d2 = (pos - proj).squaredNorm();
        if (d2 < best_dist)
        {
            best_dist = d2;
            best = (int)i;
            s_out = accum + t * std::sqrt(len2);
        }
        accum += std::sqrt(len2);
    }
    return best;
}

Eigen::Vector2d ReferencePath::sample(double s) const
{
    if (points.empty()) return Eigen::Vector2d::Zero();
    if (points.size() == 1) return points.front();

    double total = length();
    s = std::clamp(s, 0.0, total);
    double accum = 0.0;

    for (size_t i = 0; i + 1 < points.size(); i++)
    {
        Eigen::Vector2d a = points[i];
        Eigen::Vector2d b = points[i + 1];
        double seg = (b - a).norm();
        if (seg < 1e-9) continue;
        if (s <= accum + seg)
        {
            double t = (s - accum) / seg;
            return a + t * (b - a);
        }
        accum += seg;
    }
    return points.back();
}

double ReferencePath::heading(double s) const
{
    if (points.size() < 2) return 0.0;

    double total = length();
    s = std::clamp(s, 0.0, total);
    double accum = 0.0;

    for (size_t i = 0; i + 1 < points.size(); i++)
    {
        Eigen::Vector2d d = points[i + 1] - points[i];
        double seg = d.norm();
        if (seg < 1e-9) continue;
        if (s <= accum + seg)
            return std::atan2(d.y(), d.x());
        accum += seg;
    }

    Eigen::Vector2d d = points.back() - points[points.size() - 2];
    return std::atan2(d.y(), d.x());
}

double ReferencePath::headingBetween(double s0, double s1) const
{
    Eigen::Vector2d p0 = sample(s0);
    Eigen::Vector2d p1 = sample(s1);
    Eigen::Vector2d d = p1 - p0;
    if (d.norm() < 1e-6)
        return heading(s0);
    return std::atan2(d.y(), d.x());
}

ReferencePathSample ReferencePath::sampleState(double s, double remain_hint) const
{
    ReferencePathSample out;
    out.pos = sample(s);

    double total = length();
    double ds_heading = std::clamp(total * 0.06, 0.6, 2.0);
    double s_next = std::min(s + ds_heading, total);
    out.psi = headingBetween(s, s_next);

    double remain = (remain_hint >= 0.0) ? remain_hint : std::max(total - s, 0.0);
    double scale = std::clamp(remain / 6.0, 0.25, 1.0);
    out.v = nominal_speed * scale;
    return out;
}

MPCSolver::MPCSolver()
{
    tmp_states_.resize(params_.N + 1);
    tmp_controls_.resize(params_.N);
    for (auto &c : tmp_controls_) c.setZero();
    prev_traj_.dt = params_.dt;
    prev_traj_.n = params_.N;
}

void MPCSolver::clearWarmstart()
{
    prev_traj_.points.clear();
    prev_progress_.clear();
    prev_controls_.clear();
    seed_controls_.clear();
}

void MPCSolver::setCurrentHeading(double psi)
{
    current_heading_ = psi;
}

void MPCSolver::setReferencePath(const ReferencePath &path)
{
    ref_path_ = path;
    guidance_points_.clear();
    guidance_progress_.clear();
    guidance_points_.resize(params_.N + 1);
    guidance_progress_.resize(params_.N + 1, 0.0);
    int path_len = path.size();
    if (path_len < 2)
    {
        for (int k = 0; k <= params_.N; k++)
            guidance_points_[k] = path.empty() ? Eigen::Vector2d::Zero() : path.points.front();
        return;
    }

    // 只保留与当前艇首方向一致的前向路径，避免近端反向锚点导致原地打转
    ReferencePath filtered;
    filtered.points.reserve(path.points.size());
    Eigen::Vector2d origin = path.points.front();
    Eigen::Vector2d heading_vec(std::cos(current_heading_), std::sin(current_heading_));
    filtered.points.push_back(origin);
    for (size_t i = 1; i < path.points.size(); ++i)
    {
        Eigen::Vector2d delta = path.points[i] - origin;
        if (delta.dot(heading_vec) > -0.2)
            filtered.points.push_back(path.points[i]);
    }
    if (filtered.points.size() < 2)
        filtered = path;

    filtered.total_length = 0.0;
    for (size_t i = 1; i < filtered.points.size(); ++i)
        filtered.total_length += (filtered.points[i] - filtered.points[i - 1]).norm();

    double segment = params_.dt * params_.desired_speed;
    double total_len = filtered.length();
    for (int k = 0; k <= params_.N; k++)
    {
        double s = std::min(segment * k, total_len);
        ReferencePathSample smp = filtered.sampleState(s, total_len - s);
        guidance_points_[k] = smp.pos;
        guidance_progress_[k] = s;
    }
}

void MPCSolver::integrateStep(const SolverState &s, double a, double omega,
                              SolverState &s_next) const
{
    double dt = params_.dt;
    double x = s[0], y = s[1], psi = s[2], v = s[3], progress = s[4];

    // RK3 for differential drive
    auto f = [](double /*x*/, double /*y*/, double psi, double v,
                double a, double omega) -> SolverState
    {
        SolverState dot;
        dot[0] = v * std::cos(psi);
        dot[1] = v * std::sin(psi);
        dot[2] = omega;
        dot[3] = a;
        dot[4] = std::max(v, 0.0);
        return dot;
    };

    // k1
    auto k1 = f(x, y, psi, v, a, omega);
    // k2
    double x2 = x + 0.5 * dt * k1[0];
    double y2 = y + 0.5 * dt * k1[1];
    double psi2 = Pose2D::normalizeAngle(psi + 0.5 * dt * k1[2]);
    double v2 = std::clamp(v + 0.5 * dt * k1[3], 0.0, params_.max_vel);
    auto k2 = f(x2, y2, psi2, v2, a, omega);
    // k3
    double x3 = x + dt * (2.0 * k2[0] - k1[0]);
    double y3 = y + dt * (2.0 * k2[1] - k1[1]);
    double psi3 = Pose2D::normalizeAngle(psi + dt * (2.0 * k2[2] - k1[2]));
    double v3 = std::clamp(v + dt * (2.0 * k2[3] - k1[3]), 0.0, params_.max_vel);
    auto k3 = f(x3, y3, psi3, v3, a, omega);

    s_next[0] = x + dt / 6.0 * (k1[0] + 4.0 * k2[0] + k3[0]);
    s_next[1] = y + dt / 6.0 * (k1[1] + 4.0 * k2[1] + k3[1]);
    s_next[2] = Pose2D::normalizeAngle(psi + dt / 6.0 * (k1[2] + 4.0 * k2[2] + k3[2]));
    s_next[3] = std::clamp(v + dt / 6.0 * (k1[3] + 4.0 * k2[3] + k3[3]),
                           0.0, params_.max_vel);
    s_next[4] = progress + dt / 6.0 * (k1[4] + 4.0 * k2[4] + k3[4]);
}

double MPCSolver::projectProgress(double x, double y) const
{
    if (ref_path_.empty())
        return 0.0;
    double s_proj = 0.0;
    ref_path_.findClosest(Eigen::Vector2d(x, y), s_proj);
    return std::clamp(s_proj, 0.0, ref_path_.length());
}

static void rollout(const MPCSolver::Params &params,
                    const SolverState &x0,
                    const std::vector<Eigen::Vector2d> &controls,
                    std::vector<SolverState> &states,
                    const MPCSolver *solver)
{
    states[0] = x0;
    states[0][4] = solver->projectProgress(states[0][0], states[0][1]);
    for (int k = 0; k < params.N && k < (int)controls.size(); k++)
    {
        solver->integrateStep(states[k], controls[k][0], controls[k][1], states[k + 1]);
        states[k + 1][4] = solver->projectProgress(states[k + 1][0], states[k + 1][1]);
    }
}

double MPCSolver::totalCost(const std::vector<SolverState> &states,
                            const std::vector<Eigen::Vector2d> &controls,
                            const StageObstacleSet &obstacles_by_stage,
                            double goal_x, double goal_y) const
{
    double cost = 0.0;
    double s_prev = std::clamp(states[0][4], 0.0, ref_path_.length());
    double prev_abs_heading_err = std::numeric_limits<double>::infinity();
    static const std::vector<Obstacle> empty_stage_obstacles;

    for (int k = 0; k < params_.N; k++)
    {
        double x = states[k][0], y = states[k][1];
        const auto &stage_obstacles =
            (k < static_cast<int>(obstacles_by_stage.size())) ? obstacles_by_stage[k] : empty_stage_obstacles;

        // 全局目标：从早期阶段逐步增强吸引，避免避障后忘记终点
        {
            int goal_ramp_start = 3;
            if (k >= goal_ramp_start)
            {
                double ratio = static_cast<double>(k - goal_ramp_start + 1) /
                               static_cast<double>(params_.N - goal_ramp_start);
                ratio = std::clamp(ratio, 0.0, 1.0);
                double dx = x - goal_x;
                double dy = y - goal_y;
                cost += ratio * params_.weight_goal * (dx * dx + dy * dy);
            }
        }

        // 路径跟随：更接近 contour/lag，而不是纯追前瞻点
        if (!ref_path_.empty())
        {
            double s_proj = std::clamp(states[k][4], 0.0, ref_path_.length());
            double s_geom = 0.0;
            ref_path_.findClosest(Eigen::Vector2d(x, y), s_geom);
            s_geom = std::clamp(s_geom, 0.0, ref_path_.length());
            double s_eval = std::max(s_geom, std::min(s_proj, s_geom + 1.5));
            double min_forward = s_prev + params_.min_progress_step;
            double backtrack = std::max(0.0, min_forward - s_geom);
            cost += params_.weight_backtrack * backtrack * backtrack;
            double progress_consistency = s_proj - s_geom;
            cost += params_.weight_progress_consistency * progress_consistency * progress_consistency;

            StageReference ref = buildStageReference(ref_path_, params_, guidance_progress_, s_eval, s_prev, k);
            double progress_err = s_geom - ref.s;
            cost += params_.weight_progress * progress_err * progress_err;
            cost -= params_.weight_forward_progress * s_geom;

            Eigen::Vector2d err(x - ref.pos.x(), y - ref.pos.y());
            double lag_err = err.dot(ref.tangent);
            double contour_err = err.dot(ref.normal);
            double lag_weight = (lag_err < 0.0) ? 1.2 : 0.20;
            cost += params_.weight_path * (lag_weight * lag_err * lag_err + 1.2 * contour_err * contour_err);

            double psi_err = Pose2D::normalizeAngle(states[k][2] - ref.heading);
            double abs_psi_err = std::abs(psi_err);
            double heading_phase = (k < 6) ? (2.0 - 0.16 * k) : 1.0;
            cost += heading_phase * 0.9 * params_.weight_path * psi_err * psi_err;

            if (std::isfinite(prev_abs_heading_err))
            {
                double heading_regress = std::max(0.0, abs_psi_err - prev_abs_heading_err + 0.03);
                cost += params_.weight_heading_progress * heading_regress * heading_regress;
            }
            prev_abs_heading_err = abs_psi_err;

            double align = headingAlignmentScore(abs_psi_err);
            double speed_ref = ref.speed_ref *
                (params_.min_heading_speed_scale + (1.0 - params_.min_heading_speed_scale) * align);
            double speed_err = states[k][3] - std::min(speed_ref, params_.desired_speed);
            cost += 1.0 * params_.weight_speed * speed_err * speed_err;

            if (k < 4)
            {
                double forward_floor = 0.45 * params_.desired_speed;
                if (abs_psi_err > 1.20 && k < 2)
                    forward_floor = 0.22 * params_.desired_speed;
                double missing_forward = std::max(0.0, forward_floor - states[k][3]);
                cost += 35.0 * params_.weight_speed * missing_forward * missing_forward;

                double standstill_turn = (1.0 - align) * std::max(0.0, 0.38 * params_.desired_speed - states[k][3]);
                cost += 28.0 * params_.weight_speed * standstill_turn * standstill_turn;
            }

            if (k + 1 <= params_.N)
            {
                double s_follow = std::clamp(states[k + 1][4], 0.0, ref_path_.length());
                double s_follow_geom = 0.0;
                ref_path_.findClosest(Eigen::Vector2d(states[k + 1][0], states[k + 1][1]), s_follow_geom);
                s_follow_geom = std::clamp(s_follow_geom, 0.0, ref_path_.length());
                double s_follow_eval = std::max(s_follow_geom, std::min(s_follow, s_follow_geom + 1.5));
                StageReference next_ref =
                    buildStageReference(ref_path_, params_, guidance_progress_, s_follow_eval, ref.s, k + 1);
                double desired_omega =
                    Pose2D::normalizeAngle(next_ref.heading - ref.heading) / std::max(params_.dt, 1e-3);
                desired_omega = std::clamp(desired_omega, -stageOmegaLimit(params_, k), stageOmegaLimit(params_, k));
                double omega_err = controls[k][1] - desired_omega;
                cost += params_.weight_heading_rate * omega_err * omega_err;
            }

            s_prev = s_geom;
        }

        // 拓扑走廊约束（T-MPC guidance track）：偏离锚点超过 margin 则强惩罚
        if (!guidance_points_.empty() && k < (int)guidance_points_.size())
        {
            double gdx = x - guidance_points_[k].x();
            double gdy = y - guidance_points_[k].y();
            double gdist = std::sqrt(gdx * gdx + gdy * gdy);
            // 二次增长 + 线性区，确保 corridor 内部代价很小，外部代价很高
            if (gdist > params_.guidance_margin)
            {
                double excess = gdist - params_.guidance_margin;
                cost += params_.weight_guidance * excess * excess;
            }
        }

        // 障碍物斥力：每个 stage 只看对应时刻的动态障碍物
        for (const auto &obs : stage_obstacles)
        {
            double odx = x - obs.x;
            double ody = y - obs.y;
            double dist = std::sqrt(odx * odx + ody * ody);
            double hard_margin = obs.radius + params_.obstacle_clearance;
            double barrier = (dist < hard_margin) ? (hard_margin - dist) * (hard_margin - dist) : 0.0;
            cost += params_.weight_obstacle * barrier;
        }

        // 平滑性
        if (k < params_.N)
        {
            cost += params_.weight_smooth * (controls[k][0] * controls[k][0] +
                                             controls[k][1] * controls[k][1]);
        }

        if (k < 3)
        {
            cost += params_.weight_initial_omega * controls[k][1] * controls[k][1];
            cost += params_.weight_turning_speed * states[k][3] * states[k][3] *
                    controls[k][1] * controls[k][1];
        }
        if (k > 0 && k < 3)
        {
            double domega = controls[k][1] - controls[k - 1][1];
            cost += params_.weight_initial_domega * domega * domega;
        }
        if (k < (int)seed_controls_.size() && k < 4)
        {
            Eigen::Vector2d du = controls[k] - seed_controls_[k];
            cost += params_.weight_seed_alignment * du.squaredNorm();
        }

        // 期望速度
        double v = states[k][3];
        cost += 0.2 * params_.weight_speed * (v - params_.desired_speed) * (v - params_.desired_speed);
    }

    // 终端代价
    double tdx = states[params_.N][0] - goal_x;
    double tdy = states[params_.N][1] - goal_y;
    cost += params_.weight_goal * (tdx * tdx + tdy * tdy);
    if (!ref_path_.empty())
    {
        double s_term = std::clamp(states[params_.N][4], 0.0, ref_path_.length());
        double s_term_next = std::min(s_term + std::max(params_.desired_speed * params_.dt, 0.8), ref_path_.length());
        double path_heading = ref_path_.headingBetween(s_term, s_term_next);
        Eigen::Vector2d p_ref = ref_path_.sample(s_term);
        Eigen::Vector2d tangent(std::cos(path_heading), std::sin(path_heading));
        Eigen::Vector2d normal(-tangent.y(), tangent.x());
        Eigen::Vector2d terr(states[params_.N][0] - p_ref.x(), states[params_.N][1] - p_ref.y());
        double tlag = terr.dot(tangent);
        double tcontour = terr.dot(normal);
        double angle_err = Pose2D::normalizeAngle(states[params_.N][2] - path_heading);
        cost += params_.weight_terminal_angle * angle_err * angle_err;
        cost += params_.weight_terminal_contouring *
                params_.weight_path * (0.4 * tlag * tlag + 1.2 * tcontour * tcontour);
    }

    return cost;
}

void MPCSolver::computeGradient(const std::vector<SolverState> &states,
                                const std::vector<Eigen::Vector2d> &controls,
                                const StageObstacleSet &obstacles_by_stage,
                                double goal_x, double goal_y,
                                std::vector<Eigen::Vector2d> &grad) const
{
    double eps = params_.grad_eps;
    double base = totalCost(states, controls, obstacles_by_stage, goal_x, goal_y);

    for (int k = 0; k < params_.N; k++)
    {
        for (int d = 0; d < 2; d++)
        {
            auto perturbed = controls;
            perturbed[k][d] += eps;
            std::vector<SolverState> perturbed_states(params_.N + 1);
            rollout(params_, states[0], perturbed, perturbed_states, this);
            double cost_plus = totalCost(perturbed_states, perturbed, obstacles_by_stage, goal_x, goal_y);
            grad[k][d] = (cost_plus - base) / eps;
        }
    }
}

Trajectory MPCSolver::solve(const USVState &state,
                            const StageObstacleSet &obstacles_by_stage,
                            double goal_x, double goal_y)
{
    tmp_states_.resize(params_.N + 1);
    tmp_controls_.resize(params_.N);

    SolverState x0 = SolverState::Zero();
    // 初始化：优先使用拓扑走廊锚点 + 参考路径
    std::vector<Eigen::Vector2d> controls(params_.N, Eigen::Vector2d::Zero());
    std::vector<SolverState> states(params_.N + 1, SolverState::Zero());

    double ref_length = ref_path_.length();
    double s_start = 0.0;
    bool has_ref = !ref_path_.empty() && ref_length > 1e-6;
    bool has_guidance = !guidance_points_.empty() && (int)guidance_points_.size() > params_.N;
    double init_err_abs = 0.0;
    if (has_ref)
        ref_path_.findClosest(Eigen::Vector2d(state.x, state.y), s_start);
    auto heading_preview = [&](double s_ref)
    {
        double preview_ds = std::max(1.5, 0.4 * params_.lookahead_distance);
        double s_far = std::min(s_ref + preview_ds, ref_length);
        return ref_path_.headingBetween(s_ref, s_far);
    };
    auto heading_alignment = [](double err_abs)
    {
        return std::clamp(0.5 * (std::cos(err_abs) + 1.0), 0.0, 1.0);
    };

    double v0 = std::clamp(state.v, 0.0, params_.max_vel);
    if (has_ref)
    {
        double init_heading = heading_preview(s_start);
        init_err_abs = std::abs(Pose2D::normalizeAngle(init_heading - state.psi));
        double align = heading_alignment(init_err_abs);
        double target_seed_speed = params_.desired_speed * (0.18 + 0.55 * align);
        if (init_err_abs > 1.35)
        {
            v0 = std::clamp(std::min(state.v, 0.22 * params_.desired_speed), 0.0, params_.max_vel);
        }
        else if (init_err_abs > 0.80)
        {
            double cautious_speed = std::min(
                std::max(0.45 * state.v, 0.20 * params_.desired_speed),
                0.65 * params_.desired_speed);
            v0 = std::clamp(cautious_speed, 0.0, params_.max_vel);
        }
        else
        {
            v0 = std::clamp(std::max(state.v, target_seed_speed), 0.0, params_.max_vel);
        }
    }
    else
    {
        v0 = std::clamp(std::max(state.v, params_.desired_speed * 0.35), 0.0, params_.max_vel);
    }
    x0 << state.x, state.y, state.psi, v0, s_start;
    states[0] = x0;

    bool has_prev = prev_traj_.size() == params_.N + 1;
    bool has_prev_progress = prev_progress_.size() == static_cast<size_t>(params_.N + 1);
    bool has_prev_controls = prev_controls_.size() == static_cast<size_t>(params_.N);
    bool severe_heading_mismatch = has_ref && init_err_abs > 1.20;
    bool guidance_continuous = has_guidance;
    if (has_ref && has_prev_progress && !prev_progress_.empty())
    {
        double prev_s0 = std::clamp(prev_progress_.front(), 0.0, ref_length);
        double continuity_gap = std::abs(prev_s0 - s_start);
        guidance_continuous = continuity_gap < std::max(1.8, 2.5 * params_.desired_speed * params_.dt);
    }
    bool reuse_prev = has_prev && has_guidance && guidance_continuous && !severe_heading_mismatch;
    bool use_guidance_seed = has_ref && !reuse_prev;
    bool use_braking_seed = !reuse_prev && !has_ref;
    bool use_fixed_path_seed = use_guidance_seed;

    for (int k = 0; k < params_.N; k++)
    {
        if (reuse_prev)
        {
            int src = std::min(k + 1, prev_traj_.size() - 1);
            states[k + 1][0] = prev_traj_.points[src].x;
            states[k + 1][1] = prev_traj_.points[src].y;
            states[k + 1][2] = prev_traj_.points[src].psi;
            states[k + 1][3] = std::clamp(prev_traj_.points[src].v, 0.0, params_.max_vel);
            states[k + 1][4] = has_prev_progress
                ? std::clamp(prev_progress_[src], s_start, ref_length)
                : std::min(s_start + params_.desired_speed * (k + 1) * params_.dt, ref_length);
        }
        else if (use_fixed_path_seed)
        {
            StageReference stage_ref = buildStageReference(ref_path_, params_, guidance_progress_, states[k][4], s_start, k + 1);
            states[k + 1][0] = stage_ref.pos.x();
            states[k + 1][1] = stage_ref.pos.y();
            states[k + 1][4] = stage_ref.s;

            double heading_step = severe_heading_mismatch ? 0.28 : 0.24;
            double max_heading_delta = (severe_heading_mismatch ? 0.55 : 0.42) * params_.max_omega * params_.dt;
            double path_err = Pose2D::normalizeAngle(stage_ref.heading - states[k][2]);
            double dpsi_target = std::clamp(heading_step * path_err, -max_heading_delta, max_heading_delta);
            states[k + 1][2] = Pose2D::normalizeAngle(states[k][2] + dpsi_target);

            double stage_speed = stage_ref.speed_ref;
            double ablation_floor = params_.desired_speed * (severe_heading_mismatch ? 0.35 : 0.52);
            double ablation_cap = params_.desired_speed * (severe_heading_mismatch ? 0.65 : 0.95);
            stage_speed = std::clamp(stage_speed, ablation_floor, ablation_cap);
            if (severe_heading_mismatch)
            {
                if (k < 2)
                    stage_speed = std::min(stage_speed, std::max(params_.desired_speed * 0.22, stage_speed));
                else if (k < 4)
                    stage_speed = std::min(stage_speed, params_.desired_speed * 0.38);
            }
            else if (k < 2)
            {
                stage_speed = std::min(stage_speed, params_.desired_speed * (0.45 + 0.20 * k));
            }
            states[k + 1][3] = std::clamp(stage_speed, 0.0, params_.max_vel);
        }
        else if (use_braking_seed)
        {
            SolverState next = states[k];
            double braking_a = -0.8 * params_.max_acc;
            integrateStep(states[k], braking_a, 0.0, next);
            next[2] = states[k][2];
            next[3] = std::max(next[3], 0.0);
            states[k + 1] = next;
        }
        else
        {
            states[k + 1] = states[k];
        }

        double dpsi = Pose2D::normalizeAngle(states[k + 1][2] - states[k][2]);
        controls[k][0] = (states[k + 1][3] - states[k][3]) / params_.dt;
        controls[k][1] = dpsi / params_.dt;
        controls[k][1] = std::clamp(controls[k][1], -params_.max_omega, params_.max_omega);
        if (reuse_prev && has_prev_controls)
        {
            int src = std::min(k + 1, (int)prev_controls_.size() - 1);
            controls[k] = prev_controls_[src];
        }
    }

    seed_controls_ = controls;

    // 梯度下降做障碍微调（自适应步长 + 充足迭代）
    double prev_cost = std::numeric_limits<double>::max();
    double step_size = 0.008;
    const double step_grow = 1.04;
    const double step_shrink = 0.55;
    const int max_iter = params_.max_iter;

    for (int iter = 0; iter < max_iter; iter++)
    {
        rollout(params_, x0, controls, tmp_states_, this);
        double cost = totalCost(tmp_states_, controls, obstacles_by_stage, goal_x, goal_y);

        if (std::abs(prev_cost - cost) < params_.cost_tol && iter >= 8)
            break;

        // 自适应步长：cost 上升则缩减，下降则缓慢恢复
        if (cost > prev_cost && prev_cost < 1e9)
        {
            step_size *= step_shrink;
            step_size = std::max(step_size, 0.001);
        }
        else if (cost < prev_cost && step_size < 0.008)
        {
            step_size *= step_grow;
            step_size = std::min(step_size, 0.008);
        }
        prev_cost = cost;

        std::vector<Eigen::Vector2d> grad(params_.N, Eigen::Vector2d::Zero());
        computeGradient(tmp_states_, controls, obstacles_by_stage, goal_x, goal_y, grad);

        for (int k = 0; k < params_.N; k++)
        {
            controls[k][0] -= step_size * grad[k][0];
            controls[k][1] -= step_size * grad[k][1];
            controls[k][0] = std::clamp(controls[k][0], -params_.max_acc, params_.max_acc);

            double omega_limit = stageOmegaLimit(params_, k);
            omega_limit = std::min(omega_limit, 0.8);
            if (k < 2)
                omega_limit *= 0.85;
            controls[k][1] = std::clamp(controls[k][1], -omega_limit, omega_limit);

            if (k < static_cast<int>(seed_controls_.size()) && k < 4)
            {
                double trust = seedOmegaTrustWidth(params_, k);
                double omega_lo = std::max(-omega_limit, seed_controls_[k][1] - trust);
                double omega_hi = std::min(omega_limit, seed_controls_[k][1] + trust);
                controls[k][1] = std::clamp(controls[k][1], omega_lo, omega_hi);
            }
        }
    }

    // 速度下限投影：前几步强制 v>0，打破 v=0+原地转 basin
    {
        rollout(params_, x0, controls, tmp_states_, this);
        bool patched = false;
        for (int k = 0; k < 4 && k < params_.N; k++)
        {
            double v_floor = params_.desired_speed * (k < 2 ? 0.20 : 0.28);
            if (severe_heading_mismatch)
                v_floor = params_.desired_speed * (k < 2 ? 0.18 : 0.25);
            if (tmp_states_[k + 1][3] < v_floor - 0.01)
            {
                double a_needed = (v_floor - tmp_states_[k][3]) / params_.dt;
                controls[k][0] = std::max(controls[k][0], std::min(a_needed, params_.max_acc));
                patched = true;
            }
        }
        if (patched)
            rollout(params_, x0, controls, tmp_states_, this);
    }

    rollout(params_, x0, controls, tmp_states_, this);
    last_cost_ = totalCost(tmp_states_, controls, obstacles_by_stage, goal_x, goal_y);

    Trajectory traj;
    traj.dt = params_.dt;
    prev_progress_.assign(params_.N + 1, 0.0);
    for (int k = 0; k <= params_.N; k++)
    {
        double v = tmp_states_[k][3];
        double omega = (k < params_.N) ? controls[k][1] : 0.0;
        traj.add(tmp_states_[k][0], tmp_states_[k][1], tmp_states_[k][2], v, omega);
        prev_progress_[k] = tmp_states_[k][4];
    }

    prev_traj_ = traj;
    prev_controls_ = controls;
    return traj;
}

} // namespace MultiUSV
