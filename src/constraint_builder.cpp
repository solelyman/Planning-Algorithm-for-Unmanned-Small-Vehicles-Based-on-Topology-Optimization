#include <multi_usv_planner/constraint_builder.h>
#include <multi_usv_planner/acados_contouring_solver.h>

namespace MultiUSV
{

/// @brief Get reference path position at time @p tk.
static Eigen::Vector2d refPos(const ReferencePath &ref, double tk)
{
    if (ref.points.empty())
        return Eigen::Vector2d::Zero();
    double total = ref.length();
    double arclen = std::min(tk * 1.2, total);
    double accum = 0.0;
    for (size_t i = 0; i + 1 < ref.points.size(); ++i)
    {
        const Eigen::Vector2d &a = ref.points[i];
        const Eigen::Vector2d &b = ref.points[i + 1];
        double seg = (b - a).norm();
        if (accum + seg >= arclen || i + 2 >= ref.points.size())
        {
            double t = (arclen - accum) / std::max(seg, 1e-6);
            t = std::clamp(t, 0.0, 1.0);
            return a + t * (b - a);
        }
        accum += seg;
    }
    return ref.points.back();
}

StageConstraintSet ConstraintBuilder::build(
    const ReferencePath &contour_ref,
    const USVState & /*ego*/,
    const std::vector<Obstacle> &static_obstacles,
    double robot_radius,
    double obstacle_clearance,
    double dt,
    int N,
    double corridor_y_min,
    double corridor_y_max)
{
    StageConstraintSet sc_set;
    sc_set.reserve(N);

    for (int k = 0; k < N; ++k)
    {
        double tk = dt * (k + 1);
        const Eigen::Vector2d ego_pred = refPos(contour_ref, tk);

        struct Candidate {
            double ox, oy, r;
            double dist;  // distance to predicted ego position
        };
        std::vector<Candidate> cands;
        cands.reserve(32);

        // Corridor walls — DecompUtil-aware

        // 1) Static obstacles (islands) — fixed position, no velocity
        for (const auto &obs : static_obstacles)
        {
            if (obs.radius < 0.01) continue;
            double d = (Eigen::Vector2d{obs.x, obs.y} - ego_pred).norm();
            cands.push_back({
                obs.x, obs.y,
                obs.radius + robot_radius + obstacle_clearance,
                d
            });
        }

        {
            double yp = ego_pred.y();
            double y_min = corridor_y_min;
            double y_max = corridor_y_max;
            double wall_margin = robot_radius + obstacle_clearance;

            double dist_bottom = std::abs(yp - y_min);
            double dist_top    = std::abs(yp - y_max);
            cands.push_back({ego_pred.x(), y_min, wall_margin, dist_bottom});
            cands.push_back({ego_pred.x(), y_max, wall_margin, dist_top});
        }

        // 单体 Fishbot: 无动态障碍/邻居
        // Sort by distance to predicted ego, take N_ELLIPSOIDS
        std::sort(cands.begin(), cands.end(), [](const Candidate &a, const Candidate &b) {
                return a.dist < b.dist;
            });

        // 合并重叠障碍: 同一物理障碍被激光雷达扫成多个相邻点 → 多个重叠椭球
        // 会把可行域压死导致 QP 不可行. 重叠(中心距 < 半径和的一半)则合并,
        // 中心取中点, 半径取外切覆盖, 保证合并约束比原始约束更保守(更安全).
        std::vector<Candidate> merged;
        for (const auto &c : cands)
        {
            bool absorbed = false;
            for (auto &m : merged)
            {
                double dc = std::hypot(c.ox - m.ox, c.oy - m.oy);
                if (dc < 0.5 * (c.r + m.r))
                {
                    double cover = (dc + m.r + c.r) / 2.0;
                    m.r = std::max(m.r, std::max(c.r, cover));
                    m.ox = 0.5 * (m.ox + c.ox);
                    m.oy = 0.5 * (m.oy + c.oy);
                    m.dist = std::min(m.dist, c.dist);
                    absorbed = true;
                    break;
                }
            }
            if (!absorbed) merged.push_back(c);
        }
        cands = std::move(merged);

        StageConstraints sc;
        int n_to_take = std::min(static_cast<int>(cands.size()),
                                 AcadosContouringSolver::N_ELLIPSOIDS);
        for (int i = 0; i < n_to_take; ++i)
        {
            EllipsoidObstacle ell;
            ell.ox = cands[i].ox;
            ell.oy = cands[i].oy;
            ell.r  = cands[i].r;
            sc.ellipsoids.push_back(ell);
        }

        // Pad remaining slots with dummies (r=0 → no cost)
        while (static_cast<int>(sc.ellipsoids.size()) < AcadosContouringSolver::N_ELLIPSOIDS)
        {
            EllipsoidObstacle ell;
            ell.ox = 0.0; ell.oy = 0.0; ell.r = 0.0;
            sc.ellipsoids.push_back(ell);
        }

        sc_set.push_back(std::move(sc));
    }

    return sc_set;
}

}  // namespace MultiUSV
