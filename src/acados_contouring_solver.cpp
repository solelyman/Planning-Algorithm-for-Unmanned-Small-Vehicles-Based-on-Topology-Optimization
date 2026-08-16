#include <multi_ugv_planner/acados_contouring_solver.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <limits>
#include <vector>

extern "C" {
#include "acados_solver_contouring_unicycle.h"
#include "acados_c/ocp_nlp_interface.h"
}

namespace MultiUGV
{

namespace
{
void fitNaturalCubicSpline(const std::vector<double> &s,
                           const std::vector<double> &x,
                           const std::vector<double> &y,
                           int n_seg,
                           double params_out[AcadosContouringSolver::NP])
{
    const int n_knots = n_seg + 1;

    auto fit1d = [&](const std::vector<double> &v,
                     const std::vector<double> &sk,
                     double param[][4])
    {
        std::vector<double> h(n_seg, 0.0), alpha(n_seg, 0.0);
        for (int i = 0; i < n_seg; i++)
        {
            h[i] = sk[i + 1] - sk[i];
            if (h[i] < 1e-9) h[i] = 1e-6;
            if (i > 0)
                alpha[i] = 3.0 / h[i] * (v[i + 1] - v[i])
                         - 3.0 / h[i - 1] * (v[i] - v[i - 1]);
        }
        std::vector<double> l(n_knots, 1.0), mu(n_knots, 0.0), z(n_knots, 0.0);
        for (int i = 1; i < n_seg; i++)
        {
            double li = 2.0 * (sk[i + 1] - sk[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / li;
            l[i] = li;
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / li;
        }
        std::vector<double> M(n_knots, 0.0);
        for (int i = n_seg - 1; i >= 0; i--)
            M[i] = z[i] - mu[i] * M[i + 1];
        for (int i = 0; i < n_seg; i++)
        {
            param[i][0] = (M[i + 1] - M[i]) / (6.0 * h[i]);
            param[i][1] = M[i] / 2.0;
            param[i][2] = (v[i + 1] - v[i]) / h[i]
                        - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
            param[i][3] = v[i];
        }
    };

    double cx[5][4] = {}, cy[5][4] = {};
    fit1d(x, s, cx);
    fit1d(y, s, cy);

    const int base = 8;
    for (int i = 0; i < n_seg; i++)
    {
        int off = base + i * 9;
        params_out[off + 0] = cx[i][0];
        params_out[off + 1] = cx[i][1];
        params_out[off + 2] = cx[i][2];
        params_out[off + 3] = cx[i][3];
        params_out[off + 4] = cy[i][0];
        params_out[off + 5] = cy[i][1];
        params_out[off + 6] = cy[i][2];
        params_out[off + 7] = cy[i][3];
        params_out[off + 8] = s[i];
    }
}
} // anonymous namespace

AcadosContouringSolver::AcadosContouringSolver()
{
    spline_params_.resize(NP, 0.0);
}

AcadosContouringSolver::~AcadosContouringSolver()
{
    if (capsule_)
    {
        contouring_unicycle_acados_free(capsule_);
        contouring_unicycle_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }
}

bool AcadosContouringSolver::initialize()
{
    initialized_ = false;
    if (capsule_)
    {
        contouring_unicycle_acados_free(capsule_);
        contouring_unicycle_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }

    capsule_ = contouring_unicycle_acados_create_capsule();
    if (!capsule_)
    {
        status_message_ = "failed to create contouring capsule";
        return false;
    }

    int status = contouring_unicycle_acados_create(capsule_);
    if (status != 0)
    {
        status_message_ = "contouring_unicycle_acados_create failed (status="
                        + std::to_string(status) + ")";
        return false;
    }

    // Cache pointers for RTI multi-iteration loop
    _nlp_config = contouring_unicycle_acados_get_nlp_config(capsule_);
    _nlp_dims   = contouring_unicycle_acados_get_nlp_dims(capsule_);
    _nlp_in     = contouring_unicycle_acados_get_nlp_in(capsule_);
    _nlp_out    = contouring_unicycle_acados_get_nlp_out(capsule_);
    _nlp_solver = contouring_unicycle_acados_get_nlp_solver(capsule_);
    _nlp_opts   = contouring_unicycle_acados_get_nlp_opts(capsule_);

    initialized_ = true;
    status_message_ = "acados contouring solver ready (5-state, EXTERNAL+EXACT+MIRROR)";
    return true;
}

void AcadosContouringSolver::fitSpline(const ReferencePath &path,
                                       double params_out[NP]) const
{
    std::memset(params_out, 0, NP * sizeof(double));

    int n_pts = static_cast<int>(path.points.size());
    if (n_pts < 2) return;

    const int n_seg = 5;
    const int n_knots = n_seg + 1;

    std::vector<double> s(n_pts, 0.0);
    for (int i = 1; i < n_pts; i++)
        s[i] = s[i - 1] + (path.points[i] - path.points[i - 1]).norm();
    double total_len = s.back();
    if (total_len < 1e-6) return;

    std::vector<double> sk(n_knots), xk(n_knots), yk(n_knots);
    for (int i = 0; i < n_knots; i++)
    {
        double target = total_len * static_cast<double>(i) / (n_knots - 1);
        int j = 0;
        for (j = 0; j + 1 < n_pts; j++)
            if (s[j + 1] > target) break;
        j = std::min(j, n_pts - 2);
        double seg_len = s[j + 1] - s[j];
        double t = (seg_len > 1e-9) ? (target - s[j]) / seg_len : 0.0;
        t = std::clamp(t, 0.0, 1.0);
        xk[i] = path.points[j].x() + t * (path.points[j + 1].x() - path.points[j].x());
        yk[i] = path.points[j].y() + t * (path.points[j + 1].y() - path.points[j].y());
        sk[i] = target;
    }

    fitNaturalCubicSpline(sk, xk, yk, n_seg, params_out);
}

void AcadosContouringSolver::setStageParams(int stage, double reference_velocity,
                                            const std::vector<EllipsoidObstacle> &ellipsoids,
                                            bool is_k0,
                                            double anchor_x, double anchor_y,
                                            bool use_linear)
{
    if (!capsule_ || !initialized_) return;

    double p[NP] = {};
    p[0] = params_.weight_acc;
    p[1] = params_.weight_angvel;
    p[2] = params_.weight_velocity;
    p[3] = reference_velocity;
    p[4] = params_.weight_contour;
    p[5] = params_.weight_lag;
    p[6] = params_.weight_terminal_angle;
    p[7] = params_.weight_terminal_contour;

    // Spline [8..52]
    bool spline_valid = true;
    for (int i = 8; i < 8 + N_SEG * 9; i++)
    {
        if (!std::isfinite(spline_params_[i]))
        {
            spline_valid = false;
            break;
        }
    }
    if (spline_valid)
        std::memcpy(p + 8, spline_params_.data() + 8, N_SEG * 9 * sizeof(double));

    // Weight obstacle [53] — 线性约束开启时关闭软椭球罚, 纯线性效果
    p[53] = use_linear ? 0.0 : params_.weight_obstacle;

    // Ellipsoids [54..65] (4 × 3 = 12)
    if (is_k0)
    {
        // k=0: dummy (r=0 → safe, no penalty)
        for (int i = 0; i < N_ELLIPSOIDS; i++)
        {
            int off = 54 + i * 3;
            p[off+0] = 0.0; p[off+1] = 0.0; p[off+2] = 0.0;
        }
    }
    else
    {
        int n_e = std::min(static_cast<int>(ellipsoids.size()), N_ELLIPSOIDS);
        for (int i = 0; i < N_ELLIPSOIDS; i++)
        {
            int off = 54 + i * 3;
            if (i < n_e)
            {
                p[off+0] = ellipsoids[i].ox;
                p[off+1] = ellipsoids[i].oy;
                p[off+2] = ellipsoids[i].r;
            }
            else
            {
                p[off+0] = 0.0; p[off+1] = 0.0; p[off+2] = 0.0;
            }
        }
    }

    // Linear halfspaces [90..125] (12 × 3 = 36): h = a1*x + a2*y - b <= 0
    // mpc_planner LinearizedConstraints: 法向 = (锚点→障碍) 单位向量, b = a1*ox + a2*oy - r
    if (use_linear && !is_k0)
    {
        // 1) 锚点径向投影到安全区 (projectToSafety, 3 次迭代), 保证线性化/暖启动可行
        double ax = anchor_x, ay = anchor_y;
        for (int it = 0; it < 3; ++it)
        {
            for (int i = 0; i < N_LIN && i < static_cast<int>(ellipsoids.size()); ++i)
            {
                double ox = ellipsoids[i].ox, oy = ellipsoids[i].oy, r = ellipsoids[i].r;
                if (r <= 1e-6) continue;  // dummy
                double dx = ax - ox, dy = ay - oy;
                double d = std::hypot(dx, dy);
                if (d > 1e-9 && d < r)
                {
                    double scale = (r + 1e-3) / d;
                    ax = ox + dx * scale;
                    ay = oy + dy * scale;
                }
            }
        }

        // 2) 逐障碍半空间
        for (int i = 0; i < N_LIN; ++i)
        {
            int off = LIN_PARAM_OFFSET + i * 3;
            bool filled = false;
            if (i < static_cast<int>(ellipsoids.size()))
            {
                double ox = ellipsoids[i].ox, oy = ellipsoids[i].oy, r = ellipsoids[i].r;
                if (r > 1e-6)
                {
                    double dx = ox - ax, dy = oy - ay;
                    double d = std::hypot(dx, dy);
                    if (d > 1e-9)
                    {
                        double a1 = dx / d, a2 = dy / d;
                        p[off + 0] = a1;
                        p[off + 1] = a2;
                        p[off + 2] = a1 * ox + a2 * oy - r;
                        filled = true;
                    }
                }
            }
            if (!filled)
            {
                p[off + 0] = 1.0; p[off + 1] = 0.0; p[off + 2] = 100.0;  // dummy: x-100<=0
            }
        }
    }
    else
    {
        // k=0 或未启用: dummy 约束 (x-100<=0, 恒成立)
        for (int i = 0; i < N_LIN; ++i)
        {
            int off = LIN_PARAM_OFFSET + i * 3;
            p[off + 0] = 1.0; p[off + 1] = 0.0; p[off + 2] = 100.0;
        }
    }

    contouring_unicycle_acados_update_params(capsule_, stage, p, NP);
}

Trajectory AcadosContouringSolver::solve(const UGVState &state,
                                          const ReferencePath &ref_path,
                                          const StageConstraintSet &stage_constraints)
{
    Trajectory traj;
    traj.dt = params_.dt;

    if (!initialized_ || !capsule_ || ref_path.empty())
    {
        status_message_ = "contouring solve skipped: not ready or empty path";
        return traj;
    }

    // ──  1. Initial state ──
    double s0 = 0.0;
    ref_path.findClosest(Eigen::Vector2d(state.x, state.y), s0);

    double x0[5] = {state.x, state.y, state.psi,
                    std::clamp(state.v, 0.0, 1.9), s0};
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, _nlp_out,
                                  0, "lbx", x0);
    ocp_nlp_constraints_model_set(_nlp_config, _nlp_dims, _nlp_in, _nlp_out,
                                  0, "ubx", x0);

    // ──  2. Fit spline and set stage parameters ──
    double path_len = ref_path.length();
    auto speed_ref_at = [&](double s_ref) -> double
    {
        // 减速区长度按 desired_speed 缩放(适配 UGV 数米级短路径, 原 8.0 是 UGV 尺度)
        double remain = std::max(path_len - s_ref, 0.0);
        double v_scale = std::clamp(remain / std::max(1.0, 3.0 * params_.desired_speed), 0.15, 1.0);
        return params_.desired_speed * v_scale;
    };

    fitSpline(ref_path, spline_params_.data());

    // ──  3. Warm start 轨迹 (先算, 同时作为半空间锚点, mpc_planner 式) ──
    // 锚点 = 动力学可行的预测轨迹点 (上次解 shift 或直线外推) + 障碍径向投影.
    // 不用参考路径点: V-PRM 绕行路径可能离起点很远, 用它做锚点会要求轨迹
    // "第一步就跳到障碍另一侧", 与动力学冲突 → QP INFEASIBLE.
    auto project_warm = [&](int k, double &px, double &py)
    {
        if (k <= 0 || k >= static_cast<int>(stage_constraints.size())) return;
        for (int it = 0; it < 3; ++it)
        {
            for (const auto &e : stage_constraints[k].ellipsoids)
            {
                if (e.r <= 1e-6) continue;  // dummy
                double dx = px - e.ox, dy = py - e.oy;
                double d = std::hypot(dx, dy);
                if (d > 1e-9 && d < e.r)
                {
                    double scale = (e.r + 1e-3) / d;
                    px = e.ox + dx * scale;
                    py = e.oy + dy * scale;
                }
            }
        }
    };

    double guess_v = std::max(state.v, 0.3);
    bool use_prev = params_.warmstart_with_previous && has_prev_trajectory_ &&
                    prev_trajectory_.size() >= params_.N + 1;
    std::vector<std::array<double, 5>> wpts(params_.N + 1);
    for (int k = 0; k <= params_.N; k++)
    {
        wpts[k] = {x0[0], x0[1], x0[2], guess_v, x0[4]};
        if (k > 0)
        {
            if (use_prev)
            {
                // shift 上一帧解: 第 k 步用上一帧第 k+1 步 (末段重复终点)
                int prev_idx = std::min(k + 1, prev_trajectory_.size() - 1);
                const auto &pp = prev_trajectory_.points[prev_idx];
                wpts[k][0] = pp.x;
                wpts[k][1] = pp.y;
                wpts[k][2] = pp.psi;
                wpts[k][3] = std::max(pp.v, 0.3);  // 速度下限, 防止低 v 暖启动锁死蠕行
                wpts[k][4] = std::clamp(pp.s, 0.0, path_len);
            }
            else
            {
                wpts[k][0] += guess_v * std::cos(wpts[k][2]) * params_.dt * k;
                wpts[k][1] += guess_v * std::sin(wpts[k][2]) * params_.dt * k;
                wpts[k][4] = std::min(s0 + guess_v * params_.dt * k, path_len);
            }
            project_warm(k, wpts[k][0], wpts[k][1]);
        }
    }

    // ──  2b. Stage parameters (锚点 = warm start 轨迹点) ──
    for (int k = 0; k < params_.N; k++)
    {
        double s_ref = std::min(s0 + params_.desired_speed * params_.dt * (k + 1), path_len);
        double vref = speed_ref_at(s_ref);

        // Extract per-stage ellipsoid constraints
        std::vector<EllipsoidObstacle> stage_ells;
        if (k < static_cast<int>(stage_constraints.size()))
            stage_ells = stage_constraints[k].ellipsoids;

        setStageParams(k, vref, stage_ells, k == 0,
                       wpts[k][0], wpts[k][1], params_.use_linear_constraints);
    }

    // 写入 warm start 到求解器
    for (int k = 0; k <= params_.N; k++)
    {
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _nlp_in, k, "x", wpts[k].data());
    }
    for (int k = 0; k < params_.N; k++)
    {
        double uk[2] = {0.0, 0.0};
        ocp_nlp_out_set(_nlp_config, _nlp_dims, _nlp_out, _nlp_in, k, "u", uk);
    }

    // ──  4. Multi-RTI iteration loop (matching mpc_planner-main) ──
    // SQP_RTI with multiple iterations: each iteration warms from previous.
    // After each iteration, check status and reset QP memory if needed.
    int nlp_status = 0;
    int iter = 0;
    while (iter < RTI_ITERATIONS)
    {
        iter++;  // 每次迭代(含失败重试)都推进计数, 防止失败分支无限循环
        // Precompute for this RTI phase
        int rti_phase = 0;  // 0 = both prep + feedback
        ocp_nlp_solver_opts_set(_nlp_config, _nlp_opts, "rti_phase", &rti_phase);
        ocp_nlp_precompute(_nlp_solver, _nlp_in, _nlp_out);

        nlp_status = contouring_unicycle_acados_solve(capsule_);

        // Check QP status
        int qp_status = 0;
        ocp_nlp_get(_nlp_solver, "qp_status", &qp_status);

        if (nlp_status != 0 && qp_status != 0)
        {
            // QP failed — reset and retry (受 iter 计数保护, 最多 RTI_ITERATIONS 次)
            contouring_unicycle_acados_reset(capsule_, 1);
            ocp_nlp_solver_reset_qp_memory(_nlp_solver, _nlp_in, _nlp_out);
            continue;
        }

        // Check residual — even if acados says success
        double res_eq = 0.0;
        ocp_nlp_get(_nlp_solver, "res_eq", &res_eq);
        if (res_eq > 1e-2 && nlp_status == 0)
        {
            nlp_status = 1;  // Mark as soft failure
            contouring_unicycle_acados_reset(capsule_, 1);
            ocp_nlp_solver_reset_qp_memory(_nlp_solver, _nlp_in, _nlp_out);
            continue;
        }
    }

    // ──  5. Extract solution or fallback ──
    if (nlp_status != 0)
    {
        // ── 诊断: 失败帧各 stage 障碍/半空间/warm-start 约束值 ──
        int qp_status_last = 0;
        ocp_nlp_get(_nlp_solver, "qp_status", &qp_status_last);
        fprintf(stderr, "\n[SOLVE-DIAG] nlp=%d qp_status=%d x0=(%.2f,%.2f,%.2f,%.2f) s0=%.2f path_len=%.2f use_prev=%d\n",
                nlp_status, qp_status_last, x0[0], x0[1], x0[2], x0[3], x0[4], path_len, use_prev ? 1 : 0);
        for (int k = 1; k < params_.N; k++)
        {
            double ax = wpts[k][0], ay = wpts[k][1];  // warm start 轨迹点(已投影) = 锚点
            std::vector<EllipsoidObstacle> stage_ells;
            double xk[5] = {};
            ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, k, "x", xk);
            for (size_t i = 0; i < stage_ells.size(); ++i)
            {
                const auto &e = stage_ells[i];
                if (e.r <= 1e-6) continue;
                double dx = e.ox - ax, dy = e.oy - ay;
                double d = std::hypot(dx, dy);
                double a1 = dx / d, a2 = dy / d;
                double b = a1 * e.ox + a2 * e.oy - e.r;
                double h = a1 * xk[0] + a2 * xk[1] - b;
                fprintf(stderr, "  [stage %2d] obs=(%5.2f,%5.2f,r=%.3f) anchor=(%5.2f,%5.2f) hsp=(%+.3f,%+.3f,b=%+.3f) ws=(%5.2f,%5.2f) h=%+.4f\n",
                        k, e.ox, e.oy, e.r, ax, ay, a1, a2, b, xk[0], xk[1], h);
            }
            if (stage_ells.size() < static_cast<size_t>(N_LIN))
                fprintf(stderr, "  [stage %2d] +%d dummy\n", k, N_LIN - static_cast<int>(stage_ells.size()));
        }
        fprintf(stderr, "[SOLVE-DIAG] end\n\n");

        status_message_ = "contouring solve failed (all "
                        + std::to_string(RTI_ITERATIONS) + " RTI iters)";

        // Fallback to previous successful trajectory
        if (has_prev_trajectory_ && !prev_trajectory_.points.empty())
            return prev_trajectory_;

        return traj;
    }

    traj.points.clear();
    traj.controls.clear();
    for (int k = 0; k <= params_.N; k++)
    {
        double x[5] = {};
        ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, k, "x", x);
        double u[2] = {};
        if (k < params_.N)
            ocp_nlp_out_get(_nlp_config, _nlp_dims, _nlp_out, k, "u", u);
        traj.add(x[0], x[1], x[2], x[3], u[1]);
        traj.points.back().s = x[4];
        if (k < params_.N)
            traj.controls.push_back({u[0], u[1]});
    }

    status_message_ = "contouring solve ok ("
                    + std::to_string(RTI_ITERATIONS) + " RTI iters)";

    prev_trajectory_ = traj;
    has_prev_trajectory_ = true;
    return traj;
}

} // namespace MultiUGV
