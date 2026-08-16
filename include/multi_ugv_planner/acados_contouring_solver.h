#ifndef MULTI_UGV_ACADOS_CONTOURING_SOLVER_H
#define MULTI_UGV_ACADOS_CONTOURING_SOLVER_H

#include <multi_ugv_planner/types.h>
#include <multi_ugv_planner/mpc_solver.h>
#include <string>
#include <vector>

struct ocp_nlp_solver;
struct ocp_nlp_in;
struct ocp_nlp_out;
struct ocp_nlp_config;
struct ocp_nlp_dims;

/* 新版生成 API 的胶囊类型 (带 model-hash 前缀), 旧名仅为兼容 */
typedef struct ocp_contouring_unicycle_1f1aa62f_solver_capsule contouring_unicycle_solver_capsule;

namespace MultiUGV
{

/// An ellipsoid (circle) constraint: (x-ox)² + (y-oy)² - r² <= 0
struct EllipsoidObstacle
{
    double ox = 0.0;
    double oy = 0.0;
    double r  = 1e6;   // dummy: always feasible
};

/// Per-stage ellipsoid constraints (one per dynamic obstacle)
struct StageConstraints
{
    std::vector<EllipsoidObstacle> ellipsoids;
};

/// Constraints per stage: [stage][constraint_set]
using StageConstraintSet = std::vector<StageConstraints>;

class AcadosContouringSolver
{
public:
    static constexpr int N_ELLIPSOIDS = 12;
    static constexpr int N_LIN = 12;
    static constexpr int N_SEG = 5;
    // NP = 8(weights+ref_vel) + 45(spline) + 1(weight_obstacle)
    //    + 36(ellipsoid 12×3) + 36(lin halfspace 12×3) = 126
    static constexpr int NP = 8 + N_SEG * 9 + 1 + N_ELLIPSOIDS * 3 + N_LIN * 3;
    static constexpr int LIN_PARAM_OFFSET = 8 + N_SEG * 9 + 1 + N_ELLIPSOIDS * 3;  // = 90
    // 外层 SQP 迭代数. 2 次从低速度 warm start 收敛不到最优(蠕行),
    // 6 次让非线性充分收敛, 速度能爬向参考值.
    static constexpr int RTI_ITERATIONS = 6;

    struct Params
    {
        int N = 15;
        double dt = 0.4;
        int solve_timeout_ms = 1500;

        double weight_acc       = 0.3;
        double weight_angvel    = 1.0;
        double weight_velocity  = 0.3;
        double desired_speed    = 1.25;
        double weight_contour   = 0.04;
        double weight_lag       = 0.15;
        double weight_terminal_angle     = 100.0;
        double weight_terminal_contour   = 10.0;

        double weight_obstacle          = 500.0;

        double robot_radius = 0.2;   // UGV 小车车体半径, 椭球半径 = obs.r + robot_radius + clearance
        double obstacle_clearance = 3.0;
        bool use_linear_constraints = true;  // 硬线性半空间约束 (mpc_planner 式), 开启时软椭球罚权重置 0
        bool warmstart_with_previous = false;
    };

    AcadosContouringSolver();
    ~AcadosContouringSolver();

    void setParams(const Params &p) { params_ = p; }
    const Params &params() const { return params_; }
    Params &params_mutable() { return params_; }

    bool initialize();
    bool available() const { return initialized_; }
    const std::string &statusMessage() const { return status_message_; }

    /** Solve OCP with path tracking cost + ellipsoid soft penalty for dynamic obstacles. */
    Trajectory solve(const UGVState &state,
                     const ReferencePath &ref_path,
                     const StageConstraintSet &stage_constraints);

private:
    void fitSpline(const ReferencePath &path,
                   double params_out[NP]) const;

    void setStageParams(int stage, double reference_velocity,
                        const std::vector<EllipsoidObstacle> &ellipsoids,
                        bool is_k0 = false,
                        double anchor_x = 0.0, double anchor_y = 0.0,
                        bool use_linear = false);

    Params params_;
    bool initialized_ = false;
    std::string status_message_ = "not initialized";
    contouring_unicycle_solver_capsule *capsule_ = nullptr;

    ocp_nlp_config *_nlp_config = nullptr;
    ocp_nlp_dims *_nlp_dims = nullptr;
    ocp_nlp_in *_nlp_in = nullptr;
    ocp_nlp_out *_nlp_out = nullptr;
    ocp_nlp_solver *_nlp_solver = nullptr;
    void *_nlp_opts = nullptr;

    std::vector<double> spline_params_;

    Trajectory prev_trajectory_;
    bool has_prev_trajectory_ = false;
};

} // namespace MultiUGV

#endif
