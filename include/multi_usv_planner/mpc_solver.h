#ifndef MULTI_USV_MPC_SOLVER_H
#define MULTI_USV_MPC_SOLVER_H

#include <multi_usv_planner/types.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace MultiUSV
{

struct ReferencePathSample
{
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    double psi = 0.0;
    double v = 0.0;
};

struct ReferencePath
{
    std::vector<Eigen::Vector2d> points;
    double total_length = 0.0;
    double nominal_speed = 1.5;

    void clear() { points.clear(); total_length = 0.0; }
    bool empty() const { return points.empty(); }
    int size() const { return (int)points.size(); }

    int findClosest(const Eigen::Vector2d &pos, double &s_out) const;
    Eigen::Vector2d sample(double s) const;
    double heading(double s) const;
    double length() const;
    ReferencePathSample sampleState(double s, double remain_hint = -1.0) const;
    double headingBetween(double s0, double s1) const;
};

using SolverState = Eigen::Matrix<double, 5, 1>;  // [x, y, psi, v, s_progress]
using StageObstacleSet = std::vector<std::vector<Obstacle>>;

class MPCSolver
{
public:
    struct Params
    {
        int N = 12;                 // 预测步数
        double dt = 0.4;            // 步长 [s]

        double max_vel = 2.0;       // 最大速度 [m/s]
        double max_omega = 1.57;    // 最大角速度 [rad/s]
        double max_acc = 1.0;       // 最大加速度 [m/s²]
        double max_domega = 1.0;    // 最大角加速度 [rad/s²]

        double weight_goal = 8.0;        // 终端吸引，path following 为主
        double weight_path = 60.0;       // 路径跟随主导（contour/lag）
        double weight_obstacle = 30.0;   // barrier 权重，exp(10v) 形状提供硬梯度
        double obstacle_clearance = 2.0; // 船体之外的额外安全余量 [m]
        double dynamic_obstacle_growth = 0.12; // 动态障碍物时域不确定性膨胀 [m/step]
        double weight_smooth = 1.0;
        double weight_speed = 1.5;

        double desired_speed = 1.5;
        double lookahead_distance = 6.0; // 前瞻距离

        double weight_guidance = 50.0;   // 拓扑走廊约束权重
        double guidance_margin = 0.8;    // 拓扑走廊半宽 [m]
        double weight_initial_omega = 2.0; // 首段偏航动作惩罚
        double weight_initial_domega = 2.0; // 首段偏航变化惩罚
        double weight_progress = 10.0;   // 进度跟踪权重
        double weight_backtrack = 40.0;  // 进度回退惩罚
        double min_progress_step = 0.15; // 每步最小前进量 [m]
        double weight_terminal_angle = 80.0;
        double weight_terminal_contouring = 8.0;
        double weight_seed_alignment = 8.0;
        double weight_turning_speed = 7.0;
        double weight_heading_progress = 20.0;
        double weight_heading_rate = 5.0;
        double min_heading_speed_scale = 0.12;
        double weight_progress_consistency = 22.0;
        double weight_forward_progress = 5.0;
        bool unicycle_like_ablation = false;

        // acados contouring weights
        double acados_weight_acc = 0.25;
        double acados_weight_angvel = 0.50;
        double acados_weight_velocity = 0.30;
        double acados_weight_contour = 0.05;
        double acados_weight_lag = 0.10;
        double acados_weight_terminal_angle = 100.0;
        double acados_weight_terminal_contouring = 10.0;

        int max_iter = 50;           // 最大迭代次数
        double grad_eps = 1e-4;      // 梯度步长
        double cost_tol = 1e-3;      // 收敛阈值
    };

    MPCSolver();
    void setParams(const Params &p) { params_ = p; }
    Params &params() { return params_; }
    const Params &params() const { return params_; }

    // 设置参考路径（同时预计算拓扑走廊锚点）
    void setReferencePath(const ReferencePath &path);
    void setCurrentHeading(double psi);
    void clearWarmstart();

    Trajectory solve(const USVState &state,
                     const StageObstacleSet &obstacles_by_stage,
                     double goal_x, double goal_y);

    const Trajectory &lastTrajectory() const { return prev_traj_; }
    double lastCost() const { return last_cost_; }

    void integrateStep(const SolverState &s, double a, double omega,
                       SolverState &s_next) const;
    double projectProgress(double x, double y) const;

private:

    // 总成本
    double totalCost(const std::vector<SolverState> &states,
                     const std::vector<Eigen::Vector2d> &controls,
                     const StageObstacleSet &obstacles_by_stage,
                     double goal_x, double goal_y) const;

    // 数值梯度
    void computeGradient(const std::vector<SolverState> &states,
                         const std::vector<Eigen::Vector2d> &controls,
                         const StageObstacleSet &obstacles_by_stage,
                         double goal_x, double goal_y,
                         std::vector<Eigen::Vector2d> &grad) const;

    Params params_;
    ReferencePath ref_path_;
    double current_heading_ = 0.0;
    double last_cost_ = std::numeric_limits<double>::max();
    std::vector<Eigen::Vector2d> guidance_points_;  // 预计算的局部参考位置采样，仅用于 guidance 位置约束
    std::vector<double> guidance_progress_;         // 预计算的单调参考进度
    Trajectory prev_traj_;
    std::vector<double> prev_progress_;
    std::vector<Eigen::Vector2d> prev_controls_;
    std::vector<Eigen::Vector2d> seed_controls_;

    // 临时缓冲
    mutable std::vector<SolverState> tmp_states_;
    mutable std::vector<Eigen::Vector2d> tmp_controls_;
};

} // namespace MultiUSV

#endif
