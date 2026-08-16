#include <multi_ugv_planner/ugv_planner.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <numeric>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <iomanip>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

using namespace std::chrono_literals;
using namespace MultiUGV;

double accumulatePathLength(const ReferencePath &path)
{
    if (path.points.size() < 2)
        return 0.0;

    double total = 0.0;
    for (size_t i = 1; i < path.points.size(); ++i)
        total += (path.points[i] - path.points[i - 1]).norm();
    return total;
}

double dynamicStopClearance(double stop_radius)
{
    return std::max(stop_radius * 0.72, 1.8);
}

double dynamicSlowClearance(double obstacle_clearance)
{
    return std::max(obstacle_clearance + 3.8, 7.8);
}

UGVPlanner::UGVPlanner()
    : Node("ugv_planner_node")
{
    declareParams();
    loadParams();
    initCommunication();

    {
        AcadosContouringSolver::Params ap;
        const auto &mp = mpc_solver_.params();
        ap.N = mp.N;
        ap.dt = mp.dt;
        ap.desired_speed = mp.desired_speed;
        ap.obstacle_clearance = mp.obstacle_clearance;
        ap.robot_radius = this->get_parameter("mpc.robot_radius").as_double();
        ap.use_linear_constraints = this->get_parameter("mpc.use_linear_constraints").as_bool();
        ap.weight_acc = mp.acados_weight_acc;
        ap.weight_angvel = mp.acados_weight_angvel;
        ap.weight_velocity = mp.acados_weight_velocity;
        ap.weight_contour = mp.acados_weight_contour;
        ap.weight_lag = mp.acados_weight_lag;
        ap.weight_terminal_angle = mp.acados_weight_terminal_angle;
        ap.weight_terminal_contour = mp.acados_weight_terminal_contouring;
        ap.warmstart_with_previous = this->get_parameter("mpc.warmstart_with_previous").as_bool();
        acados_solver_.setParams(ap);
        bool acados_ok = acados_solver_.initialize();
        if (acados_ok)
            RCLCPP_INFO(this->get_logger(), "UGV-%d acados contouring solver ready (5-state, EXTERNAL+EXACT+MIRROR, ellipsoid soft penalty[N_ELL=%d])", agent_id_, AcadosContouringSolver::N_ELLIPSOIDS);
        else
            RCLCPP_WARN(this->get_logger(), "UGV-%d acados init failed: %s",
                        agent_id_, acados_solver_.statusMessage().c_str());
    }

    RCLCPP_INFO(this->get_logger(),
        "UGV-%d 启动 | MPC N=%d | CTRL=%s | ILOS lh=%.1f",
        agent_id_,
        mpc_solver_.params().N,
        control_mode_.c_str(),
        controller_.params().lookahead_h);
}

#define P_(name, val) this->declare_parameter(#name, val)

void UGVPlanner::declareParams()
{
    P_(agent_id, 1);
    P_(goal.x, 120.0);    P_(goal.y, 30.0);
    P_(use_direct_cmd, false);  P_(control_mode, "los");
    P_(static_obstacles_x, std::vector<double>{});  P_(static_obstacles_y, std::vector<double>{});  P_(static_obstacles_r, std::vector<double>{});

    // MPC
    P_(mpc.N, 10);        P_(mpc.dt, 0.4);         P_(mpc.max_vel, 2.0);
    P_(mpc.max_omega, 1.57);  P_(mpc.max_acc, 1.0);  P_(mpc.desired_speed, 1.5);
    P_(mpc.max_iter, 50);     P_(mpc.weight_goal, 8.0);  P_(mpc.weight_path, 60.0);
    P_(mpc.weight_obstacle, 30.0);  P_(mpc.weight_smooth, 1.0);  P_(mpc.weight_speed, 1.0);
    P_(mpc.weight_guidance, 50.0);  P_(mpc.guidance_margin, 0.8);
    P_(mpc.weight_progress, 10.0);  P_(mpc.weight_backtrack, 40.0);  P_(mpc.min_progress_step, 0.15);
    P_(mpc.weight_terminal_angle, 12.0);  P_(mpc.weight_terminal_contouring, 3.0);
    P_(mpc.weight_seed_alignment, 3.0);
    P_(mpc.obstacle_clearance, 2.0);
    P_(mpc.robot_radius, 0.5);
    P_(mpc.use_linear_constraints, true);
    P_(mpc.warmstart_with_previous, true);
    P_(mpc.dynamic_obstacle_growth, 0.12);
    P_(mpc.acados_weight_acc, 0.25);
    P_(mpc.acados_weight_angvel, 0.50);
    P_(mpc.acados_weight_velocity, 0.30);
    P_(mpc.acados_weight_contour, 0.05);
    P_(mpc.acados_weight_lag, 0.10);
    P_(mpc.acados_weight_terminal_angle, 100.0);
    P_(mpc.acados_weight_terminal_contouring, 10.0);

    // ILOS
    P_(controller.m11, 28.0);  P_(controller.m22, 28.0);   P_(controller.Izz, 10.0);
    P_(controller.d_u, -16.45);  P_(controller.d_r, -6.0); P_(controller.lookahead_h, 8.0);
    P_(controller.k_x, 2.0);   P_(controller.k_u, 5.0);    P_(controller.k_psi, 1.0);
    P_(controller.k_y, 0.5);   P_(controller.k_r, 5.0);    P_(controller.max_u, 2.0);
    P_(controller.max_r, 1.0); P_(controller.filter_omega_n, 2.2);  P_(controller.filter_zeta, 0.85);
    P_(controller.stop_radius, 2.0);
    P_(pass_x_threshold, 30.0);
    P_(episode_timeout_s, 600.0);
}

#undef P_

void UGVPlanner::loadParams()
{
    agent_id_ = this->get_parameter("agent_id").as_int();

    // CSV log (写入当前工作目录)
    std::string csv_path = "log_ugv_" + std::to_string(agent_id_) + ".csv";
    csv_out_.open(csv_path);
    csv_out_ << "t,x,y,psi,u,v,r,desired_speed,dist_to_goal,nearest_obs,cmd_u,cmd_r\n";
    csv_initialized_ = true;
    RCLCPP_INFO(this->get_logger(), "CSV ➔ %s", csv_path.c_str());

    goal_x_ = this->get_parameter("goal.x").as_double();
    goal_y_ = this->get_parameter("goal.y").as_double();
    has_goal_ = true;

    use_direct_cmd_ = this->get_parameter("use_direct_cmd").as_bool();
    control_mode_ = this->get_parameter("control_mode").as_string();

    auto obs_x = this->get_parameter("static_obstacles_x").as_double_array();
    auto obs_y = this->get_parameter("static_obstacles_y").as_double_array();
    auto obs_r = this->get_parameter("static_obstacles_r").as_double_array();
    int n_obs = std::min({(int)obs_x.size(), (int)obs_y.size(), (int)obs_r.size()});
    for (int i = 0; i < n_obs; i++)
        static_obstacles_.push_back(Obstacle{obs_x[i], obs_y[i], obs_r[i]});

    auto load_double = [&](const char *k) { return this->get_parameter(k).as_double(); };
    auto load_int = [&](const char *k) { return this->get_parameter(k).as_int(); };

    // MPC
    {
        auto &p = mpc_solver_.params();
        p.N = load_int("mpc.N");
        p.dt = load_double("mpc.dt");
        p.max_vel = load_double("mpc.max_vel");
        p.max_omega = load_double("mpc.max_omega");
        p.max_acc = load_double("mpc.max_acc");
        p.desired_speed = load_double("mpc.desired_speed");
        p.max_iter = load_int("mpc.max_iter");
        p.weight_goal = load_double("mpc.weight_goal");
        p.weight_path = load_double("mpc.weight_path");
        p.weight_obstacle = load_double("mpc.weight_obstacle");
        p.weight_smooth = load_double("mpc.weight_smooth");
        p.weight_speed = load_double("mpc.weight_speed");
        p.weight_guidance = load_double("mpc.weight_guidance");
        p.guidance_margin = load_double("mpc.guidance_margin");
        p.weight_progress = load_double("mpc.weight_progress");
        p.weight_backtrack = load_double("mpc.weight_backtrack");
        p.min_progress_step = load_double("mpc.min_progress_step");
        p.weight_terminal_angle = load_double("mpc.weight_terminal_angle");
        p.weight_terminal_contouring = load_double("mpc.weight_terminal_contouring");
        p.weight_seed_alignment = load_double("mpc.weight_seed_alignment");
        p.obstacle_clearance = load_double("mpc.obstacle_clearance");
        p.dynamic_obstacle_growth = load_double("mpc.dynamic_obstacle_growth");
        p.acados_weight_acc = load_double("mpc.acados_weight_acc");
        p.acados_weight_angvel = load_double("mpc.acados_weight_angvel");
        p.acados_weight_velocity = load_double("mpc.acados_weight_velocity");
        p.acados_weight_contour = load_double("mpc.acados_weight_contour");
        p.acados_weight_lag = load_double("mpc.acados_weight_lag");
        p.acados_weight_terminal_angle = load_double("mpc.acados_weight_terminal_angle");
        p.acados_weight_terminal_contouring = load_double("mpc.acados_weight_terminal_contouring");
    }

    // Controller
    {
        UGVController::Params cp;
        cp.m11 = load_double("controller.m11");
        cp.m22 = load_double("controller.m22");
        cp.Izz = load_double("controller.Izz");
        cp.d_u = load_double("controller.d_u");
        cp.d_r = load_double("controller.d_r");
        cp.lookahead_h = load_double("controller.lookahead_h");
        cp.k_x = load_double("controller.k_x");
        cp.k_u = load_double("controller.k_u");
        cp.k_psi = load_double("controller.k_psi");
        cp.k_y = load_double("controller.k_y");
        cp.k_r = load_double("controller.k_r");
        cp.max_u = load_double("controller.max_u");
        cp.max_r = load_double("controller.max_r");
        cp.filter_omega_n = load_double("controller.filter_omega_n");
        cp.filter_zeta = load_double("controller.filter_zeta");
        cp.stop_radius = load_double("controller.stop_radius");
    pass_x_threshold_ = load_double("pass_x_threshold");
    episode_timeout_s_ = load_double("episode_timeout_s");
    controller_.setParams(cp);
    }
}

void UGVPlanner::initCommunication()
{
    const auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", rclcpp::QoS(10).best_effort(),   // UGV 固件 /odom 为 BEST_EFFORT
        [this](const nav_msgs::msg::Odometry::SharedPtr m) { odomCallback(m); });

    sub_path_ = this->create_subscription<nav_msgs::msg::Path>(
        "reference_path", latched_qos,
        [this](const nav_msgs::msg::Path::SharedPtr m) { pathCallback(m); });

    sub_goal_ = this->create_subscription<geometry_msgs::msg::Point>(
        "final_goal", 10,
        [this](const geometry_msgs::msg::Point::SharedPtr m) { goalCallback(m); });

    // UGV 雷达: /scan (RELIABLE) -> 世界坐标障碍, 注入 MPC 约束
    sub_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::QoS(10).reliable(),
        [this](const sensor_msgs::msg::LaserScan::SharedPtr m) { scanCallback(m); });

    pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

    pub_ref_path_ = this->create_publisher<nav_msgs::msg::Path>("smooth_trajectory", 10);

    // Visualization
    pub_viz_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "trajectory_markers", 10);
    const std::string ns_prefix = "/ugv_" + std::to_string(agent_id_);
    pub_planned_traj_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/planned_trajectory", 10);
    pub_contouring_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/contouring", 10);
    pub_contouring_current_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/contouring/current", 10);
    pub_contouring_path_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/contouring/path", 10);
    pub_contouring_points_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/contouring/points", 10);
    pub_goal_module_alias_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        ns_prefix + "/goal_module", 10);

    timer_ = this->create_wall_timer(100ms, [this]() { controlLoop(); });
    RCLCPP_INFO(this->get_logger(), "UGV-%d control loop: 10 Hz", agent_id_);
}

void UGVPlanner::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (!ego_.received)
        return;

    std::vector<Obstacle> obs;
    const double max_range = 4.0;
    const double min_range = 0.05;
    const size_t n = msg->ranges.size();
    const double a0 = msg->angle_min;
    const double ainc = msg->angle_increment;

    const double bin_rad = 3.0 * M_PI / 180.0;
    std::map<int, std::pair<double, std::pair<double, double>>> bins;
    for (size_t i = 0; i < n; ++i)
    {
        double r = msg->ranges[i];
        if (std::isnan(r) || std::isinf(r) || r <= min_range || r > max_range)
            continue;
        double a = a0 + static_cast<double>(i) * ainc;
        double ox = ego_.x + std::cos(ego_.psi + a) * r;
        double oy = ego_.y + std::sin(ego_.psi + a) * r;
        int bin = static_cast<int>(std::floor(a / bin_rad));
        auto it = bins.find(bin);
        if (it == bins.end() || r < it->second.first)
            bins[bin] = {r, {ox, oy}};
    }
    for (const auto &kv : bins)
    {
        obs.push_back(Obstacle{kv.second.second.first, kv.second.second.second, 0.15});
    }
    scan_obstacles_ = std::move(obs);
}

void UGVPlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    ego_.x = msg->pose.pose.position.x;
    ego_.y = msg->pose.pose.position.y;

    tf2::Quaternion q;
    tf2::fromMsg(msg->pose.pose.orientation, q);
    double roll, pitch;
    tf2::Matrix3x3(q).getRPY(roll, pitch, ego_.psi);

    ego_.v = msg->twist.twist.linear.x;
    ego_.omega = msg->twist.twist.angular.z;
    ego_.received = true;
    has_odom_ = true;

    if (metrics_written_ && has_goal_)
    {
        double dxg = msg->pose.pose.position.x - goal_x_;
        double dyg = msg->pose.pose.position.y - goal_y_;
        double dist_to_goal = std::sqrt(dxg * dxg + dyg * dyg);
        bool still_far = dist_to_goal > 40.0;
        if (still_far)
        {
            metrics_written_ = false;
            metrics_initialized_ = false;
            done_ = false;
            collision_static_ = false;
            min_static_clearance_ = std::numeric_limits<double>::max();
            episode_start_time_ = this->now();
            metrics_initialized_ = true;
            RCLCPP_WARN(this->get_logger(),
                "UGV-%d episode reset (was done but still far from goal, dist=%.1f)",
                agent_id_, dist_to_goal);
            reset_guard_until_ = this->now() + rclcpp::Duration::from_seconds(20.0);
            near_goal_cooldown_ = 200;
            return;
        }
    }

    if (first_run_)
    {
        double dxg = ego_.x - goal_x_;
        double dyg = ego_.y - goal_y_;
        double dist2goal = std::sqrt(dxg * dxg + dyg * dyg);
        if (ego_.x > pass_x_threshold_ || dist2goal < 0.35) {   // 单体: 接近到达才停 (原 40m 为仿真出生判定)
            RCLCPP_WARN(this->get_logger(),
                "UGV-%d spawn past threshold (x=%.1f, dist2goal=%.1f), skipping",
                agent_id_, ego_.x, dist2goal);
            first_run_ = false;
            metrics_written_ = true;
            return;
        }
        controller_.reset(ego_.omega);
        last_time_ = this->now();
        episode_start_time_ = last_time_;
        reset_guard_until_ = this->now() + rclcpp::Duration::from_seconds(20.0);
        near_goal_cooldown_ = 200;
        metrics_initialized_ = true;
        RCLCPP_INFO(this->get_logger(),
            "UGV-%d 初始: [%.2f, %.2f, %.1f°] cooldown=%d",
            agent_id_, ego_.x, ego_.y, ego_.psi * 180.0 / M_PI, near_goal_cooldown_);
        first_run_ = false;
        return;
    }
}

void UGVPlanner::pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    ReferencePath received_path;
    for (const auto &pose : msg->poses)
    {
        received_path.points.push_back(
            Eigen::Vector2d(pose.pose.position.x, pose.pose.position.y));
    }
    received_path.total_length = accumulatePathLength(received_path);
    has_reference_path_ = received_path.points.size() >= 2;
    if (has_reference_path_)
    {
        smooth_ref_path_ = std::move(received_path);
        selected_reference_active_ = true;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "UGV-%d received reference_path with %zu poses", agent_id_, msg->poses.size());
    }
    else
    {
        smooth_ref_path_.clear();
        selected_reference_active_ = false;
    }
}

void UGVPlanner::goalCallback(const geometry_msgs::msg::Point::SharedPtr msg)
{
    // 单体 UGV: 收到 /final_goal 即重置 episode (nav2 风格, 点相同 goal 也重导航).
    // 必须无条件重置: episode 超时后 metrics_written_=1 会永久空转, 同 goal 不再触发.
    if (has_goal_)
    {
        metrics_written_ = false;
        metrics_initialized_ = false;
        done_ = false;
        collision_static_ = false;
        min_static_clearance_ = std::numeric_limits<double>::max();
        episode_start_time_ = this->now();
        metrics_initialized_ = true;
        last_progress_dist_to_goal_ = std::numeric_limits<double>::max();
        near_goal_cooldown_ = 50;   // 5s 防抖, 避免刚重置就误判到达
        reset_guard_until_ = this->now() + rclcpp::Duration::from_seconds(3.0);
        RCLCPP_INFO(this->get_logger(), "UGV-%d 新目标, episode 已重置: [%.1f, %.1f]",
                    agent_id_, msg->x, msg->y);
    }
    goal_x_ = msg->x;
    goal_y_ = msg->y;
    has_goal_ = true;
    RCLCPP_INFO(this->get_logger(), "UGV-%d 目标: [%.1f, %.1f]", agent_id_, goal_x_, goal_y_);
}

ReferencePath UGVPlanner::buildLocalReferenceWindow(const ReferencePath &path) const
{
    if (!has_odom_ || path.points.size() < 2)
        return path;

    const double total = path.length();
    if (total < 1e-6)
        return path;

    const Eigen::Vector2d ego_pos(ego_.x, ego_.y);
    double s_near = 0.0;
    path.findClosest(ego_pos, s_near);
    s_near = std::clamp(s_near, 0.0, total);

    const double back_margin = 0.35;
    const double start_s = std::max(0.0, s_near - back_margin);
    const double step = std::clamp(
        mpc_solver_.params().desired_speed * mpc_solver_.params().dt,
        0.5, 1.2);

    ReferencePath local;
    local.nominal_speed = path.nominal_speed;
    local.points.push_back(path.sample(start_s));

    for (double s = start_s + step; s < total; s += step)
    {
        Eigen::Vector2d sample = path.sample(s);
        if ((sample - local.points.back()).norm() > 1e-3)
            local.points.push_back(sample);
    }

    const Eigen::Vector2d tail = path.sample(total);
    if ((tail - local.points.back()).norm() > 1e-3)
        local.points.push_back(tail);

    if (local.points.size() < 2)
        return path;

    local.total_length = accumulatePathLength(local);
    return local;
}

ReferencePath UGVPlanner::buildContouringReferenceWindow(const ReferencePath &path) const
{
    if (!has_odom_ || path.points.size() < 2)
        return path;

    const double total = path.length();
    if (total < 1e-6)
        return path;

    const Eigen::Vector2d ego_pos(ego_.x, ego_.y);
    double s_near = 0.0;
    path.findClosest(ego_pos, s_near);
    s_near = std::clamp(s_near, 0.0, total);

    // 单体 UGV 无动态障碍: danger 恒为 0
    const double danger = 0.0;

    const double back_margin = 0.35 + 0.30 * danger;
    const double start_s = std::max(0.0, s_near - back_margin);
    // UGV 短路径: 固定 0.08m 步长重采样, 保留 V-PRM 绕行路径细节
    // (原 0.40m 会把 2.6m 绕行路径重采样成 ~9 点, 样条拟合后绕行段丢失,
    //  MPC 只看到"前方障碍"→ 不敢走, 即"V-PRM 绕了但 MPC 不知道";
    //  且步长不能跟 desired_speed 走: speed=1.0 时 0.25m 仍会丢绕行细节)
    const double step = 0.08;
    const double forward_horizon = std::clamp(
        mpc_solver_.params().desired_speed * mpc_solver_.params().dt *
            static_cast<double>(mpc_solver_.params().N + 5) +
            5.0 * danger,
        12.0, 21.0);
    const double end_s = std::min(total, start_s + forward_horizon);

    ReferencePath local;
    local.nominal_speed = path.nominal_speed;
    local.points.push_back(path.sample(start_s));

    for (double s = start_s + step; s < end_s; s += step)
    {
        const Eigen::Vector2d sample = path.sample(s);
        if ((sample - local.points.back()).norm() > 1e-3)
            local.points.push_back(sample);
    }

    const Eigen::Vector2d tail = path.sample(end_s);
    if ((tail - local.points.back()).norm() > 1e-3)
        local.points.push_back(tail);

    if (local.points.size() < 2)
        return path;

    local.total_length = accumulatePathLength(local);
    return local;
}

void UGVPlanner::rebuildActiveReference()
{
    if (!(has_reference_path_ && smooth_ref_path_.points.size() >= 2))
    {
        selected_reference_active_ = false;
        ref_path_.clear();
        return;
    }

    selected_reference_active_ = true;
    ref_path_ = buildLocalReferenceWindow(smooth_ref_path_);
}

void UGVPlanner::controlLoop()
{
    // [DBG] entry
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "[DBG_ENTER] UGV-%d controlLoop has_odom=%d has_goal=%d metrics_written=%d ego=(%.1f,%.1f)",
        agent_id_, has_odom_, has_goal_, metrics_written_, ego_.x, ego_.y);
    if (!has_odom_ || !has_goal_)
    {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[DBG_SKIP] UGV-%d skip: odom=%d goal=%d", agent_id_, has_odom_, has_goal_);
        return;
    }

    last_time_ = this->now();
    updateEpisodeMetrics();

    double dxg = ego_.x - goal_x_;
    double dyg = ego_.y - goal_y_;
    double dist_to_goal = std::sqrt(dxg * dxg + dyg * dyg);
    double episode_elapsed = metrics_initialized_ ? (this->now() - episode_start_time_).seconds() : 0.0;

    if (!std::isfinite(last_progress_dist_to_goal_))
    {
        last_progress_dist_to_goal_ = dist_to_goal;
        last_progress_time_ = this->now();
    }
    else if (dist_to_goal < last_progress_dist_to_goal_ - 0.8)
    {
        last_progress_dist_to_goal_ = dist_to_goal;
        last_progress_time_ = this->now();
    }

    std::vector<Obstacle> planning_obstacles = static_obstacles_;
    planning_obstacles.insert(planning_obstacles.end(),
                              scan_obstacles_.begin(), scan_obstacles_.end());

    const double stop_clearance = dynamicStopClearance(controller_.params().stop_radius);
    const double slow_clearance = dynamicSlowClearance(mpc_solver_.params().obstacle_clearance);

    if (near_goal_cooldown_ > 0) {
        near_goal_cooldown_--;
    }

    bool near_goal = (dist_to_goal < 0.35)   // 单体: UGV 到达半径
                     && (this->now() > reset_guard_until_)
                     && (near_goal_cooldown_ == 0);
    if (near_goal && !metrics_written_)
    {
        RCLCPP_INFO(this->get_logger(), "UGV-%d arrived (dist=%.2f near=%d) ✓",
            agent_id_, dist_to_goal, (int)near_goal);
        done_ = true;
        writeEpisodeMetricsIfNeeded(false);

        std::string done_path = "trial_done_" + std::to_string(agent_id_) + ".flag";
        std::ofstream done_flag(done_path);
        if (done_flag.is_open())
            done_flag << "arrived " << std::fixed << std::setprecision(1) << episode_elapsed << "s\n";
    }

    if (!metrics_written_ && episode_elapsed > episode_timeout_s_)
    {
        RCLCPP_WARN(this->get_logger(), "EP timeout ugv=%d elapsed=%.1f x=%.1f dist=%.2f — forcing stop",
            agent_id_, episode_elapsed, ego_.x, dist_to_goal);
        done_ = true;
        writeEpisodeMetricsIfNeeded(true);

        std::string done_path = "trial_done_" + std::to_string(agent_id_) + ".flag";
        std::ofstream done_flag(done_path);
        if (done_flag.is_open())
            done_flag << "timeout " << std::fixed << std::setprecision(1) << episode_elapsed << "s\n";
    }

    // ── Pass-through detection @ x=25: signal experiment complete but keep running ──
    if (!passed_threshold_ && ego_.x > pass_x_threshold_)
    {
        passed_threshold_ = true;

        std::string done_path = "trial_done_" + std::to_string(agent_id_) + ".flag";
        std::ofstream done_flag(done_path);
        if (done_flag.is_open())
            done_flag << "passed " << std::fixed << std::setprecision(1) << episode_elapsed << "s\n";

        RCLCPP_INFO(this->get_logger(), "UGV-%d PASSED x=%.1f > %.1f at %.1f s — flag written, continuing to goal",
                    agent_id_, ego_.x, pass_x_threshold_, episode_elapsed);
    }

    if (!metrics_written_)
    {
        if (optimizing_.exchange(true))
        {
            auto stuck_duration = std::chrono::steady_clock::now() - solver_start_time_;
            if (stuck_duration > std::chrono::seconds(5))
            {
                RCLCPP_WARN(this->get_logger(), "UGV-%d solver STUCK for %.1fs — force reset",
                    agent_id_, std::chrono::duration<double>(stuck_duration).count());
                acados_solver_was_reset_ = true;
                optimizing_ = false;
            }
            else if (last_output_.size() > 0)
            {
                publishCommand(last_output_);
                publishTrajectoryPath(last_output_);
            }
            return;
        }
        solver_start_time_ = std::chrono::steady_clock::now();
        acados_solver_was_reset_ = false;

    const bool has_selected_ref = has_reference_path_ && smooth_ref_path_.points.size() >= 2;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
        "[DBG_REF] UGV-%d has_selected_ref=%d ref_pts=%zu metrics_written=%d",
        agent_id_, has_selected_ref,
        smooth_ref_path_.points.size(), metrics_written_);
    if (!has_selected_ref)
    {
        mpc_solver_.clearWarmstart();
        if (last_output_.size() > 0)
            last_output_.clear();
        last_predicted_traj_.clear();
        auto cmd = geometry_msgs::msg::Twist();
        pub_cmd_vel_->publish(cmd);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1500,
            "UGV-%d waiting reference path, hold position", agent_id_);
        return;
    }


    Trajectory mpc_traj;
    Trajectory final_traj;

    {
        rebuildActiveReference();
        if (ref_path_.points.size() < 2)
        {
            mpc_solver_.clearWarmstart();
            optimizing_ = false;
            return;
        }

        mpc_traj = runMPC();
        if (mpc_traj.size() == 0)
        {
            optimizing_ = false;
            return;
        }

        final_traj = mpc_traj;
    }

    for (int i = 0; i < final_traj.size() && i < mpc_traj.size(); ++i)
    {
        final_traj.points[i].v = std::min(final_traj.points[i].v, mpc_traj.points[i].v);
        final_traj.points[i].v = std::min(final_traj.points[i].v, mpc_solver_.params().max_vel);
    }

    static int cmp_log_count = 0;
    if (cmp_log_count++ < 20 && mpc_traj.size() > 1 && final_traj.size() > 1)
    {
        const auto &m1 = mpc_traj.points[1];
        const auto &f1 = final_traj.points[1];
        RCLCPP_INFO(this->get_logger(),
            "PATHCMP ctrl=%s | p1 mpc(%.2f,%.2f,%.2f,%.2f,%.2f) final(%.2f,%.2f,%.2f,%.2f,%.2f)",
            control_mode_.c_str(),
            m1.x, m1.y, m1.psi, m1.v, mpc_traj.points[0].omega,
            f1.x, f1.y, f1.psi, f1.v, final_traj.points[0].omega);
    }

    if (final_traj.size() >= 2)
    {
        const auto &tail = final_traj.points.back();
        const auto &prev = final_traj.points[final_traj.size() - 2];
        double goal_heading = std::atan2(goal_y_ - tail.y, goal_x_ - tail.x);
        double tail_seg_heading = std::atan2(tail.y - prev.y, tail.x - prev.x);
        double heading_err = Pose2D::normalizeAngle(tail_seg_heading - goal_heading);
        if (std::abs(heading_err) > 0.6)
        {
            final_traj.points.back().psi = goal_heading;
            final_traj.points.back().omega = 0.0;
        }
    }

    double nearest_obstacle_clearance = std::numeric_limits<double>::max();
    for (const auto &obs : planning_obstacles)
    {
        double clearance = std::hypot(ego_.x - obs.x, ego_.y - obs.y) -
                           (obs.radius + controller_.params().stop_radius);
        nearest_obstacle_clearance = std::min(nearest_obstacle_clearance, clearance);
    }

    const double hazard_clearance = nearest_obstacle_clearance;
    if (final_traj.size() > 0)
    {
        double min_speed_scale = 1.0;
        double min_path_clearance = std::numeric_limits<double>::max();
        const size_t limit = std::min<size_t>(4, final_traj.points.size());

        for (size_t i = 0; i < limit; ++i)
        {
            const auto &pt = final_traj.points[i];
            double point_clearance = std::numeric_limits<double>::max();
            for (const auto &obs : planning_obstacles)
            {
                double clearance = std::hypot(pt.x - obs.x, pt.y - obs.y) -
                                   (obs.radius + controller_.params().stop_radius);
                point_clearance = std::min(point_clearance, clearance);
            }

            if (!std::isfinite(point_clearance))
                continue;

            min_path_clearance = std::min(min_path_clearance, point_clearance);
            double point_scale = 1.0;
            if (point_clearance <= 0.0)
            {
                point_scale = 0.25;
            }
            else if (point_clearance <= stop_clearance * 0.5)
            {
                point_scale = 0.25 + 0.10 * static_cast<double>(i);
                point_scale = std::min(point_scale, 0.50);
            }
            else if (point_clearance <= stop_clearance)
            {
                point_scale = (i == 0) ? 0.20 : 0.30 + 0.10 * static_cast<double>(i);
                point_scale = std::min(point_scale, 0.65);
            }
            else if (point_clearance < slow_clearance)
            {
                point_scale = std::clamp(
                    (point_clearance - stop_clearance) / std::max(slow_clearance - stop_clearance, 1e-3),
                    0.35, 1.0);
            }
            min_speed_scale = std::min(min_speed_scale, point_scale);
        }

        if (std::isfinite(hazard_clearance))
        {
            if (hazard_clearance <= stop_clearance && min_speed_scale > 0.25)
                min_speed_scale = std::max(0.25, min_speed_scale * 0.50);
            else if (hazard_clearance < slow_clearance)
                min_speed_scale = std::min(min_speed_scale, std::clamp(
                    (hazard_clearance - stop_clearance) / std::max(slow_clearance - stop_clearance, 1e-3),
                    0.40, 1.0));
        }

        if (min_speed_scale < 1.0)
        {
            double speed_floor = 0.25;
            for (size_t i = 0; i < limit; ++i)
            {
                double ramp = (i == 0) ? min_speed_scale :
                    std::max(min_speed_scale, 0.35 + 0.15 * static_cast<double>(i));
                double point_scale = std::clamp(ramp, speed_floor, 1.0);
                final_traj.points[i].v *= point_scale;
            }
        }
    }

    last_output_ = final_traj;
    publishCommand(final_traj);
    publishTrajectoryPath(final_traj);
    publishVisualization(final_traj);

    // CSV per-step log
    if (csv_initialized_ && csv_out_.is_open())
    {
        double now_s = (this->now() - episode_start_time_).seconds();
        double dist_goal = (has_goal_ && std::isfinite(ego_.x) && std::isfinite(goal_x_))
            ? std::hypot(ego_.x - goal_x_, ego_.y - goal_y_) : 999.0;
        csv_out_ << std::fixed << std::setprecision(2)
            << now_s << ","
            << ego_.x << "," << ego_.y << "," << ego_.psi << ","
            << ego_.v << ",0.0," << ego_.omega << ","
            << acados_solver_.params().desired_speed << ","
            << dist_goal << ","
            << 999.0 << ","
            << final_traj.points[0].v << "," << final_traj.points[0].omega << "\n";
    }

    double log_clr_st = (std::isfinite(nearest_obstacle_clearance) && nearest_obstacle_clearance < 900.0) ? nearest_obstacle_clearance : 999.0;
    {
        const char *csv_env = std::getenv("MULTI_UGV_RESULTS_CSV");
        std::string csv_path = csv_env ? std::string(csv_env) : "results_multi_ugv.csv";
        int fd = ::open(csv_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd >= 0) {
            ::close(fd);
            std::ofstream hdr(csv_path, std::ios::app);
            if (hdr.is_open())
                hdr << "t,agent,coll_st,clr_st,speed\n";
        }
        std::ofstream csv(csv_path, std::ios::app);
        if (csv.is_open()) {
            csv << std::fixed << std::setprecision(3) << this->now().seconds() << ',' << agent_id_ << ','
                << (collision_static_?1:0) << ','
                << log_clr_st << ','
                << ego_.v << '\n';
        }
    }

    if (agent_id_ <= 4 && mpc_traj.size() > 1)
    {
        const auto &p1 = mpc_traj.points[1];
        double min_dist_barrier = 1e9;
        for (const auto &obs : planning_obstacles)
        {
            double d = std::hypot(p1.x - obs.x, p1.y - obs.y);
            double safe = obs.radius + mpc_solver_.params().obstacle_clearance;
            double margin = d - safe;
            min_dist_barrier = std::min(min_dist_barrier, margin);
        }
    }


    optimizing_ = false;
    } // end of if (!metrics_written_)
}

void UGVPlanner::updateEpisodeMetrics()
{
    for (const auto &obs : static_obstacles_)
    {
        double clearance = std::hypot(ego_.x - obs.x, ego_.y - obs.y) - obs.radius;
        min_static_clearance_ = std::min(min_static_clearance_, clearance);
        if (clearance <= 0.0)  // center inside obstacle → collision
            collision_static_ = true;
    }
}

void UGVPlanner::writeEpisodeMetricsIfNeeded(bool timeout)
{
    if (!metrics_initialized_ || metrics_written_) return;
    metrics_written_ = true;

    const double duration = (this->now() - episode_start_time_).seconds();
    const char *csv_env = std::getenv("MULTI_UGV_RESULTS_CSV");
    const std::string path = csv_env ? std::string(csv_env) : "results_multi_ugv.csv";
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return;
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd >= 0) {
        ::close(fd);
        out << "agent_id,duration_s,timeout,collision_static,min_static_clearance\n";
    }
    out << agent_id_ << ',' << duration << ',' << (timeout ? 1 : 0) << ','
        << (collision_static_ ? 1 : 0) << ','
        << min_static_clearance_ << '\n';
}

Trajectory UGVPlanner::runMPC()
{
    auto t0 = std::chrono::steady_clock::now();
    rebuildActiveReference();
    if (ref_path_.points.size() < 2)
        return Trajectory();
    auto t1 = std::chrono::steady_clock::now();

    ReferencePath contour_ref = buildContouringReferenceWindow(ref_path_);
    auto t2 = std::chrono::steady_clock::now();

    std::vector<Obstacle> obs_all = static_obstacles_;
    obs_all.insert(obs_all.end(), scan_obstacles_.begin(), scan_obstacles_.end());

    StageConstraintSet stage_cons = constraint_builder_.build(
        contour_ref, ego_, obs_all,
        acados_solver_.params().robot_radius,
        acados_solver_.params().obstacle_clearance,
        acados_solver_.params().dt, acados_solver_.params().N,
        -18.0, 18.0);

    const double ods = acados_solver_.params().desired_speed;

    auto t3 = std::chrono::steady_clock::now();
    Trajectory traj = acados_solver_.solve(ego_, contour_ref, stage_cons);
    auto t4 = std::chrono::steady_clock::now();

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[TIMING] UGV-%d ref=%.0fms win=%.0fms cons=%.0fms solve=%.0fms ncons=%zu refpts=%zu status=%s",
        agent_id_,
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        std::chrono::duration<double, std::milli>(t2 - t1).count(),
        std::chrono::duration<double, std::milli>(t3 - t2).count(),
        std::chrono::duration<double, std::milli>(t4 - t3).count(),
        stage_cons.size(), contour_ref.points.size(),
        acados_solver_.statusMessage().c_str());

    last_predicted_traj_ = traj;

    acados_solver_.params_mutable().desired_speed = ods;
    return traj;
}

void UGVPlanner::publishCommand(const Trajectory &traj)
{
    auto cmd = geometry_msgs::msg::Twist();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 800,
        "[PUB] direct=%d size=%d v1=%.3f w0=%.3f maxv=%.3f maxr=%.3f has_goal=%d",
        use_direct_cmd_ ? 1 : 0, traj.size(),
        traj.size() > 1 ? traj.points[1].v : -99.0,
        traj.size() > 0 ? traj.points[0].omega : -99.0,
        mpc_solver_.params().max_vel, controller_.params().max_r,
        has_goal_ ? 1 : 0);

    if (use_direct_cmd_ && traj.size() > 1)
    {
        // points[0] 被 x0 固定为当前状态(v=当前速度), 必须用第一预测步 points[1]
        cmd.linear.x = std::clamp(traj.points[1].v, 0.0, mpc_solver_.params().max_vel);
        cmd.angular.z = std::clamp(traj.points[0].omega, -controller_.params().max_r, controller_.params().max_r);
    }
    else if (use_direct_cmd_ && traj.size() > 0)
    {
        cmd.linear.x = std::clamp(traj.points[0].v, 0.0, mpc_solver_.params().max_vel);
        cmd.angular.z = std::clamp(traj.points[0].omega, -controller_.params().max_r, controller_.params().max_r);
    }
    else
    {
        double dt_seconds = 0.1;
        auto [u_cmd, r_cmd] = controller_.compute(traj, ego_, dt_seconds);
        cmd.linear.x = u_cmd;
        cmd.angular.z = r_cmd;
    }

    pub_cmd_vel_->publish(cmd);
}


void UGVPlanner::publishTrajectoryPath(const Trajectory &traj)
{
    (void)traj;
    auto path = nav_msgs::msg::Path();
    path.header.stamp = this->now();
    path.header.frame_id = "odom";

    const ReferencePath *global_path = nullptr;
    if (smooth_ref_path_.points.size() >= 2)
        global_path = &smooth_ref_path_;
    else if (ref_path_.points.size() >= 2)
        global_path = &ref_path_;

    if (!global_path)
    {
        pub_ref_path_->publish(path);
        return;
    }

    for (const auto &rp : global_path->points)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = path.header;
        ps.pose.position.x = rp.x();
        ps.pose.position.y = rp.y();
        path.poses.push_back(ps);
    }
    pub_ref_path_->publish(path);
}

void UGVPlanner::publishVisualization(const Trajectory &traj)
{
    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = this->now();

    auto agent_color = [&](float alpha = 1.0f) {
        std_msgs::msg::ColorRGBA c;
        if (agent_id_ == 1) {
            c.r = 0.95f; c.g = 0.30f; c.b = 0.30f;
        } else if (agent_id_ == 2) {
            c.r = 0.25f; c.g = 0.90f; c.b = 0.35f;
        } else if (agent_id_ == 3) {
            c.r = 0.25f; c.g = 0.45f; c.b = 0.95f;
        } else {
            c.r = 0.95f; c.g = 0.80f; c.b = 0.20f;
        }
        c.a = alpha;
        return c;
    };

    visualization_msgs::msg::Marker line;
    line.header.stamp = stamp;
    line.header.frame_id = "odom";
    line.ns = "trajectory";
    line.id = agent_id_ * 10;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = 0.22;
    line.color = agent_color(0.88f);

    visualization_msgs::msg::Marker nodes;
    nodes.header = line.header;
    nodes.ns = "trajectory_nodes";
    nodes.id = agent_id_ * 10 + 1;
    nodes.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    nodes.action = visualization_msgs::msg::Marker::ADD;
    nodes.pose.orientation.w = 1.0;
    nodes.scale.x = 0.34;
    nodes.scale.y = 0.34;
    nodes.scale.z = 0.10;
    nodes.color = agent_color(0.0f);

    visualization_msgs::msg::Marker heading;
    heading.header = line.header;
    heading.ns = "trajectory_heading";
    heading.id = agent_id_ * 10 + 2;
    heading.type = visualization_msgs::msg::Marker::LINE_LIST;
    heading.action = visualization_msgs::msg::Marker::ADD;
    heading.pose.orientation.w = 1.0;
    heading.scale.x = 0.08;
    heading.color = agent_color(0.55f);

    for (size_t i = 0; i < traj.points.size(); ++i)
    {
        const auto &pt = traj.points[i];
        const float t = traj.points.size() > 1 ? static_cast<float>(i) / static_cast<float>(traj.points.size() - 1) : 1.0f;

        geometry_msgs::msg::Point p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = 0.03;
        line.points.push_back(p);
        nodes.points.push_back(p);
        nodes.colors.push_back(agent_color(0.15f + 0.75f * t));

        geometry_msgs::msg::Point p2 = p;
        p2.x += 0.65 * std::cos(pt.psi);
        p2.y += 0.65 * std::sin(pt.psi);
        heading.points.push_back(p);
        heading.points.push_back(p2);
    }
    markers.markers.push_back(line);
    markers.markers.push_back(nodes);
    markers.markers.push_back(heading);

    pub_viz_->publish(markers);

    if (pub_planned_traj_alias_)
        pub_planned_traj_alias_->publish(markers);

    if (ref_path_.points.size() >= 2)
    {
        visualization_msgs::msg::MarkerArray contour_path;
        visualization_msgs::msg::Marker path_line;
        path_line.header.stamp = stamp;
        path_line.header.frame_id = "odom";
        path_line.ns = "contouring_path";
        path_line.id = 0;
        path_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        path_line.action = visualization_msgs::msg::Marker::ADD;
        path_line.pose.orientation.w = 1.0;
        path_line.scale.x = 0.12;
        path_line.color = agent_color(0.42f);

        visualization_msgs::msg::Marker path_points = path_line;
        path_points.ns = "contouring_points";
        path_points.id = 1;
        path_points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        path_points.scale.x = 0.18;
        path_points.scale.y = 0.18;
        path_points.scale.z = 0.05;
        path_points.color = agent_color(0.0f);

        for (size_t i = 0; i < ref_path_.points.size(); ++i)
        {
            const auto &rp = ref_path_.points[i];
            const float t = ref_path_.points.size() > 1 ? static_cast<float>(i) / static_cast<float>(ref_path_.points.size() - 1) : 1.0f;
            geometry_msgs::msg::Point p;
            p.x = rp.x();
            p.y = rp.y();
            p.z = 0.02;
            path_line.points.push_back(p);
            path_points.points.push_back(p);
            path_points.colors.push_back(agent_color(0.08f + 0.45f * t));
        }
        contour_path.markers.push_back(path_line);
        contour_path.markers.push_back(path_points);
        if (pub_contouring_alias_)
            pub_contouring_alias_->publish(contour_path);
        if (pub_contouring_path_alias_)
            pub_contouring_path_alias_->publish(contour_path);
        if (pub_contouring_points_alias_)
        {
            visualization_msgs::msg::MarkerArray contour_points_only;
            contour_points_only.markers.push_back(path_points);
            pub_contouring_points_alias_->publish(contour_points_only);
        }
    }

    visualization_msgs::msg::MarkerArray contour_current;
    visualization_msgs::msg::Marker current;
    current.header.stamp = stamp;
    current.header.frame_id = "odom";
    current.ns = "contouring_current";
    current.id = 0;
    current.type = visualization_msgs::msg::Marker::ARROW;
    current.action = visualization_msgs::msg::Marker::ADD;
    current.scale.x = 0.9;
    current.scale.y = 0.22;
    current.scale.z = 0.22;
    current.color.r = 0.95f;
    current.color.g = 0.75f;
    current.color.b = 0.15f;
    current.color.a = 0.95f;
    current.pose.position.x = ego_.x;
    current.pose.position.y = ego_.y;
    current.pose.position.z = 0.05;
    current.pose.orientation.z = std::sin(ego_.psi * 0.5);
    current.pose.orientation.w = std::cos(ego_.psi * 0.5);
    contour_current.markers.push_back(current);
    if (pub_contouring_current_alias_)
        pub_contouring_current_alias_->publish(contour_current);

    if (pub_goal_module_alias_ && has_goal_)
    {
        visualization_msgs::msg::MarkerArray goal_markers;

        visualization_msgs::msg::Marker goal;
        goal.header.stamp = stamp;
        goal.header.frame_id = "odom";
        goal.ns = "goal_module";
        goal.id = 0;
        goal.type = visualization_msgs::msg::Marker::SPHERE;
        goal.action = visualization_msgs::msg::Marker::ADD;
        goal.pose.position.x = goal_x_;
        goal.pose.position.y = goal_y_;
        goal.pose.position.z = 0.15;
        goal.pose.orientation.w = 1.0;
        goal.scale.x = 0.8;
        goal.scale.y = 0.8;
        goal.scale.z = 0.8;
        goal.color.r = 0.2f;
        goal.color.g = 0.95f;
        goal.color.b = 0.2f;
        goal.color.a = 0.95f;
        visualization_msgs::msg::Marker link = goal;
        link.ns = "goal_module_link";
        link.id = 1;
        link.type = visualization_msgs::msg::Marker::LINE_STRIP;
        link.scale.x = 0.08;
        link.color.r = 0.9f;
        link.color.g = 0.9f;
        link.color.b = 0.2f;
        link.color.a = 0.75f;
        link.points.clear();
        geometry_msgs::msg::Point p0;
        p0.x = ego_.x;
        p0.y = ego_.y;
        p0.z = 0.06;
        geometry_msgs::msg::Point p1;
        p1.x = goal_x_;
        p1.y = goal_y_;
        p1.z = 0.06;
        link.points.push_back(p0);
        link.points.push_back(p1);
        goal_markers.markers.push_back(link);

        pub_goal_module_alias_->publish(goal_markers);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UGVPlanner>());
    rclcpp::shutdown();
    return 0;
}
