#ifndef MULTI_UGV_PLANNER_H
#define MULTI_UGV_PLANNER_H

#include <multi_ugv_planner/types.h>
#include <multi_ugv_planner/mpc_solver.h>
#include <multi_ugv_planner/acados_contouring_solver.h>
#include <multi_ugv_planner/controller.h>
#include <multi_ugv_planner/constraint_builder.h>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <vector>
#include <string>
#include <atomic>
#include <fstream>
#include <limits>

namespace MultiUGV
{
class UGVPlanner : public rclcpp::Node
{
public:
    UGVPlanner();

private:
    void declareParams();
    void loadParams();
    void initCommunication();
    void controlLoop();

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void goalCallback(const geometry_msgs::msg::Point::SharedPtr msg);

    void publishCommand(const Trajectory &traj);
    void publishVisualization(const Trajectory &traj);
    void publishTrajectoryPath(const Trajectory &traj);
    void updateEpisodeMetrics();
    void writeEpisodeMetricsIfNeeded(bool timeout = false);
    void rebuildActiveReference();
    ReferencePath buildLocalReferenceWindow(const ReferencePath &path) const;
    ReferencePath buildContouringReferenceWindow(const ReferencePath &path) const;
    Trajectory runMPC();

    //State
    int agent_id_ = 0;
    UGVState ego_;
    double goal_x_ = 100.0, goal_y_ = 0.0;
    bool has_goal_ = false;
    bool has_odom_ = false;
    bool first_run_ = true;
    rclcpp::Time reset_guard_until_{0, 0, RCL_ROS_TIME};
    int near_goal_cooldown_ = 0;
    bool use_direct_cmd_ = false;
    std::string control_mode_ = "los";
    rclcpp::Time last_time_;

    //Obstacle list
    std::vector<Obstacle> static_obstacles_;
    std::vector<Obstacle> scan_obstacles_;

    // Solver & Controller
    MPCSolver mpc_solver_;
    AcadosContouringSolver acados_solver_;
    ReferencePath ref_path_;
    ReferencePath smooth_ref_path_;
    bool has_reference_path_ = false;
    bool selected_reference_active_ = false;
    double last_progress_dist_to_goal_ = std::numeric_limits<double>::infinity();
    rclcpp::Time last_progress_time_{0, 0, RCL_ROS_TIME};
    UGVController controller_;

    ConstraintBuilder constraint_builder_;

    std::atomic<bool> optimizing_{false};
    std::chrono::steady_clock::time_point solver_start_time_;
    bool acados_solver_was_reset_ = false;

private:
    Trajectory last_output_;
    Trajectory last_predicted_traj_;  // full MPC prediction for DR anchor projection
    rclcpp::Time episode_start_time_;
    bool metrics_initialized_ = false;
    bool metrics_written_ = false;
    bool passed_threshold_ = false;
    bool done_ = false;  // true after passing obstacle zone → x>pass_x_threshold
    std::ofstream csv_out_;
    bool csv_initialized_ = false;
    double pass_x_threshold_ = 55.0;
    bool collision_static_ = false;
    double min_static_clearance_ = std::numeric_limits<double>::max();
    double episode_timeout_s_ = 600.0;  // generous — DYN_SPEED may crawl near obstacles

    //Publisher
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_ref_path_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_viz_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_planned_traj_alias_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_contouring_alias_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_contouring_current_alias_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_contouring_path_alias_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_contouring_points_alias_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_goal_module_alias_;
    rclcpp::Time last_pass_warn_{0, 0, RCL_ROS_TIME};

    //Subscriber
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_goal_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;

    rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace MultiUGV

#endif
