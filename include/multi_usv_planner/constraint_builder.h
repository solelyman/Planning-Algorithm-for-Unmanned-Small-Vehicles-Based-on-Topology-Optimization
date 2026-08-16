#ifndef MULTI_USV_CONSTRAINT_BUILDER_H
#define MULTI_USV_CONSTRAINT_BUILDER_H

#include <multi_usv_planner/types.h>
#include <multi_usv_planner/acados_contouring_solver.h>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <algorithm>

namespace MultiUSV
{

/**
 * Generates per-stage ellipsoid constraints for static obstacles.
 *
 * Each obstacle → EllipsoidObstacle (circle):
 *   (x-ox)² + (y-oy)² - r² <= 0
 * where r = obs.radius + robot_radius + clearance.
 *
 * Bounded con_h_expr constraints in acados solver, handled by multi-RTI loop.
 */
class ConstraintBuilder
{
public:
    StageConstraintSet build(
        const ReferencePath &contour_ref,
        const USVState &ego,
        const std::vector<Obstacle> &static_obstacles,
        double robot_radius,
        double obstacle_clearance,
        double dt,
        int N,
        double corridor_y_min = -18.0,
        double corridor_y_max = 18.0);
};

}  // namespace MultiUSV

#endif  // MULTI_USV_CONSTRAINT_BUILDER_H
