/**
 * Compatibility wrapper: 把新版 acados_template (0.5.1, 带 model-hash 前缀) 生成的
 * API 映射回旧名 (contouring_unicycle_acados_*), 使 C++ 调用侧无需改动.
 * 注意: 重新生成求解器后此文件会被清掉, 需重新拷贝/创建.
 */
#pragma once

#include "acados_solver_ocp_contouring_unicycle_1f1aa62f.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef ocp_contouring_unicycle_1f1aa62f_solver_capsule contouring_unicycle_solver_capsule;

#define contouring_unicycle_acados_create_capsule ocp_contouring_unicycle_1f1aa62f_acados_create_capsule
#define contouring_unicycle_acados_free_capsule   ocp_contouring_unicycle_1f1aa62f_acados_free_capsule
#define contouring_unicycle_acados_create         ocp_contouring_unicycle_1f1aa62f_acados_create
#define contouring_unicycle_acados_free           ocp_contouring_unicycle_1f1aa62f_acados_free
#define contouring_unicycle_acados_solve          ocp_contouring_unicycle_1f1aa62f_acados_solve
#define contouring_unicycle_acados_update_params  ocp_contouring_unicycle_1f1aa62f_acados_update_params
#define contouring_unicycle_acados_get_nlp_config ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_config
#define contouring_unicycle_acados_get_nlp_dims   ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_dims
#define contouring_unicycle_acados_get_nlp_in     ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_in
#define contouring_unicycle_acados_get_nlp_out    ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_out
#define contouring_unicycle_acados_get_nlp_solver ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_solver
#define contouring_unicycle_acados_get_nlp_opts   ocp_contouring_unicycle_1f1aa62f_acados_get_nlp_opts

/* 旧 2 参 reset (reset_qp_solver_mem) → 新 5 参 reset */
static inline int contouring_unicycle_acados_reset(contouring_unicycle_solver_capsule *capsule, int reset_qp_solver_mem)
{
    return ocp_contouring_unicycle_1f1aa62f_acados_reset(capsule, reset_qp_solver_mem, 0, 0, 0);
}

#ifdef __cplusplus
}
#endif
