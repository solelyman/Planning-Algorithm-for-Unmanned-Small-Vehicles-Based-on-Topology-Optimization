/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SIM_ocp_contouring_unicycle_1f1aa62f_H_
#define ACADOS_SIM_ocp_contouring_unicycle_1f1aa62f_H_

#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#define OCP_CONTOURING_UNICYCLE_1F1AA62F_NX     5
#define OCP_CONTOURING_UNICYCLE_1F1AA62F_NZ     0
#define OCP_CONTOURING_UNICYCLE_1F1AA62F_NU     2
#define OCP_CONTOURING_UNICYCLE_1F1AA62F_NP     126

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule
{
    // acados objects
    sim_in *acados_sim_in;
    sim_out *acados_sim_out;
    sim_solver *acados_sim_solver;
    sim_opts *acados_sim_opts;
    sim_config *acados_sim_config;
    void *acados_sim_dims;
    void *acados_sim_mem;

    /* external functions */
    // ERK
    external_function_param_casadi * sim_expl_vde_forw;
    external_function_param_casadi * sim_vde_adj_casadi;
    external_function_param_casadi * sim_expl_ode_fun_casadi;
    external_function_param_casadi * sim_expl_ode_hess;
    external_function_param_casadi * sim_expl_vde_forw_p;

    // IRK
    external_function_param_casadi * sim_impl_dae_fun;
    external_function_param_casadi * sim_impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi * sim_impl_dae_jac_x_xdot_u_z;
    external_function_param_casadi * sim_impl_dae_hess;
    external_function_param_casadi * sim_impl_dae_jac_p;

    // GNSF
    external_function_param_casadi * sim_gnsf_phi_fun;
    external_function_param_casadi * sim_gnsf_phi_fun_jac_y;
    external_function_param_casadi * sim_gnsf_phi_jac_y_uhat;
    external_function_param_casadi * sim_gnsf_f_lo_jac_x1_x1dot_u_z;
    external_function_param_casadi * sim_gnsf_get_matrices_fun;

} ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule;


ACADOS_SYMBOL_EXPORT int ocp_contouring_unicycle_1f1aa62f_acados_sim_create(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT int ocp_contouring_unicycle_1f1aa62f_acados_sim_solve(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int ocp_contouring_unicycle_1f1aa62f_acados_sim_free(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT int ocp_contouring_unicycle_1f1aa62f_acados_sim_update_params(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule, double *value, int np);

ACADOS_SYMBOL_EXPORT sim_config * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_config(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_in * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_in(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_out * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_out(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT void * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_dims(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_opts * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_opts(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_solver * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_solver(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT void * ocp_contouring_unicycle_1f1aa62f_acados_get_sim_mem(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule * ocp_contouring_unicycle_1f1aa62f_acados_sim_solver_create_capsule(void);
ACADOS_SYMBOL_EXPORT int ocp_contouring_unicycle_1f1aa62f_acados_sim_solver_free_capsule(ocp_contouring_unicycle_1f1aa62f_sim_solver_capsule *capsule);

#ifdef __cplusplus
}
#endif

#endif  // ACADOS_SIM_ocp_contouring_unicycle_1f1aa62f_H_
