#!/usr/bin/env python3
"""5-state, 2-control, EXTERNAL+EXACT+MIRROR, hard linear half-space constraints.

Obstacle avoidance: con_h_expr h = a1*x + a2*y - b <= 0 (mpc_planner-main style
LinearizedConstraints). a1/a2/b are realtime params, re-linearized each solve.
Ellipsoid soft penalty kept in cost but weight_obstacle=0 when linear is on.
NP = 8(weights) + 45(spline) + 1(w_obs) + 36(ellipsoid 12×3) + 36(lin 12×3) = 126
"""

import os, sys, shutil
import numpy as np
import casadi as cd

_acados_iface = os.path.join(os.path.expanduser("~"), ".local", "share",
                              "acados", "interfaces", "acados_template")
if _acados_iface not in sys.path: sys.path.insert(0, _acados_iface)

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

N_SEG = 5
N_ELLIPSOIDS = 12
N_LIN = 12
NP = 8 + N_SEG * 9 + 1 + N_ELLIPSOIDS * 3 + N_LIN * 3  # = 8 + 45 + 1 + 36 + 36 = 126
NC = N_LIN  # con_h_expr: hard linear half-spaces

# Parameter layout
# [0..7]    = 8 weight/reference params
# [8..52]   = 45 spline params
# [53]      = weight_obstacle
# [54..89]  = 12 ellipsoids × 3 (ox, oy, r)
# [90..125] = 12 linear halfspaces × 3 (a1, a2, b)

_OBS_WEIGHT_IDX = 53
_ELL_OFS = 54  # ellipsoid param offset


def build_param_map():
    pm = {}; idx = 0
    for nm in ["acceleration","angular_velocity","velocity",
               "reference_velocity","contour","lag",
               "terminal_angle","terminal_contouring"]:
        pm[nm]=idx; idx+=1
    for i in range(N_SEG):
        for ax in "abcd": pm[f"spline_x{i}_{ax}"]=idx; idx+=1
        for ax in "abcd": pm[f"spline_y{i}_{ax}"]=idx; idx+=1
        pm[f"spline{i}_start"]=idx; idx+=1
    pm["obstacle_weight"] = idx; idx += 1  # = 53
    for i in range(N_ELLIPSOIDS):
        pm[f"ell{i}_ox"] = idx; idx += 1
        pm[f"ell{i}_oy"] = idx; idx += 1
        pm[f"ell{i}_r"]  = idx; idx += 1
    for i in range(N_LIN):
        pm[f"lin{i}_a1"] = idx; idx += 1
        pm[f"lin{i}_a2"] = idx; idx += 1
        pm[f"lin{i}_b"]  = idx; idx += 1
    return pm


def default_parameters():
    p = np.zeros(NP)
    pm = build_param_map()
    p[pm["reference_velocity"]] = 1.0
    p[pm["obstacle_weight"]] = 1.0
    # ellipsoids default to r=0 (no penalty)
    # linear halfspaces default to x - 100 <= 0 (vacuous)
    for i in range(N_LIN):
        p[pm[f"lin{i}_a1"]] = 1.0
        p[pm[f"lin{i}_a2"]] = 0.0
        p[pm[f"lin{i}_b"]] = 100.0
    return p


class SplineSeg:
    __slots__ = ("a","b","c","d","s_start")
    def __init__(self, p, pm, nr):
        self.a = p[pm[f"spline_x{nr}_a"]]; self.b = p[pm[f"spline_x{nr}_b"]]
        self.c = p[pm[f"spline_x{nr}_c"]]; self.d = p[pm[f"spline_x{nr}_d"]]
        self.s_start = p[pm[f"spline{nr}_start"]]
    def at(self,s): ds=s-self.s_start; return self.a*ds**3+self.b*ds**2+self.c*ds+self.d
    def deriv(self,s): ds=s-self.s_start; return 3*self.a*ds**2+2*self.b*ds+self.c
class SplineSegY:
    __slots__ = ("a","b","c","d","s_start")
    def __init__(self, p, pm, nr):
        self.a = p[pm[f"spline_y{nr}_a"]]; self.b = p[pm[f"spline_y{nr}_b"]]
        self.c = p[pm[f"spline_y{nr}_c"]]; self.d = p[pm[f"spline_y{nr}_d"]]
        self.s_start = p[pm[f"spline{nr}_start"]]
    def at(self,s): ds=s-self.s_start; return self.a*ds**3+self.b*ds**2+self.c*ds+self.d
    def deriv(self,s): ds=s-self.s_start; return 3*self.a*ds**2+2*self.b*ds+self.c

def build_spline(pv, pm, s, N):
    sx=[SplineSeg(pv,pm,i) for i in range(N)]
    sy=[SplineSegY(pv,pm,i) for i in range(N)]
    lam=[1/(1+cd.exp((s-sx[i].s_start+0.02)/0.1)) for i in range(1,N)]
    px=sx[-1].at(s); pdx=sx[-1].deriv(s)
    for k in range(N-2,-1,-1): lk=lam[k]; px=lk*sx[k].at(s)+(1-lk)*px; pdx=lk*sx[k].deriv(s)+(1-lk)*pdx
    py=sy[-1].at(s); pdy=sy[-1].deriv(s)
    for k in range(N-2,-1,-1): lk=lam[k]; py=lk*sy[k].at(s)+(1-lk)*py; pdy=lk*sy[k].deriv(s)+(1-lk)*pdy
    nrm=cd.sqrt(pdx*pdx+pdy*pdy+1e-12)
    return px,py,pdx/nrm,pdy/nrm

def haar_diff(a,b): return cd.atan2(cd.sin(a-b),cd.cos(a-b))

def build_model():
    m=AcadosModel(); m.name="contouring_unicycle"
    x,y,psi,v,s=(cd.SX.sym(nm) for nm in["x","y","psi","v","s"])
    a_ctrl,w_ctrl=cd.SX.sym("a"),cd.SX.sym("w")
    m.x=cd.vertcat(x,y,psi,v,s); m.u=cd.vertcat(a_ctrl,w_ctrl)
    f=cd.vertcat(v*cd.cos(psi),v*cd.sin(psi),w_ctrl,a_ctrl,v)
    m.f_expl_expr=f
    xdot=cd.SX.sym("xdot",5); m.xdot=xdot; m.f_impl_expr=xdot-f
    m.p=cd.SX.sym("p",NP)
    # Hard linear half-space constraints h = a1*x + a2*y - b <= 0 (mpc_planner
    # LinearizedConstraints). a1/a2/b are realtime params; set per stage in C++.
    pm = build_param_map()
    h_list = []
    for i in range(N_LIN):
        a1 = m.p[pm[f"lin{i}_a1"]]
        a2 = m.p[pm[f"lin{i}_a2"]]
        b  = m.p[pm[f"lin{i}_b"]]
        h_list.append(a1 * x + a2 * y - b)
    m.con_h_expr = cd.vertcat(*h_list)
    return m

def build_cost(model):
    """Scalar EXTERNAL cost: path tracking + ellipsoid obstacle soft penalty.

    Obstacle penalty uses smooth softplus activation:
      d_sq = (x-ox)^2 + (y-oy)^2
      violation = r^2 - d_sq        (positive when inside)
      smooth_v = softplus(10*violation)/10
      cost += w_obs * smooth_v^2
    When r=0 (dummy), violation always < 0 → no penalty.
    """
    x=model.x[0]; y=model.x[1]; psi=model.x[2]; v=model.x[3]; s=model.x[4]
    a=model.u[0]; w=model.u[1]; pv=model.p; pm=build_param_map()
    px,py,pdx,pdy=build_spline(pv,pm,s,N_SEG)
    contour=pdy*(x-px)-pdx*(y-py)
    lag=pdx*(x-px)+pdy*(y-py)
    angle_err=haar_diff(psi,cd.atan2(pdy,pdx))
    wa=pv[pm["acceleration"]]; ww=pv[pm["angular_velocity"]]; wv=pv[pm["velocity"]]
    vr=pv[pm["reference_velocity"]]; wc=pv[pm["contour"]]; wl=pv[pm["lag"]]
    wta=pv[pm["terminal_angle"]]
    wobs=pv[pm["obstacle_weight"]]

    # Path tracking cost (unconstrained → 0 MINSTEP)
    cost_stage=wa*a*a+ww*w*w+wv*(v-vr)**2+wc*contour**2+wl*lag**2
    cost_terminal=wta*angle_err*angle_err

    # Obstacle soft penalty (smooth, always differentiable)
    ell_penalty = 0.0
    for i in range(N_ELLIPSOIDS):
        ox = pv[pm[f"ell{i}_ox"]]
        oy = pv[pm[f"ell{i}_oy"]]
        r  = pv[pm[f"ell{i}_r"]]
        # Squared distance
        d_sq = (x - ox)*(x - ox) + (y - oy)*(y - oy)
        r_sq = r*r + 1e-12
        # Violation = r_sq - d_sq (positive when inside obstacle)
        violation = r_sq - d_sq
        # Smooth max(0, violation) via softplus
        smooth_v = cd.log(1 + cd.exp(10 * violation)) / 10.0
        ell_penalty += smooth_v * smooth_v

    cost_stage += wobs * ell_penalty

    model.cost_expr_ext_cost = cost_stage
    model.cost_expr_ext_cost_e = cost_terminal

def main():
    N=10; DT=0.4
    OUT_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..","acados","generated","contouring_solver")
    model=build_model(); build_cost(model)
    ocp=AcadosOcp(); ocp.model=model; ocp.dims.N=N
    ocp.cost.cost_type="EXTERNAL"
    ocp.constraints.x0=np.zeros(5)
    ocp.constraints.lbx=np.array([-2000,-2000,-4*np.pi,-0.01,-1.0])
    ocp.constraints.ubx=np.array([2000,2000,4*np.pi,1.9,1000])
    ocp.constraints.idxbx=np.array(range(5))
    ocp.constraints.lbu=np.array([-2.0,-0.8])
    ocp.constraints.ubu=np.array([2.0,0.8])
    ocp.constraints.idxbu=np.array(range(2))
    # Hard linear half-space constraints: -1e6 <= h <= 0 (JSON 不能序列化 -inf)
    ocp.constraints.lh=-1e6*np.ones(N_LIN)
    ocp.constraints.uh=np.zeros(N_LIN)
    ocp.parameter_values=default_parameters()
    ocp.solver_options.tf=N*DT
    ocp.solver_options.integrator_type="ERK"
    ocp.solver_options.sim_method_num_stages=4
    ocp.solver_options.sim_method_num_steps=3
    ocp.solver_options.nlp_solver_type="SQP_RTI"
    ocp.solver_options.hessian_approx="EXACT"
    ocp.solver_options.regularize_method="MIRROR"
    ocp.solver_options.qp_solver="PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max=50
    ocp.solver_options.qp_solver_warm_start=2
    ocp.solver_options.tol=1e-2
    ocp.solver_options.qp_tol=1e-5
    ocp.solver_options.print_level=0
    ocp.code_export_directory=OUT_DIR
    os.makedirs(OUT_DIR,exist_ok=True)
    for f in os.listdir(OUT_DIR) if os.path.isdir(OUT_DIR) else []:
        fp=os.path.join(OUT_DIR,f)
        (shutil.rmtree if os.path.isdir(fp) else os.remove)(fp)
    print(f"--- Generating acados solver ---")
    print(f"  5-state contouring, N={N}, dt={DT}")
    print(f"  EXTERNAL+EXACT+MIRROR, NP={NP}, ellipsoid[{N_ELLIPSOIDS}] + hard linear[{N_LIN}]")
    solver=AcadosOcpSolver(ocp=ocp, json_file=os.path.join(OUT_DIR,f"{model.name}.json"))
    print(f"  Output: {OUT_DIR}")

if __name__=="__main__": main()
