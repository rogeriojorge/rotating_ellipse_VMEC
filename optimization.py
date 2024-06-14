#!/usr/bin/env python

import os
import glob
import time
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd import Vmec
from scipy.optimize import minimize
from simsopt import make_optimizable
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition, proc0_print, comm_world
from vmecPlot2 import main as vmecPlot2_main
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field import (InterpolatedField, SurfaceClassifier,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.geo import (curves_to_vtk, create_equally_spaced_curves, LpCurveCurvature, LinkingNumber,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier)
mpi = MpiPartition()
this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)
######################
## Choose below the flags to run stage 1 and stage 2 optimization
## RUN with: mpirun -n 9 python optimization.py for stage 1 and mpirun -n 1 python optimization.py for stage 2
run_stage_1 = False
run_stage_2 = False
do_Poincare = True
######################
### INPUT PARAMETERS
max_mode = 1
target_aspect_ratio = 5
target_iota = 0.41
max_nfev = 40
ncoils = 4
order = 12
LENGTH_CON_WEIGHT = 0.1
LENGTH_THRESHOLD = 3.5
CC_THRESHOLD = 0.05
CC_WEIGHT = 100
CURVATURE_THRESHOLD = 10
CURVATURE_WEIGHT = 1e-2
MSC_THRESHOLD = 10
MSC_WEIGHT = 1e-2
MAXITER = 600
nphi=32
ntheta=32
nfieldlines = 6
tmax_fl = 500
#############################################
if run_stage_1:
    proc0_print("#### STAGE 1 Optimization ####")
    filename = os.path.join(os.path.dirname(__file__), 'input.nfp5')
    vmec = Vmec(filename, mpi=mpi, verbose=False, ntheta=ntheta, nphi=nphi, range_surface='half period')
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    vmec.indata.mpol = max_mode+2
    vmec.indata.ntor = max_mode+2
    def circular_axis_objective(vmec):
        rc = vmec.wout.raxis_cc
        zs = vmec.wout.zaxis_cs
        return np.sum(np.abs(np.concatenate((rc[1:], zs)))**2)
    circular_axis_optimizable = make_optimizable(circular_axis_objective, vmec)
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, target_aspect_ratio, 1),
                                            (vmec.iota_axis, target_iota, 1),
                                            (circular_axis_optimizable.J, 0, 1e3)])
    prob.objective()
    proc0_print("Initial aspect ratio:", vmec.aspect())
    proc0_print("Initial iota on-axis:", vmec.iota_axis())
    proc0_print("Initial circular axis objective:", circular_axis_optimizable.J())
    proc0_print("Total objective before optimization:", prob.objective())
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-4, abs_step=1e-7, max_nfev=max_nfev)
    prob.objective()
    proc0_print("Final aspect ratio:", vmec.aspect())
    proc0_print("Final iota on-axis:", vmec.iota_axis())
    proc0_print("Final circular axis objective:", circular_axis_optimizable.J())
    proc0_print("Total objective after optimization:", prob.objective())
    mpi.comm_world.Barrier()
    if comm_world is None or comm_world.rank == 0:
        try:
            for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
            for obj_file in glob.glob("objective_*"): os.remove(obj_file)
            for obj_file in glob.glob("input.nfp5_*"): os.remove(obj_file)
            for obj_file in glob.glob("wout_*"): os.remove(obj_file)
        except: pass
        vmec.indata.ns_array[:3]    = [  16,    51,    101]
        vmec.indata.niter_array[:3] = [ 300,   500,  20000]
        vmec.indata.ftol_array[:3]  = [ 1e-9, 1e-10, 1e-14]
        vmec.indata.phiedge = 1 / vmec.wout.volavgB
        vmec.write_input("input.nfp5_opt")
        vmec_final = Vmec("input.nfp5_opt", mpi=mpi, verbose=False)
        vmec_final.run()
        vmecPlot2_main(vmec_final.output_file)
        try:
            for obj_file in glob.glob("parvmec*"): os.remove(obj_file)
            for obj_file in glob.glob("threed*"): os.remove(obj_file)
            for obj_file in glob.glob("jxbout*"): os.remove(obj_file)
            for obj_file in glob.glob("threed*"): os.remove(obj_file)
            for obj_file in glob.glob("mercier*"): os.remove(obj_file)
            for obj_file in glob.glob("input.nfp5_opt_*"): os.remove(obj_file)
        except: pass
    mpi.comm_world.Barrier()
else:
    filename = os.path.join(os.path.dirname(__file__), 'input.nfp5_opt')
    vmec = Vmec(filename, mpi=mpi, verbose=False, ntheta=ntheta, nphi=nphi, range_surface='half period')
    surf = vmec.boundary
    vmec.run()
## Set big surface for stage 2 and Poincare
nphi_big   = nphi * 2 * surf.nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi   = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(dofs=surf.dofs, nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta, stellsym=surf.stellsym)
#############################################
proc0_print("#### STAGE 2 Optimization ####")
if run_stage_2 and (comm_world is None or comm_world.rank == 0):
    R0_coils = np.sum(vmec.wout.raxis_cc)
    R1_coils = np.min((vmec.wout.Aminor_p*2.6,R0_coils/1.4))
    base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0_coils, R1=R1_coils, order=order)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    total_current_vmec = vmec.external_current() / (2 * surf.nfp)
    base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils - 1)]
    total_current = Current(total_current_vmec)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if comm_world is None or comm_world.rank == 0:
        curves_to_vtk(curves, "curves_init", close=True)
        curves_to_vtk(base_curves, "base_curves_init", close=True)
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        Bmod = bs.AbsB().reshape((nphi,ntheta,1))
        surf.to_vtk(os.path.join(this_path,"surf_init"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    proc0_print(f'Initializing stage 2 objective function')
    Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jls = [CurveLength(c) for c in base_curves]
    JF = Jf \
        + LENGTH_CON_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls) \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
        + LinkingNumber(curves, 2)
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        outstr = f" J={J:.1e}, Jf={jf:.1e}, max(B·n)/B={np.max(np.abs(BdotN)):.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad
    f = fun
    dofs = JF.x
    proc0_print(f'Performing stage 2 optimization')
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    print(res.message)
    curves_to_vtk(curves, "curves_opt", close=True)
    curves_to_vtk(base_curves, "base_curves_opt", close=True)
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi,ntheta,1))
    surf.to_vtk(os.path.join(this_path,"surf_opt"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi_big,ntheta_big,1))
    surf_big.to_vtk(os.path.join(this_path, "surf_opt_big"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    bs.save(os.path.join(this_path,"biot_savart_opt.json"))
mpi.comm_world.Barrier()
###### Poincare plot
if do_Poincare:
    proc0_print(f'Defining Poincaré plot functions')
    bs = load(os.path.join(this_path,"biot_savart_opt.json"))
    R_theta0_phi0_array = np.sort(np.sum(vmec.wout.rmnc,axis=0))
    indices_to_plot = np.array(np.linspace(1,len(R_theta0_phi0_array)-1,nfieldlines),dtype=int)
    R0 = R_theta0_phi0_array[indices_to_plot]
    degree = 4
    sc_fieldline = SurfaceClassifier(surf_big, h=0.1, p=2)
    # sc_fieldline.to_vtk('levelset', h=0.1)
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/surf.nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-14, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            # particles_to_vtk(fieldlines_tys, f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, f'poincare_fieldline_{label}.png', dpi=150, surf=surf_big)
        return fieldlines_phi_hits
    n = 40
    rs = np.linalg.norm(surf_big.gamma()[:, :, 0:2], axis=2)
    zs = surf_big.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/surf_big.nfp, n*2)
    zrange = (0, np.max(zs), n//2)
    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.3).flatten())
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip
    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=surf_big.nfp, stellsym=True, skip=skip)
    proc0_print('Done initializing InterpolatedField.')
    bsh.set_points(surf_big.gamma().reshape((-1, 3)))
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))
    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    proc0_print('Beginning field line tracing')
    fieldlines_phi_hits = trace_fieldlines(bsh, 'bsh')