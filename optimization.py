#!/usr/bin/env python

import os
import glob
import numpy as np
from mpi4py import MPI
from simsopt.mhd import Vmec
from simsopt import make_optimizable
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
from vmecPlot2 import main as vmecPlot2_main
mpi = MpiPartition()
### INPUT PARAMETERS
max_mode = 1
target_aspect_ratio = 5
target_iota = 0.41
max_nfev = 30
### END INPUT PARAMETERS
filename = os.path.join(os.path.dirname(__file__), 'input.nfp5')
vmec = Vmec(filename, mpi=mpi, verbose=False)
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
                                        (vmec.mean_iota, target_iota, 1),
                                        (circular_axis_optimizable.J, 0, 1)])
prob.objective()
proc0_print("Initial aspect ratio:", vmec.aspect())
proc0_print("Initial iota:", vmec.mean_iota())
proc0_print("Initial circular axis objective:", circular_axis_optimizable.J())
proc0_print("Total objective before optimization:", prob.objective())
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-4, abs_step=1e-7, max_nfev=max_nfev)
prob.objective()
proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Final iota:", vmec.mean_iota())
proc0_print("Final circular axis objective:", circular_axis_optimizable.J())
proc0_print("Total objective after optimization:", prob.objective())
mpi.comm_world.Barrier()
if MPI.COMM_WORLD.rank==0:
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