#!/bin/bash

# method="CBCS"
# method="EBES"
# method="EBES_new"
# operation="Slater"
# operation="Jastrow"
# operation="Backflow"
# operation="Jastrow_varmin"
# operation="Jastrow_emin"
# operation="Backflow_varmin"
# operation="Backflow_emin"
# operation="Jastrow_dmc"
# operation="Backflow_dmc"
# operation="Geminal"
# operation="Gjastrow"
# operation="Gjastrow_emin"

# path="gwfn/He/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/Be/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/${method}/${operation}/"
# path="gwfn/B/MP2-CASSCF(3.8)/def2-QZVP/${method}/${operation}/"
# path="gwfn/N/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/Ne/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/Ar/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/Kr/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/O3/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/B2H6/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/HF/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/H2O/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/NH3/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/CH4/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/C2H2/HF/cc-pVQZ/${method}/${operation}/"
# path="gwfn/C4H4/HF/cc-pVQZ/${method}/${operation}/"

# PP="HF"
# PP="DF"
# path="ppotential_${PP}/H/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/B/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/C/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/N/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/O/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/F/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/Ne/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"
# path="ppotential_${PP}/B2H6/HF/aug-cc-pVQZ-CDF/${method}/${operation}/"

# path="stowfn/He/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Be/HF/QZ4P/${method}/${operation}/"
# path="stowfn/N/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Ne/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Ar/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Kr/HF/QZ4P/${method}/${operation}/"
# path="stowfn/O3/HF/QZ4P/${method}/${operation}/"

# path="stowfn/He/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Be/HF/QZ4P/${method}/${operation}/"
# path="stowfn/N/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Ne/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Ar/HF/QZ4P/${method}/${operation}/"
# path="stowfn/Kr/HF/QZ4P/${method}/${operation}/"
# path="stowfn/O3/HF/QZ4P/${method}/${operation}/"

# ../casino/plot/plot.py stowfn/He/HF/QZ4P/CBCS/Backflow/ u

# path="geminal/Be/OO-RI-MP2/ano-pVDZ/CBCS/Jastrow_emin"

# path="stowfn/Be/HF/QZ4P/${method}/Backflow_omega_emin/0_4"

# pycasino.log and correlation.out.* are written to the current directory,
# set workdir to run in a subdirectory of $path and keep the reference output intact
# workdir="unreweighted"
# workdir="reweighted"

if [ -n "$workdir" ]; then
    path=$(realpath "$path")
    cd "${path}/${workdir}" || exit 1
fi

# export NUMBA_NUM_THREADS=1
# single MPI-process
# pycasino $path
# multiple MPI-process
# mpiexec pycasino $path
# hybrid code
# mpiexec -n 2 --map-by slot:pe=${NUMBA_NUM_THREADS} pycasino $path

# weighted nodal domain averages of the wave function a vmc run has just used
# pycasino --nodal $path

# nodal surface descriptor of the determinant part, any number of runs in one table. It forces
# use_jastrow F and picks the epsilon grid itself. -n and -d are needed because a *_dmc directory
# carries vmc_nstep 1024 and vmc_decorr_period 1, its VMC having only fed DMC, while a Slater one
# carries 1e8 and 10. 1e7 gives about 2% on the widest tube, 1e8 about 0.7%, the relative error
# being 72/sqrt(vmc_nstep) as long as the count in the flat region stays above ~1e4
# mpiexec ../casino/nodal_descriptor.py -n 100000000 -d 10 stowfn/Be/HF/QZ4P/CBCS/Slater stowfn/Be/HF/QZ4P/CBCS/Backflow_dmc
# mpiexec ../casino/nodal_descriptor.py -n 100000000 -d 10 stowfn/N/HF/QZ4P/CBCS/Slater stowfn/N/HF/QZ4P/CBCS/Backflow_dmc
# mpiexec ../casino/nodal_descriptor.py -n 10000000 -d 10 stowfn/Ne/HF/QZ4P/CBCS/Slater stowfn/Ne/HF/QZ4P/CBCS/Backflow_dmc

# mpiexec ../casino/nodal_descriptor.py -n 10000000 -d 10 noda_surface/be_c2/0.*
# mpiexec ../casino/nodal_descriptor.py -n 100000000 -d 10 noda_surface/be_c2/0.*

# the three points where the descriptor exploded, with the backflow dropped as well. It separates a
# determinant that is diffuse at a large c_2 on its own from a backflow left without the jastrow it
# was optimized with: if the potential component comes back to -14 au and the curve is smooth, the
# node is not what broke, the weight is
# mpiexec ../casino/nodal_descriptor.py -n 10000000 -d 10 -b noda_surface/be_c2/0.15 noda_surface/be_c2/0.20 noda_surface/be_c2/0.25

# the nine points on a common measure. The weight is applied to the walk's own sample, which costs
# the effective sample size - measured on Be, 100% of it at zeta 0, 37% at 0.5 and 0.6% at 2 - so
# the grid stops at 0.5 and the whole scan still comes out of one walk per point. What decides it
# is the volume term: <V> ran from -15.0 to -1.55 au over these nine with no weight at all, and a
# weight that does not depend on the wave function has to flatten that
# mpiexec ../casino/nodal_descriptor.py -n 10000000 -d 10 -z 0,0.125,0.25,0.375,0.5 noda_surface/be_c2/0.*

# where each of the nine walks actually stands, which is what zeta has to be read against. The
# potential split says C = 0.25 sits at 5.5 bohr per electron against 0.93 on a healthy point, and
# <sum of e-n distances> says it directly. It converges far faster than the surface integral, so
# 1e6 steps is enough and the epsilon columns of this run mean nothing
# mpiexec ../casino/nodal_descriptor.py -n 1000000 -d 10 noda_surface/be_c2/0.*

# does the chain carrying its own zeta work, and does it reach what reweighting cannot. Two points
# and two zetas, run twice. C = 0.15 at zeta 0.25 is the control: reweighting works there, so the
# two must agree. C = 0.25 at zeta 0.7 is the question: <sum of e-n distances> has to come down
# from 22.4 bohr towards 10 and the potential component back towards -14, and if it does not, the
# walk is not sampling what it is supposed to
# mpiexec ../casino/nodal_descriptor.py -n 200000 -d 10 -z 0.25,0.7 noda_surface/be_c2/0.15 noda_surface/be_c2/0.25
# cp pycasino.log noda_surface/be_c2_reweighted.log
# mpiexec ../casino/nodal_descriptor.py -n 200000 -d 10 -w -z 0.25,0.7 noda_surface/be_c2/0.15 noda_surface/be_c2/0.25
# cp pycasino.log noda_surface/be_c2_direct.log

# how far up zeta has to go, on the three points that decide it. Direct sampling made the effective
# sample size independent of zeta, so the grid is now free of it and bounded from above by
# something else: the volume term grows with zeta (-14.65 to -15.34 on C = 0.15 between 0.25 and
# 0.7) while E^nda stays -Z^2/8-sized, so the cancellation between surface and volume gets worse
# the further it is pushed. What to look for is not <sum of e-n distances> becoming equal - it will
# not, and it need not, that is a property of the wave functions - but the ranking of the three
# settling. E_kin^nda of C = 0.25 has gone 6.11, 5.28, 2.47 at zeta 0, 0.25, 0.7 against 1.12 for
# C = 0.15, so it is still moving. Twelve walks, about an hour and a half
# mpiexec ../casino/nodal_descriptor.py -n 1000000 -d 10 -w -z 0.7,1.0,1.4,2.0 noda_surface/be_c2/0.15 noda_surface/be_c2/0.20 noda_surface/be_c2/0.25
# cp pycasino.log noda_surface/be_c2_direct_scan.log

# the nine points on a measure that is finally common to all of them. zeta = 1.0 is where the scan
# above put the threshold: below it C = 0.25 still sits at 11.2 bohr against 6.4 for the others,
# at 1.0 all three agree within 3% and the volume term closes from a 13 au spread to 0.63 au.
# It is also the smallest zeta that does, and smaller is better - the surface and volume terms
# grow with zeta while their sum stays of order a tenth, so the cancellation only gets worse.
# 1e7 rather than 1e6 because the three-point subset spans 0.56 mHa of dE_FN, which the earlier
# calibration turns into 0.068 au of descriptor, and 1e6 measured 0.073 +/- 0.129: the signal is
# where it should be and the error is twice its size. Nine walks, about two hours
# mpiexec ../casino/nodal_descriptor.py -n 10000000 -d 10 -w -z 1.0 noda_surface/be_c2/0.*
# cp pycasino.log noda_surface/be_c2_direct_nine.log

# where the minimum of the objective actually sits, which is the question an optimiser asks and
# the ranking does not. At 1e7 the four points around the bottom are separated by 1.0 and 1.5
# sigma, so the bottom is flat and a parabola fit puts the vertex at 0.128 +/- 0.005 with
# chi2/dof = 5.8 - unusable. The reference dE_FN has its vertex at 0.1639 +/- 0.0013. 1e8 turns
# those gaps into 4.6 and 6.5 sigma and settles the displacement to about 0.005 in c_2. Four
# walks at ten times the length of the nine-point run, so roughly nine hours
# mpiexec ../casino/nodal_descriptor.py -n 100000000 -d 10 -w -z 1.0 noda_surface/be_c2/0.10 noda_surface/be_c2/0.12 noda_surface/be_c2/0.15 noda_surface/be_c2/0.20
# cp pycasino.log noda_surface/be_c2_direct_bottom.log

# E_VMC(c_2) with the jastrow and the backflow of each point kept - literally the curve emin
# would descend if c_2 were free, so the distance from its minimum to the 0.164 of E_FN is
# how far energy minimisation misses the node optimum, measured rather than argued. The
# archive cannot give it: those are vmc_dmc runs whose VMC phase is 1024 steps, 4-7 mHa of
# error against a 2.4 mHa curve. 1e7 puts the error near 0.1 mHa. The input is copied down to
# 1e7 rather than edited in place, so the descriptor's own directories stay as they are, and
# each run keeps its log beside it. Nine runs of about twenty minutes
for c in noda_surface/be_c2/0.*; do
    mkdir -p $c/vmc
    sed 's/vmc_nstep         : 100000000/vmc_nstep         : 10000000/' $c/input > $c/vmc/input
    for f in gwfn.data correlation.data parameters.casl; do ln -sf $(realpath $c/$f) $c/vmc/$f; done
    (cd $c/vmc && mpiexec pycasino .)
done
