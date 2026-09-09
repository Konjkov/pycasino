#!/bin/bash

method="CBCS"
# method="EBES"
# method="EBES_new"
# operation="Slater"
operation="Jastrow"
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

path="stowfn/He/HF/QZ4P/${method}/${operation}/"
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
