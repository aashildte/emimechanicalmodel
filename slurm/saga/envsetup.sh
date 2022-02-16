#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2021 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script sets up the environment for running job scripts for the
# EMI mechanical model on the Saga cluster.
#
# Example usage:
#
#  $ . slurm/saga/envsetup.sh
#

module unuse /cluster/modulefiles/all
module use /cluster/projects/nn2849k/jamest/ex3modules/0.6.0/modulefiles
module load python-fenics-dolfin-2019.1.0.post0 python-numpy-1.19.2 python-matplotlib-3.1.1 python-scipy-1.5.4
module unload xz-5.2.5
