#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2022 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script sets up the environment for running job scripts for the
# EMI mechanical model on the Saga cluster.
#
# Example usage:
#
#  $ . slurm/saga/envsetup.sh
#

. /cluster/projects/nn2849k/jamest/ex3modules/1.0.0/envsetup.sh
module load python-fenics-dolfin-2019.1.0.post0 python-numpy-1.19.2 python-matplotlib-3.1.1 python-scipy-1.7.3
module unload xz-5.2.5
