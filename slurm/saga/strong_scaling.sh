#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2022 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script launches a series of jobs for a strong scaling experiment
# based on the EMI Holzapfel-Ogden model on dual-socket Intel Xeon
# Gold 6138 nodes on the Saga supercomputing cluster.
#
# Example usage:
#
#  $ ./slurm/saga/strong_scaling.sh
#

account=nn2849k
partition=normal

# stretching experiments
function emimm_stretching()
{
    local nodes="${1}"
    local ntasks_per_node="${2}"
    local time="${3}"
    local mem="${4}"
    local num_steps="${5}"
    local mesh="${6}"
    local dir="${7}"

    local mesh_name=$(basename ${mesh%.*})
    sbatch \
    --account="${account}" \
    --partition="${partition}" \
    --time="${time}" \
    --mem="${mem}" \
    --nodes=${nodes} \
    --ntasks-per-node=${ntasks_per_node} \
    --job-name="emimm_stretching_${dir}_${mesh_name}_$(printf "%02d" "${num_steps}")_steps_superlu_dist_$(printf "%02d" "${nodes}")_nodes_${ntasks_per_node}_tasks_per_node" \
    slurm/saga/normal/emimm_stretching.sbatch \
    --mesh_file ${mesh} \
    --num_steps=${num_steps} \
    -d ${dir} --verbose=3;
}

# strong scaling experiments
function emimm_stretching_stretch_ff_strong_scaling()
{
    local mesh_dir=/cluster/projects/nn9249k/aashild/tiled_meshes
    local mesh=${mesh_dir}/tile_connected_5p0_4_4_4.h5
    emimm_stretching 6  1 0-07:00:00 60G 10 ${mesh} stretch_ff
    emimm_stretching 6  2 0-08:00:00 120G 10 ${mesh} stretch_ff
    emimm_stretching 6  4 0-05:00:00 148G 10 ${mesh} stretch_ff
    emimm_stretching 6  8 0-04:00:00 148G 10 ${mesh} stretch_ff
    emimm_stretching 6  16 0-04:00:00 148G 10 ${mesh} stretch_ff
    emimm_stretching 6  32 0-02:00:00 160G 10 ${mesh} stretch_ff

    # emimm_stretching 1  1 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 1  2 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 1  4 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 1  8 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 1 16 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 1 32 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 2 32 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 3 32 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 4 32 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 5 32 0-02:00:00 0 10 ${mesh} stretch_ff
    # emimm_stretching 6 32 0-02:00:00 0 10 ${mesh} stretch_ff
}

# run experiments
emimm_stretching_stretch_ff_strong_scaling
