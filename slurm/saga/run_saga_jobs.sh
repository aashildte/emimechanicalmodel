#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2022 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script launches a series of scaling experiments based on the
# EMI Holzapfel-Ogden model on dual-socket Intel Xeon Gold 6138 nodes
# on the Saga supercomputing cluster.
#
# Example usage:
#
#  $ ./run_saga_jobs.sh
#

# Set up the environment
# . slurm/saga/envsetup.sh

account=nn2849k
partition=normal
mesh_dir=/cluster/projects/nn9249k/aashild/tiled_meshes

# Stretching experiments
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

# Stretching experiments in the x-direction on a 5x5x5 mesh
function emimm_stretching_stretch_ff_tile_pad1p0_0p5_5_5_5()
{
    local mesh_dir=/cluster/projects/nn9249k/aashild/meshes_emi_labels
    local mesh=${mesh_dir}/tile_pad1p0_0p5_5_5_5.h5
    local num_steps=10
    for nodes in 6; do
    for ntasks_per_node in 32; do
        emimm_stretching ${nodes} ${ntasks_per_node} 0-02:00:00 148GB ${num_steps} ${mesh} stretch_ff
	 done
    done
}

# Stretching experiments in the y-direction on a 5x5x5 mesh
function emimm_stretching_ydir_tile_pad1p0_0p5_5_5_5()
{
    local mesh_dir=/cluster/projects/nn9249k/aashild/meshes_emi_labels
    local mesh=${mesh_dir}/tile_pad1p0_0p5_5_5_5.h5
    local num_steps=10
    for nodes in 6; do
    for ntasks_per_node in 1 2 4 8 16 32; do
        emimm_stretching ${nodes} ${ntasks_per_node} 0-06:00:00 0 ${num_steps} ${mesh} ydir
	 done
    done
}

# Weak scaling experiments for stretching in the x-direction
function emimm_stretching_stretch_ff_weak_scaling()
{
    # 2 cardiac cells per node
    # emimm_stretching 1  1 0-00:20:00 8G 10 ${mesh_dir}/tile_connected_5p0_1_1_2.h5 stretch_ff
    # emimm_stretching 2  1 0-00:30:00 8G 10 ${mesh_dir}/tile_connected_5p0_2_1_2.h5 stretch_ff
    # emimm_stretching 4  1 0-00:45:00 8G 10 ${mesh_dir}/tile_connected_5p0_2_2_2.h5 stretch_ff
    # emimm_stretching 8  1 0-01:30:00 8G 10 ${mesh_dir}/tile_connected_5p0_2_2_4.h5 stretch_ff
    # emimm_stretching 16 1 0-02:00:00 8G 10 ${mesh_dir}/tile_connected_5p0_4_2_4.h5 stretch_ff
    # emimm_stretching 32 1 0-04:00:00 16G 10 ${mesh_dir}/tile_connected_5p0_4_4_4.h5 stretch_ff
    emimm_stretching 64 1 0-04:00:00 15G 10 ${mesh_dir}/tile_connected_5p0_4_4_8.h5 stretch_ff

    # 1 cardiac cell per node
    # emimm_stretching 1  1 0-00:05:00 4G 10 ${mesh_dir}/tile_connected_5p0.h5 stretch_ff
    # emimm_stretching 2  1 0-00:05:00 4G 10 ${mesh_dir}/tile_connected_5p0_1_1_2.h5 stretch_ff
    # emimm_stretching 4  1 0-00:10:00 4G 10 ${mesh_dir}/tile_connected_5p0_2_1_2.h5 stretch_ff
    # emimm_stretching 8  1 0-00:20:00 4G 10 ${mesh_dir}/tile_connected_5p0_2_2_2.h5 stretch_ff
    # emimm_stretching 16 1 0-01:00:00 4G 10 ${mesh_dir}/tile_connected_5p0_2_2_4.h5 stretch_ff
    # emimm_stretching 32 1 0-02:00:00 8G 10 ${mesh_dir}/tile_connected_5p0_4_2_4.h5 stretch_ff
    # emimm_stretching 64 1 0-04:00:00 8G 10 ${mesh_dir}/tile_connected_5p0_4_4_4.h5 stretch_ff

    # emimm_stretching 1  1 0-00:05:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_1_1_1.h5 stretch_ff
    # emimm_stretching 2  1 0-00:05:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_2_1_1.h5 stretch_ff
    # emimm_stretching 4  1 0-00:10:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_2_2_1.h5 stretch_ff
    # emimm_stretching 8  1 0-00:20:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_2_2_2.h5 stretch_ff
    # emimm_stretching 16 1 0-01:00:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_4_2_2.h5 stretch_ff
    # emimm_stretching 32 1 0-02:00:00 4G 10 ${mesh_dir}/tile_pad1p0_0p5_4_4_2.h5 stretch_ff
    # emimm_stretching 64 1 0-04:00:00 8G 10 ${mesh_dir}/tile_pad1p0_0p5_4_4_4.h5 stretch_ff
}

# Run experiments
# emimm_stretching_stretch_ff_tile_pad1p0_0p5_5_5_5
emimm_stretching_stretch_ff_weak_scaling
# emimm_stretching 6 1 0-00:30:00 0 10 ${mesh_dir}/tile_pad1p0_0p5_10_10_10.h5 stretch_ff

#
# Active contraction experiments
#

# # Run active contraction experiments on 3x3x3 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=00:30:00 \
#     slurm/saga/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_3_3_3.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done

# # Run active contraction experiments on 5x5x5 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=01:00:00 \
#     slurm/saga/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_5_5_5.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done

# # Run active contraction experiments on 10x10x10 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=02:00:00 \
#     slurm/saga/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_10_10_10.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done
