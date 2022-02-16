#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2022 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script launches a series of scaling experiments based on the
# EMI Holzapfel-Ogden model on ‘normal’ nodes on the Betzy
# supercomputing cluster.
#
# Example usage:
#
#  $ ./run_betzy_jobs.sh
#

# Set up the environment
. slurm/betzy/envsetup.sh

account=nn2849k
partition=normal

# Stretching experiments
function emimm_stretch_holzapfel()
{
    local nodes="${1}"
    local ntasks_per_node="${2}"
    local time="${3}"
    local mem="${4}"
    local num_steps="${5}"
    local mesh="${6}"
    local rowperm="${7}"
    local dir="${8}"

    local meshname=$(basename ${mesh%.*})
    sbatch \
    --account="${account}" \
    --partition="${partition}" \
    --time="${time}" \
    --mem="${mem}" \
    --nodes=${nodes} \
    --ntasks-per-node=${ntasks_per_node} \
    --job-name="emimm_stretch_holzapfel_${dir}_${meshname}_$(printf "%02d" "${num_steps}")_steps_superlu_dist_$(printf "%02d" "${nodes}")_nodes_${ntasks_per_node}_tasks_per_node" \
    slurm/betzy/normal/emimm_stretch_holzapfel.sbatch \
    --mesh_file ${mesh} \
    --num_steps=${num_steps} \
    --mat_superlu_dist_rowperm ${rowperm} \
    --mat_superlu_dist_statprint \
    -d ${dir} --verbose=3;
}

# Stretching experiments in the x-direction on a 5x5x5 mesh
function emimm_stretch_holzapfel_xdir_tile_pad1p0_0p5_5_5_5()
{
    local meshdir=/cluster/projects/nn2849k/jamest/emimechanicalmodel/meshes/meshes_emi_labels
    local mesh=${meshdir}/tile_pad1p0_0p5_5_5_5.h5
    local num_steps=10
    local rowperm=LargeDiag_MC64
    for nodes in 6; do
    for ntasks_per_node in 32; do
        emimm_stretch_holzapfel ${nodes} ${ntasks_per_node} 0-02:00:00 148GB ${num_steps} ${mesh} ${rowperm} xdir
	 done
    done
}

# Stretching experiments in the y-direction on a 5x5x5 mesh
function emimm_stretch_holzapfel_ydir_tile_pad1p0_0p5_5_5_5()
{
    local meshdir=/cluster/projects/nn2849k/jamest/emimechanicalmodel/meshes/meshes_emi_labels
    local mesh=${meshdir}/tile_pad1p0_0p5_5_5_5.h5
    local num_steps=10
    local rowperm=LargeDiag_MC64
    for nodes in 6; do
    for ntasks_per_node in 1 2 4 8 16 32; do
        emimm_stretch_holzapfel ${nodes} ${ntasks_per_node} 0-06:00:00 0 ${num_steps} ${mesh} ${rowperm} ydir
	 done
    done
}

# Weak scaling experiments for stretching in the x-direction
function emimm_stretch_holzapfel_xdir_weak_scaling()
{
    local meshdir=/cluster/projects/nn2849k/jamest/emimechanicalmodel/meshes/meshes_emi_labels
    local num_steps=10
    emimm_stretch_holzapfel   1 1 0-00:05:00 4G 10 ${meshdir}/tile_pad1p0_0p5_1_1_1.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel   2 1 0-00:05:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_1_1.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel   4 1 0-00:10:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_2_1.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel   8 1 0-00:20:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_2_2.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel  16 1 0-01:00:00 4G 10 ${meshdir}/tile_pad1p0_0p5_4_2_2.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel  32 1 0-02:00:00 4G 10 ${meshdir}/tile_pad1p0_0p5_4_4_2.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel  64 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel 128 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel 256 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel 512 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_MC64 xdir
    emimm_stretch_holzapfel   1 1 0-00:05:00 4G 10 ${meshdir}/tile_pad1p0_0p5_1_1_1.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel   2 1 0-00:05:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_1_1.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel   4 1 0-00:10:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_2_1.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel   8 1 0-00:20:00 4G 10 ${meshdir}/tile_pad1p0_0p5_2_2_2.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel  16 1 0-01:00:00 4G 10 ${meshdir}/tile_pad1p0_0p5_4_2_2.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel  32 1 0-02:00:00 4G 10 ${meshdir}/tile_pad1p0_0p5_4_4_2.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel  64 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel 128 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel 256 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_AWPM xdir
    emimm_stretch_holzapfel 512 1 0-04:00:00 8G 10 ${meshdir}/tile_pad1p0_0p5_4_4_4.h5 LargeDiag_AWPM xdir
}

# Run experiments
emimm_stretch_holzapfel_xdir_tile_pad1p0_0p5_5_5_5
# emimm_stretch_holzapfel_xdir_weak_scaling
# emimm_stretch_holzapfel 6 1 0-00:30:00 0 10 ${meshdir}/tile_pad1p0_0p5_10_10_10.h5 xdir

#
# Active contraction experiments
#

# # Run active contraction experiments on 3x3x3 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=00:30:00 \
#     slurm/betzy/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_3_3_3.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done

# # Run active contraction experiments on 5x5x5 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=01:00:00 \
#     slurm/betzy/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_5_5_5.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done

# # Run active contraction experiments on 10x10x10 mesh
# for n in 1 2 4 8 16; do
#     sbatch --partition=normal --time=02:00:00 \
#     slurm/betzy/normal/emimm_active_contr_holzapfel_$(printf "%02d" "${n}")_nodes_$(printf "%04d" "$((${ntasks_per_node}*${n}))")_tasks.sbatch \
#     --mesh_file /cluster/projects/nn9249k/aashild/meshes_emi_labels/tile_pad1p0_0p5_10_10_10.h5 \
#     -t 138 -tm 138 --verbose=3 --superlu_dist_statprint;
# done
