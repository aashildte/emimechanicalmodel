#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2022 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This job script is used to run mechanical stretching experiments
# based on the EMI Holzapfel-Ogden model on dual-socket Intel Xeon
# Gold 6138 nodes on the Saga supercomputing cluster.
#
# Example usage:
#
#  $ sbatch --time=04:00:00 --nodes=2 --ntasks-per-node=4 slurm/saga/normal/emimm_stretching.sbatch --mesh_file meshes/tile_1p0.h5 --num_steps 10 --verbose=2
#

#SBATCH --job-name="emimm_stretching"
#SBATCH --account=nn2849k
#SBATCH --partition=normal
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --output=slurm/saga/normal/jobs/%j-%x-out.txt
#SBATCH --error=slurm/saga/normal/jobs/%j-%x-err.txt


emimm="python3 scripts/stretching.py"
partition=normal

function diagnostics ()
{
    echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}" >&2
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}" >&2
    echo "SLURM_SUBMIT_HOST=${SLURM_SUBMIT_HOST}" >&2
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}" >&2
    echo "SLURM_NTASKS=${SLURM_NTASKS}" >&2
    echo "PKG_CONFIG=${PKG_CONFIG}" >&2
    set -x
    uname -a >&2
    lscpu >&2
    numactl -H >&2
    pkg-config --exists dolfin >&2
    { set +x; } 2>/dev/null
}

function main ()
{
    diagnostics

    local job_id="${SLURM_JOB_ID:-0}"
    local job_name="${SLURM_JOB_NAME:-0}"
    local ntasks="${SLURM_NTASKS:-1}"
    local cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
    local job_dir="${SLURM_SUBMIT_DIR:-.}/slurm/saga/${partition}/jobs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
    if [[ -n "${TMPDIR}" ]]; then mkdir -p ${TMPDIR}; fi
    local workdir="$(mktemp --tmpdir -d ${job_id}-${job_name}-XXXX)"
    printf "Using working directory: ${workdir}" >&2

    # Create working and output directories
    set -x
    mkdir -p "${job_dir}"
    { set +x; } 2>/dev/null

    # Run the benchmark
    local srun_options="-l"
    local timings_path="${job_dir}/${job_name}-timings.xml"
    local emimm_options="--timings_file ${timings_path}"
    local emimm_stdout="${job_dir}/${job_name}-stdout.txt"
    local emimm_stderr="${job_dir}/${job_name}-stderr.txt"
    set -x
    (OMP_NUM_THREADS=${cpus_per_task} \
        srun ${srun_options} -- \
    	/usr/bin/time --verbose \
	${emimm} ${emimm_options} ${@} \
    	>${emimm_stdout} 2>${emimm_stderr})
    { set +x; } 2>/dev/null
}

main "${@}"
