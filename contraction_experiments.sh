#! /bin/bash

export OMP_NUM_THREADS=2
mpirun -n 2 python scripts/contraction_fibrosis.py geometries/2d_mesh_baseline_10_0.h5 500 1 1 

for seed in 1 2 3 4 5
do
	for k in 0 10
	do
		for fib in "replacement" "interstitial"
		do
	       		mesh="2d_mesh_${fib}_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/contraction_fibrosis.py geometries/$mesh 500 1 1 &
			mpirun -n 2 python scripts/contraction_fibrosis.py geometries/$mesh 500 2 2
		done
	done
done

seed=1
for k in 0 10
do
	for fib in "replacement" "interstitial"
	do
		mesh="2d_mesh_${fib}_N_10_k_${k}_seed_${seed}.h5"
		for sc in 1.5 2 2.5 3 3.5 4
		do
			mpirun -n 2 python scripts/contraction_fibrosis.py geometries/$mesh 137 $sc 1
		done
		for sc in 2 3 4 5 6 7 8
		do
			mpirun -n 2 python scripts/contraction_fibrosis.py geometries/$mesh 137 1 $sc
		done
		
		for sc in 3 4 5 6
		do
			mpirun -n 2 python scripts/contraction_fibrosis.py geometries/$mesh 137 $sc $sc
		done
	done
done
