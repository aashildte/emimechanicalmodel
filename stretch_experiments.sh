#! /bin/bash
export OMP_NUM_THREADS=1

mpirun -n 2 python scripts/stretch_fibrosis.py geometries/2d_mesh_baseline_10_0.h5 stretch_ff 1 1 True 
mpirun -n 2 python scripts/stretch_fibrosis.py geometries/2d_mesh_baseline_10_0.h5 stretch_ss 1 1 True 

for seed in 1 2 3 4 5
do
       	for k in 0 10
	do
		mesh="2d_mesh_interstitial_N_10_k_${k}_seed_${seed}.h5"
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 1 False &
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 1 False 
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 2 2 False &
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 2 2 False
		mesh="2d_mesh_replacement_N_10_k_${k}_seed_${seed}.h5"
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 1 False &
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 1 False 
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 2 2 False &
		mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 2 2 False
	done
done

for seed in 1
do
       	for k in 10
	do
		for sc in 2 3 4 5 6 7 8
		do
			mesh="2d_mesh_interstitial_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 $sc False 
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 $sc False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 $sc False 
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 $sc False
			mesh="2d_mesh_replacement_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 $sc False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 $sc False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff 1 $sc False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss 1 $sc False
		done
	
		for sc in 1.5 2 2.5 3 3.5 4
		do
			mesh="2d_mesh_interstitial_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc 1 False 
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc 1 False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc 1 False 
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc 1 False
			mesh="2d_mesh_replacement_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc 1 False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc 1 False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc 1 False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc 1 False
		done
		
		for sc in 3 4 5 6
		do
			mesh="2d_mesh_interstitial_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc $sc False  
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc $sc False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc $sc False 
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc $sc False
			mesh="2d_mesh_replacement_N_10_k_${k}_seed_${seed}.h5"
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc $sc False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc $sc False
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ff $sc $sc False &
			mpirun -n 2 python scripts/stretch_fibrosis.py geometries/$mesh stretch_ss $sc $sc False
		done	
	done
done

