#! /bin/bash

export OMP_NUM_THREADS=2

#for sc in 200.0 150.0 100.0 50.0 25.0 12.5 6.25 5.0 2.5 1.25 0.625 0.5 0.25 0.125 0.0625 0.05 0.025; do
#	m="meshes/varying_cells_buckling_with_ECM/cell2_baseline.h5"	
#	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/chatgpt_stiffness_params_isotropic -tm 137 -t 137 --z-line-scale-factor $sc
#done



for mesh in sarcomere_geometry_withECM_50_10_clength_0.6.h5 sarcomere_geometry_withECM_55_10_clength_0.6.h5 sarcomere_geometry_withECM_60_10_clength_0.6.h5 sarcomere_geometry_withECM_65_10_clength_0.6.h5 sarcomere_geometry_withECM_70_10_clength_0.6.h5 sarcomere_geometry_withECM_75_10_clength_0.6.h5 sarcomere_geometry_withECM_80_10_clength_0.6.h5; do
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m meshes/mitochondria_geometries_added_withECM/$mesh -o /data1/aashild/sarcomere_model/new_sarcomere_lesion_experiments_org_stiffness_params -tm 137 -t 137
done
