#! /bin/bash

export OMP_NUM_THREADS=2

for sc in 200.0 150.0 100.0 50.0 25.0 12.5 6.25 5.0 2.5 1.25 0.625 0.5 0.25 0.125 0.0625 0.05 0.025; do
	m="meshes/sarcomere_meshes_2D/cell1_straight_withnucleus.h5"
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/softer_ECM_experiments_v4 -tm 137 -t 137 --ECM-scale-factor $sc &
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/softer_ECM_experiments_v4_isometric -tm 137 -t 137 --ECM-scale-factor $sc --isometric &
	m="meshes/sarcomere_meshes_2D/cell1_slight_variation_withnucleus.h5"
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/softer_ECM_experiments_v4 -tm 137 -t 137 --ECM-scale-factor $sc &
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/softer_ECM_experiments_v4_isometric -tm 137 -t 137 --ECM-scale-factor $sc --isometric
done
