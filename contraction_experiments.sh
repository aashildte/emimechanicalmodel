#! /bin/bash

#m="meshes/cells_with_nucleus_v2/cell15_with_2nuclei.h5"
#mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12 -tm 500 -t 500 

export OMP_NUM_THREADS=2
   
for nuclei in "single_nucleus"; do
    #for id in "idealized" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    for id in 9 12; do
	m="meshes/cells_with_nucleus_v2/cell${id}_with_${nuclei}.h5"
	ls $m
	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12 -tm 500 -t 500 & 
    done
    wait
    
    #for id in "idealized" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    #	m="meshes/cells_with_nucleus_v2/cell${id}_with_${nuclei}.h5"
    #	ls $m
    #	mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_iso -tm 500 -t 500 --isometric &    
    #done
    wait
done

exit 0;

for id in "idealized" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    for nuclei in "single_nucleus" "2nuclei"; do
	for sc in 0.1 0.25 0.5 1.0 2.0 4.0 10.0; do
		m="meshes/cells_with_nucleus_v2/cell${id}_with_${nuclei}.h5"
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis -tm 137 -t 137 --z-line-scale-factor $sc &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis -tm 137 -t 137 --sarcomere-scale-factor $sc &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis -tm 137 -t 137 --cytoskeleton-scale-factor $sc &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis -tm 137 -t 137 --nucleus-scale-factor $sc 

		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis_iso -tm 137 -t 137 --z-line-scale-factor $sc --isometric &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis_iso -tm 137 -t 137 --sarcomere-scale-factor $sc --isometric &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis_iso -tm 137 -t 137 --cytoskeleton-scale-factor $sc --isometric &
		mpirun -n 2 python scripts/active_contraction_sarcomere_model.py -m $m -o /data1/aashild/sarcomere_model/nuclei_calibration_v12_sensitivity_analysis_iso -tm 137 -t 137 --nucleus-scale-factor $sc --isometric
	done
    done
done
