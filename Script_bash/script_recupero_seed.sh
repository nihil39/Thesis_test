#!/bin/bash

for i in {122..400};

do	
	pushd $i
	for filename in out_nve_T044_P293_run_*; do
    		seme=`sed -n 13p "$filename"`	
    		#mv "$filename" "${filename}_seed-${seme}"
    		echo "$seme" >> seeds_recuperati.txt
    	
    		#mpirun -np 64 --use-hwthread-cpus lmp -in ../run_recupero_velocita.lammps > ./out_recupero_velocita
		 # mpirun -np 12 --use-hwthread-cpus /home/linuxbrew/.linuxbrew/bin/lmp_mpi -in ../run_recupero_velocita.lammps > ./out_recupero_velocita
    	done
	popd
done
