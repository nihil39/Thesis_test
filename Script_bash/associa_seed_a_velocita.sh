#!/bin/bash

for i in {400..400}; 

do	
	pushd $i
	mpirun -np 12 --use-hwthread-cpus /home/linuxbrew/.linuxbrew/bin/lmp_mpi -in ../run_recupero_velocita.lammps > ./out_recupero_velocita
    #Server mpirun -np 64 --use-hwthread-cpus lmp < ../mio_nve.lammps > ./out_nve_T044_P293_run_${k}_seed-${seed}	
	popd
done
