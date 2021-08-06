#!/bin/bash

for i in {400..400}; 

do	
	pushd $i
	rm *MB*
	mpirun -np 12 --use-hwthread-cpus /home/linuxbrew/.linuxbrew/bin/lmp_mpi -in ../run_recupero_velocita.lammps > ./out_recupero_velocita
	
	popd
done
