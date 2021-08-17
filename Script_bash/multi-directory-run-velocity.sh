#!/bin/bash


for i in {121..199}; 
do	
	pushd $i
	
		for k in {01..20};
		do
		  bash ../genera_seed.sh # deve generare un solo seed, altrimenti c'e' ambiguita' sulla variabile seed
		  file="./seeds.txt"
		  seed=`cat "$file"`
		  #seed=$(cat "$file")
		  
		 mpirun -np 64 --use-hwthread-cpus lmp < ../mio_nve.lammps > ./out_nve_T044_P293_run_${k}_seed-${seed}
		 #mpirun -np 12 --use-hwthread-cpus /home/linuxbrew/.linuxbrew/bin/lmp_mpi < ../mio_nve.lammps > ./out_nve_T044_P293_run_${k}_seed-${seed}
		  

		done
	popd
done
