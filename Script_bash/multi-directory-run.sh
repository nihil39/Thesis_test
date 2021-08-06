#!/bin/bash

export CFG=`basename "$1"`

for i in {003..400}; do
	mkdir $i
	cp ./genera_seed.sh $i
	pushd $i
	mpirun -np 64 --use-hwthread-cpus lmp < ../"$CFG" > out_"$CFG"_random_seed_file
	popd
done

