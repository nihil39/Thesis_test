#!/bin/bash

#Questo script toglie le righe inutili dal file su cui sono salvate le configurazioni generate da lammps e lo rinomina. Da lanciare prima dei run in cui si calcolano le 20 velocita' per ogni configurazione

for i in {003..400}; do
	pushd $i
	sed '14,18d' conf_equil_lammps_T044_P293_random_seed_file > conf_equil_T044_P293
	rm conf_equil_lammps_T044_P293_random_seed_file
	mv out_mio_npt_T044_P293_configurazioni_random_seed_file out_npt_T044_P293_configurazioni
	popd
done
