#!/bin/bash

<<<<<<< HEAD:Script_bash/script_rinominazione_pre_run_velocity.sh
#Questo script toglie le righe inutili dal file su cui sono salvate le configurazioni generate da lammps e lo rinomina. Da lanciare prima dei run in cui si calcolano le 20 velocita' per ogni configurazione. Occhio ai nomi dei file, potrebbero cambiare per ogni punto di pressione e temperatura

=======
#Questo script toglie le righe inutili dal file su cui sono salvate le configurazioni generate da lammps e lo rinomina. Da lanciare prima dei run in cui si calcolano le 20 velocita' per ogni configurazione. github
>>>>>>> 460d935ec3fecbe5dc621a406282fa3cfc236b56:Script_bash/script_rinominazione_preliminare.sh
for i in {003..400}; do
	pushd $i
	sed '14,18d' conf_equil_lammps_T044_P293_random_seed_file > conf_equil_T044_P293
	rm conf_equil_lammps_T044_P293_random_seed_file
	mv out_mio_npt_T044_P293_configurazioni_random_seed_file out_npt_T044_P293_configurazioni
	popd
done
