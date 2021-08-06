#!/bin/bash

#export CFG=`basename "$1"`

for i in {001..120}; do
	pushd $i
	#mkdir configurazioni_ordinate
	perl-rename  's/_(\d\_)/_0$1/g' MB_initial_velocities*
	for filename in MB_initial_velocities_run_*; do
		paste <(sed -n '22,4117p' "$filename"| sort -nk1 | cut -f 2-5  -d' ') <(sed -n '4121,8216p' "$filename" | sort -nk1 | cut -f 2- -d' ')  > "${filename}_ordinato"
		rm ${filename}
		
	done
	for filename2 in out_nve_*; do
	sed -n '35,234p' "$filename2" > "$filename2"_msd_pulito #Controlla quali linee cancellare per il MSD!!!
	
	rm ${filename2}
	done
	popd
done


#perl-rename -n 's/_(\d\_)/_0$1/g' -- *ordinato* per aggiungere gli zeri prima delle singole cifre, -n NON esegue il comando ma fa vedere cosa accadrebbe se fosse eseguito

#(Remove flag -n to actually do the renaming.)

# where _(\d\.) matches an underscore followed by a single digit and then another underscore, and replaces the first underscore with _0, thus inserting a leading zero. $1 is a back-reference to the group within parentheses, the digit and dot, and leaves it unchanged in the new file name.
