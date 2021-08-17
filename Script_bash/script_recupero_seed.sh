#!/bin/bash

# controlla la riga di che sed estrae, non e' sempre 13


for i in {122..400};

do	
	pushd $i
	for filename in out_nve_T044_P293_run_*; do
    		seme=`sed -n 13p "$filename"` # controlla il 13	
    		#mv "$filename" "${filename}_seed-${seme}"
    		echo "$seme" >> seeds_recuperati.txt
    	done
	popd
done
