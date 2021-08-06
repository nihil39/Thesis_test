#!/bin/bash

# Rinomina gli output del msd dal 21 al 30, sostituisce la parola 'pulito' con la parola 'extra'
# -exec, opzione di find, passa a perl-rename il risultato della ricerca, le {} servono ad indicare dove va messo il risultato della ricerca, il \+ serve per indicare di eseguire perl-rename su tutto l'output una volta, non ogni volta sul singolo risultato di find (l'alternativa e' il \;)
# Levare il -n per eseguire ralmente il comando, -n significa dry run

for i in {001..120}; do
    pushd $i
    find out_nve_T044_P293_run_2[1-9]* out_nve_T044_P293_run_3* -exec perl-rename -n 's/pulito/extra/g' {} \+ # guarda il -n
    popd
done
