#!/bin/bash


#for i in {003..400}; do
#    pushd $i
#    pushd configurazioni_iniziali
#    python ../../riempi_array_configurazioni.py
#    #mv dati_configurazione.npy ../dati_configurazione_${i}.npy    
#    mv dati_configurazione.npy ../../dati_configurazione_${i}.npy    
#    popd
#    popd
#done

for i in {003..400}; do
    pushd $i
    python ../riempi_array_msd.py
    mv dati_msd.npy ../dati_msd_${i}.npy
    popd
done 
