#!/bin/bash


for i in {003..400}; do
    pushd $i
    python ../riempi_array_dopo_ordinamento.py
    mv dati_configurazione.npy ../dati_configurazione_${i}.npy    
    
    popd
done
