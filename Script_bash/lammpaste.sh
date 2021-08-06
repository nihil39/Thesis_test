#!/bin/bash
if [ -z "$1" ] ; then echo "Uso: lammpaste.sh nomefile" ; exit 0 ; fi
paste <(sed -n '22,4117p' "$1"| sort -nk1 | cut -f -5  -d' ') <(sed -n '4121,8216p' "$1" | sort -nk1 | cut -f 2- -d' ')


#sed 'm,nd' file cancella le righe da m a n

#sed -n 'x,yp' "$1" estrae da x a y del file, occhio alla p
#"$1" il primo argomento del comando?
#-z e' vuoto?

#sort -k1 = ordina per la prima colonna
#cut -f2- -d = taglia dal secondo campo (non taglia il secondo) fino alla fine ed usa come delimitatore (-d) lo spazio, -f -5 tiene i primi 5 e butta quelli successivi
#occhio ad usare <() come input per paste, questo rende il contenuto delle parentesi un file, paste prende in ingresso i file?

#Alternativa con awk
# awk '{print $1 " "  $2}' timestamps.txt > timestamps_3.txt stampa la prima colonna, uno spazio e poi la seconda colonna, interpreta gli spazi multipli come singolo ed usa quello come separatore https://stackoverflow.com/questions/2961635/using-awk-to-print-all-columns-from-the-nth-to-the-last 
