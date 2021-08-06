#!/usr/bin/bash

rm -f seeds.txt ;  
for s in {1..100} 
do 
	cat /dev/urandom | tr -cd '[:digit:]' | head -c 6 | xargs -0 >> seeds.txt 
done
