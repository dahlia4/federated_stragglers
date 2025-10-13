#!/bin/bash

for ((j = 1; j < 500; j+=1)); do
    for ((i = 500 ; i < 5001 ; i+=500)); do
	echo $i > number.txt

	echo "MISSING = True" > myclient/missing.py
	echo "COMPUTE_WEIGHTS = True" > myclient/compute.py
	flwr run . local-simulation$i

	#echo "COMPUTE_WEIGHTS = False" > myclient/compute.py
	#flwr run . local-simulation$i

	#echo "MISSING = False" > myclient/missing.py
	#flwr run . local-simulation$i
    done
done
