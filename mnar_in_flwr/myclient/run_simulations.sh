#!/bin/bash

for ((j = 1; j < 500; j+=1)); do
    for ((i = 5 ; i < 36 ; i+=5)); do
	echo $i > number.txt

	python3 test_mnist.py $i
	
	echo "MISSING = True" > myclient/missing.py
	echo "COMPUTE_WEIGHTS = True" > myclient/compute.py
	flwr run . local-simulation$i

	echo "COMPUTE_WEIGHTS = False" > myclient/compute.py
	flwr run . local-simulation$i

	echo "MISSING = False" > myclient/missing.py
	flwr run . local-simulation$i
    done
done
