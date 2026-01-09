#!/bin/bash

for ((j = 1; j < 500; j+=1)); do
    for ((i = 175 ; i < 251 ; i+=25)); do
	#echo $i > number.txt

	#python3 test_mnist.py $i
	echo "100" > number.txt

	python3 test_mnist.py 100
	
	#echo "MISSING = True" > myclient/missing.py
	#echo "COMPUTE_WEIGHTS = True" > myclient/compute.py
	#sleep 30
	#flwr run . local-simulation$i

	echo "COMPUTE_WEIGHTS = False" > myclient/compute.py
	#flwr run . local-simulation$i

	echo "MISSING = False" > myclient/missing.py
	#flwr run . local-simulation$i
	flwr run . local-simulation100
    done
done
