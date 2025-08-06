#!/bin/bash

for ((i = 5000 ; i < 5001 ; i+=500)); do
    echo $i > number.txt
    flwr run . local-simulation$i
done
