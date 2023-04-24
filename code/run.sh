#!/bin/bash
for ((it=1; it<=1; it++))
do
    for ((i=1; i<=100; i++))
        do
            echo "epoch $i"
	          python selfplay.py
	          python main.py
            python test.py
        done
done
