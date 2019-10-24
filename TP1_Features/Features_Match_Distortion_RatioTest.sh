#!/bin/bash

for d in 10 30 60 90 180
do	
	for typeD in kaze orb
	do
			python3.5 Features_Match_Distortion_RatioTest.py $typeD $d	
	done
done
