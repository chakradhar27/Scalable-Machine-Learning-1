#!/bin/bash
#$ -l h_rt=10:00:00  #time needed
#$ -pe smp 8 #number of cores
#$ -l rmem=32G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o ../Output/Q2_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M cvsangaraju1@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 30g --master local[8] ../Code/Q2_code.py

