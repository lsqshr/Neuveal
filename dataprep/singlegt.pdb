#!/bin/bash
# pbs launching script example for NAMD job
# PLEASE DO NOT CHANGE THIS FILE. TO SUBMIT QUEUE, PLS MAKE A COPY OF THIS FILE AND MAKE THE ACCORDING CHANGES

#     job name:
#PBS -N Neuveal 
#PBS -P RDS-FEI-NRMMCI-RW
#PBS -q compute 

#     how many cpus?
#PBS -l ncpus=2

#PBS -l pmem=4000mb

# How long to run the job? (hours:minutes:seconds)
#PBS -l walltime=3:0:0

#     Name of output file:
#PBS -o trainlog.txt

#     Environmental varibles to make it work:
 
module load matlab;
cd $PBS_O_WORKDIR;
 
#     Launching the job!
JOBNAME='singlegt';

DATA="/project/$PROJECTID1/SQ-Workspace/RivuletJournalData/OP/OP1/op1.v3draw";
SWC="/project/$PROJECTID1/SQ-Workspace/RivuletJournalData/OP/OP1/OP_1.v3draw";
TODIR="/project/$PROJECTID1/SQ-Workspace/RivuletJournalData/OP/OP1/OP1Feat/";

#     Run Script
matlab -nodesktop -nosplash -r "[~, ~, data] = singlegt($DATA, $SWC, 10, 7, 13, $TODIR);"