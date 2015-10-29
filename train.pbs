#!/bin/bash
# pbs launching script example for NAMD job
# PLEASE DO NOT CHANGE THIS FILE. TO SUBMIT QUEUE, PLS MAKE A COPY OF THIS FILE AND MAKE THE ACCORDING CHANGES

#     job name:
#PBS -N Neuveal 
#PBS -P RDS-FEI-NRMMCI-RW
#PBS -q compute 

#     how many cpus?
#PBS -l ncpus=4

#PBS -l pmem=16000mb

# How long to run the job? (hours:minutes:seconds)
#PBS -l walltime=6:0:0

#     Name of output file:
#PBS -o trainlog.txt

#     Environmental varibles to make it work:
 
module load torch;
cd $PBS_O_WORKDIR;
 
#     Launching the job!
JOBNAME='50bigbatch-denoise';

DATETIME=$(date +"%F-%T");

#     Transfer the trained model to data folder
cachefolder=/project/RDS-FEI-NRMMCI-RW/NeuvealCache/$JOBNAME$DATETIME;

if [ ! -d "$cachefolder" ];then
	echo "Creating cache job folder: $cachefolder";
	mkdir $cachefolder;
else
	echo "Cache job folder exists: $cachefolder";
fi

# me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
cp train.pbs $cachefolder/train.pbs;

DATAPATH=/project/RDS-FEI-NRMMCI-RW/Data/OP/OP_V3Draw

#     Run Script
# luajit cnnexample.lua
luajit cnn3d.lua --load_only 5000 --maxEpoch 30 --imgpath $DATAPATH/whole-op-noise.h5.t7\
 --synpath $DATAPATH/whole-op-img-downsample.h5.t7 --savemodelprefix $cachefolder/model\
  --dataformat t7 --optimization CG --maxIter 10 --batchsize 50\
  --nout '{30,1}'

echo "Dumping in the cached model...";
