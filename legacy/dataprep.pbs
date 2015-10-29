#!/bin/bash
# pbs launching script example for NAMD job

#     job name:
#PBS -N DataPrep
#PBS -P RDS-FEI-NRMMCI-RW
#PBS -q compute 

#     how many cpus?
#PBS -l ncpus=8

#PBS -l pmem=8000mb

# How long to run the job? (hours:minutes:seconds)
#PBS -l walltime=0:10:0

#     Name of output file:
#PBS -o dataprep-output.txt

#     Environmental varibles to make it work:
 
cd $PBS_O_WORKDIR;
 
module load matlab;
#     Launching the job!
JOBNAME='dataprep';

DATETIME=$(date +"%F-%T");

#     Transfer the trained model to data folder
cachefolder=/project/RDS-FEI-NRMMCI-RW/Cache/neudraw_cache/$JOBNAME$DATETIME;

if [ ! -d "$cachefolder" ];then
	echo "Creating cache job folder: $cachefolder";
	mkdir $cachefolder;
else
	echo "Cache job folder exists: $cachefolder";
fi

me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
cp $me $cachefolder/run.pbs;

V3DPATH='/project/RDS-FEI-NRMMCI-RW/v3d/v3d_external/matlab_io_basicdatatype';

#     Run Script
matlab -nodesktop -nodisplay -nosplash -r "dataprep('/project/RDS-FEI-NRMMCI-RW/Data/OP/OP_V3Draw', '/project/RDS-FEI-NRMMCI-RW/Data/OP/OP_ORI/GT', '$V3DPATH', 3)";

echo "Done"
