#!/usr/bin/env bash

###################################
## Definitions for batch system
#SBATCH -A cli@cpu                      # Accounting information
#SBATCH --job-name=C1D_PAPA_L22DNN      # Job name
#SBATCH --partition=cpu_p1              # Partition Name
#SBATCH --ntasks=2                      # Total number of MPI processes
#SBATCH --hint=nomultithread            # 1  MPI process per node  (no hyperthreading)
#SBATCH --time=00:10:00                 # Maximum execution time (HH:MM:SS)
#SBATCH --output=C1D_PAPA_L22DNN.out    # Name of output listing file
#SBATCH --error=C1D_PAPA_L22DNN.err     # Name of error listing file (the same)
###################################

# Process distribution
NPROC_NEMO=1
NPROC_PYTHON=1

## -------------------------------------------------------
##   End of user-defined section - modify with knowledge
## -------------------------------------------------------
set -x
ulimit -s unlimited

## Load environment
source ${HOME}/.bash_profile

# job information 
cat << EOF
------------------------------------------------------------------
Job submit on $SLURM_SUBMIT_HOST by $SLURM_JOB_USER
JobID=$SLURM_JOBID Running_Node=$SLURM_NODELIST 
Node=$SLURM_JOB_NUM_NODES Task=$SLURM_NTASKS
------------------------------------------------------------------
EOF

## Move to config directory
CONFIG_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd ${CONFIG_DIR}

## Create execution directory and move there
XXD=`date +%F%H%M%S`
echo " XXD " $XXD
mkdir -p $CONFIG_DIR/OUT/CPLTESTCASE/$XXD
cd $CONFIG_DIR/OUT/CPLTESTCASE/$XXD
echo "RUN directory " `pwd`

## Get input files for NEMO
for file in $CONFIG_DIR/*.nc 
   do
      ln -s $file . || exit 2
   done

## Get input namelist and xml files
for file in $CONFIG_DIR/*namelist*_ref $CONFIG_DIR/*namelist*_cfg $CONFIG_DIR/*.xml
do
    cp $file . || exit 3
done

## Get Executables
cp $CONFIG_DIR/nemo nemo.exe  || exit 5
cp $CONFIG_DIR/*.py . || exit 5

## Get input files for coupling, including the namcouple file
cp $CONFIG_DIR/namcouple . || exit 4

set -e
ls -l

## Run Eophis preproduction
python3 ./main.py --exec preprod
mv eophis.out eophis_preprod.out
mv eophis.err eophis_preprod.err

# write multi-prog file
touch run_file
echo 0-$((NPROC_NEMO - 1)) ./nemo.exe >> run_file
echo ${NPROC_NEMO}-$((NPROC_NEMO + NPROC_PYTHON - 1)) python3 ./main.py >> run_file

# run coupled NEMO-Python
time srun --multi-prog ./run_file
