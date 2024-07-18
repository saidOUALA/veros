#!/bin/bash
#SBATCH -c 100
#SBATCH -t 35:00:00
#SBATCH --mem=250G
#Other SBATCH commands go here

#Activating conda
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
source ~/.bashrc
conda activate veros

export OMP_NUM_THREADS=32


#Your appication commands go here --force-overwrite
mpiexec -n 32 veros run veros/setups/acc_variable_res_quarter_no_params/acc_variable_res_quarter_no_params.py -b numpy -n 8 4 
