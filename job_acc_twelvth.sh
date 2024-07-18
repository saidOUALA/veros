#!/bin/bash
#SBATCH -c 100
#SBATCH -t 48:00:00
#SBATCH --mem=100G
#Other SBATCH commands go here

#Activating conda
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
source ~/.bashrc
conda activate veros

#Your appication commands go here
mpiexec -n 96 veros run veros/setups/acc_variable_res/acc_variable_res.py -b numpy -n 8 12