#!/bin/bash
# Jay C. Rothenberger
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#

#SBATCH --partition=ai2es_v100
#SBATCH --cpus-per-task=2
# The number of cores you get
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results/exp%01a_stdout.txt
#SBATCH --error=results/exp%01a_stderr.txt
#SBATCH --time=15:00
#SBATCH --job-name=MNIST_exp

#SBATCH --mail-user=<your email>
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/<your directory>/MNIST
#SBATCH --array=[0-0]
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python experiment.py --exp $SLURM_ARRAY_TASK_ID --gpu

