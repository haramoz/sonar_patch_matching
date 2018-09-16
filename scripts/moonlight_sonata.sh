#!/bin/bash
#SBATCH --partition=hpc          # partition (queue)
#SBATCH --tasks=80               # number of tasks     <---------- this is different to above
#SBATCH --mem=8G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=2-20:00:00              # total runtime of job allocation ((format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

source activate iwin
export OMP_NUM_THREADS=4


echo python hello-world.py
echo python hyperas_contrastive_loss.py
echo python densenet_siamese_best_run.py
echo python hyperas_densenet.py
python hyperas_densenet_siamese.py
echo python densenet_simple.py
echo python keras_densenet_siamese.py
