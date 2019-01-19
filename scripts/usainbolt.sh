#!/bin/bash
#SBATCH --partition=hpc1        # partition (queue)
#SBATCH --ntasks=32             # number of tasks     <---------- this is different to above
#SBATCH --mem=70G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=2-20:30:00              # total runtime of job allocation ((format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

source activate anime
export OMP_NUM_THREADS=64

echo python keras_dn_simple_multigpu.py
echo python keras_dn_simple.py
echo python custom_gridsearch_dn_siamese.py
echo python custom_gs_dn_siamese_layers_multi.py -g 1
echo python custom_gridsearch_dn_siamese_layers_avg.py
echo python custom_gridsearch_dn_siamese_layers.py
echo python custom_gridsearch_dn_two_channel.py
python ensembles_test1.py
