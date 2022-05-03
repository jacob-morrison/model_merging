#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

## >>> conda init >>>

__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"

if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="$PATH:$HOME/anaconda3/bin"
    fi
fi
unset __conda_setup

# qrsh -g $ID_GROUP -l rt_F=1 -l h_rt=1:00:00 -v GPU_COMPUTE_MODE=1
conda activate merge-models
module load cuda/11.2
module load cudnn/8.1
nvidia-smi
python3 replicate_merging_paper_experiments.py