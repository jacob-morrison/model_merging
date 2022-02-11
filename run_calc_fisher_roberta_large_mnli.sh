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

conda activate merge-models
nvidia-smi
cd jacob/merge-vl/model_merging
python3 compute_fisher.py --model roberta-large-mnli --fisher_path ../fishers/ --glue_task mnli 