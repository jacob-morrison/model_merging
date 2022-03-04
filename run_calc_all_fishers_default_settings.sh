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
module load cuda/11.2
module load cudnn/8.1
nvidia-smi
python3 compute_fisher.py --model yoshitomo-matsubara/bert-base-uncased-cola --fisher_path ../fishers/cola-default-settings.hdf5 --glue_task cola
python3 compute_fisher.py --model textattack/bert-base-uncased-SST-2 --fisher_path ../fishers/sst-2-default-settings.hdf5 --glue_task sst-2
python3 compute_fisher.py --model textattack/bert-base-uncased-MRPC --fisher_path ../fishers/mrpc-default-settings.hdf5 --glue_task mrpc
python3 compute_fisher.py --model textattack/bert-base-uncased-QQP --fisher_path ../fishers/qqp-default-settings.hdf5 --glue_task qqp
python3 compute_fisher.py --model madlag/bert-large-uncased-mnli --fisher_path ../fishers/mnli-default-settings.hdf5 --glue_task mnli
python3 compute_fisher.py --model textattack/bert-base-uncased-QNLI --fisher_path ../fishers/qnli-default-settings.hdf5 --glue_task qnli
python3 compute_fisher.py --model textattack/bert-base-uncased-RTE --fisher_path ../fishers/rte-default-settings.hdf5 --glue_task rte