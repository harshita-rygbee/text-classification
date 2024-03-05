#!/bin/bash
#SBATCH --job-name=llama_finetuning_13
#SBATCH --output=new_out/llama_finetuning_13.log
#SBATCH --gres=gpu:2

source ~/.bashrc
cd /local/scratch3/hsahijw/llama/
conda init bash
conda activate h100_env
#source activate benv
time python finetune_llama.py


# source ~/.bashrc
# cd /local/scratch/svolokh/kgat/item_no_kg/
# cd /home/hsahijw/slurm_job/
# time python main.py --gpu $CUDA_VISIBLE_DEVICES --epochs 500 --evaluate_every 25 --slurm --model lgcn_single --uid new_2l_single --n_layers 2
