#!/usr/bin/env bash
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner cifar100_online.sh
sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner cifar100_offline.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner cifar100_finetune.sh

