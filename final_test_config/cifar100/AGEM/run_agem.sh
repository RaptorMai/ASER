#!/usr/bin/env bash
sbatch --nodes=1 --time=8:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner agem_1.sh
sbatch --nodes=1 --time=8:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner agem_2.sh
sbatch --nodes=1 --time=8:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner agem_3.sh
