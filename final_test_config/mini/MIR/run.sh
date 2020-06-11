#!/usr/bin/env bash
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner mir_1.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner mir_2.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner mir_3.sh

sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner mir_4.sh