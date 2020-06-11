#!/usr/bin/env bash
# sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner gss_1.sh
# sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner gss_2.sh
# sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner gss_3.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner gss_4.sh
