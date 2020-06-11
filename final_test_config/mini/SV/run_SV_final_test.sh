#!/usr/bin/env bash
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_1.sh
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_2.sh
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_3.sh
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_4.sh
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_5.sh
# sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_6.sh

sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_5.sh
sbatch --nodes=1 --time=24:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_6.sh
