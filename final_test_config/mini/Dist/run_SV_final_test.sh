#!/usr/bin/env bash
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_1.sh
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_2.sh
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_3.sh
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_4.sh
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_5.sh
sbatch --nodes=1 --time=18:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_6.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_7.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_mini_8.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_1.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_2.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_3.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_4.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_5.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_6.sh

# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_1.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_2.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_3.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_4.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_5.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_6.sh

# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_1.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_2.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_3.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_4.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_5.sh
# sbatch --nodes=1 --time=10:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_6.sh
