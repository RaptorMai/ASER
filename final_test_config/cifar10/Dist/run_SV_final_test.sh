#!/usr/bin/env bash
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_1.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_2.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_3.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_4.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_5.sh
sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_6.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_7.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_cifar10_8.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_1.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_2.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_3.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_4.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_5.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_6.sh

# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_1.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_2.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_3.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_4.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_5.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_sep_6.sh

# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_1.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_2.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_3.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_4.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_5.sh
# sbatch --nodes=1 --time=2:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner SV_adv_max_sep_6.sh
