 #!/bin/bash
mems=(50 20 10)

#CIFAR100
for mem in "${mems[@]}"
    do
        #ER
        python final_test.py --parameters final_test_config/cifar100/Tiny/tiny_$mem.yml
        #ASER
        python final_test.py --parameters final_test_config/cifar100/SV/SV_adv_max_$mem.yml
        python final_test.py --parameters final_test_config/cifar100/SV/SV_adv_mean_$mem.yml
        #GSS
        python final_test.py --parameters final_test_config/cifar100/GSS/gss_$mem.yml
        #MIR
        python final_test.py --parameters final_test_config/cifar100/MIR/mir_$mem.yml
    done

#mini-imagenet
for mem in "${mems[@]}"
    do
        #ER
        python final_test.py --parameters final_test_config/mini/Tiny/tiny_$mem.yml
        #ASER
        python final_test.py --parameters final_test_config/mini/SV/SV_adv_max_$mem.yml
        python final_test.py --parameters final_test_config/mini/SV/SV_adv_mean_$mem.yml
        #GSS
        python final_test.py --parameters final_test_config/mini/GSS/gss_$mem.yml
        #MIR
        python final_test.py --parameters final_test_config/mini/MIR/mir_$mem.yml
    done


mems=(100 50 20)
for mem in "${mems[@]}"
    do
        #ER
        python final_test.py --parameters final_test_config/cifar10/Tiny/tiny_$mem.yml
        #ASER
        python final_test.py --parameters final_test_config/cifar10/SV/SV_adv_max_$mem.yml
        python final_test.py --parameters final_test_config/cifar10/SV/SV_adv_mean_$mem.yml
        #GSS
        python final_test.py --parameters final_test_config/cifar10/GSS/gss_$mem.yml
        #MIR
        python final_test.py --parameters final_test_config/cifar10/MIR/mir_$mem.yml
    done