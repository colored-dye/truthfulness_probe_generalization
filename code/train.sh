#!/usr/bin/bash

# declare -a model_names=(
#     "Llama-2-7b-hf"
#     "Llama-2-7b-chat-hf"
#     "Llama-2-13b-hf"
#     "Llama-2-13b-chat-hf"
#     "Meta-Llama-3.1-8B-hf"
#     "Meta-Llama-3.1-8B-Instruct-hf"
#     "Meta-Llama-3.1-70B-hf"
#     "Meta-Llama-3.1-70B-Instruct-hf"
# )

declare -a model_names=(
    "Meta-Llama-3.1-70B-Instruct-hf"
)

declare -a seeds=(0 1 2)


for model_name in "${model_names[@]}"
do
    for seed in ${seeds[@]}
    do
        echo "model:" ${model_name} ", seed:" ${seed}
        $(which python) train.py --seed ${seed} --model_name ${model_name} --probe_save_dir probes
    done
done
