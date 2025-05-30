import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM,
    QuantoConfig, set_seed,
)

from utils import *


DEVICE = "cuda"
BATCH_SIZE = 1

# MODEL_PATH = "/Model/meta-llama/Llama-2-7b-hf"
# MODEL_PATH = "/Model/meta-llama/Llama-2-7b-chat-hf"
# MODEL_PATH = "/Model/meta-llama/Llama-2-13b-hf"
# MODEL_PATH = "/Model/meta-llama/Llama-2-13b-chat-hf"
MODEL_PATH = "/Model/meta-llama/Meta-Llama-3.1-8B-hf"
# MODEL_PATH = "/Model/meta-llama/Meta-Llama-3.1-8B-Instruct-hf"
# MODEL_PATH = "/Model/meta-llama/Meta-Llama-3.1-70B-hf"
# MODEL_PATH = "/Model/meta-llama/Meta-Llama-3.1-70B-Instruct-hf"
# MODEL_PATH = "/Model/mistralai/Mistral-7B-v0.1"
# MODEL_PATH = "/Model/mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_PATH = "/Model/mistralai/Mistral-Large-Instruct-2407"

model_name = os.path.basename(MODEL_PATH)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'

quant_config = QuantoConfig(weights='float8')
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     # torch_dtype=torch.bfloat16,
#     device_map='auto',
#     attn_implementation="eager",
#     quantization_config=quant_config
# )

set_seed(0)
model_name = "Meta-Llama-3.1-8B-hf-rand-seed=0"
config = AutoConfig.from_pretrained(MODEL_PATH)
config.quantization_config=quant_config
config.attn_implementation="eager"
config.device_map='auto'
model = LlamaForCausalLM(config).cuda()

dataset_dir = "data/"
topics = [
    "animal_class",
    "cities",
    "facts",
    "sp_en_trans",
    "inventors",
    "element_symb",
]
dataset_names = []
for t in topics:
    dataset_names.append(t)
    dataset_names.append("neg_"+t)
    dataset_names.append(t+"_conj")
    dataset_names.append(t+"_disj")
dataset_names += [
    "animal_class_disj_new",
    "cities_disj_new",
    "cities_cities_disj",
    "element_symb_new",
    "neg_element_symb_new",
    "element_symb_disj_new",
    "inventors_disj_new",
    "sp_en_trans_disj_new",
    "companies",
    "neg_companies",
    "larger_than",
    "smaller_than",
    "all_unambiguous_replies",
    # "sciq_true_false",
    # "sciq_true_false_with_options",
    # "sciq_true_false_with_options_TTT",
    # "sciq_true_false_with_options_TTF",
    # "mmlu_true_false",
    # "mmlu_true_false_with_options",
    # "mmlu_true_false_with_options_TTTTT",
    # "mmlu_true_false_with_options_TTFFF",
]
dataset_names = [
    "animal_class",
    # "sciq_true_false_mc",
    # "sciq_true_false_mc_TTT",
    # "sciq_true_false_mc_TTF",
    # "sciq_true_false_mc_FFT",
    # "mmlu_true_false_mc",
    # "mmlu_true_false_mc_TTTTT",
    # "mmlu_true_false_mc_TTFFF",
    # "triviaqa_true_false_Meta-Llama-3.1-8B-hf-shots=20",
    # "triviaqa_true_false_Llama-2-13b-hf",
    # "sciq_true_false",
    # "sciq_true_false_with_options",
    # "sciq_true_false_with_options_TTT",
    # "sciq_true_false_with_options_TTF",
    # "mmlu_true_false",
    # "mmlu_true_false_with_options",
    # "mmlu_true_false_with_options_TTTTT",
    # "mmlu_true_false_with_options_TTFFF",
    # "xsum_true_false",
    # "xsum_true_false_T",
    # "xsum_true_false_TT",
    # "xsum_true_false_TTT",
    # "boolq_true_false",
    # "boolq_true_false_T",
    # "boolq_true_false_F",
    # "boolq_true_false_with_options",
    # "boolq_true_false_with_options_T",
    # "boolq_true_false_with_options_F",
]

qa_datasets = ('trivia', 'sciq', 'mmlu')

# task_prompt = "Is the following statement is true or false?\n{}"
# prompt_option = "with_prompt"

task_prompt = '{}'
prompt_option = "no_prompt"


activations_dict = {}
labels_dict = {}

acts_dir = f"activations_and_labels/{model_name}/{prompt_option}"
os.makedirs(acts_dir, exist_ok=True)

for dataset_name in dataset_names:
    dataset = read_csvs([os.path.join(dataset_dir, dataset_name+".csv")])
    activations_by_layer, labels = layerwise_activations_and_labels(
        model=model,
        tokenizer=tok,
        dataset_name=dataset_name,
        dataset=dataset,
        text_field="statement",
        label_field="label",
        template=task_prompt,
        batch_size=BATCH_SIZE,
        use_chat_template=False,
    )
    activations_dict[dataset_name] = activations_by_layer
    labels_dict[dataset_name] = labels
    
    task_dir = os.path.join(acts_dir, dataset_name)
    os.makedirs(task_dir, exist_ok=True)
    acts = activations_dict[dataset_name]
    labels = labels_dict[dataset_name]
    for layer in range(len(acts)):
        np.save(os.path.join(task_dir, f"acts_{layer}.npy"), acts[layer])
    np.save(os.path.join(task_dir, f"labels_{layer}.npy"), labels)


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6))
relative_variances_all = 0
for k in activations_dict:
    activations_by_layer = activations_dict[k]
    labels = labels_dict[k]

    between_class_variances = []
    within_class_variances = []
    for layer_nr in range(model.config.num_hidden_layers):
        # Calculate means for each class
        false_stmnt_ids = labels == 0
        true_stmnt_ids = labels == 1

        false_acts = activations_by_layer[layer_nr, false_stmnt_ids]
        true_acts = activations_by_layer[layer_nr, true_stmnt_ids]

        mean_false = false_acts.mean(axis=0)
        mean_true = true_acts.mean(axis=0)

        # Calculate within-class variance
        within_class_variance_false = np.var(false_acts, axis=0).mean()
        within_class_variance_true = np.var(true_acts, axis=0).mean()
        within_class_variances.append((within_class_variance_false + within_class_variance_true) / 2)

        # Calculate between-class variance
        overall_mean = activations_by_layer[layer_nr].mean(axis=0)
        between_class_variances.append(((mean_false - overall_mean)**2
                                        + (mean_true - overall_mean)**2).mean().item() / 2)

    relative_variances = np.array(between_class_variances) / np.array(within_class_variances)
    ax = plt.plot(range(len(relative_variances)), relative_variances, label=k)
    plt.annotate(relative_variances.argmax(), xy=(relative_variances.argmax(), relative_variances.max()), c=ax[0].get_color())
    relative_variances_all += relative_variances
# plt.yscale('log')
plt.legend()
plt.xlabel("Layers (starting from 0)")
plt.ylabel("Between-class variance / In-class variance")

# variance_dir = "figures_relative_variance/"
# os.makedirs(variance_dir, exist_ok=True)
# figname = os.path.join(variance_dir, f"{model_name}_{prompt_option}.pdf")
# if os.path.exists(figname):
#     os.remove(figname)
# plt.savefig(figname)

print(relative_variances_all.argmax())
