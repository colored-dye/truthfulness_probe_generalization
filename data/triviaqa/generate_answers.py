import json
import numpy as np
import os
import pandas as pd
import re
import string
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    QuantoConfig,
    TextGenerationPipeline,
    set_seed,
)
from transformers.generation.utils import GenerationConfig

import utils


DEVICE = "cuda"

MODEL_PATH = "/Model/meta-llama/Meta-Llama-3.1-8B-hf"

FEW_SHOT_TEMPLATE = """Question: {question}
Answer: {answer}"""

GENERATION_TEMPLATE = """Question: {question}
Answer:"""

N_TEST_SAMPLES = 1000
SEED = 0
N_SHOTS = 20
STOP_WORDS = ["\n", ".", ","]
MAX_NEW_TOKENS = 24
TEMPERATURE = 1.0
TOP_K = 50
N_SAMPLES = 20
BATCH_SIZE = 1

RESULTS_DIR = "results/"


def generate_answers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    few_shot_dataset,
    test_dataset,
    few_shot_seed,
    few_shot_template,
    n_shots,
    generation_template,
    max_new_tokens,
    temperature,
    top_k,
    stop_words,
    num_return_sequences,
    batch_size,
    results_filename,
):
    few_shot_rng = np.random.RandomState(few_shot_seed)
    n_batches = (len(test_dataset)+batch_size-1) // batch_size
    with open(results_filename, 'w') as fp:
        for batch_idx in tqdm(range(n_batches)):
            batch = []
            for rec in test_dataset[batch_idx*batch_size:(batch_idx+1)*batch_size]:
                few_shot_prompt = utils.get_few_shot_prompt(few_shot_dataset, few_shot_template, few_shot_rng, n_shots)
                prompt = few_shot_prompt + generation_template.format(question=rec['question'])
                batch.append(prompt)
            genconfig = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                stop_strings=stop_words,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
            )
            pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
            generated_answers = pipe(
                batch,
                tokenizer=tokenizer,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                prefix=tokenizer.bos_token,
                generation_config=genconfig,
            )
            for answers in generated_answers:
                answers = [a['generated_text'].strip("".join(stop_words+[' '])) for a in answers]
                data = json.dumps(answers)
                fp.write(data+"\n")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    set_seed(SEED)
    quant_config = QuantoConfig(weights='float8')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        attn_implementation="eager",
        quantization_config=quant_config,
    )
    model_name = os.path.basename(MODEL_PATH)
    
    dataset = load_dataset("/Dataset/mandarjoshi/trivia_qa")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, f"{model_name}-seed={SEED}-temperature={TEMPERATURE}-top_k={TOP_K}-num_seq={N_SAMPLES}-shots={N_SHOTS}.jsonl")
    generate_answers(
        model=model,
        tokenizer=tok,
        few_shot_dataset=dataset['train'].to_list(),
        test_dataset=dataset['validation'].to_list()[:N_TEST_SAMPLES],
        few_shot_seed=SEED,
        few_shot_template=FEW_SHOT_TEMPLATE,
        n_shots=N_SHOTS,
        generation_template=GENERATION_TEMPLATE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        stop_words=STOP_WORDS,
        num_return_sequences=N_SAMPLES,
        batch_size=BATCH_SIZE,
        results_filename=results_file,
    )
    

if __name__ == "__main__":
    main()

