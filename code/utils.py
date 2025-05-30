import numpy as np
import pandas as pd
from tqdm import tqdm

import torch


def read_csvs(filenames, shuffle=False, seed=0):
    csvs = []
    for f in filenames:
        csv = pd.read_csv(f)[["statement", "label"]]
        csvs.append(csv)
    csv = pd.concat(csvs)
    if shuffle:
        rng = np.random.RandomState(seed)
        csv = csv.sample(frac=1, random_state=rng)
    return csv


def accuracy(clf, x, y):
    return np.count_nonzero(clf.predict(x)==y) / y.size


def accuracy_prob(clf, x, y, t=0.5):
    return np.count_nonzero((clf.predict_proba(x)[:,1]>t)==y) / y.size


def gather_activations_and_labels(model, tokenizer, dataset, text_field, label_field, template, batch_size):
    activations = []
    labels = []

    n_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches)):
        batch_text = [template.format(rec[text_field]) for _, rec in dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()]
        batch_labels = [rec[label_field] for _, rec in dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()]

        ids = tokenizer(batch_text, padding='longest', return_tensors='pt', return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            hidden_states = model(**ids).last_hidden_state
        h_last_position = hidden_states[:, -1, :].detach().cpu().float().cpu()
        for h in h_last_position:
            activations.append(h)
        labels.extend(batch_labels)

    activations =  np.array(activations)
    labels = np.array(labels)
    return activations, labels


def layerwise_activations_and_labels(model, tokenizer, dataset_name, dataset, text_field, label_field, template, batch_size, use_chat_template=False):
    activations_by_layer = [[] for _ in range(model.config.num_hidden_layers)]
    labels = []

    n_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=dataset_name):
        batch_text = []
        for _, rec in dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows():
            text = template.format(rec[text_field])
            if use_chat_template:
                msgs = []
                for t in text.split("Question: ")[1:]:
                    q, a = t.strip(' \n').split("\nAnswer: ")
                    msgs.append({'role': 'user', 'content': "Question: " + q})
                    msgs.append({'role': 'assistant', 'content': "Answer: " + a})
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                text = text[:-len(tokenizer.eos_token)].strip() if text.endswith(tokenizer.eos_token) else text
            batch_text.append(text)
        batch_labels = [rec[label_field] for _, rec in dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()]

        def do_job(model, tokenizer, batch_text):
            ids = tokenizer(batch_text, padding='longest', return_tensors='pt', return_token_type_ids=False).to(model.device)
            with torch.no_grad():
                hidden_states = model(**ids, output_hidden_states=True, use_cache=False).hidden_states
            return hidden_states
        
        hidden_states = do_job(model, tokenizer, batch_text)
        for i, batch_hidden_states in enumerate(hidden_states[1:]):
            for h in batch_hidden_states:
                activations_by_layer[i].append(h[-1, :].detach().cpu().float().numpy())
        del hidden_states
        labels.extend(batch_labels)

    activations_by_layer =  np.array(activations_by_layer)
    labels = np.array(labels)
    return activations_by_layer, labels


def calibration_graph(probs, labels, n_bins=10):
    probs_dict = {}
    for i in range(len(probs)):
        probs_dict[f"{i}-0"] = probs[i, 0]
        probs_dict[f"{i}-1"] = probs[i, 1]
    probs_dict = dict(sorted(probs_dict.items(), key=lambda item: item[1]))

    n_per_bin = len(probs_dict) // n_bins

    probs_bin = [{} for _ in range(n_bins)]
    b = 0
    for k, v in probs_dict.items():
        probs_bin[b][k] = v
        if b < n_bins-1 and len(probs_bin[b]) == n_per_bin:
            b += 1
            if b == n_bins:
                break

    accs_bin = [0 for _ in range(n_bins)]

    for i, prob_bin in enumerate(probs_bin):
        for k in prob_bin:
            idx, lbl = k.split('-')
            idx = int(idx)
            lbl = int(lbl)
            if lbl == labels[idx]:
                accs_bin[i] += 1
        accs_bin[i] /= len(prob_bin)

    probs_avg = []
    for prob_bin in probs_bin:
        probs_avg.append(np.mean(list(prob_bin.values())))
    return probs_avg, accs_bin


def calibration_error_expectation(probs, accs):
    assert len(probs) == len(accs)
    ce = 0
    for p, a in zip(probs, accs):
        ce += abs(p-a)
    ce /= len(probs)
    return ce


def calibration_error_rms(probs, accs):
    assert len(probs) == len(accs)
    ce = 0
    for p, a in zip(probs, accs):
        ce += (p-a)**2
    ce /= len(probs)
    ce = np.sqrt(ce)
    return ce


def brier_score(probs, labels):
    assert len(probs) == len(labels)
    brier = 0
    for p, l in zip(probs, labels):
        brier += (p[1]-l)**2
    brier /= len(probs)
    return brier


def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
