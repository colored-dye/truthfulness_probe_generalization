import fire
from loguru import logger
import joblib
import os
import tabulate

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC

from probes import MMProbe, TTPD
from utils import *


MODEL_NAMES_LAYER_INDICES = {
    "Llama-2-7b-hf": 12,
    "Llama-2-7b-chat-hf": 13,
    "Llama-2-13b-hf": 13,
    "Llama-2-13b-chat-hf": 13,
    "Meta-Llama-3.1-8B-hf": 12,
    "Meta-Llama-3.1-8B-Instruct-hf": 13,
    "Meta-Llama-3.1-70B-hf": 33,
    "Meta-Llama-3.1-70B-Instruct-hf": 33,
    "Mistral-Large-Instruct-2407": 43,
}

TOPICS = [
    "animal_class",
    "neg_animal_class",
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "inventors",
    "neg_inventors",
    "element_symb",
    "neg_element_symb",
    "facts",
    "neg_facts",
]

TRAINSET_RATIO = 0.7


def prepare_data(
    seed,
    model_name,
    prompt_option,
    layer_index,
):
    activations_dir = f"activations_and_labels/{model_name}/{prompt_option}"

    train_activations = []
    train_activations_centered = []
    train_polarities = []
    train_labels = []
    dev_activations = []
    dev_activations_centered = []
    dev_labels = []

    for topic in TOPICS:
        acts = np.load(f"{activations_dir}/{topic}/acts_{layer_index}.npy")
        acts_centered = acts-acts.mean(axis=0)
        labels = np.array(read_csvs([f"data/{topic}.csv"])['label'].tolist())
        pol = np.full((acts.shape[0],), -1 if topic.startswith("neg_") else 1)

        trainset_size = int(len(acts)*TRAINSET_RATIO)
        np.random.RandomState(seed).shuffle(acts)
        np.random.RandomState(seed).shuffle(labels)
        np.random.RandomState(seed).shuffle(pol)
        train_activations.append(acts[:trainset_size])
        train_activations_centered.append(acts_centered[:trainset_size])
        train_labels.append(labels[:trainset_size])
        train_polarities.append(pol[:trainset_size])
        dev_activations.append(acts[trainset_size:])
        dev_activations_centered.append(acts_centered[trainset_size:])
        dev_labels.append(labels[trainset_size:])
    train_activations = np.concatenate(train_activations)
    train_activations_centered = np.concatenate(train_activations_centered)
    train_polarities = np.concatenate(train_polarities)
    train_labels = np.concatenate(train_labels)
    dev_activations = np.concatenate(dev_activations)
    dev_activations_centered = np.concatenate(dev_activations_centered)
    dev_labels = np.concatenate(dev_labels)
    return train_activations, train_activations_centered, train_polarities, train_labels, \
        dev_activations, dev_activations_centered, dev_labels


def main(
    seed: int,
    model_name: str,
    probe_save_dir: str,
    prompt_option: str = "no_prompt",
):
    if model_name not in MODEL_NAMES_LAYER_INDICES:
        raise ValueError(f"Unindexed model: {model_name}")
    layer_index = MODEL_NAMES_LAYER_INDICES[model_name]
    train_activations, train_activations_centered, train_polarities, train_labels, \
        dev_activations, dev_activations_centered, dev_labels = prepare_data(
        seed=seed,
        model_name=model_name,
        prompt_option=prompt_option,
        layer_index=layer_index,
    )
    logger.info("Loaded activations.")

    logger.info("TTPD start")
    ttpd = TTPD.from_data(train_activations_centered, train_activations, train_labels, train_polarities, seed)
    logger.info("TTPD finish")
    
    logger.info("LR start")
    lr = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=seed)
    lr.fit(train_activations, train_labels)
    logger.info("LR finish")
    
    logger.info("MLP start")
    mlp = MLPClassifier(hidden_layer_sizes=(512, 128, 64), solver='adam', activation='tanh', learning_rate='constant', random_state=seed)
    mlp.fit(train_activations, train_labels)
    logger.info("MLP finish")
    
    logger.info("SVM start")
    svm = NuSVC(kernel='linear', nu=0.5, probability=True, gamma='auto', random_state=seed)
    svm.fit(train_activations, train_labels)
    logger.info("SVM finish")
    
    mm = MMProbe(train_activations, train_labels)
    
    logger.info("Dev set accuracy:")
    dev_accs = []
    probe_names =  ("lr", "mlp", "svm", "mm", "ttpd")
    for name in probe_names:
        probe = locals()[name]
        dev_accs.append([name, accuracy(probe, dev_activations_centered, dev_labels), accuracy(probe, dev_activations, dev_labels)])
    print(tabulate.tabulate(dev_accs, headers=("Probe", "Centered acc", "Acc"), floatfmt='.4f'))
    
    save_dir = os.path.join(probe_save_dir, model_name, f"seed={seed}")
    os.makedirs(save_dir, exist_ok=True)
    for name in probe_names:
        probe = locals()[name]
        joblib.dump(probe, os.path.join(save_dir, f"{name}.joblib"))
        logger.info(f"Saved {name}.")


if __name__ == "__main__":
    fire.Fire(main)
