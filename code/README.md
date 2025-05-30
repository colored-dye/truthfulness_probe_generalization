# Truthfulness probes

Implementation of the TTPD probe is taken from https://github.com/sciai-lab/Truth_is_Universal, and mass-mean probe is taken from https://github.com/saprmarks/geometry-of-truth.

## Layout

```
.
├── activations_and_labels          -- Model internal activations (no labels)
├── data                            -- Text format datasets and **binary truth labels**
├── figures_calibration             -- Calibration graphs
├── figures_distribution            -- Output distribution histogram plots
├── figures_heatmap                 -- Heatmap figures
├── figures_qa                      -- Barplots for QA tasks
├── figures_relative_variance       -- Separation between true/false statements across layers
├── generate_activations.py         -- Gather activations from an LLM
├── main.ipynb                      -- Flexible experiments with truthfulness probes & visualization
├── neg_generalization_results      -- AUROC of probes trained on affirmative statements and tested on negative ones
├── plots.ipynb                     -- Plotting
├── probes                          -- Probes (*.joblib) stored on disk
├── probes.py                       -- Probe definitions
├── train.py                        -- Script to train probes
├── train.sh                        -- Bash script to run `train.py`
└── utils.py                        -- Miscellaneous utility functions
```


## Environment
CUDA 12.7

Python 3.10.14. package dependencies in `requirements.txt`.


## Prepare data
Extract datasets to `data/xxx.csv` from `data.zip`. Each record has two fields: `statement` and `label`.

For the TriviaQA dataset, we need to sample answers from an LLM, using code in `data/triviaqa/`. We present our sampled answers in `data.zip`. The MMLU, SciQ, BoolQ and XSum datasets do not require generation. The sampled questions are also in `data.zip`.


## Gather activations
Manually change model paths and datasets in `generate_activations.py`. Dataset names are the base names in `data/`.
```sh
python generate_activations.py
```


## Train probes
Modify `train.sh` to specify target models and seeds.
```sh
bash train.sh
```


## Test truthfulness probes as you want
In the first section of `main.ipynb`, you may use the probes stored on disk or train probes on your designated datasets. The test sets are also free to choose as long as their activations are ready.

Visualizations such as ROC plots, PRC (precision-recall curve) plots, output probability distribution plots and calibration graphs are all available.


## Generalization across negation
In `main.ipynb`, go to the "Batch-run affirmative -> negation generalization" section. The code automatically generates results under a specific seed.

Results are visualized in `plots.ipynb`.

## Generalization across logical conjunctions/disjunctions
Refer to the section in `plots.ipynb`.

## Generalization on (few-shot) QAs
Refer to the section for MMLU and TriviaQA in `plots.ipynb`, respectively.

## Generalization on (few-shot) in-context generation tasks
Refer to the relevant section in `plots.ipynb`.
