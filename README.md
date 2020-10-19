# Transparent Sexism Classficiation [CODE]

## Pre
python version 3.6.11 <br>
`pip install -r requirements.txt`

## Train models
`python _training/train_models.py` <br>
result: pickle files of the models will be saved to "_trained_models/" <br>
(fyi: fast-bert is not in the repo due to a size of 418 MB. fast-bert was pretrained in google colab)

## Generate explanations
`python save_combinations.py`
result: explanations will be saved to "_explanations/unsex"

## Print example explanations
`python _explanations/print_explanations.py`

