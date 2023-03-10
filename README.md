# QA_MODEL_TASK

# DESCRIPTION

This project implements a question answering model based on a spin-off of the coattention encoder model. 

# PREREQUISITES

The training data come from the SQuAD dataset found here: https://rajpurkar.github.io/SQuAD-explorer/

The fasttext embeddings can be downloaded via https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz Unzipped before use.


# HOW TO RUN

python --train-path=train_path.json --dev-path=dev_path.json --model=model_name_of_choice --model_save_path=model_save_path

The model can be one of fasttext, big_bert or smooth_bert

example: python --train-path=squad.json --dev_path=squad_dev.json --model=big_bert --model_save_path=models/new_model
