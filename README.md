# QA_MODEL_TASK

# DESCRIPTION

This project implements a question answering model based on a spin-off of the coattention encoder model. 

# PREREQUISITES

The training data come from the SQuAD dataset found here: https://rajpurkar.github.io/SQuAD-explorer/

The fasttext embeddings can be downloaded via https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz Unzipped before use.

# DEPENDECIES

numpy, pandas, tensorflow, tensorflow-text, imblearn, scikit-learn, json


# HOW TO RUN

python --action=train_run_or_train_run --train-path=filepath_to_train_set --dev-path=filepath_to_dev_set --model=model_name_of_choice 
--model_save_path=path_to_model_file

The model can be one of fasttext, big_bert or smooth_bert

The action parameter can be either train, run or train_run based on the necessary task

If the action parameter is train the model is trained and saved on the train path.

If the action parameter is run, the model is loaded from the train path and evaluated on the dev set, evaluating the model and saving the model predictions

If the action parameter is train_run, both actions are performed in succession

The model predictions are saved at "./fasttext_coattention_preds.npy", "./bert_coattention_preds.npy" or "./small_bert_coattention_preds.npy" depending on the model.

example: python --action=train_run --train-path=squad.json --dev_path=squad_dev.json --model=big_bert --model_save_path=models/new_model

defaults to train_run action, ./squad.json trainset, squad_dev.json devset, smooth_bert model and q_a_model path
