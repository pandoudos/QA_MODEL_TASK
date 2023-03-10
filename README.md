# QA_MODEL_TASK

# DESCRIPTION

This is an attempt to make a running question answering model from scratch in a week. The model implemented is a spin-off on the coattention encoder model. 

# PREREQUISITES

The data used were the data from the SQuAD dataset found here: https://rajpurkar.github.io/SQuAD-explorer/

the fasttext embeddings can be downloaded through wget from here https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz and must be unzipped to use in code


# HOW TO RUN

python --train-path=train_path.json --dev-path=dev_path.json --model=model_name_of_choice --model_save_path=model_save_path

the model names of choice can be either fasttext big_bert or smooth_bert
