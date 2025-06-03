# Project Description
This is a project for training and evaluating models for loan default prediction task.

# Prerequisites
- Python >= 3.10
- requirements.txt (```pip install -r requirements.txt```)

# Project structure
- ./data folder: contains csv data (Loan_Default.csv)
- ./models folder: contains saved models
- ./pipeline folder: contains python scripts:
    - data_split.py:
        splits data into 90%/10% split of train and validation data.
        Example of usage:  
        ```python pipeline/data_split.py ./data/Loan_Default.csv ./data/train.csv ./data/new_input.csv```
    - train.py: trains Naive Bayes model. Example of usage:
        ```python pipeline/train.py ./data/train.csv ./models/model.pkl```
    - predict.py: Predicts the defaulting of loans. Example of usage: ```python pipeline/predict.py ./models/model.pkl ./data/new_input.csv ./data/datapred.csv```

- ./src folder: contains python notebooks:
    - EDA.ipynb: lab1
    - FE.ipynb: lab2
    - Modeling.ipynb: lab3
    - Hyperparams.ipynb: lab4
