import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import joblib
import sys


def train(input_train_file, output_model_file):
    pipeline = Pipeline([
        ('model', GaussianNB(var_smoothing=1e-12))
    ])

    df = pd.read_csv(input_train_file)

    X_train = df.drop(columns=['Status'])
    y_train = df['Status']
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, output_model_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_train_file> <output_model_file>")
        sys.exit(1)

    input_train_file = sys.argv[1]
    output_model_file = sys.argv[2]
    train(input_train_file, output_model_file)
