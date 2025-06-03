import pandas as pd
import joblib
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def predict(model_name, input_val_filename, output_val_filename):
    loaded_model = joblib.load(model_name)

    df = pd.read_csv(input_val_filename)

    X_val = df.drop(columns=['Status'])
    y_val = df['Status']

    y_pred = loaded_model.predict(X_val)
    y_proba = loaded_model.predict_proba(X_val)[:, 1]

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec  = recall_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred)
    roc  = roc_auc_score(y_val, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")

    pred_df = df.copy()
    pred_df['StatusPred'] = y_pred
    pred_df.to_csv(output_val_filename, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <model_name> <input_val_file> <output_res_file>")
        sys.exit(1)

    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    predict(model_name, input_file, output_file)
