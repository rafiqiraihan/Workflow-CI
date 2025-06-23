# File: MLProject/modelling.py
import argparse
import pandas as pd
import mlflow
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Ambil parameter dari MLflow Project
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--subsample", type=float, default=0.6)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--scale_pos_weight", type=float, default=2.77)
    parser.add_argument("--dataset", type=str, default="telco_preprocessing.csv")
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.dataset)
    X = data.drop("is_churn", axis=1)
    y = data["is_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            scale_pos_weight=args.scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

if __name__ == "__main__":
    main()
